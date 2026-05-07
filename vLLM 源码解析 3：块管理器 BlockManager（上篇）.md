
---

# vLLM 源码解析 3：BlockManager 上篇笔记

## 1. 这篇博客主要讲什么？

这篇博客继续接着前面的 **Scheduler 调度器** 往下看。

前面你已经看过：

请求进入 vLLM
  ↓
Scheduler 决定哪些请求进入本轮推理
  ↓
但是调度前必须先判断：GPU 上还有没有 KV Cache 空间？

这个“判断和管理 KV Cache 空间”的组件就是：

BlockManager / BlockSpaceManager

它的核心职责可以概括为一句话：

> BlockManager 不负责真正计算 KV Cache，而是负责管理 KV Cache 应该放在哪些物理 block 里。



也就是说，它更像是一个 KV Cache 显存地址管理器。


---

## 2. 为什么需要 BlockManager？

LLM 推理时，每个 token 都会产生 KV Cache。

随着请求越来越多、序列越来越长，KV Cache 会不断增长。如果直接给每个请求分配一整块连续显存，会有几个问题：

1. 显存碎片严重


2. 请求长度动态变化，不好提前分配


3. beam search / prefix sharing 等场景会产生共享需求


4. GPU 显存不够时，需要把部分 KV Cache 换到 CPU



vLLM 的 PagedAttention 思路就是把 KV Cache 拆成固定大小的 block 来管理，类似操作系统的分页机制。vLLM 论文也明确说，PagedAttention 借鉴了操作系统中的虚拟内存和分页思想，用于减少 KV Cache 的碎片和重复拷贝。

所以 BlockManager 就是 PagedAttention 在代码层面的关键组件之一。


---

## 3. 先区分两个概念：逻辑 block 和物理 block

3.1 逻辑 block

逻辑 block 是从 Sequence 视角 看的。

假设：

block_size = 16
prompt token 数 = 35

那么这个 sequence 逻辑上需要：

ceil(35 / 16) = 3 个 block

可以理解为：

Sequence 认为自己需要 3 页 KV Cache 空间

早期版本中，Sequence 里会维护 logical_token_blocks，用来记录 prompt token 如何被切成逻辑块。相关资料中提到，Sequence 初始化时会通过 _append_tokens_to_blocks 把 token 放入 LogicalBlock，如果当前 block 满了，就创建新的 LogicalBlock。

不过博客里也强调，后续 vLLM 的实现中很多地方不再显式依赖 logical table，而是通过 seq.n_blocks 这类字段直接表示当前 sequence 需要多少个 block。


---

3.2 物理 block

物理 block 是从 真实内存资源 视角看的。

一个 PhysicalTokenBlock 大致包含：

device        # block 在 CPU 还是 GPU
block_number  # block 编号
block_size    # 每个 block 能放多少 token 的 KV Cache
ref_count     # 引用计数

其中 ref_count 很重要：

ref_count = 0：这个 block 空闲
ref_count = 1：只有一个 seq 使用它
ref_count > 1：多个 seq 共享它

例如 beam search 中，多个候选序列一开始共享同一个 prompt 的 KV Cache，所以这些 block 的 ref_count 可能大于 1。资料中也明确提到，PhysicalTokenBlock 的 ref_count 用来表示 block 被使用的次数，可以用于 beam search 等场景复用 cache。


---

## 4. BlockManager 的核心数据结构

4.1 BlockAllocator

BlockAllocator 是真正负责分配物理 block 的类。

博客里主要讲的是：

UncachedBlockAllocator

它不做 prefix caching，只做普通 block 分配和释放。

它内部维护一个空闲 block 列表：

self.free_blocks = []

初始化时，会提前创建好所有物理 block：

for i in range(num_blocks):
    block = PhysicalTokenBlock(...)
    self.free_blocks.append(block)

真正分配时：

block = self.free_blocks.pop()
block.ref_count = 1
return block

释放时：

block.ref_count -= 1
if block.ref_count == 0:
    self.free_blocks.append(block)

也就是说：

BlockAllocator = 物理 block 池

用的时候从池子里 pop 一个，不用了再放回去。转载内容中也展示了 UncachedBlockAllocator 初始化、allocate、free、get_num_free_blocks 的核心逻辑。


---

4.2 gpu_allocator 和 cpu_allocator

BlockManager 里面通常会有两个 allocator：

self.gpu_allocator = ...
self.cpu_allocator = ...

含义是：

gpu_allocator：管理 GPU 上的 KV Cache block
cpu_allocator：管理 CPU 上的 KV Cache block

GPU block 用于当前正在推理的请求。

CPU block 用于被抢占、被 swap out 的请求。比如 GPU 空间不够时，vLLM 可以把某些 running 请求的 KV Cache 从 GPU 换到 CPU，然后让更优先的请求继续推理。


---

4.3 block_tables

这是 BlockManager 里最关键的映射表：

self.block_tables: Dict[int, BlockTable] = {}

它的含义是：

seq_id -> 这个 seq 使用的物理 block 列表

例如：

block_tables = {
    101: [GPU block 7, GPU block 3, GPU block 19],
    102: [GPU block 7, GPU block 3, GPU block 25],
}

这里说明：

seq 101 和 seq 102 前两个 block 共享
第三个 block 不同

这很像操作系统里的页表：

虚拟页号 -> 物理页号

在 vLLM 里就是：

逻辑 block -> 物理 block

转载内容也明确说，block_tables 负责维护每个 seq 下的物理块列表，本质上是一个字典；因为 seq_id 全局唯一，所以可以记录调度系统中所有待推理 seq 的物理块。


---

## 5. BlockManager 和 Scheduler 的关系

整体关系可以这样理解：

Scheduler：
    决定请求能不能进入本轮推理

BlockManager：
    告诉 Scheduler：这个请求有没有足够的 KV Cache block

例如调度 prefill 阶段时：

waiting 队列中有一个 seq_group
  ↓
Scheduler 调用 block_manager.can_allocate(seq_group)
  ↓
如果返回 OK
  ↓
Scheduler 调用 block_manager.allocate(seq_group)
  ↓
这个 seq_group 进入 running 队列，开始 prefill

调度 decode 阶段时：

running 队列中的 seq_group 要继续生成下一个 token
  ↓
Scheduler 调用 block_manager.can_append_slots(seq_group)
  ↓
如果可以继续生成
  ↓
调用 block_manager.append_slots(seq)
  ↓
给新 token 的 KV Cache 找到存储位置

所以，Scheduler 是“调度决策者”，BlockManager 是“显存空间管理员”。


---

## 6. AllocStatus：三种分配状态

BlockManager 判断一个请求能不能分配 block 时，会返回三种状态：

class AllocStatus(enum.Enum):
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()

含义是：

状态	含义

OK	现在 GPU block 足够，可以分配
LATER	现在不够，但以后可能够，可以先等等
NEVER	这个请求太大，GPU 永远放不下


vLLM 官方文档中对 AllocStatus 的解释也是这三个状态：OK 表示现在可分配，LATER 表示当前不能分配但 allocator 容量大于需求，NEVER 表示 seq_group 太大，无法在 GPU 中分配。


---

## 7. 代码执行流程重点解析

下面是你说“代码部分没细看”的重点。

我按真实推理过程拆成几条主线。


---

## 主线一：prefill 阶段如何分配 block？

7.1 can_allocate：先判断能不能分配

入口大概是：

block_manager.can_allocate(seq_group)

它用于判断：

当前 waiting 状态的 seq_group 能不能进入 prefill

核心逻辑是：

num_required_blocks = 当前 seq_group 需要的 block 数
num_free_gpu_blocks = 当前 GPU 空闲 block 数

if num_total_gpu_blocks - num_required_blocks < watermark_blocks:
    return AllocStatus.NEVER

if num_free_gpu_blocks - num_required_blocks >= watermark_blocks:
    return AllocStatus.OK

return AllocStatus.LATER

对应含义：

情况 1：NEVER

num_total_gpu_blocks - num_required_blocks < watermark_blocks

意思是：

就算 GPU 上所有 block 都给你，也会低于安全水位线

也就是说，这个请求太长了，GPU 根本装不下。

所以返回：

AllocStatus.NEVER

这个请求后面也没必要继续等了。


---

情况 2：OK

num_free_gpu_blocks - num_required_blocks >= watermark_blocks

意思是：

当前空闲 block 足够给这个请求用，而且分配完之后还高于水位线

所以可以立刻分配。


---

情况 3：LATER

其他情况：

GPU 总容量理论上够，但当前空闲 block 不够

可能是因为现在 running 队列里有别的请求正在占用 GPU block。

所以先返回：

AllocStatus.LATER

让这个请求继续待在 waiting 队列。

博客转载中也解释了这个判断：先根据 prompt 长度计算所需 block 数，然后看 GPU 空闲 block 是否足够，同时通过 watermark_blocks 保留安全水位，避免频繁 cache eviction。


---

7.2 allocate：真正分配物理 block

当 can_allocate 返回 OK 后，Scheduler 才会调用：

block_manager.allocate(seq_group)

执行流程：

1. 取出 waiting 状态的 seq
2. 根据 seq.n_blocks 计算需要多少物理 block
3. 调用 _allocate_sequence
4. 从 gpu_allocator 里 pop 出对应数量的物理 block
5. 把 seq_id -> block_table 写入 self.block_tables

核心代码逻辑是：

seq = seq_group.get_seqs(status=WAITING)[0]

block_table = self._allocate_sequence(
    seq,
    seq_group.num_seqs(),
    is_encoder_decoder
)

for seq in seq_group.get_seqs(status=WAITING):
    self.block_tables[seq.seq_id] = block_table.copy()

这里有一个关键点：

seq_group.num_seqs()

为什么分配时要传这个？

因为一个 seq_group 里面可能有多个 sequence，例如 beam search。它们一开始共享相同 prompt，所以这些 prompt 对应的物理 block 可以共享。

因此 _allocate_sequence 里面会设置：

block.ref_count = ref_count

如果 ref_count = 4，表示这个 block 被 4 个 sequence 共享。


---

7.3 _allocate_sequence：从空闲 GPU block 池里取 block

核心逻辑：

num_prompt_blocks = seq.n_blocks
block_table = []

for logical_idx in range(num_prompt_blocks):
    block = self.gpu_allocator.allocate()
    block.ref_count = ref_count
    block_table.append(block)

return block_table

举个例子：

prompt 长度 = 35
block_size = 16
seq.n_blocks = 3

那么会分配 3 个 GPU 物理 block：

block_table = [GPU block 8, GPU block 2, GPU block 15]

然后记录：

self.block_tables[seq.seq_id] = block_table

所以 prefill 后，BlockManager 知道：

这个 seq 的 KV Cache 应该写到哪些 GPU block 里

注意：BlockManager 这里只是建立映射，不是真正把 KV Cache 写进去。真正写 KV Cache 是后面模型 forward / attention kernel 做的。

转载内容中也明确说，_allocate_sequence 会从空闲物理 blocks 中取出 num_prompt_blocks 个 block，并映射给当前 seq_group 中的 seq。


---

## 主线二：decode 阶段如何追加新 token？

prefill 完后，seq_group 会进入 running 队列。

decode 阶段每次生成一个新 token，这个 token 也要产生新的 KV Cache。

所以需要判断：

这个新 token 的 KV Cache 放在哪里？


---

8.1 can_append_slots：能不能继续 decode？

入口：

block_manager.can_append_slots(seq_group)

核心逻辑很简单：

num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
num_seqs = seq_group.num_seqs(status=RUNNING)

return num_seqs <= num_free_gpu_blocks

为什么是 num_seqs <= num_free_gpu_blocks？

因为 decode 阶段每个 running sequence 最坏情况下都可能需要新开一个 block。

例如：

block_size = 16
某个 seq 当前最后一个 block 已经满了
现在又生成一个 token

那就必须开一个新 block。

如果一个 seq_group 里有 4 个 running seq，最坏情况需要 4 个新 block。

所以只要：

空闲 GPU block 数 >= running seq 数

就保守地认为可以继续 decode。

博客转载中也专门对比了 can_allocate 和 can_append_slots：prefill 阶段需要根据 prompt 长度计算 block 数；decode 阶段每个 seq 一次最多新开 1 个 block，所以只要空闲 block 数不少于 running seq 数即可。


---

8.2 append_slots：给新 token 找位置

入口：

block_manager.append_slots(seq)

这个函数处理两种情况。


---

情况 1：当前 block_table 数量不够，需要新开 block

代码逻辑：

n_blocks = seq.n_blocks
block_table = self.block_tables[seq.seq_id]

if len(block_table) < n_blocks:
    new_block = self._allocate_last_physical_block(seq)
    block_table.append(new_block)
    return []

为什么会出现：

len(block_table) < n_blocks

因为 sequence 在生成新 token 后，逻辑上需要的 block 数可能增加了。

例如：

原来已有 16 个 token，刚好占满 1 个 block
现在生成第 17 个 token
逻辑上需要 2 个 block
但是物理 block_table 里还只有 1 个 block

所以要新增一个物理 block。


---

情况 2：最后一个 block 没被共享，可以直接写

代码逻辑：

last_block = block_table[-1]

if last_block.ref_count == 1:
    return []

含义：

最后一个物理 block 只有当前 seq 使用

那么新 token 的 KV Cache 可以直接追加到这个 block 里，不需要新分配 block，也不需要 copy。


---

情况 3：最后一个 block 被多个 seq 共享，触发 Copy-on-Write

代码逻辑：

else:
    new_block = self._allocate_last_physical_block(seq)
    block_table[-1] = new_block
    self.gpu_allocator.free(last_block)
    return [(last_block.block_number, new_block.block_number)]

这里很关键。

如果：

last_block.ref_count > 1

说明这个 block 被多个 seq 共享。

比如 beam search：

seq A 和 seq B 一开始共享 prompt block

但是 decode 之后，它们生成的新 token 可能不一样。

如果继续往同一个 block 里写，就会互相覆盖。

所以 vLLM 使用 Copy-on-Write：

读的时候共享
写的时候复制

流程是：

1. 发现 last_block 被多个 seq 共享
2. 给当前 seq 分配一个新的 block
3. 当前 seq 的 block_table 最后一个位置改成新 block
4. 原 block 的 ref_count - 1
5. 返回 old_block -> new_block 的映射

这个返回值：

[(old_block_number, new_block_number)]

是为了告诉后续执行模块：

需要把旧 block 里的内容复制到新 block

然后当前 seq 后续生成的新 KV Cache 才能安全写到新 block 里。

转载内容中也解释了这一点：当 last_block.ref_count > 1 时，说明有别的 seq 正在使用它，不能向同一个位置追加 KV Cache，因此触发 copy-on-write，分配新 block 替换旧 block，并释放旧 block 的一次引用。


---

## 主线三：GPU 显存不够时，如何 swap out？

当 GPU block 不够时，Scheduler 可能会把某些 running 请求抢占掉。

这时需要把它们的 KV Cache 从 GPU 转移到 CPU。

入口：

block_manager.swap_out(seq_group)


---

9.1 swap_out 的目标

GPU block -> CPU block

也就是：

把当前 seq_group 使用的 GPU 物理块换成 CPU 物理块

同时更新：

self.block_tables[seq.seq_id]

让它从：

seq_id -> [GPU block 1, GPU block 5, GPU block 9]

变成：

seq_id -> [CPU block 3, CPU block 8, CPU block 12]


---

9.2 swap_out 执行流程

核心逻辑：

mapping = {}

for seq in seq_group.get_seqs(status=RUNNING):
    self.block_tables[seq.seq_id] = self._swap_block_table(
        self.block_tables[seq.seq_id],
        self.gpu_allocator,
        self.cpu_allocator,
        mapping
    )

其中 _swap_block_table 做的事是：

遍历原来的 GPU block_table
  ↓
为每个 GPU block 分配一个 CPU block
  ↓
记录 GPU block -> CPU block 的映射
  ↓
释放原 GPU block
  ↓
返回新的 CPU block_table

注意 mapping 的作用：

mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock]

它是为了处理共享 block。

如果多个 seq 共享同一个 GPU block，那么 swap 到 CPU 时也应该共享同一个 CPU block，而不是重复拷贝多份。

流程大概是：

如果 from_block 已经在 mapping 里：
    直接复用对应的 to_block
    to_block.ref_count += 1

否则：
    新分配一个 CPU block
    mapping[from_block] = to_block

最后返回的是 block 编号映射，用于后续真正执行 KV Cache 数据拷贝：

return [(cpu_block.block_number, gpu_block.block_number)
        for cpu_block, gpu_block in mapping.items()]

虽然这里变量名看着有点绕，但核心意思就是：

告诉 worker：哪些 GPU block 的内容要搬到哪些 CPU block

转载内容中也说明，swap_out 会遍历当前 seq_group 中的 running seq，把它们的 block table 从 GPU allocator 转移到 CPU allocator，并释放 GPU block。


---

## 主线四：被 swap out 的请求如何 swap in？

当后面 GPU 空间又够了，被换到 CPU 的请求可以重新进入 running。

这时调用：

block_manager.can_swap_in(seq_group)
block_manager.swap_in(seq_group)


---

10.1 can_swap_in：判断能不能换回 GPU

它要判断：

这个 seq_group 从 CPU 换回 GPU 需要多少 block？
当前 GPU 空闲 block 够不够？

核心逻辑：

blocks = self._get_physical_blocks(seq_group)
num_swapped_seqs = seq_group.num_seqs(status=SWAPPED)
num_free_blocks = self.gpu_allocator.get_num_free_blocks()

num_required_blocks = len(blocks) + num_swapped_seqs

这里为什么要加：

num_swapped_seqs

因为 swap in 之后，这些 seq 很快还要继续 decode。

每个 seq 最坏情况下还需要额外新开一个 block 来存下一步生成 token 的 KV Cache。

所以它保守估计：

需要换回来的已有 block 数 + 每个 swapped seq 预留 1 个 block

然后判断：

if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
    return AllocStatus.NEVER

elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
    return AllocStatus.OK

else:
    return AllocStatus.LATER

转载内容中也解释了：len(blocks) 是已有 prompt + 已完成 output 的 KV Cache 所需 block 数，num_swapped_seqs 是为接下来 decode 预留的最坏情况 block 数。


---

10.2 swap_in：CPU block 换回 GPU block

swap_in 和 swap_out 很像，只是方向反过来：

CPU block -> GPU block

核心逻辑：

self._swap_block_table(
    self.block_tables[seq.seq_id],
    self.cpu_allocator,
    self.gpu_allocator,
    mapping
)

也就是：

从 CPU allocator 释放
从 GPU allocator 分配
更新 block_tables
返回 CPU block -> GPU block 的映射

博客转载中也指出，swap_in 和 swap_out 的代码结构几乎一样，只是 _swap_block_table 里 CPU/GPU allocator 的顺序反了。


---

## 11. 整体执行流程图

可以把博客里的代码流程浓缩成这样：

新请求进入 waiting 队列
        │
        ▼
Scheduler._schedule_prefills()
        │
        ▼
block_manager.can_allocate(seq_group)
        │
        ├── NEVER：请求太大，不能执行
        │
        ├── LATER：现在 GPU block 不够，继续等待
        │
        └── OK
              │
              ▼
        block_manager.allocate(seq_group)
              │
              ▼
        gpu_allocator.allocate()
              │
              ▼
        建立 block_tables:
        seq_id -> [physical_block_1, physical_block_2, ...]
              │
              ▼
        prefill forward，真正写入 KV Cache
              │
              ▼
        seq_group 进入 running 队列
              │
              ▼
Scheduler._schedule_running()
              │
              ▼
block_manager.can_append_slots(seq_group)
              │
              ├── false：GPU block 不够，可能触发 preemption / swap_out
              │
              └── true
                    │
                    ▼
            block_manager.append_slots(seq)
                    │
                    ├── last block 未满且未共享：直接追加
                    │
                    ├── block 不够：分配新 block
                    │
                    └── block 被共享：Copy-on-Write

如果 GPU block 不够：

running seq_group 被抢占
        │
        ▼
block_manager.swap_out(seq_group)
        │
        ▼
GPU block -> CPU block
        │
        ▼
seq_group 进入 swapped 队列
        │
        ▼
后续 GPU 空间够了
        │
        ▼
block_manager.can_swap_in(seq_group)
        │
        ▼
block_manager.swap_in(seq_group)
        │
        ▼
CPU block -> GPU block
        │
        ▼
seq_group 回到 running 队列


---

## 12. 用一个小例子串起来

假设：

block_size = 16
GPU 一共有 10 个 block
watermark_blocks = 1
请求 A 的 prompt 有 35 个 token

prefill 阶段

请求 A 需要：

ceil(35 / 16) = 3 个 block

can_allocate 判断：

空闲 GPU block = 10
需要 block = 3
分配后剩余 = 7
7 >= watermark 1

所以：

AllocStatus.OK

然后 allocate 分配 3 个物理 block：

A -> [GPU block 9, GPU block 8, GPU block 7]

写入：

block_tables[A.seq_id] = [9, 8, 7]


---

decode 阶段

A 继续生成 token。

如果第 36 个 token 还能放在第 3 个 block 中：

不需要新 block

如果生成到第 49 个 token：

前 48 个 token 刚好占满 3 个 block
第 49 个 token 需要第 4 个 block

于是 append_slots 会分配一个新 block：

A -> [9, 8, 7, 6]


---

beam search 场景

假设 A 分裂成两个 beam：

A1 -> [9, 8, 7]
A2 -> [9, 8, 7]

此时 block 9、8、7 的 ref_count = 2。

如果 A1 要继续写新 token，而且最后一个 block 7 被共享：

不能直接写 block 7
否则会影响 A2

所以触发 Copy-on-Write：

给 A1 分配新 block 6
把 block 7 的内容复制到 block 6
A1 -> [9, 8, 6]
A2 -> [9, 8, 7]
block 7.ref_count -= 1

这就是 vLLM 中共享 KV Cache 又避免互相污染的关键机制。


---

## 13. 这篇博客的核心结论

13.1 BlockManager 不做计算，只做管理

它不负责真正生成 KV Cache。

它只负责：

这个 seq 的 KV Cache 应该放在哪些 block 中

真正写入 KV Cache 的动作是在模型 forward 和 attention kernel 中完成的。

博客转载最后也强调：BlockManager 只是调度 seq.id 与 block 索引号映射、GPU/CPU block 间 KV Cache 转移等，并不是实际填充 KV Cache；实际填充发生在推理过程的 attention 计算中。


---

13.2 can_allocate 和 can_append_slots 的区别

方法	阶段	判断对象	核心逻辑

can_allocate	prefill	waiting seq_group	prompt 需要多少 block
can_append_slots	decode	running seq_group	每个 seq 最坏新开 1 个 block


简单说：

prefill：一次性处理整段 prompt，所以需要根据 prompt 长度算 block 数
decode：每次只生成一个 token，所以每个 seq 最坏多需要一个 block


---

13.3 block_tables 是理解源码的关键

只要抓住这个映射：

seq_id -> List[PhysicalTokenBlock]

很多代码就顺了。

例如：

allocate

就是创建这个映射。

append_slots

就是扩展或修改这个映射。

swap_out

就是把映射里的 GPU block 换成 CPU block。

swap_in

就是把映射里的 CPU block 换回 GPU block。

free

就是删除映射并释放对应 block。


---

## 14. 最后给你一个记忆版总结

这篇博客你可以这样记：

Scheduler 负责“选谁推理”
BlockManager 负责“有没有 KV Cache 空间”
BlockAllocator 负责“真正拿一个物理 block”
block_tables 负责“seq_id 到物理 block 的映射”

再进一步：

can_allocate：prefill 前判断 prompt 能不能放下
allocate：给 prompt 分配物理 block
can_append_slots：decode 前判断新 token 有没有位置
append_slots：给新 token 找 slot，必要时新开 block
copy-on-write：共享 block 要写入时，先复制再写
swap_out：GPU block 换到 CPU
swap_in：CPU block 换回 GPU

一句话总结：

> BlockManager 是 vLLM 中 KV Cache 的“页表 + 内存分配器 + swap 管理器”。它通过 block_tables 维护 seq 到物理 block 的映射，让 vLLM 能像操作系统分页一样高效管理动态增长的 KV Cache。