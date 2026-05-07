
> 这篇博客是 **vLLM 源码解析系列第 2 篇**，主题是：  
> **vLLM 的 Scheduler 如何决定每一轮推理应该调度哪些请求。**

上一篇主要讲 vLLM 的整体架构，这一篇开始深入到核心模块：

```text
LLMEngine
  ↓
Scheduler
  ↓
BlockManager
  ↓
Worker / ModelRunner
```

其中这篇重点讲的是：

> **Scheduler 如何根据 waiting / running / swapped 三个队列，以及 GPU KV Cache block 的使用情况，决定本轮推理是做 prefill 还是 decode。**

---

# 1. 这篇博客主要讲什么？

这篇文章从 vLLM 离线批处理入口开始，逐步讲到调度器的核心逻辑。

最核心的两个入口函数是：

```python
LLMEngine.add_request()
LLMEngine.step()
```

可以先这样理解：

```text
add_request()
    负责把用户请求包装成 SequenceGroup
    然后放入 Scheduler 的 waiting 队列

step()
    负责执行一次调度
    决定本轮哪些 SequenceGroup 能送进模型推理
```

也就是说：

```text
add_request：负责“把请求放进系统”
step：负责“每一轮选谁执行”
```

这篇文章其实就是围绕一个问题展开：

> **vLLM 在每一轮推理中，到底是怎么决定哪些请求可以被送到 GPU 上执行的？**

---

# 2. 先建立整体调用链

从用户调用 vLLM 开始，整体流程大概是：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)
```

表面上看，用户只是调用：

```python
llm.generate(prompts, sampling_params)
```

但内部会进入：

```text
LLM.generate()
  ↓
LLMEngine.add_request()
  ↓
Scheduler.add_seq_group()
  ↓
LLMEngine.step()
  ↓
Scheduler.schedule()
  ↓
Worker / ModelRunner 执行模型推理
```

整体执行图可以这样记：

```text
用户输入 prompts
        │
        ▼
LLM.generate()
        │
        ▼
为每个 prompt 创建 request_id
        │
        ▼
LLMEngine.add_request()
        │
        ▼
把 prompt 包装成 SequenceGroup
        │
        ▼
Scheduler waiting 队列
        │
        ▼
LLMEngine.step()
        │
        ▼
Scheduler 决定本轮调度哪些 seq_group
        │
        ▼
Worker / ModelRunner 执行推理
        │
        ▼
返回 RequestOutput
```

---

# 3. 为什么需要 SequenceGroup？

## 3.1 原始输入不一定是一对一

一般推理时，我们会觉得：

```text
1 个 prompt -> 1 个 output
```

但在 vLLM 中，情况可能更复杂。

例如 **Parallel Sampling**：

```python
SamplingParams(n=4)
```

表示：

```text
1 个 prompt -> 生成 4 个 output
```

再比如 **Beam Search**：

```python
SamplingParams(n=3, best_of=3, use_beam_search=True)
```

表示：

```text
1 个 prompt -> 多条候选序列共同搜索
```

所以 vLLM 不能简单地把一个 prompt 当成一个普通字符串处理，而是要把它包装成更统一的结构。

这个结构就是：

```python
SequenceGroup
```

---

## 3.2 Sequence 和 SequenceGroup 的关系

可以这样理解：

```text
SequenceGroup：
    管理同一个 prompt 下产生的一组 Sequence

Sequence：
    一条具体的生成路径
```

例如：

```text
prompt = "What is the meaning of life?"
SamplingParams(n=4)
```

推理前：

```text
SequenceGroup
  └── seq0：prompt，状态 WAITING
```

prefill 后，可能变成：

```text
SequenceGroup
  ├── seq0：output 路径 1，状态 RUNNING
  ├── seq1：output 路径 2，状态 RUNNING
  ├── seq2：output 路径 3，状态 RUNNING
  └── seq3：output 路径 4，状态 RUNNING
```

所以：

> **SequenceGroup 是为了统一管理“一个 prompt 可能对应多个输出序列”的情况。**

---

## 3.3 Sequence 的几种状态

文章中提到，Sequence 会有多种状态，其中和调度最相关的是这几个：

| 状态 | 含义 |
|---|---|
| `WAITING` | 还没做 prefill，正在 waiting 队列中等着 |
| `RUNNING` | 已经开始推理，正在 running 队列中 |
| `SWAPPED` | 因为 GPU 资源不足，被换出到 CPU，等待 swap in |
| `FINISHED_STOPPED` | 正常结束，比如遇到停止符 |
| `FINISHED_LENGTH_CAPPED` | 达到最大长度限制而结束 |
| `FINISHED_ABORTED` | 异常终止，比如客户端断开 |
| `FINISHED_IGNORED` | prompt 太长，无法处理，被忽略 |

最重要的是前三个：

```text
WAITING
RUNNING
SWAPPED
```

它们正好对应 Scheduler 里的三个核心队列。

---

# 4. Scheduler 的三个核心队列

Scheduler 维护了三个重要队列：

```python
self.waiting
self.running
self.swapped
```

它们的含义是：

| 队列 | 存放内容 | 当前阶段 |
|---|---|---|
| `waiting` | 还没做过 prefill 的 seq_group | 等待 prefill |
| `running` | 已经进入推理流程的 seq_group | 正在 decode |
| `swapped` | 被抢占并换出到 CPU 的 seq_group | 等待 swap in |

可以这样记：

```text
waiting：
    新来的请求，还没开始算 prompt

running：
    已经算过 prompt，正在逐 token 生成

swapped：
    原来在 running，但 GPU KV Cache 不够，被换到 CPU 了
```

整体状态流转图：

```text
新请求
  ↓
waiting 队列
  ↓ prefill 成功
running 队列
  ↓ decode 过程中 GPU block 不够
swapped 队列
  ↓ GPU block 够了，swap in
running 队列
  ↓ 生成结束
finished
```

---

# 5. Scheduler 和 BlockManager 的关系

这篇文章虽然重点是 Scheduler，但它一直会和 BlockManager 交互。

因为 Scheduler 做调度时必须知道：

```text
GPU 上还有没有足够的 KV Cache block？
```

所以它需要调用 BlockManager。

整体关系是：

```text
Scheduler：
    决定本轮哪些 seq_group 可以推理

BlockManager：
    判断有没有 KV Cache 空间
    负责分配 / 释放 / swap KV Cache block

Worker / ModelRunner：
    真正执行模型 forward
```

也就是说：

```text
Scheduler 是“调度决策者”
BlockManager 是“KV Cache 空间管理员”
Worker 是“真正干活的执行者”
```

---

# 6. Scheduler 一次调度到底在做什么？

文章中有一个非常关键的定义：

> **一次 step 表示一个推理阶段。**

在 vLLM 里：

```text
prefill 算一次 step
每一次 decode 也算一次 step
```

但注意一个非常重要的规则：

> **在一次 step 中，所有被调度的 seq_group 要么全部做 prefill，要么全部做 decode。**

也就是说，不会在同一轮里一部分请求做 prefill，另一部分请求做 decode。

可以这样记：

```text
一次 step 只能是两种模式之一：

模式 1：prefill step
    从 waiting 队列中选请求
    一次性处理 prompt

模式 2：decode step
    从 running / swapped 队列中选请求
    每个 seq 生成下一个 token
```

---

# 7. 整体调度策略：先看大图

Scheduler 每一轮的核心判断逻辑大概是：

```text
调用 Scheduler._schedule()
        │
        ▼
先看 swapped 队列是否为空
        │
        ├── swapped 为空
        │       │
        │       ▼
        │    尝试调度 waiting 队列
        │       │
        │       ├── 成功调度 waiting
        │       │       │
        │       │       ▼
        │       │    本轮做 prefill
        │       │
        │       └── 没有调度 waiting
        │               │
        │               ▼
        │            转去调度 running
        │
        └── swapped 非空
                │
                ▼
             不调度新的 waiting
             优先考虑 running / swapped
             本轮做 decode
```

文章里总结得很关键：

```text
如果 swapped 队列为空：
    可以尝试从 waiting 队列调度新请求做 prefill

如果 swapped 队列非空：
    说明之前有请求被抢占
    根据 FCFS 思路，应该优先让它们恢复
    所以不要继续调度新的 waiting 请求
```

所以核心原则是：

> **如果系统里已经有被抢占的请求，就不要急着接新请求，先尽量恢复旧请求。**

---

# 8. 为什么要区分 prefill 和 decode？

LLM 推理分为两个阶段：

```text
prefill：
    处理整个 prompt
    一次性计算 prompt 所有 token 的 KV Cache

decode：
    每次只生成一个新 token
    并追加这个 token 的 KV Cache
```

这两个阶段对资源的需求很不一样。

## 8.1 prefill 阶段

假设 prompt 有 1000 个 token。

那么 prefill 要一次性处理：

```text
1000 个 token
```

所以它需要的 KV Cache block 数量和 prompt 长度有关。

```text
prompt 越长
  ↓
需要的 block 越多
  ↓
prefill 越重
```

---

## 8.2 decode 阶段

decode 每次只生成一个 token。

如果一个 seq_group 中有 n 个正在运行的 seq，那么这一轮 decode 最多新产生：

```text
n 个 token
```

因此最坏情况下，每个 seq 需要新开一个 block。

```text
1 个 seq 最坏新增 1 个 block
n 个 seq 最坏新增 n 个 block
```

所以：

```text
prefill 判断：这个 prompt 一共需要多少 block
decode 判断：这个 seq_group 最坏还需要多少新 block
```

---

# 9. _passed_delay：为什么不马上调度 waiting？

## 9.1 问题背景

如果 waiting 队列里来了新请求，Scheduler 是否应该立刻调度它？

不一定。

因为一旦调度 waiting 队列，就意味着本轮要做 prefill。

而做 prefill 会打断 running 队列中正在 decode 的请求。

所以这里有一个平衡：

```text
不能完全不管新请求
否则 waiting 队列会越积越多

也不能每次都处理新请求
否则 running 中的旧请求迟迟不能继续生成
```

因此 vLLM 设计了：

```python
_passed_delay()
```

它用来判断：

> **现在是否已经到了可以调度 waiting 队列的时间。**

---

## 9.2 _passed_delay 的核心逻辑

相关变量：

```text
prev_prompt：
    上一次调度是否调度了 waiting 队列

prev_time：
    上一次调度发生的时间

last_prompt_latency：
    当前时间 - 上一次调度 waiting 的时间

delay_factor：
    用户配置的延迟因子
```

核心判断：

```python
passed_delay = (
    (now - earliest_arrival_time)
    > delay_factor * last_prompt_latency
    or not self.running
)
```

意思是：

```text
如果 waiting 中最早到达的请求已经等得足够久
    可以调度 waiting

或者当前 running 队列为空
    也可以调度 waiting
```

换句话说：

```text
如果旧请求还在 running：
    waiting 请求需要等一会儿，避免频繁打断 decode

如果没有旧请求在 running：
    waiting 请求可以立即调度
```

---

## 9.3 _passed_delay 流程图

```text
Scheduler 准备判断是否调度 waiting
        │
        ▼
调用 _passed_delay(now)
        │
        ▼
delay_factor 是否 > 0？
        │
        ├── 否
        │      │
        │      ▼
        │   直接返回 True
        │
        └── 是
               │
               ▼
        waiting 队列是否非空？
               │
               ├── 否：返回 True
               │
               └── 是
                    │
                    ▼
        找到 waiting 中最早到达的请求
                    │
                    ▼
        计算它已经等待了多久
                    │
                    ▼
        是否等待超过 delay_factor * last_prompt_latency？
                    │
                    ├── 是：可以调度 waiting
                    └── 否：
                         │
                         ▼
                  running 是否为空？
                         │
                         ├── 是：可以调度 waiting
                         └── 否：暂时不调度 waiting
```

---

# 10. can_allocate：prefill 前判断能不能分配 block

当 Scheduler 想从 waiting 队列中调度一个 seq_group 时，需要先判断：

```text
这个 seq_group 的 prompt 能不能放进 GPU KV Cache？
```

这时调用：

```python
block_manager.can_allocate(seq_group)
```

它会返回三种状态：

| 状态 | 含义 |
|---|---|
| `OK` | 现在可以分配 |
| `LATER` | 当前不够，但以后可能够 |
| `NEVER` | 这个请求太大，永远放不下 |

---

## 10.1 can_allocate 核心逻辑

简化版代码：

```python
seq = seq_group.get_seqs(status=WAITING)[0]

num_required_blocks = len(seq.logical_token_blocks)

num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
    return AllocStatus.NEVER

if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
    return AllocStatus.OK

return AllocStatus.LATER
```

含义是：

```text
1. 先计算这个 prompt 需要多少个 block

2. 如果 GPU 总 block 数都不够：
       返回 NEVER

3. 如果当前空闲 block 够：
       返回 OK

4. 否则：
       返回 LATER
```

---

## 10.2 NEVER 和 LATER 的区别

这两个状态都表示现在不能调度，但原因不同。

### NEVER

```text
这个请求太长
就算把 GPU 所有 block 都给它
也放不下
```

所以后续会把它标记为：

```python
FINISHED_IGNORED
```

也就是直接忽略，不再处理。

---

### LATER

```text
这个请求本身不是太长
只是当前 GPU block 被别的请求占着
暂时放不下
```

所以它继续留在 waiting 队列里，等后面有空间再调度。

---

## 10.3 can_allocate 流程图

```text
waiting 队列中取一个 seq_group
        │
        ▼
计算 prompt 需要多少 block
        │
        ▼
判断 GPU 总 block 是否足够
        │
        ├── 总量都不够
        │       │
        │       ▼
        │    AllocStatus.NEVER
        │    标记 FINISHED_IGNORED
        │
        └── 总量理论上够
                │
                ▼
        判断当前 free blocks 是否够
                │
                ├── 够
                │      │
                │      ▼
                │   AllocStatus.OK
                │   可以进入 prefill
                │
                └── 不够
                       │
                       ▼
                    AllocStatus.LATER
                    继续等待
```

---

# 11. can_append_slot：decode 前判断能不能继续生成

running 队列里的 seq_group 已经做过 prefill，现在要继续 decode。

decode 阶段每轮只生成一个 token，但一个 seq_group 里可能有多个 seq。

所以需要判断：

```text
当前 GPU 空闲 block 是否足够支持这个 seq_group 继续 decode？
```

这时调用：

```python
block_manager.can_append_slot(seq_group)
```

---

## 11.1 为什么 decode 阶段判断方式不同？

prefill 是处理整段 prompt：

```text
一个 prompt 可能需要很多 block
```

decode 是每个 seq 新增一个 token：

```text
一个 seq 最坏只需要新增一个 block
```

如果一个 seq_group 里有 n 个 running seq：

```text
最坏需要 n 个新 block
```

所以 `can_append_slot` 的保守判断就是：

```text
当前空闲 GPU block 数 >= running seq 数
```

---

## 11.2 can_append_slot 流程图

```text
从 running 队列取一个 seq_group
        │
        ▼
统计这个 seq_group 中还在 running 的 seq 数量 n
        │
        ▼
获取当前 GPU free block 数
        │
        ▼
free blocks >= n ?
        │
        ├── 是
        │      │
        │      ▼
        │   可以继续 decode
        │   后面 append_slot
        │
        └── 否
               │
               ▼
            GPU block 不够
            需要抢占别的 seq_group
```

---

# 12. allocate 和 append_slot：真正分配 block

当 Scheduler 判断资源足够后，才会调用真正的分配逻辑。

## 12.1 prefill 阶段：allocate

如果 `can_allocate` 返回 `OK`：

```python
block_manager.allocate(seq_group)
```

作用是：

```text
为 prompt 的所有逻辑块分配对应的物理块
建立逻辑块 -> 物理块的映射
```

大概流程：

```text
waiting seq_group
        │
        ▼
计算 prompt 需要的 block 数
        │
        ▼
从 gpu_allocator 中分配 physical blocks
        │
        ▼
写入 block_tables
        │
        ▼
seq_group 进入 running
```

---

## 12.2 decode 阶段：append_slot

如果 `can_append_slot` 判断可以继续 decode：

```python
block_manager.append_slot(seq)
```

作用是：

```text
为本轮新生成 token 的 KV Cache 准备位置
```

decode 阶段可能出现三种情况：

```text
1. 最后一个 block 还没满：
       直接追加 token

2. 最后一个 block 已经满：
       新分配一个 block

3. 最后一个 block 被多个 seq 共享：
       触发 copy-on-write
```

流程图：

```text
decode 生成新 token 前
        │
        ▼
调用 append_slot(seq)
        │
        ▼
检查最后一个 physical block
        │
        ├── 还有空位
        │       │
        │       ▼
        │    直接把新 token 的 KV Cache 放进去
        │
        ├── 已经满了
        │       │
        │       ▼
        │    分配新 physical block
        │
        └── 被多个 seq 共享
                │
                ▼
             copy-on-write
             分配新 block
             把旧内容复制过去
```

---

# 13. preempt：抢占策略

## 13.1 为什么需要抢占？

在 decode 阶段，running 队列里的请求会不断生成新 token。

这意味着：

```text
KV Cache 会不断增长
```

如果 GPU KV Cache block 不够了，就必须腾出空间。

这时 Scheduler 会选择一些 seq_group 进行抢占：

```python
_preempt(seq_group)
```

抢占的目的：

```text
释放 GPU KV Cache block
让更合适的请求继续执行
```

---

## 13.2 两种抢占方式

vLLM 有两种抢占方式：

| 抢占方式 | 适用场景 | 做法 |
|---|---|---|
| `RECOMPUTE` | seq_group 中最大并行 seq 数为 1 | 释放 GPU block，把请求放回 waiting，之后重新 prefill |
| `SWAP` | seq_group 中有多个 seq，比如 beam search | 把 GPU block 换到 CPU，之后再 swap in |

---

## 13.3 为什么单 seq 用 recompute？

如果一个 seq_group 只有一个 seq，那么它的 KV Cache 相对简单。

抢占时可以直接：

```text
释放 GPU block
重新放回 waiting 队列
以后重新 prefill
```

这叫：

```text
重计算 recomputation
```

原因是：

```text
单条 seq 重算成本相对低
不用占 CPU swap 空间
```

---

## 13.4 为什么多 seq 用 swap？

如果一个 seq_group 里有多个 seq，比如：

```text
parallel sampling
beam search
```

它们可能共享 prompt block，也可能已经分叉出多条输出路径。

如果直接丢掉 KV Cache，后续重算会很复杂、成本也更高。

所以 vLLM 选择：

```text
把 GPU 上的 KV Cache block 搬到 CPU
等 GPU 资源够了再搬回来
```

这叫：

```text
swap out / swap in
```

---

## 13.5 preempt 流程图

```text
decode 阶段发现 GPU block 不够
        │
        ▼
选择一个 seq_group 进行抢占
        │
        ▼
调用 _preempt(seq_group)
        │
        ▼
判断 seq_group.get_max_num_running_seqs()
        │
        ├── == 1
        │      │
        │      ▼
        │   RECOMPUTE
        │      │
        │      ├── 释放 GPU block
        │      ├── seq 状态改为 WAITING
        │      ├── 重置 computed tokens
        │      └── seq_group 放回 waiting 队首
        │
        └── > 1
               │
               ▼
            SWAP
               │
               ├── 检查 CPU swap 空间
               ├── GPU block -> CPU block
               ├── seq 状态改为 SWAPPED
               └── seq_group 放入 swapped 队列
```

---

# 14. _schedule：调度器核心代码主线

现在把前面的内容串起来。

Scheduler 的核心函数是：

```python
_schedule()
```

它会维护三类 block 迁移信息：

```python
blocks_to_swap_in = {}
blocks_to_swap_out = {}
blocks_to_copy = {}
```

含义是：

| 变量 | 含义 |
|---|---|
| `blocks_to_swap_in` | CPU block -> GPU block |
| `blocks_to_swap_out` | GPU block -> CPU block |
| `blocks_to_copy` | copy-on-write 产生的 block 拷贝 |

这些信息会被传给 Worker / CacheEngine，后面真正执行 KV Cache 的移动或复制。

---

## 14.1 主线一：优先尝试调度 waiting 做 prefill

条件：

```text
swapped 队列为空
并且 _passed_delay() 返回 True
```

流程：

```text
进入 _schedule()
        │
        ▼
swapped 队列是否为空？
        │
        ├── 否：跳过 waiting
        │
        └── 是
             │
             ▼
        _passed_delay(now) 是否通过？
             │
             ├── 否：跳过 waiting
             │
             └── 是
                  │
                  ▼
            尝试从 waiting 队列取 seq_group
                  │
                  ▼
            block_manager.can_allocate(seq_group)
                  │
                  ├── NEVER：标记 FINISHED_IGNORED
                  ├── LATER：停止调度 waiting
                  └── OK：
                        │
                        ▼
                  block_manager.allocate(seq_group)
                        │
                        ▼
                  seq_group 放入本轮 scheduled
                        │
                        ▼
                  本轮做 prefill
```

如果成功从 waiting 调度到了 seq_group，那么本轮就是：

```text
prefill step
```

---

## 14.2 主线二：调度 running 做 decode

如果本轮没有走 waiting prefill，或者 swapped 不为空，就进入 decode 调度。

流程：

```text
进入 running 调度
        │
        ▼
从 running 队列按顺序取 seq_group
        │
        ▼
判断 can_append_slot(seq_group)
        │
        ├── 可以
        │      │
        │      ▼
        │   append_slot
        │   加入本轮 decode batch
        │
        └── 不可以
               │
               ▼
            需要抢占
               │
               ▼
            _preempt(victim_seq_group)
               │
               ├── RECOMPUTE：放回 waiting 队首
               └── SWAP：放入 swapped 队列
```

这里有一个很关键的点：

> **running 队列中的请求不一定每一轮都能继续执行。**

因为它们继续 decode 也需要 KV Cache 空间，如果空间不够，就可能被抢占。

---

## 14.3 主线三：调度 swapped 做 swap in

如果本轮 running 调度过程中没有新发生抢占，并且 swapped 队列不为空，就可以尝试把之前被换出的请求换回来。

流程：

```text
running 调度结束
        │
        ▼
本轮是否发生新的 preemption？
        │
        ├── 是
        │      │
        │      ▼
        │   不调度 swapped
        │
        └── 否
               │
               ▼
        swapped 队列是否非空？
               │
               ├── 否：结束
               │
               └── 是
                    │
                    ▼
             尝试 can_swap_in(seq_group)
                    │
                    ├── 不可以：继续留在 swapped
                    └── 可以：
                         │
                         ▼
                    block_manager.swap_in(seq_group)
                         │
                         ▼
                    状态 SWAPPED -> RUNNING
                         │
                         ▼
                    加入本轮 decode batch
```

为什么如果本轮发生了新的抢占，就不调度 swapped？

因为：

```text
本轮已经因为 GPU 空间紧张而抢占了请求
说明资源压力仍然存在
此时再 swap in 旧请求风险更大
```

所以 vLLM 选择保守处理。

---

# 15. 整体执行流程图：一张图串起来

```text
LLMEngine.step()
        │
        ▼
Scheduler._schedule()
        │
        ▼
初始化 blocks_to_swap_in / swap_out / copy
        │
        ▼
判断 swapped 队列是否为空
        │
        ├── swapped 为空
        │       │
        │       ▼
        │    判断 _passed_delay()
        │       │
        │       ├── 通过
        │       │       │
        │       │       ▼
        │       │    尝试调度 waiting
        │       │       │
        │       │       ├── can_allocate = NEVER
        │       │       │       └── 标记 FINISHED_IGNORED
        │       │       │
        │       │       ├── can_allocate = LATER
        │       │       │       └── 停止调度 waiting
        │       │       │
        │       │       └── can_allocate = OK
        │       │               ├── allocate
        │       │               ├── 加入 scheduled
        │       │               └── 本轮做 prefill
        │       │
        │       └── 不通过
        │               │
        │               ▼
        │            进入 running 调度
        │
        └── swapped 非空
                │
                ▼
             跳过 waiting
             进入 running 调度

running 调度
        │
        ▼
遍历 running seq_group
        │
        ├── can_append_slot = True
        │       │
        │       ├── append_slot
        │       └── 加入本轮 decode
        │
        └── can_append_slot = False
                │
                ▼
             触发 preempt
                │
                ├── RECOMPUTE：释放 block，放回 waiting
                └── SWAP：GPU block -> CPU block，放入 swapped

running 调度结束
        │
        ▼
如果本轮没有新抢占，并且 swapped 非空
        │
        ▼
尝试调度 swapped
        │
        ├── can_swap_in = False
        │       └── 继续等待
        │
        └── can_swap_in = True
                ├── CPU block -> GPU block
                ├── 状态改为 RUNNING
                └── 加入本轮 decode
```

---

# 16. LLMEngine.step() 和 Scheduler._schedule() 的关系

Scheduler 只是决定：

```text
本轮哪些 seq_group 可以推理
以及需要做哪些 KV Cache block 操作
```

但它本身不执行模型 forward。

`LLMEngine.step()` 后续还会把 Scheduler 的输出交给 Worker / ModelRunner。

整体可以这样理解：

```text
LLMEngine.step()
        │
        ▼
scheduler_outputs = scheduler.schedule()
        │
        ▼
拿到本轮要执行的 seq_group
拿到 blocks_to_swap_in
拿到 blocks_to_swap_out
拿到 blocks_to_copy
        │
        ▼
交给 Worker / ModelRunner
        │
        ▼
CacheEngine 执行 KV Cache 搬运 / 拷贝
        │
        ▼
模型 forward
        │
        ▼
Sampler 采样新 token
        │
        ▼
更新 SequenceGroup 状态
```

所以：

```text
Scheduler：
    决定谁执行

CacheEngine：
    执行 KV Cache 搬运

ModelRunner：
    执行模型 forward

Sampler：
    采样下一个 token
```

---

# 17. 用一个例子串起来

假设当前有三个队列：

```text
waiting:
    A, B

running:
    C, D

swapped:
    空
```

## 17.1 第一步：swapped 为空

Scheduler 先看：

```text
swapped 是否为空？
```

现在是空的，所以可以考虑调度 waiting。

---

## 17.2 第二步：判断 _passed_delay

如果 `_passed_delay()` 返回 True：

```text
可以从 waiting 里调度新请求
```

于是检查 A。

---

## 17.3 第三步：can_allocate(A)

假设 A 的 prompt 需要 3 个 block。

当前 GPU free block 足够，于是：

```text
can_allocate(A) = OK
```

于是：

```text
allocate(A)
A 从 waiting 移到 scheduled
本轮进入 prefill
```

再检查 B。

如果 B 太长：

```text
can_allocate(B) = NEVER
```

那么：

```text
B 标记 FINISHED_IGNORED
```

最终本轮可能只调度 A 做 prefill。

---

## 17.4 下一轮：A 进入 running

prefill 后，A 进入 running。

现在队列可能是：

```text
waiting:
    空

running:
    C, D, A

swapped:
    空
```

下一次 step，如果没有 waiting 要调度，就进入 running decode。

---

## 17.5 decode 时发现 block 不够

假设正在调度 C、D、A。

C 可以继续 decode。

D 可以继续 decode。

轮到 A 时发现：

```text
can_append_slot(A) = False
```

说明 GPU block 不够了。

于是需要抢占某个 seq_group，比如 D。

如果 D 只有一个 running seq：

```text
D 被 RECOMPUTE
释放 GPU block
D 回到 waiting 队首
```

如果 D 有多个 running seq，比如 beam search：

```text
D 被 SWAP
GPU block -> CPU block
D 状态变成 SWAPPED
D 进入 swapped 队列
```

---

## 17.6 后续再 swap in

之后某一轮 GPU 空间够了，并且本轮没有新抢占。

Scheduler 会尝试：

```text
can_swap_in(D)
```

如果可以：

```text
CPU block -> GPU block
D 状态 SWAPPED -> RUNNING
D 继续 decode
```

这就是 Scheduler + BlockManager + CacheEngine 配合完成连续推理的过程。

---

# 18. 这篇博客和后面 BlockManager 的关系

这篇文章讲 Scheduler，但很多地方只讲到“调用 BlockManager”。

比如：

```python
can_allocate()
allocate()
can_append_slot()
append_slot()
swap_out()
swap_in()
free_seq()
```

这些函数真正的内部细节，是下一篇 BlockManager 文章讲的内容。

可以这样衔接：

```text
Scheduler 文章：
    讲什么时候调用 can_allocate / allocate / append_slot / swap_out / swap_in

BlockManager 文章：
    讲这些函数里面到底如何管理 physical block
```

也就是说：

```text
Scheduler：
    负责调度策略

BlockManager：
    负责 KV Cache block 的实际管理

二者关系：
    Scheduler 每做一个调度决策，都要问 BlockManager：
        “这个请求有没有地方放 KV Cache？”
```

---

# 19. 这篇博客的核心结论

## 19.1 Scheduler 的核心任务

Scheduler 的核心任务不是执行模型，而是：

```text
决定每一轮推理要执行哪些 seq_group
```

它要在这些因素之间做平衡：

```text
waiting 中的新请求
running 中正在生成的请求
swapped 中被抢占的请求
GPU KV Cache block 是否足够
CPU swap 空间是否足够
调度延迟阈值
FCFS 公平性
```

---

## 19.2 一次 step 只能是一种阶段

这是理解 vLLM Scheduler 的关键：

```text
一次 step 中：
    要么全部做 prefill
    要么全部做 decode
```

也就是说：

```text
prefill 和 decode 不混在同一轮调度里
```

---

## 19.3 waiting / running / swapped 是理解调度器的主线

这三个队列就是整篇文章的主线：

```text
waiting：
    新请求，等待 prefill

running：
    已经 prefill，正在 decode

swapped：
    被抢占，KV Cache 暂时放到 CPU
```

只要抓住这三个队列，就能看懂 Scheduler 大部分逻辑。

---

## 19.4 can_allocate 和 can_append_slot 的区别

| 函数 | 阶段 | 判断内容 |
|---|---|---|
| `can_allocate` | prefill | prompt 需要的 block 是否放得下 |
| `can_append_slot` | decode | 新生成 token 的 KV Cache slot 是否够 |

简单说：

```text
can_allocate：
    为整段 prompt 判断空间

can_append_slot：
    为下一轮 decode token 判断空间
```

---

## 19.5 抢占策略分为 recompute 和 swap

当 GPU block 不够时，vLLM 会抢占某些 seq_group。

```text
单 seq：
    用 recompute
    释放 GPU block
    以后重新 prefill

多 seq：
    用 swap
    GPU block 换到 CPU
    以后再 swap in
```

---

# 20. 最后给你一个记忆版总结

这篇博客可以这样记：

```text
LLMEngine.add_request()
    把请求变成 SequenceGroup
    放进 waiting 队列

LLMEngine.step()
    触发一次调度

Scheduler._schedule()
    决定本轮是 prefill 还是 decode
```

三个队列：

```text
waiting：
    还没 prefill 的新请求

running：
    正在 decode 的旧请求

swapped：
    被抢占，KV Cache 在 CPU 上的请求
```

两个关键判断：

```text
can_allocate：
    waiting -> running 前判断 prompt 能不能放下

can_append_slot：
    running 继续 decode 前判断新 token 有没有 KV Cache 空间
```

两种抢占：

```text
RECOMPUTE：
    单 seq，释放 block，回 waiting 队首，以后重算

SWAP：
    多 seq，GPU block 搬到 CPU，进入 swapped，以后 swap in
```

一句话总结：

> **vLLM 的 Scheduler 是一个围绕 waiting / running / swapped 三个队列运转的调度器。它每一轮 step 都会根据 KV Cache block 资源、调度延迟和抢占策略，决定本轮是从 waiting 中取请求做 prefill，还是从 running / swapped 中取请求做 decode。Scheduler 不真正执行模型，它只决定“谁能上车”；真正的 KV Cache 管理交给 BlockManager，真正的模型计算交给 Worker / ModelRunner。**