 **vLLM 整体怎么跑起来：调用入口 → LLMEngine → Scheduler → Worker → CacheEngine → 模型推理**。
# vLLM 源码解析 1：整体架构笔记

> 原文主题：从源码视角理解 vLLM 的整体架构  
> 核心问题：一个请求进入 vLLM 后，到底经过哪些模块，最后如何完成推理？

---

## 1. 这篇文章在讲什么？

这篇文章不是重点讲 PagedAttention 的数学原理，而是讲 vLLM 的整体工程架构。

vLLM 是一个高性能 LLM 推理框架，它的核心不是简单地“调用模型生成文本”，而是围绕高并发推理做了很多工程设计：

- 动态 batch
- Scheduler 调度
- KV Cache 物理块管理
- GPU/CPU 之间的 swap
- Worker 分布式执行
- CacheEngine 管理 KV Cache
- PagedAttention 执行实际 attention 计算

可以把 vLLM 理解成：

> vLLM = LLMEngine + Scheduler + Worker + CacheEngine + PagedAttention

其中最核心的是 **LLMEngine**。

---

## 2. vLLM 的两种调用方式

文章首先说，vLLM 对用户主要提供两种调用方式：

### 2.1 Offline Batched Inference：离线批处理

示例形式大概是：

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
````

表面看起来，这是一个普通的 batch 推理：

> 给一批 prompt，然后一起生成，最后一起返回。

但是文章强调：  
**vLLM 内部并不是死板的静态 batch，而是动态 batch。**

也就是说，虽然用户传进来的是一整个 batch，但 vLLM 内部会根据当前显存、KV Cache 物理块、请求状态动态决定：

- 哪些请求先进 running 队列
- 哪些请求暂时放在 waiting 队列
- 哪些请求生成结束后释放显存
- 新请求什么时候补进 running 队列

所以 Offline Batched Inference 表面是同步的，但底层仍然是动态调度。

---

### 2.2 API Server For Online Serving：在线服务

在线服务一般是这样启动：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf
```

然后客户端用 OpenAI 兼容 API 请求：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-2-7b-hf",
      "prompt": "San Francisco is a",
      "max_tokens": 7,
      "temperature": 0
  }'
```

在线服务的特点是：

- 请求是一条一条来的
- 不能等所有请求到齐再统一推理
- 谁先生成完，谁就可以先返回
- 如果是流式输出，可以边生成边返回

vLLM 在线服务底层使用的是 `AsyncLLMEngine`，它继承自 `LLMEngine`。所以理解 `LLMEngine`，就能理解离线批处理和在线服务的共同核心。

---

## 3. LLMEngine 是 vLLM 的核心

文章里非常关键的一句话是：

> LLMEngine 是 vLLM 的核心逻辑。

LLMEngine 最重要的两个函数是：

### 3.1 add_request()

作用：

> 把用户请求包装成 vLLM 内部能处理的数据结构，然后放进 Scheduler 的 waiting 队列。

也就是说，用户传进来的 prompt 不会直接扔给模型，而是会先被封装成内部对象，比如 `SequenceGroup`。

可以理解成：

```text
用户请求
  ↓
add_request()
  ↓
SequenceGroup
  ↓
Scheduler waiting 队列
```

---

### 3.2 step()

作用：

> 执行一次推理阶段。

这里的“一次推理阶段”包括：

- 一个 prefill 阶段
- 或者一个 decode 阶段

也就是说：

```text
prefill 算一次 step
每生成一个 token 的 decode 也算一次 step
```

在 `step()` 中，vLLM 会做几件事：

1. Scheduler 决定这次要处理哪些请求
2. Scheduler 给这些请求分配 KV Cache 物理块 ID
3. Worker 执行模型前向推理
4. 模型根据 block 信息使用 PagedAttention
5. 返回生成结果
6. 更新请求状态

所以可以粗略理解为：

```text
LLMEngine.step()
  ↓
Scheduler 调度
  ↓
分配 KV Cache block
  ↓
Worker 执行模型
  ↓
PagedAttention 读取/写入 KV Cache
  ↓
返回本轮输出
```

---

## 4. vLLM 整体架构

文章把 LLMEngine 分成两大部分：

```text
LLMEngine
├── Centralized Controller
└── Distributed Workers
```

---

## 5. Centralized Controller：中央控制器

Centralized Controller 本质上就是调度器所在的部分。

它运行在 CPU 上，主要负责：

- 管理请求队列
- 决定哪些请求进入当前推理阶段
- 给请求分配 KV Cache 物理块 ID
- 维护 waiting / running / swapped 等队列
- 管理 BlockSpaceManager

注意一个非常关键的点：

> Scheduler 分配的是 KV Cache 物理块的 ID，不是真正直接操作 GPU 显存。

真正管理 GPU/CPU 上 KV Cache 数据的是 Worker 里的 **CacheEngine**。

---

## 6. Scheduler 维护的几个队列

vLLM 调度过程中主要涉及三个队列：

```text
waiting 队列
running 队列
swapped 队列
```

### 6.1 waiting 队列

还没有开始推理的请求会先进入 waiting 队列。

例如 batch 太大，GPU 一次放不下，剩下的请求就先留在 waiting 里。

---

### 6.2 running 队列

当前正在 GPU 上参与推理的请求会放在 running 队列。

每次 `step()` 时，vLLM 会从 running 队列里取请求执行一次 prefill 或 decode。

---

### 6.3 swapped 队列

如果 GPU 上 KV Cache 空间不够，vLLM 会把一部分请求的 KV Cache 暂时换到 CPU 上。

这些被换出的请求就放在 swapped 队列。

等 GPU 显存重新充足后，再把它们换回 GPU，继续推理。

---

## 7. Distributed Workers：分布式执行部分

Distributed Workers 可以理解成真正干活的 GPU 执行层。

文章里说，可以粗略把每个 worker 理解成一块 GPU。

Worker 主要负责：

- 加载模型
- 执行模型前向推理
- 管理真实的 KV Cache 数据
- 调用 PagedAttention 相关逻辑

它里面有两个重要对象：

```text
Worker
├── CacheEngine
└── model_runner / model
```

---

## 8. CacheEngine 是干什么的？

CacheEngine 的作用是：

> 根据 Scheduler 分配的 block ID，真正管理 GPU/CPU 上的 KV Cache 物理块。

也就是说：

- Scheduler：决定分配哪些 block ID
- CacheEngine：根据这些 ID 管理真实的 KV Cache tensor

可以类比成操作系统：

```text
Scheduler / BlockManager：像虚拟内存页表管理器
CacheEngine：像真正操作物理内存的人
KV Cache block：像物理页
```

---

## 9. 模型加载与显存预分配

vLLM 在正式处理请求前，会先做两件事：

```text
1. 加载模型
2. 预分配 KV Cache 显存
```

---

## 10. 加载模型

加载模型比较直观：

> 把 base model 加载到 worker 上。

如果是单卡，就加载到一个 GPU worker 上。  
如果是多卡，vLLM 支持 TP / PP 等并行方式，把模型分布到多张卡上。

---

## 11. 预分配 KV Cache 显存

这是这篇文章里非常重要的一部分。

vLLM 不会等请求来了以后再一点点申请 KV Cache 显存，而是在初始化阶段提前估算：

> 当前 GPU 上到底能放多少个 KV Cache 物理块？

这个过程叫：

```text
profile_num_available_blocks
```

---

## 12. vLLM 如何估算可用 KV Cache 显存？

文章给出的核心公式是：

```text
分配给 KV Cache 的显存
= GPU 总显存 - 不使用 KV Cache 时做一次 FWD 的显存占用
```

也就是说，vLLM 会先模拟一次前向推理，看看模型权重和中间激活大概占多少显存。

然后剩下的显存就尽量分配给 KV Cache。

---

## 13. 怎么模拟一次“不使用 KV Cache 的前向推理”？

vLLM 会根据两个参数杜撰一批假数据：

```text
max_num_seqs
max_num_batched_tokens
```

含义分别是：

```text
max_num_seqs：
一个推理阶段最多处理多少条序列

max_num_batched_tokens：
一个推理阶段最多处理多少个 token
```

例如：

```text
max_num_batched_tokens = 10
max_num_seqs = 3
```

那么 vLLM 可以构造出 3 条假序列：

```text
seq1: 4 tokens
seq2: 3 tokens
seq3: 3 tokens
```

然后用这批假数据跑一次前向推理，测出模型在不考虑 KV Cache 预分配时大概占多少显存。

---

## 14. KV Cache 物理块大小怎么算？

文章给出的公式是：

```text
K_cache_block_size
= block_size * num_heads * head_size * num_layers * dtype_size
```

因为 KV Cache 里既有 K 又有 V，所以：

```text
V_cache_block_size = K_cache_block_size
```

最终一个 KV Cache 物理块大小是：

```text
cache_block_size
= block_size * num_heads * head_size * num_layers * dtype_size * 2
```

---

## 15. 公式里的每一项是什么意思？

### 15.1 block_size

一个 KV Cache block 中能放多少个 token。

vLLM 默认常见值是：

```text
block_size = 16
```

意思是：

> 一个物理块可以存 16 个 token 的 KV Cache。

---

### 15.2 num_heads

注意这里更准确地说，应该关注的是 **KV heads 的数量**。

普通 MHA 中：

```text
num_q_heads = num_kv_heads
```

但如果是 GQA / MQA：

```text
num_kv_heads < num_q_heads
```

所以在 GQA 模型中，KV Cache 的大小应该按照 KV head 数量算，而不是 Q head 数量算。

你之前问过：

> 一个 token 不是只有一组 K/V 吗？为什么还要乘 num_heads？

原因是：  
一个 token 在每一层里，不是只有一个 K 向量和一个 V 向量，而是每个 KV head 都有自己的 K/V 表示。

所以一个 token 的 K cache 形状大概可以理解成：

```text
[num_kv_heads, head_size]
```

一个 token 的 V cache 也是：

```text
[num_kv_heads, head_size]
```

所以必须乘：

```text
num_kv_heads * head_size
```

---

### 15.3 head_size

每个 attention head 的维度。

一般有：

```text
hidden_size = num_heads * head_size
```

例如：

```text
hidden_size = 4096
num_heads = 32
head_size = 128
```

意思是整个 hidden 向量 4096 维，被切成 32 个头，每个头 128 维。

---

### 15.4 num_layers

Transformer 层数。

因为每一层都要保存自己的 KV Cache。

所以如果模型有 32 层，那么每个 token 的 KV Cache 不是只存一份，而是每层都存一份。

---

### 15.5 dtype_size

每个元素占多少字节。

常见情况：

```text
fp16 / bf16: 2 bytes
fp32: 4 bytes
int8: 1 byte
```

如果 KV Cache 用 fp16，那么 `dtype_size = 2`。

---

### 15.6 为什么最后乘 2？

因为 KV Cache 同时存：

```text
K cache
V cache
```

所以最后要乘 2。

---

## 16. 用一句话理解 KV Cache block 大小

一个 KV Cache block 存的是：

> 若干个 token，在所有层、所有 KV head 上的 K 和 V 向量。

所以公式可以理解为：

```text
一个 block 的 token 数
× 每个 token 的 KV head 数
× 每个 head 的维度
× 层数
× 每个数占用的字节数
× K/V 两份
```

也就是：

```text
cache_block_size
= block_size
  * num_kv_heads
  * head_size
  * num_layers
  * dtype_size
  * 2
```

---

## 17. KV Cache block 数量怎么算？

文章里的逻辑是：

```text
总物理块数量
= 可用于 KV Cache 的显存大小 / 单个物理块大小
```

也就是：

```text
num_gpu_blocks
= available_kv_cache_memory / cache_block_size
```

如果 CPU swap 空间默认是 4GB，那么 CPU 上可用 block 数量就是：

```text
num_cpu_blocks
= cpu_swap_space / cache_block_size
```

---

## 18. 为什么 vLLM 初始化后显存占用很高？

因为 vLLM 会提前创建 KV Cache tensor，把 GPU 上的一大块显存预留出来。

文章中给出的代码形状大概是：

```python
kv_cache_shape = (
    2,
    num_blocks,
    block_size * num_kv_heads * head_size
)
```

然后每一层都会创建一个这样的 tensor：

```python
for _ in range(num_layers):
    kv_cache.append(torch.empty(kv_cache_shape, ...))
```

这里第一维的 `2` 表示：

```text
K 和 V
```

所以整体可以理解为：

```text
每一层都有一组 KV Cache
每组 KV Cache 包含 K 和 V
每个 K/V 由很多 block 组成
每个 block 能存 block_size 个 token 的 KV 向量
```

因此你看到 vLLM 一启动显存就很高，不一定是异常，而是因为：

> vLLM 预先把 KV Cache 空间占住了。

---

## 19. Scheduler 调度逻辑

当初始化完成后，请求进入 vLLM，Scheduler 开始工作。

整体流程可以理解为：

```text
请求进入
  ↓
add_request()
  ↓
waiting 队列
  ↓
Scheduler 选择一部分请求
  ↓
分配 KV Cache block
  ↓
进入 running 队列
  ↓
Worker 执行 prefill / decode
  ↓
生成结束则释放 block
  ↓
显存不足则部分请求 swap 到 CPU
```

---

## 20. Preemption：后来先抢占

文章提到 vLLM 调度里有一个策略：

```text
Preemption
```

可以理解为：

> 如果 GPU KV Cache 空间不够，就把后来的请求先抢占出去。

具体做法是：

1. 当前 running 队列里有多个请求
2. GPU 空间不够继续处理所有请求
3. vLLM 选择后来的请求
4. 把它的 KV Cache 从 GPU swap 到 CPU
5. 请求从 running 队列移动到 swapped 队列
6. 等 GPU 空间够了，再换回来继续推理

这就是为什么 vLLM 日志里可能会看到：

```text
running
waiting
swapped
```

这些状态。

---

## 21. 这篇文章的核心主线

可以总结成一句话：

> vLLM 的核心是 LLMEngine，LLMEngine 通过 Scheduler 动态选择请求，通过 BlockManager 分配 KV Cache block ID，通过 Worker 和 CacheEngine 在 GPU 上执行模型推理，并用 PagedAttention 高效读写 KV Cache。

---

## 22. 对初学者最重要的理解

如果你刚开始看 vLLM，不要一上来就钻源码细节。

可以先抓住这条链路：

```text
用户请求
  ↓
LLM / API Server
  ↓
LLMEngine.add_request()
  ↓
Scheduler waiting 队列
  ↓
LLMEngine.step()
  ↓
Scheduler 选择本轮请求
  ↓
BlockManager 分配 KV Cache block ID
  ↓
Worker 执行模型推理
  ↓
CacheEngine 管理真实 KV Cache
  ↓
PagedAttention 读取/写入 KV Cache
  ↓
返回生成结果
```

只要这条链路理解了，后面再看：

- Scheduler
- BlockManager
- CacheEngine
- PagedAttention
- Prefix Caching
- Continuous Batching

都会更清楚。

---

## 23. 和 PagedAttention 那篇文章的区别

上一篇 PagedAttention 原理文主要讲：

```text
为什么要把 KV Cache 分成 block？
逻辑块和物理块如何映射？
如何减少显存碎片？
如何支持共享和 copy-on-write？
```

而这篇源码整体架构文主要讲：

```text
vLLM 代码整体怎么组织？
请求如何进入系统？
Scheduler 如何调度？
Worker 如何执行？
KV Cache 显存如何预分配？
```

所以两篇关系是：

```text
PagedAttention 原理文：讲核心算法思想
vLLM 源码架构文：讲工程系统怎么把这个思想落地
```

---

## 24. 面试版总结

vLLM 的核心是 LLMEngine，它统一支撑离线批处理和在线服务。请求进入后会被封装成 SequenceGroup，并加入 Scheduler 的 waiting 队列。每次 step 时，Scheduler 根据当前显存和 KV Cache block 使用情况，选择部分请求进入 running 队列，并通过 BlockManager 分配 KV Cache 物理块 ID。实际的模型推理由 Worker 执行，Worker 中的 CacheEngine 负责管理 GPU/CPU 上真实的 KV Cache 数据。vLLM 在初始化阶段会通过模拟一次前向推理估算模型和中间变量显存占用，从而计算剩余显存能分配多少 KV Cache block，并提前预分配这些显存。推理过程中，如果 GPU KV Cache 不足，vLLM 会把部分请求的 KV Cache swap 到 CPU，等资源充足后再换回 GPU。通过动态 batching、PagedAttention 和 KV Cache block 管理，vLLM 能显著提升高并发场景下的大模型推理吞吐。

```

补一句：你之前问的“模拟不用 KV Cache 推理时占用的显存”就在这篇文章的 **3.2 预分配显存** 部分，逻辑就是先构造假输入跑一次 forward，测出模型权重 + 中间变量占用，然后用 GPU 总显存减掉这部分，剩下的尽量给 KV Cache 预分配。1
```