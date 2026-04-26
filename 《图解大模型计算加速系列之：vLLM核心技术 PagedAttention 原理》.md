
# vLLM 核心技术：PagedAttention 原理笔记

## 1. 背景：为什么大模型推理需要优化 KV Cache？

大模型推理分为两个阶段：

1. **Prefill 阶段**
   - 输入一整段 prompt。
   - 模型并行计算 prompt 中所有 token 的表示。
   - 会为每个 token 生成对应的 Key 和 Value，并存入 KV Cache。

2. **Decode 阶段**
   - 每次只生成一个新 token。
   - 当前 token 会拿自己的 Query 去和历史所有 token 的 Key 做 attention。
   - 历史 token 的 Key、Value 不需要重复计算，直接从 KV Cache 中读取。

所以，KV Cache 的作用是：

> 用显存换计算，避免每生成一个 token 都重新计算历史 token 的 K、V。

但是问题也很明显：

> 并发请求越多、上下文越长，KV Cache 占用显存越大。

vLLM 论文指出，传统系统在管理 KV Cache 时容易出现显存碎片和重复存储，限制 batch size，最终影响吞吐量。vLLM 提出的 PagedAttention 借鉴操作系统分页思想，目标是减少 KV Cache 显存浪费。1

---

## 2. 传统 KV Cache 管理的问题

传统推理框架通常会给每个请求分配一段连续显存来存 KV Cache。

可以类比成：

```text
请求 A: [--------------------]
请求 B: [--------------------]
请求 C: [--------------------]
````

这种方式的问题有两个：

### 2.1 内部碎片

生成长度是不确定的。

比如系统给某个请求预留了 2048 token 的 KV Cache 空间，但最后它只生成了 300 个 token。

那么剩下的空间就是浪费。

```text
实际使用: [######              ]
预留空间: [####################]
```

这叫 **内部碎片**。

---

### 2.2 外部碎片

因为传统方式要求 KV Cache 连续存放，所以哪怕总显存还有很多，只要找不到一整块连续空间，也可能无法接收新请求。

类似操作系统里的内存碎片问题：

```text
显存状态:
[已用][空闲][已用][空闲][已用][空闲]

虽然空闲总量够，但没有一整块连续空间。
```

这叫 **外部碎片**。

---

## 3. PagedAttention 的核心思想

PagedAttention 的核心思想是：

> 不再要求一个请求的 KV Cache 连续存放，而是把 KV Cache 切成固定大小的 block，分散存到显存中。

vLLM 官方文档里也说明，vLLM 会把 Key 和 Value cache 切分成 blocks，每个 block 存固定数量 token 的 KV 数据。

可以类比操作系统：

|操作系统|vLLM / PagedAttention|
|---|---|
|虚拟内存|逻辑上的 token 序列|
|物理内存|GPU 显存|
|Page / 页|KV Cache block|
|页表|Block Table|
|虚拟地址到物理地址映射|逻辑 block 到物理 block 映射|

---

## 4. Block Table：逻辑连续，物理不连续

对于模型来说，一个请求的上下文 token 是连续的：

```text
Token 0, Token 1, Token 2, Token 3, ...
```

但在显存中，它们对应的 KV Cache 不一定连续。

例如一个请求有 10 个 token，block size = 4：

```text
逻辑 block:
Block 0: token 0~3
Block 1: token 4~7
Block 2: token 8~9
```

实际显存中可能是：

```text
逻辑 Block 0 -> 物理 Block 7
逻辑 Block 1 -> 物理 Block 2
逻辑 Block 2 -> 物理 Block 9
```

中间靠 **Block Table** 记录映射关系：

```text
Block Table:
[0] -> Physical Block 7
[1] -> Physical Block 2
[2] -> Physical Block 9
```

这样做的好处是：

> 逻辑上看起来连续，物理上可以不连续。

这就和操作系统的虚拟内存分页非常像。

---

## 5. PagedAttention 怎么做 Attention？

普通 Attention 逻辑是：

```text
当前 Query 和历史所有 Key 做相似度计算
再用 softmax 得到权重
最后加权求和历史 Value
```

PagedAttention 并没有改变 Attention 的数学公式，只是改变了 KV Cache 的存储方式。

传统方式：

```text
从一整段连续 KV Cache 中读取 K、V
```

PagedAttention：

```text
根据 Block Table 找到多个物理 block
逐 block 读取 K、V
完成 attention 计算
```

也就是说：

> Attention 的计算结果不变，只是 K、V 的读取方式变了。

---

## 6. 为什么 PagedAttention 可以节省显存？

### 6.1 按需分配

传统方式可能要提前预留一大段连续显存。

PagedAttention 是随着 token 增长，逐 block 分配。

比如 block size = 16，一个请求当前只有 20 个 token，那么只需要 2 个 block：

```text
Block 0: 16 tokens
Block 1: 4 tokens 有效，剩余 12 token 空间浪费
```

最多只浪费最后一个 block 的部分空间。

所以内部碎片大幅减少。

---

### 6.2 不要求连续显存

每个 block 可以放在任意物理位置。

所以只要显存里还有空闲 block，就可以继续分配。

这减少了外部碎片问题。

---

### 6.3 支持更大的 batch size

KV Cache 显存利用率提高后，同一张 GPU 可以同时容纳更多请求。

所以 vLLM 的吞吐量提升并不是因为单次 attention 计算一定更快，而是因为：

> 显存利用率提高后，可以塞进更大的 batch，从系统整体上提高吞吐量。

vLLM 论文中提到，相同延迟水平下，vLLM 相比 FasterTransformer、Orca 等系统能取得约 2～4 倍吞吐提升。

---

## 7. PagedAttention 和操作系统分页的类比

可以这样理解：

### 传统 KV Cache

类似程序必须申请一大块连续物理内存：

```text
我要一整块连续空间。
如果没有连续空间，即使总空闲内存够，也申请失败。
```

### PagedAttention

类似操作系统分页：

```text
我不需要连续物理空间。
你给我多个小块就行。
我用页表记录它们的对应关系。
```

所以 PagedAttention 的本质是：

> 把 KV Cache 管理问题，从“连续数组管理”变成“分页内存管理”。

---

## 8. 共享前缀与 Copy-on-Write

PagedAttention 还有一个重要优势：方便共享 KV Cache。

比如多个请求有相同 prompt：

```text
请求 1: 请总结以下文章：xxx
请求 2: 请总结以下文章：xxx
请求 3: 请总结以下文章：xxx
```

它们前面的 token 是一样的，对应的 KV Cache 也一样。

传统方法可能会重复存三份。

PagedAttention 可以让多个请求共享同一批 block：

```text
Request A -> Block 1, Block 2
Request B -> Block 1, Block 2
Request C -> Block 1, Block 2
```

当某个请求后续生成不同 token 时，再给它分配新的 block。

如果共享 block 需要被修改，则使用类似操作系统的 **Copy-on-Write** 思想：

> 读的时候共享，写的时候复制。

这对于 beam search、多采样、prefix caching 等场景很有用。

---

## 9. PagedAttention 的收益总结

PagedAttention 主要解决的是 **KV Cache 显存管理问题**。

它的收益包括：

1. **减少内部碎片**
    
    - 不再为每个请求预留过大的连续空间。
    - 按 block 动态增长。
2. **减少外部碎片**
    
    - 物理 block 不需要连续。
    - 显存中零散的空闲 block 也能被利用。
3. **提升显存利用率**
    
    - 更多显存真正用于存有效 KV Cache。
4. **支持更大 batch size**
    
    - 并发请求更多。
    - GPU 利用率更高。
    - 整体吞吐量更高。
5. **方便共享 KV Cache**
    
    - 相同 prefix 可以共享 block。
    - beam search、多采样场景下更省显存。

---

## 10. 需要注意：PagedAttention 不是减少计算量

PagedAttention 不是像 FlashAttention 那样主要优化 attention 计算过程中的访存和 IO。

它的重点是：

> 优化 KV Cache 的显存组织方式。

也就是说：

|技术|主要优化点|
|---|---|
|FlashAttention|Attention 计算过程中的显存读写|
|PagedAttention|KV Cache 的显存管理|
|Continuous Batching|请求调度和 batch 组织|
|Prefix Caching|复用相同前缀的 KV Cache|

PagedAttention 可能会带来一点 block 查表和非连续访问的额外开销，但整体上因为 batch size 可以变大，吞吐量反而显著提升。

---

## 11. 一句话总结

PagedAttention 的核心可以概括为：

> 把每个请求的 KV Cache 切成固定大小的 block，通过 Block Table 管理逻辑 block 到物理 block 的映射，让 KV Cache 不必连续存放，从而减少显存碎片、提升显存利用率，并支持更大的并发 batch。

---

## 12. 面试表达版

如果面试官问：vLLM 的 PagedAttention 是什么？

可以这样回答：

> PagedAttention 是 vLLM 里用于高效管理 KV Cache 的技术。传统推理框架通常给每个请求分配连续显存来存 KV Cache，但由于生成长度不确定，会产生严重的内部碎片和外部碎片。PagedAttention 借鉴操作系统虚拟内存分页的思想，把 KV Cache 切成固定大小的 block，并通过 block table 维护逻辑 block 到物理 block 的映射。这样一个请求的 KV Cache 在逻辑上连续，但物理上可以分散存放，从而减少显存浪费，提高显存利用率，支持更大的 batch size，最终提升大模型推理吞吐量。

---

## 13. 和你之前学的 KV Cache 对起来理解

你可以把它们串起来：

1. **KV Cache**
    
    - 存历史 token 的 K、V。
    - 避免 decode 阶段重复计算历史 token。
2. **问题**
    
    - KV Cache 很大。
    - 请求长度不确定。
    - 并发请求多时显存碎片严重。
3. **PagedAttention**
    
    - 不改变 attention 数学公式。
    - 改变 KV Cache 的存储布局。
    - 用 block + block table 管理 KV Cache。
4. **效果**
    
    - 显存利用率更高。
    - batch 可以更大。
    - 推理服务吞吐量更高。



你可以重点记住这句：**FlashAttention 主要优化 Attention 计算过程，PagedAttention 主要优化 KV Cache 显存管理。**