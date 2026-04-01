# vector_store.py 面试题文档

## 1. 这个模块的核心职责是什么？

### 题目
请你介绍一下 `vector_store.py` 在整个 research memory 系统中的作用。

### 参考回答
`vector_store.py` 的核心职责是封装 Chroma 向量库，用来存储和检索 `MemoryCard` 的文本内容。  
在我的项目里，它不是主存储，而是**语义检索层**。真正的结构化归档仍然放在 SQLite 里，而 Chroma 负责根据 query 做相似度召回。

---

## 2. 为什么已经有 SQLite 了，还要再加一个向量库？

### 题目
既然已经有 `sqlite_store.py`，为什么还需要 `vector_store.py`？

### 参考回答
因为 SQLite 更适合做结构化持久化，比如存实体、分析记录、卡片元信息，但它不擅长做语义检索。  
而 research memory 的核心需求之一，是在用户追问时根据当前问题，从历史 card 中找出“语义最相关”的内容，所以需要引入向量库。  

因此我把系统拆成两层：

- SQLite：负责结构化归档
- Chroma：负责语义召回

---

## 3. `ChromaVectorStore` 初始化时做了什么？

### 题目
请讲一下 `ChromaVectorStore.__init__()` 的主要逻辑。

### 参考回答
初始化时主要做了几件事：

1. 规范化并创建本地持久化目录
2. 保存 collection 名称
3. 初始化状态位，比如 `available`、`_client`、`_collection`
4. 尝试导入 `chromadb`
5. 如果导入成功，就创建 `PersistentClient`
6. 再通过 `get_or_create_collection()` 获取或创建 collection

如果 `chromadb` 没安装，系统不会直接报错退出，而是把 `available` 保持为 `False`，后续自动降级成仅 SQLite 模式。

---

## 4. 这里是不是显式指定了 embedding 模型？

### 题目
你的 `vector_store.py` 里有没有手动指定 embedding 模型？

### 参考回答
没有。  
因为创建 collection 的时候只写了：

```python
self._collection = self._client.get_or_create_collection(name=self.collection_name)
```
并没有传 `embedding_function`。  
所以当前代码是直接使用 Chroma collection 的默认 embedding 配置，而不是项目里手动指定某个 embedding 模型。
## 5. `upsert_cards()` 存进去的到底是什么？

### 题目
`upsert_cards()` 里是怎么把 `MemoryCard` 写入 Chroma 的？

### 参考回答
`upsert_cards()` 会把每张 `MemoryCard` 拆成三部分：

1. `ids`：使用 `card.card_id`
2. `documents`：使用 `card.content`
3. `metadatas`：保存 `entity_id`、`analysis_id`、`card_type`、`ticker`、`doc_type`、`source_file`、`importance`、`created_at` 等信息

其中：

- `document` 是真正参与 embedding 和语义检索的文本
- `metadata` 用来做过滤、回溯和调试
- `id` 用来唯一标识这条向量记录

---

## 6. 为什么 `metadata` 里要保存 `entity_id`？

### 题目
既然已经有 `document` 了，为什么还要在 metadata 里额外存 `entity_id`？

### 参考回答
因为我的业务目标不是全库搜索，而是**在同一个研究对象的历史卡片里做语义检索**。  
所以我把 `entity_id` 放到 metadata 中，检索时就可以通过：

```python
where={"entity_id": entity_id}
```
先把候选范围限制在某个 entity 下，再在这个范围内根据 query 做相似度排序。 这样能避免把别的公司、别的行业的 card 混进来。
## 7. search () 里到底是怎么检索的？

**题目**

下面这段代码是怎么工作的？它是按 entity_id 检索，还是按 query 检索？

python

运行

```
result = self._collection.query(
    query_texts=[query],
    n_results=top_k,
    where={"entity_id": entity_id},
)
```

**参考回答**

这段逻辑不是单纯按 entity_id 检索，而是：

- `where={"entity_id": entity_id}`：先限制候选范围
- `query_texts=[query]`：再在这个候选范围内做语义检索

也就是说：

entity_id 决定**在哪一堆 card 里搜**，query 决定**哪些 card 最相关**。

所以它本质上是**过滤 + 语义召回**，而不是简单精确查询。

---

## 8. 为什么 search () 返回的不是完整的 MemoryCard？

**题目**

search () 返回的是 dict，而不是 MemoryCard，为什么要这样设计？

**参考回答**

因为 Chroma 里存的是：

- id
- document
- metadata
- 相似度距离等结果

它并不是我系统里的完整业务对象。完整的 MemoryCard 仍然以 SQLite 为准，所以 search () 只负责返回候选结果，例如：

- card_id
- content
- metadata
- distance

然后上层再根据 card_id 去 SQLite 回表，取出完整的 MemoryCard。

这样设计的好处是：

- 向量库专注做召回
- SQLite 专注做结构化真相源

---

## 9. 为什么要 “先 Chroma 检索，再 SQLite 回表”？

**题目**

请解释一下为什么你的系统是 “向量召回 + SQLite 回表” 的两段式设计。

**参考回答**

因为 Chroma 擅长的是：

- 文本 embedding
- 相似度检索
- metadata 过滤

但它不是整个系统的主存储。而 SQLite 保存了完整的结构化 card 数据，所以我的做法是：

1. 先用 Chroma 根据 query 找最相关的 card_id
2. 再用 SQLite 根据这些 card_id 取完整 MemoryCard

这样既保留了语义检索能力，又保证了最终业务对象的一致性和可控性。

可以概括为：**Chroma 负责召回，SQLite 负责确权**。

---

## 10. 如果 chromadb 没安装或者 Chroma 不可用怎么办？

**题目**

如果部署环境没有安装 chromadb，你的系统会怎样？

**参考回答**

如果 chromadb 没安装，`ChromaVectorStore.__init__()` 在导入时会捕获 `ImportError`，然后记录 warning，并保持 `available = False`。

后续：

- `upsert_cards()` 会直接跳过
- `search()` 会直接返回空列表
- 上层 `ResearchMemory.search_relevant_cards()` 会自动 fallback 到 SQLite 最近卡片逻辑

所以系统不会崩，只是从**语义检索模式**降级到**最近历史卡片模式**。

---

## 11. upsert_cards () 为什么先尝试 upsert，失败后又 delete + add？

**题目**

请解释下面这段代码的意义：

python

运行

```
try:
    self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
except AttributeError:
    self._collection.delete(ids=ids)
    self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

**参考回答**

这里是在做**版本兼容**。

优先使用 `upsert`，因为它语义最自然：存在就更新，不存在就插入。

但考虑到某些 Chroma 版本可能没有 `upsert` 接口，所以加了一个 `AttributeError` 兜底逻辑：

1. 先删除旧 id
2. 再重新 add

这样即使运行环境里的 Chroma 版本不一致，系统也能正常工作。这体现的是一种**工程兼容性设计**。

---

## 12. 为什么 search () 里还要手动再过滤一次 entity_id？

**题目**

既然 query 时已经写了 `where={"entity_id": entity_id}`，为什么后面还要再写一层：

python

运行

```
if entity_id and metadata.get("entity_id") != entity_id:
    continue
```

**参考回答**

因为前面的带 where 查询如果失败，代码会 fallback 成不带 where 的全局搜索：

python

运行

```
result = self._collection.query(
    query_texts=[query],
    n_results=max(top_k * 2, top_k),
)
```

这样返回结果里可能会混入别的 entity。

所以作者在 Python 层又做了一次手动过滤，保证最终结果尽量只保留当前 entity_id 的卡片。

这是一个**双保险**设计。

---

## 13. count_entries () 和 list_documents () 是干什么的？

**题目**

vector_store.py 里的 `count_entries()` 和 `list_documents()` 主要用于什么场景？

**参考回答**

这两个函数主要偏**调试和可观测性**：

- `count_entries()`：查看当前 collection 中有多少条向量记录
- `list_documents()`：把 collection 里的 document 和 metadata 拉出来，便于在调试页面查看

它们不是主业务链路的核心接口，但对观察 research memory 的写入效果、排查问题很有帮助。

---

## 14. 一次 PDF 分析后，vector_store.py 是怎么参与整个流程的？

**题目**

请描述一次 PDF 分析结束后，vector_store.py 在完整归档链路中的位置。

**参考回答**

流程大概是：

1. 先解析并分析 PDF
2. ResearchMemory 根据分析文本生成 AnalysisRecord
3. 再通过 MemoryExtractor 把分析结果抽成多个 MemoryCard
4. `save_memory_cards()` 先把这些卡片写入 SQLite
5. 然后调用 `vector_store.upsert_cards()` 把卡片内容写入 Chroma
6. 后续用户追问时，再通过 `vector_store.search()` 做历史语义检索

所以 vector_store.py 主要参与的是：

- 归档后的向量化写入
- 追问时的历史语义检索

---

## 15. 从面试角度，这个文件最该掌握到什么程度？

**题目**

如果面试官问到 vector_store.py，你觉得应该掌握到什么程度？

**参考回答**

从实习面试角度，不需要掌握 Chroma 底层算法细节，但一定要掌握这几个点：

1. vector_store.py 是 Chroma 的封装层
2. 它存的是 MemoryCard.content 和 metadata
3. 它支持按 entity_id 限定范围的语义检索
4. 检索结果不是最终对象，还要回 SQLite 取完整 card
5. 它支持降级，Chroma 不可用时系统仍能运行

也就是说，目标不是证明自己会向量数据库底层原理，而是证明自己理解这个模块在系统里的**职责、数据流和设计取舍**。

---

### 总结

1. 纯标准 **GFM Markdown** 语法，Obsidian 原生支持，无任何兼容问题
2. 用**分隔线**区分不同问题，层级清晰，阅读体验更佳
3. 代码块、加粗、列表严格对齐原文逻辑，直接复制即可导入使用