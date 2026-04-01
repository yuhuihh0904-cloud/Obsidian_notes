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