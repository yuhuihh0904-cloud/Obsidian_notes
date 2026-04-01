# sqlite_store.py 面试必会 10 点

## 1. 这个模块的核心职责是什么

`sqlite_store.py` 是 research memory 的**结构化持久化层**，负责把研究对象、分析记录、memory card 存到 SQLite，并提供基础查询接口。 :contentReference[oaicite:0]{index=0}

---

## 2. 为什么这里要用 SQLite

因为 research memory 需要**长期保存**，不能只放在内存里；而 entity、analysis record、memory card 这些数据本身是**结构化数据**，适合用 SQLite 存储。  
同时 SQLite 比较轻量，不需要额外部署数据库，适合个人项目和本地原型。 

---

## 3. 这部分一共存了哪三类数据

### `entities`
存研究对象档案，比如公司或行业。  
主要字段包括：

- `entity_id`
- `entity_type`
- `entity_name`
- `ticker`
- `aliases_json`
- `tags_json`
- `doc_count`
- `analysis_count` 

### `analysis_records`
存一次完整分析记录。  
比如这次分析的是哪份 PDF、什么类型、摘要是什么。 

### `memory_cards`
存从分析结果里抽出来的可复用卡片，比如 `summary / insight / risk`。 

可以概括为：

- `entities`：对象层
- `analysis_records`：分析层
- `memory_cards`：复用层

---

## 4. 三张表之间是什么关系

三张表的关系是：

- 一个 `entity` 可以对应多条 `analysis_record`
- 一条 `analysis_record` 可以对应多张 `memory_card`

也就是：

`entity -> analysis_records -> memory_cards`

代码里通过外键建立了这种关系：

- `analysis_records.entity_id` -> `entities.entity_id`
- `memory_cards.analysis_id` -> `analysis_records.analysis_id` :contentReference[oaicite:5]{index=5}

---

## 5. 为什么有些字段要存成 JSON

比如：

- `aliases_json`
- `tags_json`
- `focus_topics_json`
- `key_conclusions_json`
- `source_pages_json` :contentReference[oaicite:6]{index=6}

这样设计是因为这些字段本质上是 Python 里的 `list`。  
为了不把表拆得太复杂，项目里选择把 list 序列化成 JSON 字符串存到 SQLite 里，读取时再反序列化回来。 :contentReference[oaicite:7]{index=7}

这是一个工程上的折中：实现简单，足够支撑当前项目需求。

---

## 6. 至少要理解这几个 SQL 概念

不用学很深，但这些概念一定要懂：

- `CREATE TABLE`：建表
- `PRIMARY KEY`：主键，唯一标识一行
- `FOREIGN KEY`：外键，表示表之间关联
- `SELECT`：查询
- `INSERT`：插入
- `WHERE`：筛选条件
- `ORDER BY`：排序
- `LIMIT`：限制返回条数
- `ON CONFLICT ... DO UPDATE`：冲突时更新，也就是 upsert
- `INSERT OR REPLACE`：插入或替换 :contentReference[oaicite:8]{index=8}

---

## 7. 这几个核心函数必须知道干什么

### `get_entity(entity_id)`
按实体 id 查一个研究对象。 :contentReference[oaicite:9]{index=9}

### `list_entities(entity_type="", ticker="")`
按条件列出实体。 :contentReference[oaicite:10]{index=10}

### `upsert_entity(entity)`
如果实体不存在就插入，存在就更新。  
这是长期维护 entity 档案的关键函数。 :contentReference[oaicite:11]{index=11}

### `insert_analysis_record(record)`
存一条分析记录。 :contentReference[oaicite:12]{index=12}

### `insert_memory_cards(cards)`
批量存 memory card。 :contentReference[oaicite:13]{index=13}

### `get_recent_cards(entity_id, limit=5)`
取某个实体最近的几张卡片。  
这是向量检索失败时的兜底方案之一。 

### `get_memory_cards_by_ids(card_ids)`
按卡片 id 批量取完整 card。  
这个函数是“Chroma 检索后回 SQLite 取完整卡片”的关键。 

### `count_entity_stats(entity_id)`
统计这个实体关联了多少文档、多少分析。 

---

## 8. SQLite 和 Chroma 是怎么配合的

SQLite 和 Chroma 的分工是：

- SQLite：负责**结构化持久化**
- Chroma：负责**语义检索**

真正检索时：

1. 先用 Chroma 根据 query 召回相关 `card_id`
2. 再去 SQLite 里根据这些 id 把完整的 `MemoryCard` 取出来
3. 如果 Chroma 不可用，就降级成直接从 SQLite 取最近卡片 

---

## 9. 一次 PDF 分析之后，这个模块是怎么被调用的

`ResearchMemory.archive_pdf_analysis()` 的流程大概是：

1. `resolve_entity()` 识别研究对象
2. `get_or_create_entity()` 写入或更新实体
3. `save_analysis_record()` 保存分析记录
4. `extract_memory_cards()` 抽取卡片
5. `save_memory_cards()` 保存卡片到 SQLite，再尝试写入 Chroma
6. `update_entity_stats()` 更新实体统计信息 :contentReference[oaicite:18]{index=18}

---

## 10. 面试时对这部分的目标，不是证明自己会 SQL，而是证明自己理解这个模块

更好的表达方式是：

> 我没有把 SQLite 当成单纯的“存一下结果”，而是把它作为 research memory 的结构化底座。  
> 我设计了实体表、分析记录表和卡片表三层结构，分别承载研究对象身份、单次分析归档和可复用记忆片段。  
> 检索链路上 SQLite 和 Chroma 分工明确：Chroma 做语义召回，SQLite 做精确回表和长期归档。 

---

# 一句话总结

对 `sqlite_store.py`，你的目标不是把 SQL 学成后端专家级别，而是：

- 能讲清楚模块职责
- 能讲清楚三张表分别存什么
- 能讲清楚核心函数做什么
- 能讲清楚它和 Chroma 的配合关系
- 能讲清楚一次分析后数据是怎么流进去的

做到这些，已经足够支撑实习面试中的项目讲解。