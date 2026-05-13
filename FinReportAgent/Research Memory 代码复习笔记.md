
> 核心文件：`app/memory/research_memory.py`  
> 数据结构：`app/memory/schemas.py`  
> Memory Card 抽取：`app/memory/memory_extractor.py`  
> 结构化存储：`app/storage/sqlite_store.py`  
> 向量检索：`app/storage/vector_store.py`

---

## 1. Research Memory 是什么

Research Memory 是围绕 `company` / `industry` 这个研究对象沉淀长期研究结论的系统。

它不保存完整 PDF 原文，也不保存当前会话聊天记录。

它保存的是：

- 研究对象档案
- 每次分析记录
- 从分析结果中提炼出来的长期研究卡片

一句话：

> Research Memory 把每次 PDF 分析结果压缩成可复用的结构化研究记忆，供后续分析和追问参考。

---

## 2. 和 Session Memory / RAG 的区别

```text
Session Memory = 当前文档会话状态
Research Memory = 跨会话、跨文档的长期研究结论
RAG = PDF 原文证据检索
```

### 2.1 Session Memory

- 当前 Gradio 会话内有效。
- 新上传 PDF 会覆盖旧状态。
- 保存当前文档 metadata、上一轮摘要、最近问题、关注主题。
- 服务当前文档连续追问。

### 2.2 Research Memory

- 跨会话持久化。
- 围绕 company / industry entity 组织。
- 保存 summary / insight / risk 等历史研究卡片。
- 服务历史背景补充、趋势比较、风险延续判断。

### 2.3 RAG

- 保存 PDF 原文 chunk。
- 按 `document_id` 或 `entity_id` 检索原文证据。
- 服务当前文档事实、数字、页码依据。

原则：

> 当前事实和数字以 RAG / 当前 PDF 原文为准；Research Memory 只能作为历史背景和趋势参考。

---

# 3. 核心数据结构

数据结构定义在：

```text
app/memory/schemas.py
```

---

## 3.1 `ResolvedEntity`

```python
@dataclass
class ResolvedEntity:
    entity_id: str
    entity_type: str
    entity_name: str
    ticker: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
```

作用：

> 临时的研究对象识别结果。

它还不一定已经写入数据库。

例如：

```text
entity_id = company:AAPL
entity_type = company
entity_name = Apple Inc
ticker = AAPL
```

---

## 3.2 `EntityProfile`

```python
@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str
    entity_name: str
    ticker: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)
    doc_count: int = 0
    analysis_count: int = 0
```

作用：

> 长期研究对象档案。

它代表一个公司或行业。

例如：

```text
company:0700
company:腾讯控股
industry:半导体行业
```

里面记录：

- entity 名称
- 股票代码
- 别名
- 标签
- 关联文档数
- 分析次数

---

## 3.3 `AnalysisRecord`

```python
@dataclass
class AnalysisRecord:
    analysis_id: str
    entity_id: str
    doc_id: str
    doc_type: str
    source_file: str
    report_title: str
    report_period: str = ""
    page_range: str = ""
    summary: str = ""
    sentiment: str = "neutral"
    focus_topics: list[str] = field(default_factory=list)
    key_conclusions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_timestamp)
```

作用：

> 一次完整 PDF 分析的结构化归档记录。

它保存的是“这次分析”的概览信息。

字段含义：

- `analysis_id`：本次分析 ID。
- `entity_id`：属于哪个研究对象。
- `doc_id`：这份文档 ID。
- `doc_type`：文档类型。
- `source_file`：原文件路径。
- `report_title`：报告标题。
- `report_period`：报告期。
- `summary`：分析摘要。
- `sentiment`：情绪判断，bullish / bearish / neutral。
- `focus_topics`：关注主题。
- `key_conclusions`：关键结论。

---

## 3.4 `MemoryCard`

```python
@dataclass
class MemoryCard:
    card_id: str
    entity_id: str
    analysis_id: str
    card_type: str
    content: str
    importance: int = 3
    tags: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    source_file: str = ""
    doc_type: str = ""
    ticker: str = ""
    created_at: str = field(default_factory=current_timestamp)
```

作用：

> 适合长期复用和语义检索的研究卡片。

`card_type` 当前主要有三类：

```text
summary
insight
risk
```

可以理解为：

```text
summary = 过去总结指出什么
insight = 过去分析认为有什么经营/财务洞察
risk = 过去记录了哪些风险
```

---

# 4. Research Memory 初始化

核心入口：

```python
class ResearchMemory:
```

初始化逻辑：

```python
self.sqlite_path = data/research_memory.db
self.chroma_path = data/chroma
self.sqlite_store = SQLiteStore(self.sqlite_path)
self.vector_store = ChromaVectorStore(self.chroma_path, collection_name=collection_name)
self.extractor = MemoryExtractor()
```

默认路径：

```text
data/research_memory.db
data/chroma/
```

也就是说，Research Memory 有两层存储：

```text
SQLite = 结构化持久化
Chroma = memory cards 向量检索
```

如果 Chroma 不可用，会自动降级：

```text
Chroma 可用：语义检索 memory cards
Chroma 不可用：回退 SQLite 最近 cards
```

---

# 5. Research Memory 的真实执行链路

Research Memory 主要出现在三个阶段：

```text
PDF 分析前
PDF 分析后
PDF 追问时
```

---

## 5.1 PDF 分析前：准备 RAG metadata

PDF 文本提取和文档分类之后，会调用：

```python
research_memory.build_rag_index_metadata(...)
```

它的作用：

> 为 RAG 原文入库准备 entity 和报告元数据。

执行流程：

```text
build_rag_index_metadata()
  -> resolve_entity()
  -> _extract_report_period()
  -> _extract_report_year()
  -> _map_doc_type_to_report_type()
  -> 返回 metadata
```

返回内容：

```python
{
    "entity_id": resolved.entity_id,
    "entity_name": resolved.entity_name,
    "entity_type": resolved.entity_type,
    "report_period": report_period,
    "report_year": report_year,
    "report_type": report_type,
}
```

这一步虽然在 Research Memory 里，但它服务的是 RAG 入库。

因为 RAG chunk 需要知道：

- 属于哪个 entity
- 是哪一期报告
- 是年报还是季报
- 后续是否能按同一 entity 检索多文档原文

---

## 5.2 PDF 分析前：检索历史研究背景

首轮 PDF 分析前，还会调用：

```python
research_memory.build_context_for_document(...)
```

执行流程：

```text
build_context_for_document()
  -> resolve_entity()
  -> build_research_context()
      -> search_relevant_cards()
      -> _prioritize_context_cards()
      -> _card_to_context_line()
  -> 返回历史研究背景文本
```

返回内容类似：

```text
历史研究背景（供参考）：
- 过去总结指出，...
- 过去分析认为，...
- 过去记录的主要风险包括，...
```

这段内容会被放入首轮 PDF 分析 prompt，作为历史研究背景。

注意：

> 这只是补充参考，不能替代当前 PDF 原文。

---

## 5.3 PDF 分析后：归档本次分析

PDF 分析完成后，会调用：

```python
archive_pdf_analysis(...)
```

这是 Research Memory 最核心的写入链路。

执行流程：

```text
archive_pdf_analysis()
  -> resolve_entity()
  -> get_or_create_entity()
  -> save_analysis_record()
  -> extract_memory_cards()
  -> save_memory_cards()
  -> 更新 entity tags
  -> update_entity_stats()
```

最终会生成并保存：

```text
EntityProfile
AnalysisRecord
MemoryCard[]
```

---

## 5.4 PDF 追问时：按需检索历史卡片

追问时会调用：

```python
build_research_context_for_followup(question, memory)
```

它不是每个问题都会触发 Research Memory。

先判断问题类型：

```python
detect_followup_research_mode(question)
```

如果问题包含：

```text
对比
比较
变化
趋势
改善
恶化
环比
同比
上季度
上一季
前几季
主线
反转
背后原因
怎么看
```

才会检索历史 memory cards。

执行流程：

```text
build_research_context_for_followup()
  -> detect_followup_research_mode()
  -> resolve_entity()
  -> 用 question + focus_topics + last_summary 构造 search_query
  -> search_relevant_cards()
  -> format_followup_research_cards()
```

---

# 6. 研究对象解析：`resolve_entity()`

`resolve_entity()` 是 Research Memory 的核心函数之一。

它回答：

> 这份文档属于哪个 company / industry？

执行顺序如下。

---

## 6.1 优先使用 ticker

如果用户传了 ticker：

```python
entity_id = f"company:{ticker}"
```

例如：

```text
AAPL -> company:AAPL
NVDA -> company:NVDA
0700 -> company:0700
```

这是最稳定的路径。

因为 ticker 比公司名更适合作为 entity ID。

如果 SQLite 里已经有这个 ticker 的 entity，就复用已有 entity name。  
否则根据候选公司名选择一个 canonical name。

---

## 6.2 判断是否为行业文档

如果没有 ticker，会判断是不是行业文档：

```python
self._looks_like_industry(doc_type, file_name, candidate_text)
```

判断依据：

- `doc_type` 是 `industry_weekly`
- 文件名或文本里出现行业关键词：

```python
INDUSTRY_KEYWORDS = ["行业", "板块", "赛道", "景气", "周报", "月报", "专题"]
```

如果是行业文档，会生成：

```python
entity_id = f"industry:{normalized_industry_id}"
```

---

## 6.3 公司候选名提取

如果不是行业文档，就按公司文档处理。

候选公司名来自：

```text
用户传入 company
文件名
文档文本
```

对应函数：

```python
_collect_company_candidate_names()
```

它会：

1. 使用用户传入的 company。
2. 从文件名里清理出公司名。
3. 从文本中用正则提取公司名。
4. 去重。

例如文件名：

```text
腾讯控股2024年第一季度报告.pdf
```

会尝试清理掉：

```text
2024年
第一季度报告
季报
年报
公告
```

得到：

```text
腾讯控股
```

---

## 6.4 匹配已有 company entity

如果已有 entity，系统会尝试匹配：

```python
_find_existing_company_entity(candidate_names)
```

匹配逻辑包括：

- 完全相等：100 分
- 去公司后缀后相等：96 分
- 简称是正式名开头且正式名有公司后缀：88 分
- 简称包含于正式名且正式名有公司后缀：82 分
- 简称是正式名前缀且正式名有后缀：80 分

如果最高分大于等于 80，就认为是同一家公司。

目的：

> 避免 “腾讯 / 腾讯控股 / 腾讯控股有限公司” 被归成多个 entity。

---

## 6.5 新建 entity

如果没有 ticker，也没匹配到已有公司，就选择一个最正式的候选公司名：

```python
_select_canonical_company_name()
```

它偏好：

- 带公司后缀的名称
- 更正式的名称
- 更长、更完整的名称

然后生成：

```python
entity_id = f"company:{normalized_company_id}"
```

---

# 7. 归档链路：`archive_pdf_analysis()`

这个函数负责把一次 PDF 分析完整归档成长期研究记忆。

---

## 7.1 `resolve_entity()`

先识别研究对象：

```python
resolved = self.resolve_entity(...)
```

---

## 7.2 `get_or_create_entity()`

```python
entity = self.get_or_create_entity(resolved)
```

如果 entity 已存在：

- 更新 entity_type
- 更新 entity_name
- 更新 ticker
- 合并 aliases
- 合并 tags
- 更新时间

如果不存在：

- 创建新的 `EntityProfile`
- 写入 SQLite

---

## 7.3 `save_analysis_record()`

```python
analysis_record = self.save_analysis_record(...)
```

保存一次完整分析记录。

内部会生成：

```python
summary = self.extractor.build_brief_summary(analysis_text)
sentiment = self._detect_sentiment(analysis_text)
focus_topics = self.extractor.extract_focus_topics(analysis_text)
key_conclusions = self.extractor.extract_key_conclusions(analysis_text)
```

也就是说，`AnalysisRecord` 不是简单保存全文，而是保存结构化摘要信息。

---

## 7.4 `extract_memory_cards()`

```python
cards = self.extract_memory_cards(...)
```

这一步调用 `MemoryExtractor`，从完整分析文本中抽取长期可复用卡片。

这是 Research Memory 最值得重点理解的部分，下面单独展开。

---

## 7.5 `save_memory_cards()`

```python
self.save_memory_cards(cards)
```

内部执行：

```python
persisted_cards = self.sqlite_store.insert_memory_cards(cards)
self.vector_store.upsert_cards(persisted_cards)
```

注意：

- 先写 SQLite。
- 再写 Chroma。
- 如果 Chroma 失败，只记录 warning，不影响 SQLite。

这保证了：

```text
结构化记忆可靠保存；
语义检索可用时增强，不可用时降级。
```

---

## 7.6 更新 entity tags 和统计

```python
entity.tags = self._merge_unique(entity.tags, self.extractor.extract_focus_topics(analysis_text))
self.sqlite_store.upsert_entity(entity)
self.update_entity_stats(entity.entity_id)
```

`update_entity_stats()` 会更新：

```text
doc_count
analysis_count
```

---

# 8. Memory Card 抽取详解

核心文件：

```text
app/memory/memory_extractor.py
```

入口：

```python
MemoryExtractor.extract_memory_cards()
```

---

## 8.1 Memory Card 抽取目标

一次完整分析报告可能很长，不适合每次都作为历史上下文塞回 prompt。

所以系统会把它压缩成几张短卡片：

```text
summary card：整体结论
insight card：经营 / 财务 / 业务洞察
risk card：风险点
```

当前最多大致会产生：

```text
1 张 summary
最多 2 张 insight
最多 2 张 risk
```

---

## 8.2 `extract_memory_cards()` 主流程

```python
def extract_memory_cards(
    self,
    entity_id: str,
    analysis_id: str,
    analysis_text: str,
    source_file: str = "",
    doc_type: str = "",
    ticker: str = "",
    source_pages: list[int] | None = None,
) -> list[MemoryCard]:
```

执行过程：

```python
normalized_text = self._normalize_text(analysis_text)
sections = self._split_sections(analysis_text)

summary_content = self._build_summary_content(sections, normalized_text)
insight_contents = self._build_insight_contents(sections, normalized_text, limit=2)
risk_contents = self._build_risk_contents(sections, normalized_text, limit=2)
```

然后：

```python
如果 summary_content 存在 -> 创建 summary card
每个 insight_content -> 创建 insight card
每个 risk_content -> 创建 risk card
```

---

## 8.3 第一步：文本标准化 `_normalize_text()`

```python
def _normalize_text(text: str) -> str:
```

处理逻辑：

```python
normalized = re.sub(r"\*\*", "", (text or ""))
normalized = re.sub(r"`", "", normalized)
normalized = re.sub(r"\[(?:第\s*\d+\s*页)\]", "", normalized)
normalized = re.sub(r"\s+", " ", normalized)
return normalized.strip()
```

它会去掉：

- Markdown 加粗符号 `**`
- 代码符号 ```
- 页码标记 `[第 3 页]`
- 多余空白

目的：

> 让后面的句子提取、关键词匹配、压缩更稳定。

---

## 8.4 第二步：按 Markdown 标题切 section `_split_sections()`

```python
def _split_sections(text: str) -> list[tuple[str, str]]:
```

它会逐行扫描分析文本：

```python
if stripped.startswith("#"):
    当前行作为新 section 标题
else:
    当前行加入 section body
```

例如报告：

```markdown
# 核心观点
公司收入增长，利润改善。

# 主营业务
核心业务保持增长。

# 风险提示
需求放缓可能影响增长。
```

会被切成：

```python
[
    ("核心观点", "公司收入增长，利润改善。"),
    ("主营业务", "核心业务保持增长。"),
    ("风险提示", "需求放缓可能影响增长。"),
]
```

如果文本开头没有标题，默认 section title 是：

```text
概览
```

为什么要切 section？

因为标题能帮助判断内容类型：

- “总结 / 核心观点” 更适合 summary
- “主营业务 / 财务 / 估值” 更适合 insight
- “风险提示 / 不确定性” 更适合 risk

---

## 8.5 第三步：构造 summary card

函数：

```python
_build_summary_content(sections, normalized_text)
```

它的目标：

> 从报告中提取整体结论。

优先级：

### 8.5.1 优先找 summary 类标题

标题关键词：

```python
SUMMARY_TITLE_HINTS = [
    "总结",
    "投资参考观点",
    "研究参考观点",
    "核心观点",
    "核心结论",
    "核心信息摘要",
]
```

如果某个 section 的标题包含这些词，就从这个 section 里压缩内容：

```python
snippet = self._compress_text(body, max_sentences=2, max_chars=180)
```

### 8.5.2 如果没有 summary 标题

就按 section 顺序找第一个能压缩出内容的 section。

### 8.5.3 如果 sections 也没有合适内容

就直接压缩全文：

```python
self._compress_text(normalized_text, max_sentences=2, max_chars=180)
```

summary card 的特点：

- 最多 1 张。
- `card_type = "summary"`
- `importance = 4`

---

## 8.6 第四步：构造 insight cards

函数：

```python
_build_insight_contents(sections, normalized_text, limit=2)
```

目标：

> 抽取非风险类的经营、财务、业务、估值等洞察。

---

### 8.6.1 对 section 排优先级

```python
prioritized_sections = sorted(
    sections,
    key=lambda item: self._insight_section_priority(item[0]),
)
```

优先级函数：

```python
_insight_section_priority(title)
```

规则：

1. 如果标题包含 insight 类关键词，优先级最高。
2. 普通标题其次。
3. summary 类标题靠后。

insight 标题关键词：

```python
INSIGHT_TITLE_HINTS = [
    "主营业务",
    "核心经营",
    "营收",
    "利润",
    "估值",
    "研究逻辑",
    "核心信息",
    "主题概述",
    "财务",
    "业务",
]
```

---

### 8.6.2 跳过风险 section

如果标题包含风险关键词，就跳过：

```python
if any(keyword in title for keyword in RISK_TITLE_HINTS):
    continue
```

风险标题关键词：

```python
RISK_TITLE_HINTS = ["风险", "风险因素", "风险提示", "不确定性"]
```

原因：

> 风险内容应该进入 risk card，而不是 insight card。

---

### 8.6.3 从 section body 中抽 snippet

```python
snippets = self._extract_snippets(body)
```

`_extract_snippets()` 会：

1. 按行遍历。
2. 去掉 bullet 符号：

```text
- xxx
* xxx
• xxx
```

3. 去掉编号：

```text
1. xxx
2) xxx
1、xxx
```

4. 丢掉太短且没有数字的内容。
5. 返回候选片段。

---

### 8.6.4 压缩 snippet

每个 snippet 会调用：

```python
candidate = self._compress_text(snippet, max_sentences=2, max_chars=180)
```

`_compress_text()` 会：

1. 标准化文本。
2. 去掉开头的“本次分析页码范围”。
3. 按句号、问号、感叹号切句。
4. 取前 `max_sentences` 句。
5. 如果超过 `max_chars`，截断加 `...`。

---

### 8.6.5 过滤风险内容

insight card 不希望混入风险内容。

所以如果 candidate 里包含：

```python
RISK_KEYWORDS = [
    "风险",
    "不确定性",
    "波动",
    "承压",
    "放缓",
    "下滑",
    "监管",
    "竞争加剧",
    "信用损失",
]
```

就跳过。

---

### 8.6.6 fallback 抽取

如果从优先 section 里抽不够 `limit` 个 insight，会从全文或所有 section body 里继续抽：

```python
fallback_text = "\n\n".join(body for _, body in sections) or normalized_text
```

然后再次 `_extract_snippets()`、`_compress_text()`。

最终：

```python
return self._deduplicate(candidates)[:limit]
```

insight card 特点：

- 最多 2 张。
- `card_type = "insight"`
- `importance = 4`

---

## 8.7 第五步：构造 risk cards

函数：

```python
_build_risk_contents(sections, normalized_text, limit=2)
```

目标：

> 抽取风险、不确定性、下滑、承压等历史风险点。

---

### 8.7.1 优先从风险 section 中抽

```python
risk_bodies = [
    body
    for title, body in sections
    if any(keyword in title for keyword in RISK_TITLE_HINTS)
]
```

也就是优先找标题包含：

```text
风险
风险因素
风险提示
不确定性
```

的部分。

---

### 8.7.2 抽风险句子 `_extract_risk_sentences()`

```python
for sentence in self._extract_risk_sentences(body):
```

`_extract_risk_sentences()` 逻辑：

1. 按中文/英文句号、问号、感叹号切句。
2. 保留包含 `RISK_KEYWORDS` 的句子。

风险关键词包括：

```text
风险
不确定性
波动
承压
放缓
下滑
监管
竞争加剧
信用损失
```

每个风险句子会被压缩成：

```python
self._compress_text(sentence, max_sentences=1, max_chars=160)
```

---

### 8.7.3 从风险 section 里抽 snippet

除了风险句子，还会从风险 section body 中抽 snippet：

```python
for snippet in self._extract_snippets(body):
    candidate = self._compress_text(snippet, max_sentences=2, max_chars=180)
```

这样可以兼容 bullet list 风格的风险提示。

例如：

```markdown
# 风险提示
- 下游需求不及预期。
- 原材料价格波动。
```

这种更适合按 snippet 抽。

---

### 8.7.4 fallback 从全文找风险

如果风险 section 里抽不够，就从全文中找包含风险关键词的 snippet：

```python
fallback_text = "\n\n".join(body for _, body in sections) or normalized_text
```

然后：

```python
if not any(keyword in snippet for keyword in RISK_KEYWORDS):
    continue
```

只保留包含风险关键词的片段。

最终：

```python
return self._deduplicate(candidates)[:limit]
```

risk card 特点：

- 最多 2 张。
- `card_type = "risk"`
- `importance = 5`

risk 的重要性比 summary / insight 更高，因为风险记忆在后续研究里通常更敏感。

---

## 8.8 `_make_card()`：生成 MemoryCard 对象

当 summary / insight / risk 内容抽出来之后，会调用：

```python
_make_card(...)
```

生成真正的 `MemoryCard`：

```python
return MemoryCard(
    card_id=f"card_{uuid4().hex[:16]}",
    entity_id=entity_id,
    analysis_id=analysis_id,
    card_type=card_type,
    content=content,
    importance=importance,
    tags=self._extract_tags(content, card_type, doc_type),
    source_pages=source_pages,
    source_file=source_file,
    doc_type=doc_type,
    ticker=ticker,
)
```

这里会生成：

- 唯一 `card_id`
- 所属 `entity_id`
- 所属 `analysis_id`
- 卡片类型
- 卡片内容
- 重要性
- tags
- 来源页码
- 来源文件
- 文档类型
- ticker

---

## 8.9 `_extract_tags()`：给 card 打标签

```python
def _extract_tags(self, content: str, card_type: str, doc_type: str) -> list[str]:
```

初始 tags：

```python
tags = [card_type]
if doc_type:
    tags.append(doc_type)
```

然后根据 `CARD_TOPIC_KEYWORDS` 匹配主题：

```python
CARD_TOPIC_KEYWORDS = {
    "盈利能力": ["净利润", "毛利率", "净利率", "盈利能力", "利润率", "业绩"],
    "营收增长": ["营业收入", "营收", "增长", "同比", "环比"],
    "现金流": ["现金流", "经营活动现金流", "自由现金流"],
    "资产负债": ["资产负债", "负债", "存货", "应收账款", "偿债"],
    "行业景气": ["行业", "景气", "需求", "供给", "交投", "市场回暖"],
    "估值": ["估值", "pe", "pb", "市盈率", "市净率"],
    "风险": ["风险", "不确定性", "压力", "波动", "下滑", "放缓", "监管"],
}
```

最后去重并最多保留 6 个：

```python
return self._deduplicate(tags)[:6]
```

例如一张卡可能得到：

```python
["risk", "company_quarterly_report", "风险", "现金流"]
```

---

## 8.10 Memory Card 抽取流程总结

完整流程：

```text
extract_memory_cards()
  -> _normalize_text()
      -> 去 markdown 符号、页码标记、多余空白
  -> _split_sections()
      -> 按 markdown 标题切 section
  -> _build_summary_content()
      -> 优先找“总结 / 核心观点 / 核心结论”等 section
      -> 压缩为 1 张 summary card 内容
  -> _build_insight_contents()
      -> 优先找“主营业务 / 财务 / 营收 / 利润 / 估值”等 section
      -> 跳过风险 section
      -> 抽 snippet
      -> 压缩
      -> 过滤风险关键词
      -> 最多生成 2 条 insight 内容
  -> _build_risk_contents()
      -> 优先找“风险 / 风险提示 / 不确定性”等 section
      -> 抽风险句子
      -> 抽风险 snippet
      -> 不够则从全文 fallback 查风险关键词
      -> 最多生成 2 条 risk 内容
  -> _make_card()
      -> 生成 MemoryCard
      -> 打 tags
      -> 绑定 entity_id / analysis_id / source_file / source_pages
```

---

# 9. 存储层：SQLite 和 Chroma

## 9.1 SQLite 存结构化数据

文件：

```text
app/storage/sqlite_store.py
```

主要表：

```text
entities
analysis_records
memory_cards
```

### `entities`

保存研究对象：

```text
entity_id
entity_type
entity_name
ticker
aliases_json
tags_json
doc_count
analysis_count
```

### `analysis_records`

保存每次分析：

```text
analysis_id
entity_id
doc_id
doc_type
source_file
report_title
report_period
summary
sentiment
focus_topics_json
key_conclusions_json
```

### `memory_cards`

保存长期研究卡片：

```text
card_id
entity_id
analysis_id
card_type
content
importance
tags_json
source_pages_json
source_file
doc_type
ticker
```

---

## 9.2 Chroma 存 memory card 向量

文件：

```text
app/storage/vector_store.py
```

写入时：

```python
ids = [card.card_id for card in cards]
documents = [card.content for card in cards]
metadatas = [
    {
        "card_id": card.card_id,
        "entity_id": card.entity_id,
        "analysis_id": card.analysis_id,
        "card_type": card.card_type,
        "ticker": card.ticker,
        "doc_type": card.doc_type,
        "source_file": card.source_file,
        "importance": card.importance,
        "created_at": card.created_at,
    }
]
```

Chroma 保存的核心是：

```text
card.content 的向量
```

用于后续语义检索。

---

## 9.3 为什么 SQLite 和 Chroma 都要有

SQLite 负责：

- 可靠持久化
- 结构化查询
- 调试页面展示
- 按 card_id 取完整卡片
- Chroma 不可用时 fallback

Chroma 负责：

- 语义检索
- 根据 query 找最相关 memory cards

两者关系：

```text
Chroma 负责找 card_id；
SQLite 负责存完整结构化 card。
```

---

# 10. 检索链路

## 10.1 `search_relevant_cards()`

```python
def search_relevant_cards(self, entity_id: str, query: str, top_k: int = 5) -> list[MemoryCard]:
```

执行逻辑：

```text
如果 query 为空
  -> 返回最近 cards

否则
  -> 调 Chroma 搜索
  -> 拿 card_id
  -> 回 SQLite 查完整 MemoryCard
  -> 如果成功，返回 top_k cards
  -> 如果 Chroma 失败或没结果，返回最近 cards
```

代码容错：

```python
try:
    matches = self.vector_store.search(...)
except Exception:
    matches = []
```

兜底：

```python
return self.get_recent_cards(entity_id, limit=max(top_k * 2, 5))
```

---

## 10.2 `build_research_context()`

```python
def build_research_context(self, entity_id: str, query: str | None = None, top_k: int = 3) -> str:
```

执行逻辑：

```text
查 cards
  -> _prioritize_context_cards()
  -> _card_to_context_line()
  -> 拼接成 prompt 文本
```

输出格式：

```text
历史研究背景（供参考）：
- 过去总结指出，...
- 过去分析认为，...
- 过去记录的主要风险包括，...
```

---

## 10.3 `_prioritize_context_cards()`

```python
priority_map = {"summary": 0, "insight": 1, "risk": 2}
```

它会尽量保证上下文里有：

```text
summary
insight
risk
```

而不是全是同一种卡片。

排序考虑：

- card_type
- importance
- created_at

---

## 10.4 `_card_to_context_line()`

根据 card 类型加前缀：

```python
if card.card_type == "risk":
    prefix = "过去记录的主要风险包括"
elif card.card_type == "summary":
    prefix = "过去总结指出"
else:
    prefix = "过去分析认为"
```

这样注入 prompt 时，模型能区分：

- 哪些是历史总结
- 哪些是历史洞察
- 哪些是历史风险

---

# 11. 追问时的 Research Memory 使用

追问时并不是无脑注入历史记忆。

先判断问题类型：

```python
detect_followup_research_mode(question)
```

模式：

```text
off
light
strong
```

### off

普通事实问题，不注入 Research Memory。

例如：

```text
这份报告里营业收入是多少？
```

这类问题应该优先看当前文档原文，不需要历史背景。

### light

解释型问题，轻量注入。

例如：

```text
这说明什么？
怎么看？
背后原因是什么？
```

### strong

比较、趋势、变化类问题，强注入。

例如：

```text
和上季度比有什么变化？
现金流恶化是延续之前趋势吗？
风险主线有没有变化？
```

strong 模式下会：

- 提高 top_k
- 加入当前文档摘要
- 如果语义检索没结果，会 fallback 最近 cards

---

# 12. Research Memory 的边界和原则

## 12.1 不保存完整 PDF 原文

PDF 原文属于 RAG。

Research Memory 只保存从分析结果里提炼出来的长期结论。

---

## 12.2 不保存完整聊天记录

聊天状态属于 Session Memory。

Research Memory 保存的是跨会话可复用的研究知识。

---

## 12.3 不能替代当前文档证据

Research Memory 是历史背景。

如果当前 PDF 原文和历史记忆冲突：

```text
以当前 PDF 原文 / RAG evidence 为准
```

---

# 13. 面试可答版本

## Q1：Research Memory 解决什么问题？

Research Memory 解决的是跨会话、跨文档的研究结论复用问题。

每次 PDF 分析后，系统会把完整分析结果压缩成结构化记录和 memory cards，围绕 company / industry entity 存入 SQLite 和 Chroma。后续分析同一家公司或行业时，可以检索这些历史 summary、insight、risk，作为背景补充。

---

## Q2：Research Memory 和 Session Memory 的区别？

Session Memory 是当前会话内的当前文档状态，刷新或重新上传 PDF 后会变化。  
Research Memory 是跨会话持久化的长期研究档案，围绕 entity 组织。

一句话：

```text
Session Memory 管当前这轮对话；
Research Memory 管过去对这个研究对象积累了什么结论。
```

---

## Q3：Research Memory 和 RAG 的区别？

Research Memory 保存的是历史分析结论卡片。  
RAG 保存的是 PDF 原文 chunk。

```text
Research Memory = 过去分析认为了什么
RAG = 原文实际写了什么
```

回答事实、数字、页码时优先 RAG；回答趋势、延续、历史风险时可以用 Research Memory。

---

## Q4：为什么要抽 MemoryCard，而不是直接存完整分析报告？

因为完整分析报告太长，不适合每次注入 prompt，也不利于语义检索。

MemoryCard 更短、更聚焦、更适合复用：

- summary card 保存整体观点
- insight card 保存经营/财务洞察
- risk card 保存风险点

这样可以控制上下文长度，并提高检索质量。

---

## Q5：MemoryCard 是怎么抽取的？

系统先把分析报告按 Markdown 标题切成 sections，然后分别抽取：

- summary：优先从“总结、核心观点、核心结论”等 section 中提取。
- insight：优先从“主营业务、财务、营收、利润、估值”等 section 中提取，并过滤风险内容。
- risk：优先从“风险提示、不确定性”等 section 中提取风险句子；如果不够，再从全文中找包含风险关键词的句子。

每条内容会被压缩到较短长度，再生成 `MemoryCard`，写入 SQLite，并同步到 Chroma 做向量检索。

---

## Q6：Chroma 不可用怎么办？

Chroma 不可用时，Research Memory 会降级到 SQLite 最近卡片。

这保证：

- 主流程不因为向量库不可用而失败。
- 历史记忆仍然有基础可用性。
- 只是语义相关性会下降。

---

## Q7：entity 是怎么归一化的？

优先使用 ticker 构造稳定 ID：

```text
company:AAPL
company:NVDA
```

如果没有 ticker，会从 company 参数、文件名、文本中提取公司名，并和已有 entity 的 aliases 做匹配，尽量避免“腾讯 / 腾讯控股 / 腾讯控股有限公司”被识别成多个对象。

行业文档则会生成：

```text
industry:xxx
```

---

# 14. 一句话总结

Research Memory 的核心价值是：

> 把每次 PDF 分析沉淀成围绕 company / industry 的长期研究卡片，让后续分析和追问能够复用历史 summary、insight、risk，同时又不把完整报告或原文塞进上下文。
