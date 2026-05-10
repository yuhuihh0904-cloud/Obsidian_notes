> 代码位置：`app/memory/session_memory.py`  
> 主要使用位置：`gradio_app.py` 的 PDF 分析和 PDF 追问链路  
> 核心作用：保存当前 Gradio 会话中“当前 PDF 文档”的轻量上下文，让用户可以围绕同一份文档连续追问。

---

## 1. Session Memory 是什么

`SessionMemory` 不是长期记忆，也不是完整聊天记录。

它是当前 Gradio 会话内，围绕当前 PDF 文档保存的一份轻量任务状态。

它主要回答这些问题：

- 当前正在追问哪份文档？
- 当前文档的 `document_id` 是什么？
- 当前文档属于哪个 `entity_id`？
- 当前文档类型是什么？
- 当前文档页码范围是什么？
- 上一轮分析或追问的简短摘要是什么？
- 最近用户问了哪些问题？
- 当前关注主题有哪些？
- RAG 检索应该使用当前文档，还是同实体多文档？

一句话：

> Session Memory 负责当前文档追问的短期上下文承接。

---

## 2. 和 Research Memory / RAG 的区别

### 2.1 Session Memory

当前会话级别。

特点：

- 只在当前 Gradio 会话内有效。
- 刷新页面或重启服务后会丢失。
- 新上传一份 PDF 后会覆盖旧文档状态。
- 保存当前文档 metadata、上一轮摘要、最近问题、关注主题。
- 用于当前文档连续追问。

### 2.2 Research Memory

长期研究记忆。

特点：

- 跨会话持久化。
- 围绕 `company` 或 `industry` entity 组织。
- 保存历史分析结论、summary、risk、insight 等 memory cards。
- 用 SQLite 保存结构化数据，用 Chroma 做向量检索。
- 用于给 LLM 补充历史研究背景。

### 2.3 RAG

原文证据检索层。

特点：

- 保存 PDF 原文 chunk。
- 按 `document_id` 或 `entity_id` 检索原文证据。
- 用于追问时提供和 query 相似的原文片段。
- 解决“当前文档原文到底怎么说”的问题。

三者边界：

```text
Session Memory = 当前文档会话状态
Research Memory = 跨会话历史研究结论
RAG = PDF 原文证据检索
```

---

# 3. 数据结构：`SessionMemory`

代码位置：

```python
@dataclass
class SessionMemory:
```

核心字段分三类。

---

## 3.1 当前文档 metadata

```python
current_doc_id: str = ""
current_entity_id: str = ""
current_file_name: str = ""
current_doc_type: str = ""
current_doc_type_label: str = ""
current_company: str = ""
current_ticker: str = ""
current_page_range: str = ""
current_report_period: str = ""
current_report_year: str = ""
current_report_type: str = ""
current_rag_retrieval_mode: str = "current_document"
```

这些字段用于描述当前文档是谁。

例如：

- `current_doc_id`：当前文档在 RAG 中的 document_id。
- `current_entity_id`：当前文档对应的公司或行业 entity。
- `current_file_name`：PDF 文件名。
- `current_doc_type`：内部文档类型，例如 `company_quarterly_report`。
- `current_doc_type_label`：展示用文档类型，例如 `单家公司季报`。
- `current_company`：公司名。
- `current_ticker`：股票代码。
- `current_page_range`：分析页码范围。
- `current_report_period`：报告期。
- `current_rag_retrieval_mode`：当前 RAG 检索模式。

---

## 3.2 会话承接信息

```python
last_summary: str = ""
focus_topics: list[str] = field(default_factory=list)
last_questions: list[str] = field(default_factory=list)
```

这些字段用于让后续追问承接前一轮状态。

- `last_summary`：上一轮分析或追问回答的简短摘要。
- `focus_topics`：当前关注主题，例如风险、现金流、盈利能力。
- `last_questions`：最近用户追问的问题，最多保留 5 条。

注意：

> 这里不保存完整 Q&A，只保存压缩后的摘要和最近问题。

---

## 3.3 当前文档文本

```python
current_document_text: str = ""
```

这是当前文档提取后的文本。

注意：

- 它只保存在当前会话中。
- 不做长期持久化。
- 不会默认完整塞进 prompt。
- 当 RAG 不可用或未命中时，会截取一部分作为 fallback 文档摘录。

---

# 4. Session Memory 的真实执行链路

## 4.1 初始化和兜底：`ensure_session_memory()`

在 Gradio 里，PDF 页面会保存一个 state：

```python
gr.State(SessionMemory())
```

每次进入 PDF 分析或追问函数时，都会先调用：

```python
memory = ensure_session_memory(memory)
```

函数逻辑：

```python
def ensure_session_memory(memory: SessionMemory | None) -> SessionMemory:
    if isinstance(memory, SessionMemory):
        return memory
    return SessionMemory()
```

作用：

- 如果 Gradio 正常传回 `SessionMemory`，继续使用。
- 如果传回来的是 `None`，新建一个空的 `SessionMemory`。
- 保证后续代码一定有可用 memory 对象。

---

# 5. 首轮 PDF 分析后的写入链路

PDF 首轮分析成功后，会执行：

```python
memory.reset().update_document_context(...).update_after_analysis(analysis)
```

这句是链式调用，真实执行顺序是：

```text
reset()
  -> update_document_context(...)
  -> update_after_analysis(analysis)
```

---

## 5.1 `reset()`：清空旧文档状态

```python
def reset(self) -> "SessionMemory":
```

它会清空当前 memory 中所有旧状态：

```python
self.current_doc_id = ""
self.current_entity_id = ""
self.current_file_name = ""
self.current_doc_type = ""
self.current_doc_type_label = ""
self.current_company = ""
self.current_ticker = ""
self.current_page_range = ""
self.current_report_period = ""
self.current_report_year = ""
self.current_report_type = ""
self.current_rag_retrieval_mode = "current_document"
self.last_summary = ""
self.focus_topics = []
self.last_questions = []
self.current_document_text = ""
```

为什么要先 `reset()`？

因为当前页面只维护一份“当前文档上下文”。

如果用户重新上传一份 PDF，旧文档的：

- `doc_id`
- `entity_id`
- 文档类型
- 上一轮摘要
- 最近追问
- 关注主题
- 文档文本

都不能继续混用。

否则后续追问会串文档。

---

## 5.2 `update_document_context()`：写入当前文档基础信息

```python
def update_document_context(...)
```

首轮 PDF 分析完成后，调用时传入：

```python
file_name=pdf_result.file_name
doc_type=document_classification.document_type
doc_type_label=document_classification.display_name
company=normalized_company
ticker=normalized_ticker
page_range=pdf_result.page_range_display
extracted_text=pdf_result.text
doc_id=rag_index_result.document_id
entity_id=rag_document_metadata.get("entity_id", "")
report_period=rag_document_metadata.get("report_period", "")
report_year=rag_document_metadata.get("report_year", "")
report_type=rag_document_metadata.get("report_type", "")
rag_retrieval_mode=RETRIEVAL_MODE_CURRENT_DOCUMENT
```

它会写入当前文档 metadata：

```python
self.current_doc_id = doc_id or self._generate_doc_id(file_name, extracted_text)
self.current_entity_id = (entity_id or "").strip()
self.current_file_name = Path(file_name).name
self.current_doc_type = (doc_type or "").strip()
self.current_doc_type_label = (doc_type_label or "").strip()
self.current_company = (company or "").strip()
self.current_ticker = (ticker or "").strip().upper()
self.current_page_range = (page_range or "").strip()
self.current_report_period = (report_period or "").strip()
self.current_report_year = str(report_year or "").strip()
self.current_report_type = (report_type or "").strip()
self.current_rag_retrieval_mode = ...
self.current_document_text = (extracted_text or "").strip()
```

重点：

### 5.2.1 `current_doc_id`

优先使用 RAG 入库得到的 `document_id`：

```python
doc_id=rag_index_result.document_id
```

如果没有，就使用 `_generate_doc_id()` 生成兜底 ID。

### 5.2.2 `current_document_text`

这里保存的是：

```python
pdf_result.text
```

也就是经过 PDF 提取、清洗、压缩并带页码标记的文本。

它不是原始 PDF 文件，也不是未清洗文本。

它的主要作用：

- 当前会话内备用。
- RAG 未命中时作为 fallback 文档摘录来源。

---

## 5.3 `_generate_doc_id()`：生成兜底文档 ID

```python
def _generate_doc_id(file_name: str, extracted_text: str = "") -> str:
```

逻辑：

```python
safe_name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", Path(file_name).stem).strip("_")
safe_name = safe_name or "document"
raw_key = f"{Path(file_name).name.lower()}::{(extracted_text or '').strip()}"
digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
return f"doc_{safe_name}_{digest}"
```

作用：

- 根据文件名和提取文本生成稳定 ID。
- 同一个文件名和同一份文本，生成结果一致。
- 避免 RAG 没返回 document_id 时没有当前文档标识。

---

## 5.4 `update_after_analysis()`：写入上一轮摘要和关注主题

```python
def update_after_analysis(
    self,
    analysis_text: str,
    focus_topics: list[str] | None = None,
) -> "SessionMemory":
```

首轮 PDF 分析结束后，它会处理 LLM 的分析结果。

执行顺序：

```python
normalized_text = self._normalize_text(analysis_text)
self.last_summary = self._build_summary(normalized_text)
topics = focus_topics or self._extract_focus_topics(normalized_text)
self.focus_topics = self._merge_topics(self.focus_topics, topics)
```

也就是说：

1. 清理分析文本。
2. 生成上一轮摘要。
3. 从分析文本中抽取关注主题。
4. 合并到 `focus_topics`。

---

## 5.5 `_normalize_text()`：清理文本

```python
def _normalize_text(text: str) -> str:
```

处理逻辑：

```python
normalized = re.sub(r"#+\s*", "", (text or "").strip())
normalized = re.sub(r"\*\*", "", normalized)
normalized = re.sub(r"\s+", " ", normalized)
return normalized.strip()
```

它会：

- 去掉 Markdown 标题符号 `#`
- 去掉加粗符号 `**`
- 压缩连续空白

目的：

- 让摘要和关键词匹配更稳定。
- 避免 Markdown 格式影响文本判断。

---

## 5.6 `_build_summary()`：生成简短摘要

```python
def _build_summary(text: str, max_length: int = 160) -> str:
```

逻辑：

```python
if not text:
    return ""
if len(text) <= max_length:
    return text
return text[: max_length - 3].rstrip() + "..."
```

注意：

> 这个摘要不是模型重新总结，而是简单截断。

优点：

- 快。
- 稳定。
- 不需要额外 LLM 调用。

缺点：

- 可能截断在不完整的位置。
- 不一定保留最重要的信息。
- 可能丢失后文中的关键结论。

---

## 5.7 `_extract_focus_topics()`：抽取关注主题

```python
def _extract_focus_topics(text: str) -> list[str]:
```

依赖顶部的关键词表：

```python
FOCUS_TOPIC_KEYWORDS = {
    "风险": ["风险", "不确定性", "波动", "压力", "下行"],
    "现金流": ["现金流", "经营活动现金流", "自由现金流"],
    "主营业务": ["主营业务", "核心业务", "业务结构", "收入构成"],
    "盈利能力": ["净利润", "毛利率", "净利率", "盈利能力", "利润率"],
    "营收增长": ["营业收入", "营收", "增长", "同比", "环比"],
    "资产负债": ["资产负债", "负债", "存货", "应收账款", "偿债"],
    "估值": ["估值", "pe", "pb", "市盈率", "市净率"],
    "行业景气": ["行业", "景气", "需求", "供给", "竞争格局"],
}
```

匹配逻辑：

```python
for topic, keywords in FOCUS_TOPIC_KEYWORDS.items():
    for keyword in keywords:
        target = lowered_text if keyword.isascii() else text
        if keyword in target:
            matched_topics.append(topic)
            break
```

特点：

- 中文关键词在原文里匹配。
- 英文关键词在 lower 后文本里匹配。
- 一个 topic 命中一个关键词后就加入，不重复加入同一 topic。

---

## 5.8 `_merge_topics()`：合并关注主题

```python
def _merge_topics(existing_topics: list[str], new_topics: list[str]) -> list[str]:
```

逻辑：

```python
ordered_topics = []
for topic in existing_topics + new_topics:
    if not topic or topic in ordered_topics:
        continue
    ordered_topics.append(topic)
return ordered_topics[:6]
```

作用：

- 保持原有顺序。
- 去重。
- 最多保留 6 个主题。

---

# 6. 首轮分析后展示状态

首轮分析结束后，页面会调用：

```python
memory.build_display_markdown()
```

---

## 6.1 `build_display_markdown()`：给用户看的状态

```python
def build_display_markdown(self) -> str:
```

如果没有当前文档：

```python
return "### 当前文档状态\n暂无文档上下文。请先完成一次 PDF 分析。"
```

如果有当前文档，会展示：

```text
文档 ID
Entity ID
文件名
文档类型
公司
股票代码
报告期
报告年份
报告类型
页码范围
当前 RAG 检索模式
当前关注主题
最近摘要
最近追问
```

注意：

> `build_display_markdown()` 是给用户看的，不是给模型看的。

---

# 7. PDF 追问链路

追问入口：

```python
run_pdf_followup(followup_question, retrieval_mode, memory)
```

真实执行顺序：

```text
ensure_session_memory()
  -> 校验问题是否为空
  -> memory.has_document()
  -> memory.add_question()
  -> memory.build_prompt_brief()
  -> build_research_context_for_followup()
  -> build_rag_context_for_followup()
  -> RAG 未命中则 fallback 到 current_document_text
  -> build_document_followup_question()
  -> Agent 回答
  -> sanitize_followup_response()
  -> memory.update_after_analysis()
  -> memory.build_display_markdown()
```

---

## 7.1 `has_document()`：判断是否可以追问

```python
def has_document(self) -> bool:
    return bool(self.current_doc_id and self.current_document_text)
```

必须同时满足：

- 有当前文档 ID。
- 有当前文档文本。

否则返回：

```text
当前暂无可追问的文档，请先完成一次 PDF 分析。
```

---

## 7.2 `add_question()`：记录本轮追问

```python
def add_question(self, question: str) -> "SessionMemory":
```

执行逻辑：

```python
normalized_question = self._normalize_text(question)
if not normalized_question:
    return self

self.last_questions.append(normalized_question)
self.last_questions = self.last_questions[-5:]

self.focus_topics = self._merge_topics(
    self.focus_topics,
    self._extract_focus_topics(normalized_question),
)
```

它做三件事：

1. 清理用户问题。
2. 保存到最近问题列表。
3. 从问题中抽取关注主题并合并。

注意：

- 最近问题最多保留 5 条。
- 它不保存完整对话。
- 它只保存用户最近问过什么。

---

## 7.3 `build_prompt_brief()`：给模型看的 Session Memory

```python
def build_prompt_brief(self, current_question: str) -> str:
```

如果没有当前文档：

```python
return "当前暂无可用文档状态摘要。"
```

如果有当前文档，会构造：

```python
doc_parts = [self.current_file_name or "未记录文件"]
if self.current_doc_type_label:
    doc_parts.append(self.current_doc_type_label)
if self.current_company:
    doc_parts.append(f"公司 {self.current_company}")
if self.current_ticker:
    doc_parts.append(f"代码 {self.current_ticker}")
if self.current_page_range:
    doc_parts.append(f"页码 {self.current_page_range}")
if self.current_report_period:
    doc_parts.append(f"报告期 {self.current_report_period}")
if self.current_report_type:
    doc_parts.append(f"报告类型 {self.current_report_type}")
```

最终输出类似：

```text
当前文档: 腾讯控股.pdf；单家公司季报；公司 腾讯；代码 0700；页码 第 1-6 页；报告期 2024Q1
当前 entity_id: company:0700
上一轮结论: ...
当前问题: 现金流为什么下降？
当前关注点: 现金流、盈利能力、风险
```

注意：

> `build_prompt_brief()` 是给 LLM 的，不是给用户看的。

它的作用是让 LLM 明白：

- 当前追问是哪份文档。
- 上一轮结论是什么。
- 当前问题是什么。
- 当前关注点是什么。

但它不会把完整聊天历史塞进去。

---

# 8. Session Memory 如何辅助 Research Memory

追问时会调用：

```python
build_research_context_for_followup(question, memory)
```

它会使用 Session Memory 中的字段：

```python
memory.current_file_name
memory.current_doc_type
memory.current_company
memory.current_ticker
memory.last_summary
memory.current_document_text
memory.focus_topics
```

逻辑大致是：

1. 判断当前有没有文档。
2. 判断问题是否需要历史研究背景。
3. 根据当前文档信息 resolve entity。
4. 用用户问题、关注主题、上一轮摘要构造 query。
5. 去 Research Memory 里检索相关 memory cards。

并不是每次追问都会查 Research Memory。

只有问题类似：

- 对比
- 变化
- 趋势
- 改善 / 恶化
- 和上季度比
- 背后原因
- 怎么看

才会触发历史研究背景。

---

# 9. Session Memory 如何辅助 RAG

追问时会调用：

```python
build_rag_context_for_followup(
    normalized_question,
    memory,
    normalized_retrieval_mode,
)
```

内部会使用：

```python
memory.current_doc_id
memory.current_entity_id
```

然后调用 RAG：

```python
rag_service.build_evidence_context(
    query=question,
    retrieval_mode=retrieval_mode,
    document_id=memory.current_doc_id,
    entity_id=memory.current_entity_id,
)
```

如果是当前文档检索：

```text
按 current_doc_id 检索当前 PDF 原文证据
```

如果是同实体多文档检索：

```text
按 current_entity_id 检索同一公司 / 行业下多份文档证据
```

---

# 10. RAG 不可用时的 fallback

如果没有拿到 RAG evidence：

```python
fallback_document_context = build_fallback_document_context(memory.current_document_text)
```

`build_fallback_document_context()` 会截取当前文档文本前 2400 字：

```python
clipped_text = normalized_text[:max_chars]
```

如果被截断：

```python
clipped_text = clipped_text.rstrip() + "..."
```

优点：

- RAG 不可用时，追问仍有最低限度的文档依据。
- 不会因为 Milvus、embedding 或检索失败就完全无法回答。

缺点：

- 固定截取的文本不一定和 query 相关。
- 可能遗漏真正相关的页码。
- 可能信息密度较低。
- 效果不如 RAG 精准检索。

---

# 11. 组装追问 Prompt

追问时最终会调用：

```python
build_document_followup_question(
    followup_question=normalized_question,
    session_context=session_brief,
    research_context=research_brief,
    rag_context=rag_brief,
    fallback_document_context=fallback_document_context,
)
```

其中：

- `session_context` 来自 `memory.build_prompt_brief()`
- `research_context` 来自 Research Memory
- `rag_context` 来自 RAG
- `fallback_document_context` 来自 `current_document_text` 的截取

Session Memory 在这里的核心贡献是：

```text
Session Memory Context（当前对话状态）
```

让模型知道当前追问应该承接哪份文档、上一轮结论和当前关注点。

---

# 12. 追问完成后更新 Session Memory

Agent 回答后：

```python
analysis = sanitize_followup_response(analysis)
memory.update_after_analysis(analysis)
```

也就是说追问回答也会更新：

- `last_summary`
- `focus_topics`

所以 Session Memory 是滚动更新的：

```text
首轮 PDF 分析结果
  -> 第一轮追问回答
  -> 第二轮追问回答
  -> 第三轮追问回答
```

它始终保留“上一轮结果的短摘要”，而不是保留完整聊天记录。

---

# 13. 手动清空当前文档状态

用户点击重置时，会调用：

```python
reset_pdf_memory(memory)
```

逻辑：

```python
memory = ensure_session_memory(memory)
memory.reset()
```

然后页面显示：

```text
当前文档状态已清空。
```

---

# 14. 完整链路总结

## 14.1 首轮 PDF 分析

```text
用户上传 PDF
  -> run_pdf_analysis()
  -> ensure_session_memory()
  -> PDF 提取与文档分类
  -> RAG 入库得到 document_id
  -> Research Memory 解析 entity_id / report_period
  -> Agent 生成首轮分析
  -> memory.reset()
  -> memory.update_document_context(...)
      -> 写入当前文档 metadata
      -> 写入 current_document_text
  -> memory.update_after_analysis(analysis)
      -> 生成 last_summary
      -> 抽取 focus_topics
  -> memory.build_display_markdown()
      -> 页面展示当前文档状态
```

---

## 14.2 当前文档追问

```text
用户输入追问
  -> run_pdf_followup()
  -> ensure_session_memory()
  -> memory.has_document()
  -> memory.add_question(question)
      -> 记录最近问题
      -> 从问题抽取 focus_topics
  -> memory.build_prompt_brief(question)
      -> 生成给模型看的当前状态摘要
  -> build_research_context_for_followup()
      -> 必要时检索长期研究背景
  -> build_rag_context_for_followup()
      -> 根据 document_id / entity_id 检索原文证据
  -> 如果 RAG 未命中
      -> fallback 到 current_document_text 摘录
  -> build_document_followup_question()
      -> 组装最终追问 prompt
  -> Agent 生成回答
  -> memory.update_after_analysis(answer)
      -> 更新 last_summary 和 focus_topics
  -> memory.build_display_markdown()
      -> 页面刷新当前文档状态
```

---

# 15. 常见面试题与修正版回答

## Q1：Session Memory 和 Research Memory 有什么区别？

Session Memory 是当前 Gradio 会话内、围绕当前 PDF 的轻量任务状态。

它保存：

- 当前文档 metadata
- 当前 document_id / entity_id
- 上一轮回答的简短摘要
- 最近几个问题
- 当前关注主题
- 当前文档提取文本

它服务的是当前文档连续追问。

Research Memory 是跨会话、围绕 company 或 industry entity 的长期研究档案。

它保存：

- entity profile
- analysis record
- summary / risk / insight 等 memory cards

它服务的是历史研究背景补充，让 LLM 在回答趋势、比较、风险延续等问题时有长期上下文。

简洁回答：

```text
Session Memory 管当前会话里的当前文档；
Research Memory 管跨会话、跨文档的长期 entity 研究结论。
```

---

## Q2：为什么首轮 PDF 分析完成后要先 reset Session Memory？

因为当前页面只维护一份“当前文档上下文”。

当用户重新上传 PDF 时，旧文档的：

- document_id
- entity_id
- 文档类型
- 公司名
- 报告期
- 上一轮摘要
- 最近追问
- 关注主题
- 当前文档文本

都不能继续混用。

所以首轮分析成功后先 `reset()`，再写入新文档状态，避免后续追问串文档。

---

## Q3：如果 PDF 提取失败或文档为空，应该怎么处理？

应该直接报错提示，而不是继续分析。

原因：

- 空文档没有可靠上下文。
- 如果继续让 LLM 分析，容易产生幻觉。
- 用户需要知道当前 PDF 可能是扫描件、加密文档、空文档或非文本型 PDF。

代码中会检查：

```python
if not pdf_result.text:
    raise ValueError(...)
```

---

## Q4：Session Memory 里应该放什么？不应该放什么？

应该放：

- 当前文档 ID
- 当前 entity ID
- 当前文档类型
- 公司名 / ticker
- 页码范围
- 报告期
- 上一轮简短摘要
- 最近几个问题
- 当前关注主题
- 当前文档提取文本，用于 fallback

不应该默认把完整聊天历史和完整 PDF 文本都塞进 prompt。

原因：

- LLM 上下文窗口有限。
- 完整聊天历史可能包含无关内容，污染当前问题。
- 完整 PDF 文本太长，不适合每轮追问都注入。

更准确地说：

> 当前代码会临时保存 `current_document_text`，但不会默认完整注入 prompt。追问时优先使用 RAG 检索相关原文片段；只有 RAG 未命中时，才截取一部分文档文本作为 fallback。

---

## Q5：focus_topics 是怎么生成和更新的？

`focus_topics` 通过关键词匹配生成。

关键词表包括：

- 风险
- 现金流
- 主营业务
- 盈利能力
- 营收增长
- 资产负债
- 估值
- 行业景气

来源包括三类：

1. 首轮 PDF 分析结果  
   调用 `update_after_analysis(analysis)`，从首轮 LLM 分析中抽主题。

2. 用户追问  
   调用 `add_question(question)`，从用户问题中抽主题。

3. 追问回答  
   追问结束后再次调用 `update_after_analysis(answer)`，从本轮回答中抽主题。

最后通过 `_merge_topics()` 去重并最多保留 6 个主题。

作用：

> 让后续追问 prompt 能带上当前关注点，帮助 LLM 承接上下文并避免偏离主题。

---

## Q6：RAG 不可用或未命中时，Session Memory 怎么兜底？

如果 RAG 没有返回原文证据，系统会从：

```python
memory.current_document_text
```

里截取固定长度文本，作为 fallback 文档摘录放入 prompt。

优点：

- RAG 不可用时仍有最低限度的文档依据。
- 不会因为 Milvus、embedding 或检索失败导致追问完全不可用。

缺点：

- 固定截取的文本不一定和 query 相关。
- 可能遗漏真正相关内容。
- 可能引导 LLM 关注错误位置。
- 效果不如 RAG 精准检索。

---

## Q7：Session Memory 保存完整聊天记录吗？

不保存。

它只保存：

- 最近几个用户问题
- 上一轮回答的简短摘要
- 当前关注主题

这样做的原因：

- 控制 prompt 长度。
- 避免无关对话污染上下文。
- 当前任务主要是“围绕同一 PDF 追问”，不需要完整聊天日志。
- 更容易让模型聚焦当前问题。

---

## Q8：`build_display_markdown()` 和 `build_prompt_brief()` 有什么区别？

`build_display_markdown()` 是给用户看的页面状态。

它展示：

- 文档 ID
- Entity ID
- 文件名
- 文档类型
- 公司
- 股票代码
- 报告期
- 页码范围
- 关注主题
- 最近摘要
- 最近追问

`build_prompt_brief()` 是给 LLM 看的压缩上下文。

它包含：

- 当前文档描述
- 当前 entity_id
- 上一轮结论
- 当前问题
- 当前关注点

简洁回答：

```text
display_markdown 面向用户展示；
prompt_brief 面向 LLM 注入 prompt。
```

---

## Q9：为什么最近问题只保留 5 条？

因为 Session Memory 不是完整聊天日志。

只保留最近问题可以：

- 控制上下文长度。
- 保留短期追问线索。
- 减少远期无关问题干扰。
- 适合当前文档连续追问场景。

---

## Q10：这个设计有什么局限？

主要局限：

1. `last_summary` 是简单截断，不是语义总结。
2. `focus_topics` 是关键词匹配，可能漏召或误召。
3. `current_document_text` fallback 只是固定截取，和 query 相关性不稳定。
4. 不保存完整对话，复杂多轮推理可能丢失细节。
5. 页面刷新或服务重启后 Session Memory 会丢失。
6. 当前页面只维护一个当前文档，不支持同时追问多份独立文档状态。

---

# 16. 一句话总结

Session Memory 的核心价值是：

> 在不保存完整聊天历史、不长期持久化的前提下，用一份轻量的当前文档状态，让金融助手能够围绕同一份 PDF 连续追问，并把当前文档、上一轮结论、关注主题、RAG 检索目标稳定地传递给后续回答流程。
