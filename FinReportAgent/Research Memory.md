# Research Memory 面试标准答案（20 题）

下面这版按**面试可直接回答**的口径整理，尽量做到：
- 说法准确
- 逻辑清楚
- 不陷入源码细枝末节
- 能体现你对设计和工程取舍的理解

---

## 1. 一次 PDF 分析结束后，research memory 归档的大致主链路是什么？

一次 PDF 分析完成后，research memory 的归档主链路大致是：

1. 先通过 `resolve_entity` 识别当前文档对应的研究对象是谁，统一成一个 `entity_id`
2. 再通过 `get_or_create_entity` 创建或更新这个研究对象的长期档案
3. 然后用 `save_analysis_record` 保存这一次分析的结构化归档记录
4. 接着通过 `extract_memory_cards` 从分析结果里提炼出 `summary / insight / risk` 三类 memory cards
5. 再通过 `save_memory_cards` 把 cards 写入 SQLite，并同步写入 Chroma 做语义检索索引
6. 最后更新 entity 的 tags、文档数和分析次数等统计信息

本质上就是把“一次分析结果”沉淀成可长期复用的研究记忆。

---

## 2. 在归档之前，为什么一定要先做 `resolve_entity`？如果不做，会有什么问题？

因为后续所有归档和检索，都是围绕“研究对象”组织的。  
无论是 `AnalysisRecord` 还是 `MemoryCard`，都要挂到某个统一的 `entity_id` 下面。

如果不先做 `resolve_entity`，就会出现两个问题：

第一，同一家公司或行业可能被拆成多个不同实体，长期 memory 会散掉；  
第二，后续检索时无法把“同一研究对象的历史信息”准确召回。

所以 `resolve_entity` 的核心价值，是把不同文档、不同时间的分析统一归并到同一个研究对象下面。

---

## 3. `archive_pdf_analysis` 最终返回了什么？这几个返回值分别代表什么意义？

`archive_pdf_analysis` 最终返回的是一个 `dict`，主要包含三个部分：

- `entity`：研究对象档案，也就是这份文档最终归属到哪个实体
- `analysis_record`：这一次分析的结构化归档记录
- `cards`：从本次分析中提炼出来的 memory cards 列表

它们分别对应三个层次：

- `entity` 代表长期研究对象
- `analysis_record` 代表某一次完整分析
- `cards` 代表后续真正可检索、可注入 prompt 的知识片段

要注意，`cards` 只有一套，不是 SQLite card 和 Chroma card 两种；只是同一套 cards 会同时写入 SQLite 和 Chroma。

---

## 4. `resolve_entity` 里，如果用户提供了 `ticker`，为什么这条分支优先级最高？

因为 `ticker` 是比公司名更稳定、更唯一的标识。  
公司名可能存在简称、别名、全称、英文名等多种写法，但 ticker 一般不会有这种歧义。

所以一旦用户提供了 ticker，系统会优先把 `entity_id` 锚定为 `company:{ticker}`，这样能最大程度减少实体识别错误，并提升后续归档和检索的一致性。

---

## 5. 没有 `ticker` 的时候，代码怎么判断当前研究对象更像“公司”还是“行业”？

没有 `ticker` 时，代码会先通过 `_looks_like_industry(...)` 判断当前文档是不是行业文档。

判断依据主要有三类：

- 文档类型是不是 `industry_weekly`
- 文件名或正文里是否包含“行业、板块、赛道、周报、景气”等行业关键词
- 同时还会参考是否已经识别出明显公司名，避免把单一公司文档误判成行业文档

如果判断更像行业，就构造 `industry:xxx` 的实体；  
如果不像行业，再走公司分支，继续提取公司候选名并做实体匹配。

---

## 6. 为什么要设计 `aliases`？它主要是在解决什么实际问题？

`aliases` 的作用，是解决同一研究对象可能有多种名称写法的问题。  
比如：

- 腾讯
- 腾讯控股
- 腾讯控股有限公司

它们本质上可能是同一个实体，但如果只靠一个名字匹配，系统很容易把它们建成多个不同实体。

所以 `aliases` 的主要作用不是展示，而是帮助实体归并和匹配打分，保证后续 memory card 的存储和召回都能统一到同一个研究对象下面。

---

## 7. `_find_existing_company_entity` 和 `_score_company_name_match` 这一套，本质上是在防什么问题？

它们本质上是在防止**同一家公司因为名称写法不同而被重复建实体**。

比如这次识别出了“腾讯控股”，而数据库里原来存的是“腾讯控股有限公司”或“腾讯”，这套逻辑就会通过名称归一化和打分匹配，把它们识别成同一个实体，而不是新建一个重复 entity。

所以这不是“没识别出名字时的兜底”，而是“已经有候选公司名后，为了归并老实体、防止重复建档”的机制。

---

## 8. 为什么 research memory 不直接只存“一整篇分析结果”，而是要拆成 `EntityProfile`、`AnalysisRecord`、`MemoryCard` 这三层？

因为这三层负责的粒度不同：

- `EntityProfile` 管研究对象本身
- `AnalysisRecord` 管某一次分析的结构化归档
- `MemoryCard` 管后续真正可检索、可复用的知识片段

这种分层设计有几个好处：

第一，方便同一实体持续更新，而不是每次都把整篇分析重新塞进去；  
第二，方便把“对象档案”和“历史记录”分开管理；  
第三，便于做轻量化检索和 prompt 注入。

如果只存整篇分析，后续会面临更新复杂、噪声大、检索不精准、token 浪费等问题。

---

## 9. `save_analysis_record` 和 `extract_memory_cards` 的区别是什么？它们分别存的是什么粒度的信息？

`save_analysis_record` 保存的是**一次分析的结构化记录**，比如：

- 报告期
- 页码范围
- summary
- sentiment
- focus topics
- key conclusions

它更偏向“归档”和“记录这次分析”。

而 `extract_memory_cards` 是从分析文本里继续提炼出 `summary / insight / risk` 三类卡片，属于更细粒度的知识抽取。  
这些 cards 后续才会真正用于语义检索和 prompt 背景注入。

所以前者偏“分析记录”，后者偏“知识片段”。

---

## 10. 为什么 `save_memory_cards` 里要同时写 SQLite 和 Chroma，而不是只写一个？

因为它们承担的职责不同：

- SQLite 负责稳定持久化和完整数据读取
- Chroma 负责基于 query 的语义检索

实际检索时，一般会先用 `entity_id + query` 到 Chroma 找相关 `card_id`，再回 SQLite 取完整的 `MemoryCard` 数据。  
如果 Chroma 检索失败，就退化成 SQLite 最近卡片。

所以这套设计不是谁替代谁，而是“Chroma 负责找，SQLite 负责稳和取完整数据”，这样既能做语义召回，又能保证系统稳定性。

---

## 11. `build_context_for_document` 在整个链路里的作用是什么？它和 `build_research_context` 是什么关系？

`build_context_for_document` 的作用是：  
针对一份当前文档，先识别它属于哪个研究对象，再为它构造可注入的历史 research context。

它本质上是一个封装器，内部主要做两件事：

1. 调用 `resolve_entity(...)` 确定 `entity_id`
2. 调用 `build_research_context(...)` 生成最终的历史背景文本

所以可以理解为：

- `build_context_for_document` 负责“面向当前文档”
- `build_research_context` 负责“真正检索和组织历史卡片”

---

## 12. `search_relevant_cards` 为什么不是直接把 Chroma 的结果原样返回给上层，而是还要再去 SQLite 取一次？

因为 Chroma 主要承担语义召回，返回的是轻量检索结果，而不是完整的 `MemoryCard` 对象。  
完整的数据结构、字段和稳定存储都在 SQLite 里。

所以实际流程是：

1. 先到 Chroma 里根据 query 找相关结果
2. 拿到对应的 `card_id`
3. 再去 SQLite 里按 `card_id` 取完整卡片

这样做的好处是职责清晰，也便于在 Chroma 不可用时回退。

---

## 13. 如果 Chroma 检索失败了，这套系统怎么退化，为什么这样设计比较稳？

如果 Chroma 检索失败，或者没有返回结果，系统会退化到 SQLite 的 `get_recent_cards(...)`，直接返回该实体最近的几张卡片。

这样设计稳的原因是：

- 向量检索只是增强项，不是系统生死依赖
- 即使语义检索挂了，research memory 仍然能工作
- 系统只是从“智能召回”退化成“最近记录召回”，不会直接空返回

这保证了系统在最坏情况下也能提供基本的历史背景能力。

---

## 14. `_prioritize_context_cards` 为什么要尽量优先挑 `summary / insight / risk`，而不是纯按相似度直接取前几条？

因为纯按相似度取前几条，容易出现结果过于同质化的问题。  
比如返回的几张卡片可能全是 `insight`，或者内容高度重复，只是措辞略有不同。

而 `_prioritize_context_cards` 会尽量优先保证：

- 一张 `summary`
- 一张 `insight`
- 一张 `risk`

这样注入给 LLM 的上下文会更均衡，让模型同时看到：

- 过去的总体结论
- 过去的具体观点
- 过去的主要风险

所以它是在相关性基础上进一步追求信息多样性和结构完整性。

---

## 15. 在追问时，为什么不能无脑把历史 research memory 全塞进 prompt？

因为 memory 不是越多越好。  
无脑把历史 research memory 全塞进去，会带来两个主要问题：

第一，prompt token 有限，历史信息太多会挤占当前问题和当前文档的空间；  
第二，历史背景里可能包含与当前问题无关、重复甚至冲突的信息，会把模型回答带偏。

所以 research memory 的正确定位是：  
按需补充比较、趋势和背景，而不是替代当前文档。

---

## 16. 如果“当前文档结论”和“历史 research background”冲突，应该以谁为准？为什么？

应该以**当前文档**为准。  
因为当前文档是这次问题的直接证据来源，而 historical research memory 只是补充参考。

历史卡片可能存在：

- 时间滞后
- 上下文缺失
- 不同阶段的判断口径差异

所以一旦当前文档和历史背景不一致，必须优先相信当前文档，而不能让历史 memory 覆盖当前事实。

---

## 17. 哪类追问最适合引入 research memory？哪类追问其实更适合只看当前文档？

最适合引入 research memory 的，一般是这些类型的问题：

- 对比型：和上季度比怎么样
- 趋势型：这个主线是延续还是反转
- 变化型：风险、盈利、现金流有没有改善或恶化
- 延续型：某个事件的影响现在还在不在
- 归纳型：过去几期一直在讲什么

这些问题天然需要“当前 + 历史”两个参照系。

而更适合只看当前文档的，是：

- 单一事实型问题
- 具体数字型问题
- 当前结论型问题
- 当前公告内容定位型问题

比如“这份文档里营收是多少”“当前风险有哪些”，这类问题主要依赖当前文档，历史背景反而可能造成干扰。

---

## 18. 你这个项目里，`session memory` 和 `research memory` 的职责边界分别是什么？

`session memory` 主要负责**当前会话内的连续上下文**，尤其是当前文档状态和同一会话中的连续追问。  
它的信息更完整，但生命周期短，会话结束或重置后就清空。

`research memory` 主要负责**研究对象的长期记忆**，支持跨文档、跨时间、跨对话的历史背景补充。  
它更轻量，不追求保存完整上下文，而追求可检索、可复用、可长期沉淀。

简单说：

- `session memory` 解决“这轮对话连续性”
- `research memory` 解决“研究对象长期知识沉淀”

---

## 19. 为什么要把长分析结果抽成 `memory cards`，而不是把整篇历史分析直接拼到 prompt 里？

因为整篇历史分析通常太长、噪声太大，而且很多内容和当前 query 不相关。  
如果直接整篇拼进 prompt，会导致：

- token 浪费
- 信息冗余
- 当前问题被淹没
- 检索不够精准

而把长分析抽成 `memory cards` 后，就可以把历史分析压缩成：

- 轻量化
- 可检索
- 可复用
- 可持续更新

的知识片段。这样既方便增量维护，也更适合按 query 精准召回。

---

## 20. 如果面试官问你：“你这个 research memory 最大的工程价值是什么？”你会怎么用 2 句话回答？

我给研究对象建立了跨文档、跨时间的长期记忆，让系统不只依赖当前会话，而是能够复用历史研究结论。  
同时我把长分析压缩成可检索的 memory cards，并结合 SQLite + Chroma 做稳定存储和语义召回，让历史信息既可用又不会把 prompt 塞爆。

---

# 复习建议

准备面试时，建议你重点把下面 6 题背熟，因为最容易被追问：

- 第 1 题：归档主流程
- 第 4 题：为什么 ticker 优先
- 第 8 题：为什么拆三层
- 第 10 题：为什么 SQLite + Chroma
- 第 15 题：为什么不能乱塞 memory
- 第 20 题：最大工程价值