
> 代码位置：`app/utils/pdf_parser.py`  
> 核心入口：`extract_pdf_text()`  
> 这部分负责把 PDF 转成适合送入 LLM 和 RAG 的结构化文本。它不是单纯读取 PDF，而是完成：页码范围选择、逐页提取、文本清洗、页眉页脚去重、高价值段落保留、最终字符数裁剪。

---

## 1. 调用入口

PDF 分析时，外部会调用：

```python
pdf_result = extract_pdf_text(
    pdf_path,
    max_pages=DEFAULT_PDF_MAX_PAGES,
    max_chars=DEFAULT_PDF_MAX_CHARS,
    start_page=normalized_start_page,
    end_page=normalized_end_page,
)
```

调用方主要有两个：

- `gradio_app.py` 的 `run_pdf_analysis()`
- `main.py` 的命令行 PDF 分析分支

`extract_pdf_text()` 返回的是 `PDFExtractResult`，后续会被用于：

- 首轮 PDF 分析 Prompt
- RAG 原文入库
- 页面状态展示
- Session Memory 更新

---

## 2. 返回数据结构：`PDFExtractResult`

```python
@dataclass
class PDFExtractResult:
    file_name: str
    total_pages: int
    extracted_pages: int
    actual_start_page: int
    actual_end_page: int
    used_custom_page_range: bool
    raw_char_count: int
    cleaned_char_count: int
    compressed_char_count: int
    char_count: int
    truncated: bool
    text: str
    cleaned_page_map: dict[int, str]
```

重点字段：

- `text`：最终送给 LLM 的文本，带页码标记。
- `cleaned_page_map`：按页保存的清洗文本，后续 RAG 切 chunk 会用。
- `raw_char_count`：PDF 原始抽取文本字符数。
- `cleaned_char_count`：去噪后的字符数。
- `compressed_char_count`：高价值段落压缩后的字符数。
- `char_count`：最终真正送模的字符数。
- `truncated`：是否只分析了部分页或部分文本。

还有一个展示属性：

```python
page_range_display
```

用于把页码范围显示成：

```text
第 1-6 页
```

---

## 3. `extract_pdf_text()` 执行主流程

### 3.1 参数校验

函数开始先检查：

```python
if not pdf_path:
    raise ValueError("缺少 PDF 文件路径")
if max_pages <= 0:
    raise ValueError("max_pages 必须大于 0")
if max_chars <= 0:
    raise ValueError("max_chars 必须大于 0")
```

目的：

- 防止空路径。
- 防止页数上限非法。
- 防止字符上限非法。

---

### 3.2 延迟导入 PyMuPDF

```python
import fitz
```

`fitz` 是 PyMuPDF 的模块名。

这里采用延迟导入：只有真正处理 PDF 时才导入。这样如果用户只跑股票分析，不会因为 PDF 依赖缺失而直接失败。

---

### 3.3 路径处理与文件检查

```python
path = Path(pdf_path).expanduser().resolve()
if not path.exists():
    raise FileNotFoundError(...)
```

这里做了两件事：

- `expanduser()`：支持 `~/xxx.pdf`
- `resolve()`：转成绝对路径

---

### 3.4 打开 PDF，并保证关闭

```python
document = fitz.open(path)
try:
    ...
finally:
    document.close()
```

`finally` 保证无论中途是否报错，PDF 文件都会关闭，避免文件句柄泄漏。

---

## 4. 页码范围确定

### 4.1 获取总页数

```python
total_pages = document.page_count
```

如果 PDF 没有页面，直接返回空的 `PDFExtractResult`。

---

### 4.2 判断是否使用自定义页码

```python
use_custom_page_range = start_page is not None and end_page is not None
```

注意：只有 `start_page` 和 `end_page` 都存在，才算用户指定页码范围。

如果用户只传一个，目前不会进入自定义页码逻辑。

---

### 4.3 自定义页码逻辑

```python
normalized_start_page = max(1, int(start_page))
normalized_end_page = max(1, int(end_page))
```

保证页码最小为 1。

如果起始页大于结束页：

```python
raise ValueError("起始页不能大于结束页。")
```

然后防止超过 PDF 总页数：

```python
actual_start_page = min(normalized_start_page, total_pages)
actual_end_page = min(normalized_end_page, total_pages)
```

---

### 4.4 默认页码逻辑

如果用户没有指定页码：

```python
actual_start_page = 1
actual_end_page = min(total_pages, max_pages)
```

也就是默认只读取前 `max_pages` 页。

这个设计是为了避免一整份年报过长，直接塞爆 LLM 上下文。

---

### 4.5 是否截断

```python
truncated = actual_start_page != 1 or actual_end_page != total_pages
```

只要不是完整覆盖整份 PDF，就认为发生了截断。

例如：

- 只读前 6 页：`truncated = True`
- 用户只读第 10-20 页：`truncated = True`
- PDF 总共 5 页，读取 1-5 页：`truncated = False`

---

## 5. 逐页提取文本

```python
for page_index in range(actual_start_page - 1, actual_end_page):
    page = document.load_page(page_index)
    page_text = _normalize_page_text(page.get_text("text"))
```

这里有一个重要细节：

- 用户看到的页码是从 1 开始。
- PyMuPDF 的页码索引是从 0 开始。

所以代码使用：

```python
actual_start_page - 1
```

提取出的每页文本会进入辅助函数 `_normalize_page_text()`。

---

## 6. 辅助函数：`_normalize_page_text()`

```python
def _normalize_page_text(text: str) -> str:
```

作用：做最基础的文本规整。

它处理：

1. 替换 PDF 里可能出现的空字符：

```python
text.replace("\x00", " ")
```

2. 删除换行前多余空格：

```python
re.sub(r"[ \t]+\n", "\n", cleaned)
```

3. 压缩连续空格：

```python
re.sub(r"[ \t]{2,}", " ", cleaned)
```

4. 压缩过多空行：

```python
re.sub(r"\n{3,}", "\n\n", cleaned)
```

5. 去掉首尾空白：

```python
strip()
```

这个函数不判断内容是否有价值，只负责把 PDF 抽出来的脏文本变得规整。

---

## 7. 保存原始页文本

```python
raw_page_texts.append((page_index + 1, page_text))
```

这里保存的是：

```python
真实页码 + 当前页文本
```

例如：

```python
[
    (1, "第一页文本..."),
    (2, "第二页文本...")
]
```

后面所有清洗、压缩、RAG 入库都依赖这个页码关系。

---

## 8. 统计原始字符数

```python
raw_char_count = len("\n\n".join(page_text for _, page_text in raw_page_texts).strip())
```

这一步只是统计原始抽取文本长度，用于页面展示和调试。

---

# 9. 多页文本清洗：`clean_extracted_pages()`

```python
cleaned_page_texts = clean_extracted_pages(
    [page_text for _, page_text in raw_page_texts]
)
```

这是 PDF 清洗的核心之一。

它做两轮处理：

1. 按规则删除明显噪声行。
2. 识别并删除重复页眉页脚。

---

## 10. 第一轮清洗：`_clean_page_lines()`

```python
page_lines_list = [_clean_page_lines(page_text) for page_text in page_texts]
```

`_clean_page_lines()` 会把单页文本拆成多行：

```python
for raw_line in page_text.splitlines():
```

每一行都会交给：

```python
_should_drop_line_by_rule(line)
```

如果判断为噪声，就跳过；否则保留。

---

## 11. 噪声判断总入口：`_should_drop_line_by_rule()`

```python
def _should_drop_line_by_rule(line: str) -> bool:
```

它是“这一行要不要删除”的总开关。

删除条件包括：

```python
if not stripped:
    return True
if _is_page_number_line(stripped):
    return True
if _is_toc_noise_line(stripped):
    return True
if _is_disclaimer_noise_line(stripped):
    return True
if _is_low_value_short_line(stripped):
    return True
```

也就是说，只要命中一个规则，这行就会被删除。

---

## 12. 页码行识别：`_is_page_number_line()`

```python
def _is_page_number_line(line: str) -> bool:
```

识别常见页码格式：

```text
第 3 页
page 3
page 3 of 20
- 3 -
3/20
3
```

对应正则：

```python
r"^第\s*\d+\s*页$"
r"^page\s*\d+$"
r"^page\s*\d+\s*of\s*\d+$"
r"^-\s*\d+\s*-$"
r"^\d+\s*/\s*\d+$"
r"^\d+$"
```

注意：单独一行数字也会被认为是页码。

这可能误删一些单独成行的数字，但在财报 PDF 清洗里通常是可接受的取舍。

---

## 13. 目录噪声识别：`_is_toc_noise_line()`

```python
def _is_toc_noise_line(line: str) -> bool:
```

它删除：

```text
目录
contents
content
```

以及目录中常见的点线：

```text
第一节 公司简介 .... 3
第一节 公司简介 …… 3
```

判断逻辑：

```python
if lowered in {"目录", "contents", "content"}:
    return True
if "...." in stripped or "……" in stripped:
    return True
```

---

## 14. 免责声明识别：`_is_disclaimer_noise_line()`

```python
def _is_disclaimer_noise_line(line: str) -> bool:
```

它会检查当前行是否包含 `DISCLAIMER_KEYWORDS` 中的关键词。

例如：

```python
"本报告仅供"
"仅供参考"
"不构成投资建议"
"请务必阅读正文之后的免责声明"
"法律声明"
```

这些内容对财务分析价值较低，而且会污染模型注意力，所以提前删除。

---

## 15. 低价值短行识别：`_is_low_value_short_line()`

```python
def _is_low_value_short_line(line: str) -> bool:
```

这个函数主要处理类似：

```text
---
***
·
|
```

这类没有信息价值的短符号行。

它的逻辑比较保守：

- 空行：删除
- 长度大于 4：不删除
- 包含数字：不删除
- 包含中文：不删除
- 包含英文字母：不删除
- 剩下的短符号行：删除

所以它不会轻易删除有文字、有数字的短行。

---

# 16. 第二轮清洗：重复页眉页脚识别

第一轮清洗完后，代码会调用：

```python
repeated_margin_lines = _detect_repeated_margin_lines(page_lines_list)
```

这个函数用来识别重复出现在多页顶部或底部的内容。

---

## 17. 辅助函数：`_detect_repeated_margin_lines()`

```python
def _detect_repeated_margin_lines(page_lines_list: list[list[str]]) -> set[str]:
```

执行逻辑：

1. 每页取前 3 行和后 3 行：

```python
candidates = page_lines[:3] + page_lines[-3:]
```

2. 对候选行做标准化：

```python
normalized = _normalize_line_for_repeat_check(line)
```

3. 如果标准化后内容过短，跳过：

```python
if len(normalized) < 4:
    continue
```

4. 如果本身是页码行，跳过。

5. 用 `Counter` 统计出现次数。

6. 出现次数大于等于 2 的行，认为是重复页眉页脚：

```python
if count >= 2
```

这个设计的假设是：页眉页脚通常会在多页重复出现，且通常位于每页开头或结尾。

---

## 18. 页眉页脚标准化：`_normalize_line_for_repeat_check()`

```python
def _normalize_line_for_repeat_check(line: str) -> str:
```

它用于把看起来相似的页眉页脚转成统一形式。

处理逻辑：

```python
line.strip().lower()
re.sub(r"\s+", "", normalized)
re.sub(r"[·•\-.:_|/\\]+", "", normalized)
```

也就是：

- 去首尾空白
- 转小写
- 删除所有空白
- 删除一些常见分隔符

例如：

```text
Company Report - 2024
CompanyReport2024
```

可能会被归一化成接近的形式。

---

## 19. 删除重复页眉页脚

回到 `clean_extracted_pages()`：

```python
for line in page_lines:
    normalized = _normalize_line_for_repeat_check(line)
    if normalized in repeated_margin_lines:
        continue
    cleaned_lines.append(line)
```

最终每页会重新拼回文本：

```python
page_content = "\n".join(cleaned_lines).strip()
```

返回：

```python
cleaned_pages
```

---

# 20. 统计清洗后字符数

```python
cleaned_char_count = len("\n\n".join(cleaned_page_texts).strip())
```

这一步用来观察清洗效果：

```text
原始字符数 -> 清洗后字符数
```

如果差异很大，说明噪声较多。

---

# 21. 高价值段落压缩：`prioritize_high_value_content()`

```python
prioritized_page_texts = prioritize_high_value_content(
    cleaned_page_texts,
    max_chars=max_chars
)
```

这个函数的目的：

> 当清洗后的文本仍然超过 `max_chars` 时，不是简单从头截断，而是优先保留财务价值更高的段落。

---

## 22. 不超长时直接返回

```python
full_text = "\n\n".join(page_texts).strip()
if len(full_text) <= max_chars:
    return page_texts
```

如果全文没有超过限制，就不做压缩。

---

## 23. 按段落切分：`_split_into_paragraphs()`

```python
paragraphs = _split_into_paragraphs(page_text)
```

函数定义：

```python
def _split_into_paragraphs(page_text: str) -> list[str]:
```

它按照空行切分段落：

```python
re.split(r"\n\s*\n+", page_text.strip())
```

所以它依赖文本中存在段落空行。

如果 PDF 抽取出来的文本没有空行，可能一整页会被视为一个段落。

---

## 24. 段落打分：`_score_paragraph()`

```python
score = _score_paragraph(paragraph)
```

这个函数根据财务信息密度打分。

### 24.1 命中高价值关键词

关键词来自：

```python
HIGH_VALUE_KEYWORDS
```

包括：

```text
营业收入
营收
归母净利润
净利润
毛利率
净利率
同比
环比
现金流
资产负债率
应收账款
存货
主营业务
风险提示
政策风险
市场风险
```

每命中一次加分：

```python
score += min(hits, 3) * 3
```

同一个关键词最多按 3 次计分，避免重复词过度放大权重。

---

### 24.2 数字加分

```python
if re.search(r"\d", paragraph):
    score += 2
```

财报中有数字通常意味着有财务指标、金额、比例等信息。

---

### 24.3 百分比加分

```python
if re.search(r"\d+(?:\.\d+)?%", paragraph):
    score += 2
```

比如：

```text
同比增长 15.3%
毛利率 42.1%
```

---

### 24.4 金额单位加分

```python
if re.search(r"\d[\d,\.]*\s*(?:亿元|万元|百万元|千万元|元|亿美元|million|billion)", paragraph, re.IGNORECASE):
    score += 3
```

命中金额单位说明段落大概率包含财务数据。

---

### 24.5 长段落加分

```python
if len(paragraph) >= 80:
    score += 1
```

较长段落通常信息量更高，所以轻微加分。

---

# 25. 段落排序和选择

每个段落会被保存成：

```python
(score, page_index, paragraph_index, paragraph)
```

然后排序：

```python
ranked_items = sorted(
    paragraph_items,
    key=lambda item: (item[0], -item[1], -item[2], len(item[3])),
    reverse=True,
)
```

排序优先级大致是：

1. 分数越高越优先。
2. 分数相同的情况下，页码更靠前的更优先。
3. 同页内，段落更靠前的更优先。
4. 段落更长的略优先。

然后逐个选择段落，直到接近 `max_chars`。

如果已经使用了 70% 字符预算，则低价值段落不再补充：

```python
if score <= 0 and selected_chars >= max_chars * 0.7:
    continue
```

如果当前段落会超过字符预算，也跳过：

```python
if selected_chars > 0 and selected_chars + paragraph_length > max_chars:
    continue
```

最终被选中的段落用 `(page_index, paragraph_index)` 记录。

---

## 26. 恢复原始顺序

虽然选择段落时是按分数排序，但输出时会恢复原始页码和段落顺序：

```python
for page_index, page_text in enumerate(page_texts):
    ...
```

这样模型看到的文本不会变成乱序。

最终返回：

```python
compressed_pages
```

---

# 27. 统计压缩后字符数

```python
compressed_char_count = len("\n\n".join(prioritized_page_texts).strip())
```

这个指标用于观察高价值段落压缩后还剩多少内容。

---

# 28. 生成 `cleaned_page_map`

```python
cleaned_page_map = {
    page_number: cleaned_text
    for (page_number, _), cleaned_text in zip(raw_page_texts, prioritized_page_texts)
}
```

这是一个非常关键的结构：

```python
{
    1: "第一页清洗后的文本",
    2: "第二页清洗后的文本",
}
```

它的价值：

- 保留页码信息。
- 后续 RAG 入库时可以按页切 chunk。
- 追问时可以返回“第几页”的证据。
- 避免只有一整坨纯文本，丢失来源定位。

---

# 29. 最终送模文本拼接

接下来代码开始生成 `PDFExtractResult.text`：

```python
chunks = []
for page_number, _ in raw_page_texts:
    page_text = cleaned_page_map.get(page_number, "").strip()
```

每页都会加页码标记：

```python
chunks.append(f"[第 {page_number} 页]\n{clipped_text}")
```

最终文本类似：

```text
[第 1 页]
公司简介...

[第 2 页]
主要财务数据...
```

这个格式对 LLM 很友好，因为模型能知道每段内容来自哪一页。

---

## 30. 最后一层字符裁剪

即使前面做过高价值段落压缩，这里仍然会做最后兜底裁剪：

```python
remaining = max_chars - total_chars
clipped_text = page_text[:remaining]
```

如果当前页文本超过剩余字符预算，就只截取前 `remaining` 个字符。

如果发生裁剪：

```python
truncated = True
```

如果总字符数达到 `max_chars`：

```python
break
```

这保证最终送给 LLM 的文本不会超过预算。

---

# 31. 返回最终结果

最终返回：

```python
return PDFExtractResult(
    file_name=path.name,
    total_pages=total_pages,
    extracted_pages=extracted_pages,
    raw_char_count=raw_char_count,
    cleaned_char_count=cleaned_char_count,
    compressed_char_count=compressed_char_count,
    char_count=len(full_text),
    truncated=truncated,
    text=full_text,
    cleaned_page_map=cleaned_page_map,
    actual_start_page=actual_start_page,
    actual_end_page=actual_end_page,
    used_custom_page_range=use_custom_page_range,
)
```

到这里，PDF 切分和清洗阶段结束。

---

# 32. 整体执行顺序总结

```text
extract_pdf_text()
  -> 参数校验
  -> 导入 fitz
  -> 解析 PDF 路径
  -> 打开 PDF
  -> 获取总页数
  -> 判断是否使用自定义页码范围
  -> 确定 actual_start_page / actual_end_page
  -> 逐页 load_page()
  -> page.get_text("text")
  -> _normalize_page_text()
  -> 保存 raw_page_texts
  -> 统计 raw_char_count
  -> clean_extracted_pages()
      -> _clean_page_lines()
          -> _should_drop_line_by_rule()
              -> _is_page_number_line()
              -> _is_toc_noise_line()
              -> _is_disclaimer_noise_line()
              -> _is_low_value_short_line()
      -> _detect_repeated_margin_lines()
          -> _normalize_line_for_repeat_check()
      -> 删除重复页眉页脚
  -> 统计 cleaned_char_count
  -> prioritize_high_value_content()
      -> _split_into_paragraphs()
      -> _score_paragraph()
      -> 按分数选择高价值段落
      -> 恢复原始页码顺序
  -> 统计 compressed_char_count
  -> 构造 cleaned_page_map
  -> 拼接带页码的最终 text
  -> 最后一层 max_chars 裁剪
  -> 返回 PDFExtractResult
```

---

# 33. 这段代码的设计亮点

## 33.1 保留页码信息

最终文本和 `cleaned_page_map` 都保留了页码。

这对金融文档很重要，因为后续回答追问时需要引用原文来源。

---

## 33.2 不是暴力截断

代码没有简单取前 `max_chars` 个字符，而是：

```text
先清洗噪声
再识别高价值段落
最后才做兜底裁剪
```

这比粗暴截断更适合财报分析。

---

## 33.3 清洗逻辑比较轻量

所有清洗都是规则实现，没有引入复杂 NLP 或 OCR。

优点：

- 快
- 可解释
- 易调试
- 不依赖模型

缺点：

- 对复杂 PDF 布局不够强
- 对扫描版 PDF 无效
- 对表格结构保留有限

---

## 33.4 RAG 和 LLM 共用同一份清洗结果

`text` 给 LLM 首轮分析用。  
`cleaned_page_map` 给 RAG 入库用。

这保证了首轮分析和后续原文检索使用的是一致的文本来源。

---

# 34. 可能的面试问题

## Q1：为什么 PDF 文本提取后还要做清洗？

答：PDF 抽取文本通常包含页码、页眉页脚、目录点线、免责声明、空字符、重复行等噪声。如果不清洗，这些内容会浪费 LLM 上下文窗口，并影响模型关注财务核心信息。

---

## Q2：为什么默认只读取前几页，而不是整份 PDF？

答：财报和研报通常很长，直接读取整份文档容易超过模型上下文限制，也会增加成本。当前 MVP 默认用 `max_pages` 控制页数，并通过 `truncated` 告诉模型本次分析范围有限。

---

## Q3：`truncated` 字段什么时候会是 True？

答：两种情况：

1. 页码范围没有覆盖整份 PDF。
2. 文本长度超过 `max_chars`，最终发生字符裁剪。

---

## Q4：为什么需要 `cleaned_page_map`，只返回一整段文本不行吗？

答：不够。`cleaned_page_map` 保留了“页码 -> 文本”的映射，后续 RAG 入库需要按页切 chunk，并在检索结果中返回页码。如果只保留整段文本，后续无法定位证据来源。

---

## Q5：如何识别页眉页脚？

答：代码取每页前 3 行和后 3 行作为候选，把它们标准化后计数。如果某一行在多页重复出现至少 2 次，就认为它是页眉或页脚。

---

## Q6：这种页眉页脚识别有什么风险？

答：如果某些重要标题在多页重复出现，可能被误删。反过来，如果页眉页脚每页略有不同，例如包含动态页码或日期，也可能识别不出来。

---

## Q7：为什么不直接用 OCR？

答：当前项目主要处理文本型 PDF，PyMuPDF 已能直接提取文本。OCR 会增加依赖、成本和处理时间。对于扫描版 PDF，当前方案确实不支持，需要后续扩展 OCR。

---

## Q8：`_score_paragraph()` 为什么要给数字、百分比、金额单位加分？

答：财务分析最关注数据，比如收入、利润、毛利率、同比增速、现金流等。包含数字、百分比和金额单位的段落更可能包含关键财务信息，所以优先保留。

---

## Q9：为什么高价值段落选择后还要恢复原始顺序？

答：如果按分数排序直接输出，文本顺序会混乱，模型难以理解上下文。恢复原始页码和段落顺序可以兼顾信息密度和可读性。

---

## Q10：当前 PDF 切分方案对表格支持好吗？

答：一般。PyMuPDF 的 `page.get_text("text")` 会把表格抽成纯文本，可能丢失表格结构、列关系和对齐信息。如果要更好支持财报表格，可以考虑 `get_text("blocks")`、`get_text("dict")` 或专门的表格解析工具。

---

## Q11：如果面试官问你“这个模块的输入和输出是什么”，怎么回答？

答：

输入是 PDF 路径、最大页数、最大字符数、可选页码范围。  
输出是 `PDFExtractResult`，其中最重要的是最终送模文本 `text` 和按页清洗文本 `cleaned_page_map`。

---

## Q12：如果面试官问你“怎么保证不会超过模型上下文”，怎么回答？

答：

代码通过两层控制：

1. `prioritize_high_value_content()` 在文本过长时优先保留高价值段落。
2. 最终拼接 `chunks` 时使用 `remaining = max_chars - total_chars` 做硬裁剪。

所以最终文本长度会被限制在 `max_chars` 附近。

---

## Q13：如果面试官问你“这个模块最值得优化的地方是什么”，怎么回答？

答：

可以从几方面优化：

- 支持扫描版 PDF 的 OCR。
- 更好地保留表格结构。
- 页眉页脚识别可以更精细，降低误删。
- 高价值段落打分可以结合文档类型动态调整。
- 自定义页码范围目前要求起止页都存在，可以支持只填起始页或只填结束页。
- 可以加入 token 级预算，而不是字符级预算。

---

## Q14：如果面试官问你“为什么不用简单字符串截断”，怎么回答？

答：

简单截断容易保留封面、目录、免责声明等低价值内容，反而丢掉关键财务段落。当前方案先去噪，再按财务关键词、数字、金额、百分比给段落打分，能更大概率把有限上下文留给有分析价值的信息。

---

# 35. 一句话总结

这个 PDF 切分模块的核心目标是：

> 把一份可能很长、很脏、结构复杂的金融 PDF，转换成一份带页码、去噪声、控制长度、尽量保留财务重点的文本结果，同时保留按页文本映射，为首轮 LLM 分析和后续 RAG 原文检索服务。
