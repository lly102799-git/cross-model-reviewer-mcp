# Cross Model Reviewer MCP

跨模型审校 MCP 服务器——在**同一个对话内**实现「生成 → 跨模型独立审校 → 修订」闭环。

解决的核心问题：AI 编程助手（Claude Code / WorkBuddy / Cursor 等）的底层模型既当运动员又当裁判，自我审校存在系统性盲区。本项目把"审校"封装为 MCP 工具，主模型完成产出后调用它，由**另一个厂商的模型**做独立评审，结构化反馈自动回流主对话，主模型据此修订——用户全程不离开对话框。

## 工作原理

```
用户任务（含主题 / 场景 / 技术路线 / 约束）
        │
        ▼
┌─────────────────┐   产物(content)      ┌──────────────────┐
│  主模型（生成者）  │ ──────────────────▶ │  审校模型（独立）    │
│  Claude/GLM/...  │                     │  DeepSeek / 智谱   │
│                 │ ◀────────────────── │  另一套"大脑"      │
└─────────────────┘   结构化反馈JSON      └──────────────────┘
        │              {verdict, issues, summary}
        ▼
  据反馈修订 → 再送审 → verdict=pass → 交付
```

关键设计：审校模型**只看到产物本身 + 任务验收标准**，看不到主模型的对话历史与推理过程——既保证错误模式去相关（另一套"大脑"的盲区互补），又消除锚定效应。

## 核心特性

| 特性 | 说明 |
|---|---|
| 跨厂商多模型路由 | 内置 DeepSeek 与智谱 GLM 两厂商，按模型名自动路由到各自 `base_url` / `api_key`，两套 key 共存；同名模型用 `厂商:模型` 消歧（如 `zhipu:glm-5.3`） |
| 按次选择模型 | 单次调用可指定 `deepseek-v4-flash`（快）/ `deepseek-v4-pro`（深度推理）/ `glm-5.3`（旗舰）等 |
| 模型白名单 | `*_MODELS` 控制可选模型，防误传昂贵模型 |
| 符合性审查通道 | `requirements` 参数传入任务边界条件与验收标准，审校模型逐条核对（区分"做得对不对"与"是不是你要的"） |
| 两阶段确认 | `draft_review`（起草+确认清单，**零费用**）→ 用户确认 → `submit_review`；确认什么就送审什么 |
| 结构化反馈 | 强制 JSON 输出：`verdict / issues(severity, location, problem, suggestion) / summary` |
| 对抗性提示词 | 默认"至少存在 3 个问题"，要求逐项排查后才可给 pass |
| **异步 + 流式** | 送审秒回任务号，后台执行；SSE 流式让进度可见（实时字符数），同时根治"客户端先超时" |
| **跨厂商并发双审** | 两位不同厂商模型**盲审**同一产物，再由第三个模型仲裁合并，标出共识与分歧 |
| **结果缓存** | 内容+模型+参数全同则直接复用，**不重复计费** |
| **审校报告落盘** | 一键导出 Markdown 报告（分级问题表 / 分歧点 / 原始 JSON），便于归档与交付 |

## 工具一览（8 个）

| 工具 | 作用 |
|---|---|
| `list_models` | 列出可用模型、建议参数、缓存状态、双审方案。不确定用什么先调它 |
| `draft_review` | **默认流程第一步**：打包请求 + 生成确认清单，不调用模型、不产生费用 |
| `submit_review` | 第二步：确认后提交，秒回 `task_id` |
| `check_review` | 查进度与结果（含实时正文字符数） |
| `list_tasks` | 列出最近任务，找回忘记的 task_id |
| `dual_review` | 重型通道：跨厂商并发双审 + 仲裁合并（支持 `dry_run` 先看配置与耗时估算） |
| `export_review` | 把已完成任务导出为 Markdown 报告 |
| `review_artifact` | 免确认直通道，仅当用户明确说"跳过确认直接送审"时使用 |

## 安装

```bash
# 1. 准备 Python 环境（3.10+）
python -m venv .venv
.venv/Scripts/pip install fastmcp   # Windows
# .venv/bin/pip install fastmcp     # macOS/Linux

# 2. 注册到 MCP 宿主
# 见 config/mcp.example.json —— 填入你的 API Key 与路径
```

最小可跑配置（单厂商）：

```json
{
  "mcpServers": {
    "cross-model-reviewer": {
      "command": "<venv>/Scripts/python.exe",
      "args": ["<repo>/deepseek_reviewer.py"],
      "env": {
        "REVIEWER_API_KEY": "<你的 DeepSeek API Key>",
        "REVIEWER_DEFAULT_MODEL": "deepseek-v4-flash"
      }
    }
  }
}
```

加上第二厂商后即可使用双审：

```json
"ZHIPU_API_KEY": "<你的智谱 API Key>",
"ZHIPU_MODELS": "glm-5.3,glm-5.3-flash,glm-5.3-flashx,glm-4.6"
```

更换/新增审校厂商：在脚本 `PROVIDERS` 里加一项（label / base_url / api_key / models），env 里配对应变量即可。任何 OpenAI 兼容端点都能接（OpenRouter / LiteLLM / 本地 vLLM / Ollama）。

## 环境变量

<details>
<summary>点开完整列表（含默认值）</summary>

| 变量 | 默认 | 说明 |
|---|---|---|
| `REVIEWER_BASE_URL` | `https://api.deepseek.com/v1` | 厂商 1 端点（别名 `DEEPSEEK_BASE_URL`） |
| `REVIEWER_API_KEY` | — | 厂商 1 密钥（别名 `DEEPSEEK_API_KEY`） |
| `REVIEWER_ALLOWED_MODELS` | flash / pro / vision-exp | 厂商 1 白名单（别名 `DEEPSEEK_MODELS`） |
| `REVIEWER_DEFAULT_MODEL` | `deepseek-v4-flash` | 不指定模型时的默认值 |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 厂商 2 端点 |
| `ZHIPU_API_KEY` | — | 厂商 2 密钥；不配则该厂商不出现在可选列表 |
| `ZHIPU_MODELS` | glm-5.3 系列 + glm-4.6 | 厂商 2 白名单 |
| `REVIEWER_TEMPERATURE` | `0.1` | 低温度，求稳定复现 |
| `REVIEWER_MAX_TOKENS` | `16384` | 默认输出上限（可按次覆盖） |
| `REVIEWER_REASONING_EFFORT` | `medium` | 默认思考强度（可按次覆盖） |
| `REVIEWER_TIMEOUT` | `1200` | 非流式总超时（秒） |
| `REVIEWER_STREAM_IDLE` | `300` | 流式"无数据"超时（秒） |
| `REVIEWER_NET_RETRY` | `1` | 网络类失败自动重试次数 |
| `REVIEWER_TASK_WAIT_MAX` | `300` | 单审同步等待上限，超过转轮询 |
| `REVIEWER_DUAL_MODELS` | 空（自动选各厂商 flash 档） | 双审固定用哪两个模型 |
| `REVIEWER_ARBITRATE_MODEL` | `deepseek-v4-pro` | 仲裁模型 |
| `REVIEWER_CACHE` | `true` | 结果缓存开关 |
| `REVIEWER_CACHE_TTL_DAYS` | `30` | 缓存有效期（天） |
| `REVIEWER_CACHE_MAX` | `300` | 缓存条目上限，超出按最旧淘汰 |
| `REVIEWER_REPORT_DIR` | `.reviewer_state/reports` | 报告默认落盘目录 |

</details>

## 使用流程（对话内）

```
用户：帮我写 XX（附应用场景、技术路线、约束条件）
主模型：产出交底书 / 数学模型 / 代码
用户：用 cross-model-reviewer 校审
主模型：调用 draft_review → 向用户展示确认清单
       （审校模型 / 产物预览 / 侧重维度 / 提炼的验收标准 /
         max_tokens 与 reasoning_effort 的当前值、来源与推荐值）
用户：确认（或提出修改）
主模型：submit_review → 秒回任务号 → check_review 轮询
       → 拿到问题清单 → 逐条修订 → 再送审 → pass 交付
```

重要产物可改走双审：

```
用户：对这份产物做跨厂商并发双审 + 仲裁合并，先 dry_run 给我看配置
主模型：dual_review(dry_run=True) → 展示模型 / 参数 / 耗时估算 / 调用次数
用户：确认
主模型：dual_review(...) → 两位审校者并发盲审 → 第三个模型仲裁合并
       → 报告自动落盘
```

免确认直通道：`review_artifact`（仅当用户明确说"跳过确认直接送审"时使用）。

提示词模板见 [`examples/prompt-template.md`](examples/prompt-template.md) —— 可直接复制粘贴使用。

## 实测记录

全部为真实 API 调用实测，非模拟。

**审校有效性**（测试产物：故意带缺陷的复合年均增长率模型 `r = (V_end/V_begin)/n - 1`）

1. flash 档准确识别公式错误（应为开 n 次方而非线性除法）、代码与公式不一致、边界条件缺失
2. 同产物 pro 档多发现 1 个问题（4 vs 3）——复杂数学推导建议用 pro
3. **符合性审查对照**：数学正确但依赖 sympy 的实现，不带约束送审时未提依赖问题；带上"嵌入式实时 / 禁第三方库 / <1ms"约束后，首个 blocker 即为 sympy 违规与延迟风险
4. 两阶段机制：起草（零费用）→ 提交 → 重复提交拦截 → 无效 ID 拦截，全部通过

**性能与耗时**

| 场景 | 配置 | 实测 |
|---|---|---|
| 智谱初筛（133 字符） | `glm-5.3-flashx` / 8192 / low | **6.7s** |
| 单审（237 字符） | `deepseek-v4-flash` / 8192 / low | 8.3s |
| 单审（237 字符） | `glm-5.3-flash` / 8192 / low | 15.0s |
| 双审**不仲裁**（237 字符） | 两厂商并发 / 8192 / low | **18.0s** |
| 双审**+仲裁**（237 字符） | 上述 + pro 档仲裁 | **135.1s**（仲裁独占 116s） |
| 同请求第二次调用 | 任意 | **0.002s**（命中缓存） |

**异步化的效果**

- `submit_review` 墙钟 **0.09s** 返回 `task_id`（改造前同步阻塞直至审校结束）
- 进度可见：正文字符数实时增长（实测 0 → 387 → 763 → 1215 → 1981 → 2009）
- 缓存往返：MISS 1.30s → HIT 0.002s，**528×**

**双审的价值**

两厂商命中集互补度很高。同一份带缺陷文档，一方给 6 条问题、另一方 5 条，仲裁合并后 7 条，
其中 6 条被标记为双方共识（`agreement: both`）、1 条为单方补充。

## 设计要点：为什么这样实现

**为什么异步 + 流式，而不是调大 timeout**

MCP 工具调用对客户端是**同步阻塞**的，宿主客户端侧的工具超时远小于服务端 `REVIEWER_TIMEOUT`——所以调大 timeout 只会让客户端断得更久。正解是把审校放到后台线程、立即返回任务号，再用 `check_review` 轮询。而 SSE 流式让思考/生成过程持续有字节流动（socket 不空闲超时），天然产出进度指标，还顺带保住 `total_tokens` / `reasoning_tokens` 统计。

**思考强度档位因厂商而异**

| 厂商 | low | medium | high | max |
|---|---|---|---|---|
| DeepSeek | ✓ | ✓ | ✓ | ✓ |
| 智谱 GLM | ✓ | **✗ HTTP 400（code 1210）** | ✓ | ✓ |

智谱档位只有 low/high/max，传 `medium` 直接 400。脚本按厂商映射（智谱 `medium` → `high`），并把映射结果写进确认清单，用户能看到"实际发出去的是什么"。

**GLM 强制思考且无法关闭**

`thinking:{"type":"disabled"}` 会 400。思维链占用 `max_tokens` 额度，设太小会返回空字符串；脚本在正文为空时自动加倍额度重试一次。实测 18k 字符文档：8192 → 正文为空；32768 → 903s 后被服务端断开；**16384 + 低档 → 400s 成功**。即额度一味调大反而会因生成超时失败。

**缓存做在调用层而非工具层**

缓存放在 `call_model()` 内部，因此单审 / 双审 / 仲裁三条路径**自动全部受益**，新增功能无需再关心缓存。key = `sha256(provider|model|budget|effort|temperature|prompt)`，任一变化即穿透。

## 目录结构

```
├── deepseek_reviewer.py            # MCP 服务器（fastmcp 实现，唯一核心文件；文件名沿用历史名）
├── config/mcp.example.json         # 宿主注册配置示例（占位符版本）
├── examples/
│   ├── prompt-template.md          # 提示词模板（最简 / 单审 / 双审，可直接粘）
│   ├── code-reviewer.md            # 替代方案：子代理定义（Claude Code .claude/agents/ 风格）
│   └── orchestrator.py             # 替代方案：零依赖函数调用闭环（理解原理 / 自研集成用）
└── .gitignore
```

运行时会生成 `.reviewer_state/`（草稿、任务、缓存、报告、事件日志），**已加入 .gitignore**——
其中会包含被审校的稿件正文，请勿入库。

## 版本历史

| 版本 | 主要变化 |
|---|---|
| v1 | 单厂商（DeepSeek），同步阻塞调用，3 个工具 |
| v2 | 多厂商路由（DeepSeek + 智谱 GLM），模型白名单，`list_models` |
| v3 | 异步任务化 + SSE 流式 + 按次参数 + 状态落盘 + 网络重试 |
| v3.1 | 确认清单补强：参数来源标注、推荐值、风险提示 |
| v3.2 | 默认思考强度改 medium；厂商档位映射（智谱 medium → high） |
| **v3.3** | **跨厂商并发双审 + 仲裁合并；结果缓存；审校报告落盘** |
| v3.3.1 | 修正确认清单中推荐值映射说明的措辞歧义 |

## 安全提示

- API Key 仅通过环境变量注入子进程，不会进入对话上下文；仓库内所有配置均为占位符
- `.reviewer_state/` 含被审校内容与结果，已默认不入库
- 审校内容会发送到所配置的远端 API，敏感内容请配置本地模型（vLLM / Ollama 的 OpenAI 兼容接口）
- 任务与草稿上限分别为 50 / 30 条，超出按最旧淘汰**仅终态**任务（运行中的不会被清理）
