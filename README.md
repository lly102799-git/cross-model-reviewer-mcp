# Cross Model Reviewer MCP

跨模型审校 MCP 服务器——在**同一个对话内**实现「生成 → 跨模型独立审校 → 修订」闭环。

解决的核心问题：AI 编程助手（Claude Code / WorkBuddy / Cursor 等）的底层模型既当运动员又当裁判，自我审校存在系统性盲区。本项目把"审校"封装为 MCP 工具，主模型完成产出后调用它，由**另一个厂商的模型**（默认 DeepSeek）做独立评审，结构化反馈自动回流主对话，主模型据此修订——用户全程不离开对话框。

## 工作原理

```
用户任务（含主题 / 场景 / 技术路线 / 约束）
        │
        ▼
┌─────────────────┐   产物(content)      ┌──────────────────┐
│  主模型（生成者）  │ ──────────────────▶ │  审校模型（独立）    │
│  Claude/GLM/...  │                     │  DeepSeek V4      │
│                 │ ◀────────────────── │  flash / pro      │
└─────────────────┘   结构化反馈JSON      └──────────────────┘
        │              {verdict, issues, summary}
        ▼
  据反馈修订 → 再送审 → verdict=pass → 交付
```

关键设计：审校模型**只看到产物本身 + 任务验收标准**，看不到主模型的对话历史与推理过程——既保证错误模式去相关（另一套"大脑"的盲区互补），又消除锚定效应。

## 核心特性

| 特性 | 说明 |
|---|---|
| 跨模型硬保证 | 审校模型由配置指定，与主对话模型无关厂商 |
| 按次选择模型 | 单次调用可指定 `deepseek-v4-flash`（快）/ `deepseek-v4-pro`（深度推理）等 |
| 模型白名单 | `REVIEWER_ALLOWED_MODELS` 控制可选模型，防误传昂贵模型 |
| 符合性审查通道 | `requirements` 参数传入任务边界条件与验收标准，审校模型逐条核对（区分"做得对不对"与"是不是你要的"） |
| 两阶段确认 | `draft_review`（起草+确认清单，零费用）→ 用户确认 → `submit_review`（提交即作废，防重放）；确认什么就送审什么 |
| 结构化反馈 | 强制 JSON 输出：`verdict / issues(severity, location, problem, suggestion) / summary` |
| 对抗性提示词 | 默认"至少存在 3 个问题"，要求逐项排查后才可给 pass |

## 安装

```bash
# 1. 准备 Python 环境（3.10+）
python -m venv .venv
.venv/Scripts/pip install fastmcp   # Windows
# .venv/bin/pip install fastmcp     # macOS/Linux

# 2. 注册到 MCP 宿主（以 ~/.workbuddy/mcp.json 或 claude mcp 配置为例）
# 见 config/mcp.example.json，填入你的 API Key
```

`config/mcp.example.json`：

```json
{
  "mcpServers": {
    "cross-model-reviewer": {
      "command": "<你的venv路径>/Scripts/python.exe",
      "args": ["<本仓库路径>/deepseek_reviewer.py"],
      "env": {
        "REVIEWER_BASE_URL": "https://api.deepseek.com/v1",
        "REVIEWER_MODEL": "deepseek-v4-flash",
        "REVIEWER_ALLOWED_MODELS": "deepseek-v4-flash,deepseek-v4-pro,deepseek-v4-flash-vision-exp",
        "REVIEWER_API_KEY": "<你的API Key>",
        "REVIEWER_TEMPERATURE": "0.1"
      }
    }
  }
}
```

更换审校厂商：`REVIEWER_BASE_URL` 指向任意 OpenAI 兼容接口（OpenRouter / LiteLLM / 本地 vLLM 等），并相应调整模型白名单。

## 使用流程（对话内）

```
用户：帮我写 XX（附应用场景、技术路线、约束条件）
主模型：产出交底书 / 数学模型 / 代码
用户：用 DeepSeek V4 Pro 校审
主模型：调用 draft_review → 向用户展示确认清单
       （审校模型 / 产物预览 / 侧重维度 / 提炼的验收标准）
用户：确认（或提出修改）
主模型：submit_review → DeepSeek 返回问题清单 → 逐条修订 → 再送审 → pass 交付
```

免确认直通道：`review_artifact`（仅当用户明确说"跳过确认直接送审"时使用）。

## 实测记录

全部为真实 API 调用实测（测试产物：故意带缺陷的复合年均增长率模型 `r = (V_end/V_begin)/n - 1`）：

1. **审校有效性**：flash 模型准确识别公式错误（应为开 n 次方而非线性除法）、代码与公式不一致、边界条件缺失
2. **模型分级**：同一产物 pro 模型多发现 1 个问题（4 vs 3）——复杂数学推导建议用 pro
3. **符合性审查对照**：数学正确但依赖 sympy 的实现，不带约束送审时审校未提依赖问题；带上"嵌入式实时 / 禁第三方库 / <1ms"约束后，首个 blocker 即为 sympy 违规与延迟风险
4. **两阶段机制**：起草（零费用）→ 提交 → 重复提交拦截 → 无效 ID 拦截，全部通过

## 目录结构

```
├── deepseek_reviewer.py        # MCP 服务器（fastmcp 实现，唯一核心文件）
├── config/mcp.example.json     # 宿主注册配置示例（占位符版本）
└── examples/
    ├── code-reviewer.md        # 替代方案：子代理定义（Claude Code .claude/agents/ 风格）
    └── orchestrator.py         # 替代方案：零依赖函数调用闭环（理解原理 / 自研集成用）
```

## 安全提示

- API Key 保存在本地 MCP 配置文件中，仅通过环境变量注入子进程，不会进入对话上下文
- 审校内容会发送到所配置的远端 API，敏感内容请配置本地模型（vLLM / Ollama 的 OpenAI 兼容接口）
