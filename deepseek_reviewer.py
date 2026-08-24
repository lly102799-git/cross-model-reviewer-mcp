"""DeepSeek 跨模型审校 MCP 服务器（两阶段确认版）

默认流程（带用户确认）：
  1. 主模型调用 draft_review 打包审校请求 -> 返回 request_id + 确认清单
  2. 主模型把确认清单展示给用户，用户确认或修改
  3. 用户确认后调用 submit_review(request_id) -> 实际调用审校模型，反馈回流
     （用户有修改时重新 draft_review 再提交）

免确认直通道：review_artifact（仅当用户明确说"不用确认直接送审"时使用）。

环境变量（在 mcp.json 的 env 中配置）:
  REVIEWER_BASE_URL       审校模型 API 地址，默认 DeepSeek 官方
  REVIEWER_MODEL          默认审校模型 ID（未按次指定时使用）
  REVIEWER_ALLOWED_MODELS 允许按次切换的模型白名单，逗号分隔
  REVIEWER_API_KEY        审校模型的 API Key（必填）
  REVIEWER_TEMPERATURE    审校温度，默认 0.1
"""
import json
import os
import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("cross-model-reviewer")

BASE_URL = os.environ.get("REVIEWER_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("REVIEWER_MODEL", "deepseek-v4-flash")
API_KEY = os.environ.get("REVIEWER_API_KEY", "")
TEMPERATURE = float(os.environ.get("REVIEWER_TEMPERATURE", "0.1"))
ALLOWED_MODELS = [
    m.strip()
    for m in os.environ.get(
        "REVIEWER_ALLOWED_MODELS",
        "deepseek-v4-flash,deepseek-v4-pro,deepseek-v4-flash-vision-exp",
    ).split(",")
    if m.strip()
]

_drafts = {}
_draft_seq = 0
_MAX_DRAFTS = 20

REVIEW_PROMPT = """你是独立的资深审校专家，与产出者不是同一个模型，也不共享任何上下文。
请以挑剔的视角审查以下{kind}，只基于产物本身判断，不要臆测作者意图。
默认产物中至少存在 3 个问题，逐项排查后确实没有再给 pass。
{requirements_block}
输出严格 JSON（不要输出其他任何内容）：
{{"verdict": "pass 或 fail",
  "issues": [{{"severity": "blocker/major/minor",
               "location": "具体位置（函数名/章节/行号）",
               "problem": "问题描述",
               "suggestion": "具体修改建议"}}],
  "summary": "一句话总体评价"}}

审查维度（按需侧重）：数学推导正确性、边界条件、假设合理性、代码实现与公式一致性、数值稳定性、文档结构。
若给出了任务边界条件与验收标准，须逐条核对符合性；不满足任意一条 blocker 级要求即 fail。

{kind}内容：
{content}"""


def build_requirements_block(requirements: str) -> str:
    if not requirements.strip():
        return ""
    return (
        "\n任务边界条件与验收标准（这是产物必须满足的要求，逐条核对符合性，"
        "不满足任意一条即为 blocker 问题）：\n" + requirements.strip() + "\n"
    )


def _resolve_model(reviewer_model: str):
    model = (reviewer_model or "").strip() or MODEL
    if model not in ALLOWED_MODELS:
        return None
    return model


def call_reviewer(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def _make_prompt(artifact_type, content, focus, requirements, kind):
    prompt = REVIEW_PROMPT.format(
        kind=kind,
        content=content,
        requirements_block=build_requirements_block(requirements),
    )
    if focus:
        prompt += "\n额外侧重维度：" + focus
    return prompt


@mcp.tool()
def draft_review(artifact_type: str, content: str, focus: str = "", reviewer_model: str = "", requirements: str = "") -> str:
    """跨模型审校 · 默认流程第一步：打包审校请求，生成给用户确认的清单（不会调用审校模型、不产生费用）。
    返回 request_id 和 confirm_checklist。请把 confirm_checklist 完整展示给用户：
    用户确认无误后调用 submit_review(request_id)；用户提出修改则按修改意见重新 draft_review。

    Args:
        artifact_type: 产物类型，"code"（代码）或 "doc"（文档）
        content: 待审校的完整产物内容（只传产物本身，不要附带作者解释）
        focus: 可选，额外侧重维度，逗号分隔，如 "数值稳定性,边界条件"
        reviewer_model: 可选，审校模型：deepseek-v4-flash（默认，快）、
            deepseek-v4-pro（深度推理，复杂数学模型推荐）、deepseek-v4-flash-vision-exp（视觉实验版）
        requirements: 强烈建议：任务的边界条件与验收标准（发明主题、应用场景、技术路线、
            精度/性能/合规等硬性指标、必须满足的假设）。从用户对话中提炼，送审前须让用户确认。
    """
    global _draft_seq
    if not API_KEY:
        return json.dumps({"error": "REVIEWER_API_KEY 未配置，请在 mcp.json 的 env 中填入"}, ensure_ascii=False)
    model = _resolve_model(reviewer_model)
    if model is None:
        return json.dumps({"error": f"模型 {reviewer_model} 不在允许列表内，可选: {ALLOWED_MODELS}"}, ensure_ascii=False)
    kind = "代码" if artifact_type == "code" else "文档"
    _draft_seq += 1
    rid = f"rev-{_draft_seq:04d}"
    _drafts[rid] = {
        "model": model,
        "prompt": _make_prompt(artifact_type, content, focus, requirements, kind),
    }
    while len(_drafts) > _MAX_DRAFTS:
        _drafts.pop(next(iter(_drafts)))
    return json.dumps({
        "request_id": rid,
        "confirm_checklist": {
            "reviewer_model": model,
            "artifact_type": artifact_type,
            "artifact_chars": len(content),
            "artifact_preview": content[:150] + ("..." if len(content) > 150 else ""),
            "focus": focus or "（无）",
            "requirements": requirements.strip() or "（无——强烈建议补充任务边界条件后重新起草）",
        },
        "next_step": "将 confirm_checklist 展示给用户确认；确认后 submit_review(request_id)，有修改则重新 draft_review",
    }, ensure_ascii=False)


@mcp.tool()
def submit_review(request_id: str) -> str:
    """跨模型审校 · 默认流程第二步：提交已经用户确认的审校请求。
    只提交用户在确认清单中看到过的内容（与 draft_review 的 request_id 一一对应），提交后草稿作废。
    """
    if not API_KEY:
        return json.dumps({"error": "REVIEWER_API_KEY 未配置"}, ensure_ascii=False)
    params = _drafts.pop(request_id, None)
    if params is None:
        return json.dumps(
            {"error": f"草稿 {request_id} 不存在或已提交，请重新 draft_review"},
            ensure_ascii=False,
        )
    try:
        review = call_reviewer(params["prompt"], params["model"])
    except Exception as e:
        return json.dumps({"error": f"审校调用失败（{params['model']}）: {e}"}, ensure_ascii=False)
    return json.dumps(
        {"reviewer_model": params["model"], "review": review},
        ensure_ascii=False,
    )


@mcp.tool()
def review_artifact(artifact_type: str, content: str, focus: str = "", reviewer_model: str = "", requirements: str = "") -> str:
    """跨模型审校 · 免确认直通道：跳过用户确认直接送审。
    仅当用户明确说"不用确认直接送审/跳过确认"时使用；默认流程请用 draft_review + submit_review。

    Args:
        artifact_type: 产物类型，"code"（代码）或 "doc"（文档）
        content: 待审校的完整产物内容
        focus: 可选，额外侧重维度
        reviewer_model: 可选，审校模型（deepseek-v4-flash / deepseek-v4-pro / deepseek-v4-flash-vision-exp）
        requirements: 强烈建议：任务边界条件与验收标准
    """
    if not API_KEY:
        return json.dumps({"error": "REVIEWER_API_KEY 未配置，请在 mcp.json 的 env 中填入"}, ensure_ascii=False)
    model = _resolve_model(reviewer_model)
    if model is None:
        return json.dumps({"error": f"模型 {reviewer_model} 不在允许列表内，可选: {ALLOWED_MODELS}"}, ensure_ascii=False)
    kind = "代码" if artifact_type == "code" else "文档"
    prompt = _make_prompt(artifact_type, content, focus, requirements, kind)
    try:
        review = call_reviewer(prompt, model)
    except Exception as e:
        return json.dumps({"error": f"审校调用失败（{model}）: {e}"}, ensure_ascii=False)
    return json.dumps({"reviewer_model": model, "review": review}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
