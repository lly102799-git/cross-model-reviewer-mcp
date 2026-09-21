"""
路线 C：裸函数调用编排（理解原理用，零依赖，只需 Python 标准库）

完整演示「模型A生成 -> 模型B审校 -> 反馈回流 -> 模型A修订」闭环。
两个模型都走 OpenAI 兼容接口，可以是任意厂商（OpenAI / DeepSeek / Kimi / OpenRouter / 本地 vLLM）。

注意：本文件是**原理演示**，刻意保持最小。正式使用请走 MCP 服务器
（deepseek_reviewer.py），它额外提供：多厂商路由、两阶段确认、异步+流式、
跨厂商并发双审、结果缓存、报告落盘、状态落盘与网络重试。

运行前设置环境变量：
  A_BASE_URL / A_MODEL  / A_API_KEY   生产者（模型A）
  B_BASE_URL / B_MODEL  / B_API_KEY   审校者（模型B，建议与A不同厂商，实现错误去相关）

运行：python orchestrator.py
"""
import json
import os
import urllib.request

# ---------- 模型配置：A 生产，B 审校，各自独立 ----------
BASE_URL_A = os.environ.get("A_BASE_URL", "https://api.openai.com/v1")
MODEL_A = os.environ.get("A_MODEL", "gpt-4o")
KEY_A = os.environ["A_API_KEY"]

BASE_URL_B = os.environ.get("B_BASE_URL", "https://api.deepseek.com/v1")
MODEL_B = os.environ.get("B_MODEL", "deepseek-chat")
KEY_B = os.environ["B_API_KEY"]


def chat(messages, model, base_url, key, tools=None, temperature=0.3):
    """调用任意 OpenAI 兼容端点的最小封装"""
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["choices"][0]["message"]


# ---------- 审校工具的定义（挂在模型A上的 tool schema） ----------
REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "review_artifact",
        "description": (
            "调用独立的审校模型对产出物做评审，返回结构化反馈"
            "（verdict/issues/summary）。每次生成或修订完成后必须调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_type": {"type": "string", "enum": ["code", "doc"]},
                "content": {
                    "type": "string",
                    "description": "待审校的完整产出物。只传产物本身，不要附带任何作者解释",
                },
                "focus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "审查维度，如 correctness / security / edge-cases / structure",
                },
                "requirements": {
                    "type": "string",
                    "description": (
                        "任务的边界条件与验收标准（场景、硬性指标、必须满足的假设）。"
                        "审校者会逐条核对符合性——区分『做得对不对』与『是不是你要的』。"
                    ),
                },
            },
            "required": ["artifact_type", "content"],
        },
    },
}

# ---------- 审校模型B的提示词：只看产物、结构化输出 ----------
REVIEW_PROMPT = """你是独立的资深审校专家，与产出者不是同一人。
以挑剔视角审查以下{kind}，只基于产物本身判断，不要臆测作者意图。
默认产物中至少存在 3 个问题，逐项排查后确实没有再给 pass。
重点维度：{focus}。
{requirements_block}
输出严格 JSON（不要输出其他内容）：
{{"verdict": "pass 或 fail",
  "issues": [{{"severity": "blocker/major/minor",
               "location": "指到具体位置",
               "problem": "问题描述",
               "suggestion": "具体可执行的修改建议"}}],
  "summary": "一句话总体评价"}}

若给出了任务边界条件与验收标准，须逐条核对符合性；不满足任意一条 blocker 级要求即 fail。

{kind}内容：
{content}"""


def review_artifact(artifact_type, content, focus=None, requirements=None):
    """真正调用模型B执行审校——在MCP路线里，这一段就是MCP服务器的工具实现"""
    kind = "代码" if artifact_type == "code" else "文档"
    req_block = ""
    if requirements and str(requirements).strip():
        req_block = ("\n该产物的任务边界条件与验收标准如下，请逐条核对：\n"
                     + str(requirements).strip() + "\n")
    prompt = REVIEW_PROMPT.format(
        kind=kind, content=content,
        focus="、".join(focus) if focus else "自行判断",
        requirements_block=req_block,
    )
    msg = chat(
        [{"role": "user", "content": prompt}],
        MODEL_B, BASE_URL_B, KEY_B,
        temperature=0.1,  # 审校要稳定，低温度
    )
    return msg["content"]


# ---------- 主循环：A 与 B 的对话由消息历史串联 ----------
def run_task(task, max_rounds=3):
    messages = [
        {
            "role": "system",
            "content": (
                "你是生产者模型A。工作流程："
                "1) 产出初稿；"
                "2) 必须调用 review_artifact 工具送审（content 只放产物本身）；"
                "3) 若 verdict=fail，根据 issues 逐条修订后再次送审；"
                "4) verdict=pass 后，向用户输出最终版。"
            ),
        },
        {"role": "user", "content": task},
    ]
    for round_no in range(1, max_rounds + 1):
        msg = chat(messages, MODEL_A, BASE_URL_A, KEY_A, tools=[REVIEW_TOOL])
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # 模型A没有再调用审校工具 = 已拿到 pass，直接交付
            print(f"[完成] 共 {round_no} 轮")
            return msg["content"]

        for tc in tool_calls:
            args = json.loads(tc["function"]["arguments"])
            feedback = review_artifact(**args)
            print(f"--- 第 {round_no} 轮审校反馈（{MODEL_B}）---")
            print(feedback[:600], "\n")
            # 关键一步：审校反馈作为 tool 结果回流进对话历史，成为A修订的依据
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": feedback}
            )

    raise RuntimeError(f"超过 {max_rounds} 轮仍未通过，建议人工介入")


if __name__ == "__main__":
    final = run_task(
        "写一个 Python 函数 parse_csv_line：解析单行 CSV 字符串，"
        "容忍引号包裹内的逗号与转义引号，附 3 个测试用例。"
    )
    print("=== 最终交付 ===")
    print(final)
