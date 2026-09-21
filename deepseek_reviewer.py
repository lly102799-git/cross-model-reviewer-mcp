"""跨模型审校 MCP 服务器（多厂商 · 异步任务 · 缓存 · 双审 · 报告）

支持厂商：DeepSeek / 智谱 GLM。按模型 ID 自动路由到对应厂商的 base_url + api_key。

流程（带用户确认）：
  1. draft_review   打包审校请求 -> 返回 request_id + 确认清单（不调用模型、不产生费用）
  2. 把确认清单展示给用户确认或修改
  3. submit_review  用户确认后送审 -> 立刻返回 task_id（后台线程执行，不再阻塞）
  4. check_review   轮询进度与结果（未完成时返回已耗时与已生成字符数）

免确认直通道：review_artifact（仅在用户明确说"跳过确认"时使用）。
重型通道：dual_review —— 跨厂商并发双审 + 第三模型仲裁合并，适合重要产物。

模型寻址：直接写模型名（如 glm-5.3）即可；若两厂商出现同名模型，
用 "厂商:模型" 形式消歧（如 zhipu:glm-5.3）。调用 list_models 可查全部可用模型。

结果缓存：相同 内容+模型+参数 会命中缓存直接复用（不重复计费），meta 里标 cached=true。
报告落盘：export_review 可把任意任务导出为 Markdown；各审校工具也可传 report_path 自动落盘。

环境变量（在 mcp.json 的 env 中配置）:
  # --- DeepSeek ---
  REVIEWER_BASE_URL        DeepSeek API 地址，默认 https://api.deepseek.com/v1
  REVIEWER_API_KEY         DeepSeek API Key
  REVIEWER_ALLOWED_MODELS  DeepSeek 可用模型白名单，逗号分隔
  # --- 智谱 GLM ---
  ZHIPU_BASE_URL           智谱 API 地址，默认 https://open.bigmodel.cn/api/paas/v4
  ZHIPU_API_KEY            智谱 API Key
  ZHIPU_MODELS             智谱可用模型白名单，逗号分隔
  # --- 通用 ---
  REVIEWER_DEFAULT_MODEL   默认审校模型，默认 deepseek-v4-flash
  REVIEWER_TEMPERATURE     审校温度，默认 0.1
  REVIEWER_MAX_TOKENS      默认输出上限，默认 32768；单次可用 max_tokens 参数覆盖
                           （智谱 GLM-5.x 始终思考、思维链占用该额度，
                             低于 4096 基本不可用；返回空正文时会自动加倍重试一次）
  REVIEWER_REASONING_EFFORT 默认思考强度，默认 medium；单次可用参数覆盖。
                           注意智谱只支持 low/high/max，收到 medium 会自动映射为 high。
  REVIEWER_TIMEOUT         非流式调用的总超时秒数，默认 1200
  REVIEWER_STREAM_IDLE     流式调用的「无数据」超时秒数，默认 300
  REVIEWER_NET_RETRY       网络类失败自动重试次数，默认 1
  REVIEWER_TASK_WAIT_MAX   submit 时 wait_seconds 的上限，默认 300
  # --- 缓存 ---
  REVIEWER_CACHE           结果缓存开关，默认 1（设 0 关闭）
  REVIEWER_CACHE_TTL_DAYS  缓存有效期天数，默认 30（0 = 不过期）
  REVIEWER_CACHE_MAX       缓存条目上限，默认 300，超出按最旧淘汰
  # --- 双审 ---
  REVIEWER_DUAL_MODELS     双审默认模型（逗号分隔 2 个），默认自动选各厂商 flash 档
  REVIEWER_ARBITRATE_MODEL 仲裁模型，默认 deepseek-v4-pro；设 none 则只并排返回不仲裁
  # --- 报告 ---
  REVIEWER_REPORT_DIR      报告默认落盘目录，默认 <脚本目录>/.reviewer_state/reports
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("cross-model-reviewer")


# ---------------------------------------------------------------- 配置

def _env(*names, default=""):
    """按顺序取第一个非空环境变量，用于兼容旧变量名。"""
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _split_list(raw, fallback):
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    return items or list(fallback)


def _flag(name, default=True):
    raw = _env(name, default="1" if default else "0").lower()
    return raw not in ("0", "false", "no", "off", "none")


TEMPERATURE = float(_env("REVIEWER_TEMPERATURE", default="0.1"))
MAX_TOKENS = int(_env("REVIEWER_MAX_TOKENS", default="32768"))
MAX_ESCALATE_TOKENS = 131072          # 空正文时递增额度的天花板
TIMEOUT = int(_env("REVIEWER_TIMEOUT", default="1200"))
STREAM_IDLE_TIMEOUT = int(_env("REVIEWER_STREAM_IDLE", default="300"))
NET_RETRY = int(_env("REVIEWER_NET_RETRY", default="1"))
TASK_WAIT_MAX = int(_env("REVIEWER_TASK_WAIT_MAX", default="300"))
REASONING_EFFORT = _env("REVIEWER_REASONING_EFFORT").lower()

CACHE_ENABLED = _flag("REVIEWER_CACHE", default=True)
CACHE_TTL_DAYS = int(_env("REVIEWER_CACHE_TTL_DAYS", default="30"))
CACHE_MAX = int(_env("REVIEWER_CACHE_MAX", default="300"))

DUAL_MODELS = _env("REVIEWER_DUAL_MODELS")
ARBITRATE_MODEL = _env("REVIEWER_ARBITRATE_MODEL", default="deepseek-v4-pro")

VALID_EFFORTS = ("low", "medium", "high", "max")
MIN_SAFE_TOKENS = 4096               # 低于此值 GLM 的思维链会吃光额度

# 各厂商实际支持的思考强度档位。智谱不接受 medium（HTTP 400 / code 1210）。
EFFORT_SUPPORT = {
    "deepseek": ("low", "medium", "high", "max"),
    "zhipu": ("low", "high", "max"),
}
# 厂商不支持的档位 -> 就近映射。智谱的 low 实测几乎不思考（reasoning_tokens 仅个位数），
# 故把 medium 向上映射为 high，保住「比最低档更充分」的意图；嫌慢可显式传 low。
EFFORT_FALLBACK = {"zhipu": {"medium": "high"}}

MAX_TASKS = 50
MAX_DRAFTS = 30

DEFAULT_MODELS = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"],
    "zhipu": ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx", "glm-4.6"],
}

MODEL_DESC = {
    "deepseek-v4-flash": "快速通用审校（默认）",
    "deepseek-v4-pro": "深度推理，复杂数学模型推荐",
    "deepseek-v4-flash-vision-exp": "视觉实验版",
    "glm-5.3": "智谱旗舰，深度审校",
    "glm-5.3-flash": "智谱快速版",
    "glm-5.3-flashx": "智谱极速版",
    "glm-4.6": "智谱上一代稳定版",
}

# 每个模型的参数建议，供 list_models 展示、供主模型决定 max_tokens / reasoning_effort
MODEL_ADVICE = {
    "deepseek-v4-flash": "默认即可；长文档可加 reasoning_effort=low 提速",
    "deepseek-v4-pro": "复杂数学模型建议 reasoning_effort=high、max_tokens>=32768",
    "deepseek-v4-flash-vision-exp": "实验模型，仅作交叉验证用",
    "glm-5.3": "强制思考、最慢；深度审校用 high，赶时间降到 low",
    "glm-5.3-flash": "强制思考；low 约 20s 量级，medium 约 1.5 倍",
    "glm-5.3-flashx": "智谱最快通道，适合快速初筛；长文档仍建议 max_tokens>=16384",
    "glm-4.6": "上一代，兼容性好；同样强制思考",
}

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": _env("DEEPSEEK_BASE_URL", "REVIEWER_BASE_URL",
                         default="https://api.deepseek.com/v1"),
        "api_key": _env("DEEPSEEK_API_KEY", "REVIEWER_API_KEY"),
        "models": _split_list(_env("DEEPSEEK_MODELS", "REVIEWER_ALLOWED_MODELS"),
                              DEFAULT_MODELS["deepseek"]),
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url": _env("ZHIPU_BASE_URL", default="https://open.bigmodel.cn/api/paas/v4"),
        "api_key": _env("ZHIPU_API_KEY"),
        "models": _split_list(_env("ZHIPU_MODELS"), DEFAULT_MODELS["zhipu"]),
    },
}

DEFAULT_MODEL_SPEC = _env("REVIEWER_DEFAULT_MODEL", "REVIEWER_MODEL",
                          default="deepseek-v4-flash")

# 模型名 -> 厂商；同名模型不入索引，需用 "厂商:模型" 消歧
MODEL_INDEX = {}
_ambiguous = set()
for _pname, _pcfg in PROVIDERS.items():
    for _mid in _pcfg["models"]:
        if _mid in MODEL_INDEX:
            _ambiguous.add(_mid)
        else:
            MODEL_INDEX[_mid] = _pname
for _mid in _ambiguous:
    MODEL_INDEX.pop(_mid, None)


# ---------------------------------------------------------------- 落盘状态

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".reviewer_state")
DRAFTS_FILE = os.path.join(STATE_DIR, "drafts.json")
TASKS_FILE = os.path.join(STATE_DIR, "tasks.json")
EVENTS_LOG = os.path.join(STATE_DIR, "events.log")
CACHE_DIR = os.path.join(STATE_DIR, "cache")
REPORT_DIR = _env("REVIEWER_REPORT_DIR") or os.path.join(STATE_DIR, "reports")

_lock = threading.RLock()


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _fmt_ts(ts):
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _log_event(msg):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(EVENTS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_now_str()} | {msg}\n")
    except Exception:
        pass


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _atomic_write(path, obj):
    """原子写盘，避免中途崩溃留下半个文件。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        _log_event(f"写盘失败 {path}: {e}")


# ---------------------------------------------------------------- 结果缓存

def _cache_key(provider_name, model_id, prompt, budget, effort):
    h = hashlib.sha256()
    h.update(f"v1|{provider_name}|{model_id}|{budget}|{effort}|{TEMPERATURE}|{prompt}"
             .encode("utf-8"))
    return h.hexdigest()[:32]


def _cache_get(key):
    if not CACHE_ENABLED:
        return None
    path = os.path.join(CACHE_DIR, key + ".json")
    if not os.path.isfile(path):
        return None
    if CACHE_TTL_DAYS > 0:
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days > CACHE_TTL_DAYS:
            try:
                os.remove(path)
            except OSError:
                pass
            return None
    return _read_json(path, None)


def _prune_cache():
    try:
        files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)
                 if f.endswith(".json")]
        if len(files) <= CACHE_MAX:
            return
        files.sort(key=os.path.getmtime)
        for path in files[:len(files) - CACHE_MAX]:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:
        pass


def _cache_put(key, content, meta):
    if not CACHE_ENABLED:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _atomic_write(os.path.join(CACHE_DIR, key + ".json"), {
            "key": key,
            "cached_at": _now_str(),
            "cached_ts": time.time(),
            "content": content,
            "meta": meta,
        })
        _prune_cache()
    except Exception as e:
        _log_event(f"缓存写入失败: {e}")


def cache_stats():
    try:
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
        total = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    except Exception:
        files, total = [], 0
    return {
        "enabled": CACHE_ENABLED,
        "entries": len(files),
        "size_kb": round(total / 1024, 1),
        "ttl_days": CACHE_TTL_DAYS,
        "max_entries": CACHE_MAX,
        "dir": CACHE_DIR,
    }


# ---------------------------------------------------------------- 模型寻址

def resolve_model(spec):
    """解析模型标识 -> (provider_name, model_id, error)。"""
    spec = (spec or "").strip() or DEFAULT_MODEL_SPEC
    if ":" in spec:
        pname, mid = spec.split(":", 1)
        pname, mid = pname.strip().lower(), mid.strip()
        if pname not in PROVIDERS:
            return None, None, f"未知厂商 {pname}，可选: {list(PROVIDERS)}"
        if mid not in PROVIDERS[pname]["models"]:
            return None, None, (
                f"模型 {mid} 不在 {pname} 白名单内，可选: {PROVIDERS[pname]['models']}"
            )
        return pname, mid, None
    if spec in MODEL_INDEX:
        return MODEL_INDEX[spec], spec, None
    if spec in _ambiguous:
        return None, None, f"模型 {spec} 在多个厂商下存在，请用 厂商:模型 形式指定"
    all_models = {p: c["models"] for p, c in PROVIDERS.items()}
    return None, None, f"模型 {spec} 不在白名单内。各厂商可用模型: {all_models}"


def map_effort(provider_name, effort):
    """把请求的思考强度适配到该厂商真正支持的档位。

    返回 (实际使用值, 说明或 None)。厂商不支持且无映射时返回 ("", 错误说明)。
    """
    if not effort:
        return effort, None
    support = EFFORT_SUPPORT.get(provider_name)
    if not support or effort in support:
        return effort, None
    fallback = EFFORT_FALLBACK.get(provider_name, {}).get(effort)
    if fallback:
        return fallback, (
            f"{PROVIDERS[provider_name]['label']} 不支持 reasoning_effort={effort}，"
            f"已自动映射为 {fallback}（该厂商支持: {'/'.join(support)}）"
        )
    return "", (
        f"{PROVIDERS[provider_name]['label']} 不支持 reasoning_effort={effort}，"
        f"可选: {'/'.join(support)}"
    )


def resolve_params(max_tokens, reasoning_effort, provider_name=""):
    """校验并适配按次传入的 max_tokens / reasoning_effort。

    返回 (budget, effort, error, note)：effort 是**适配后真正会发出去**的值，
    note 说明是否发生过厂商档位映射。
    """
    try:
        budget = int(max_tokens) if max_tokens else MAX_TOKENS
    except (TypeError, ValueError):
        return 0, "", f"max_tokens 必须是整数，收到: {max_tokens!r}", None
    if budget < 256:
        return 0, "", f"max_tokens={budget} 过小，至少 256；建议 >= 8192", None
    if budget > MAX_ESCALATE_TOKENS:
        return 0, "", f"max_tokens={budget} 超出上限 {MAX_ESCALATE_TOKENS}", None

    effort = (reasoning_effort or "").strip().lower()
    if effort and effort not in VALID_EFFORTS:
        return 0, "", (
            f"reasoning_effort={reasoning_effort!r} 不合法，可选: "
            f"{list(VALID_EFFORTS)}，或留空用默认值"
        ), None
    if not effort:
        effort = REASONING_EFFORT

    effort, note = map_effort(provider_name, effort)
    if not effort:
        return 0, "", note, None
    return budget, effort, None, note


# 实测得出的保守推荐值（GLM 强制思考，思维链会占额度；16384 是已验证的稳妥额度）
RECOMMENDED_BUDGET = 16384
RECOMMENDED_EFFORT = "medium"
LONG_DOC_CHARS = 12000


def suggest_params(provider_name, model_id, content_chars, budget, effort,
                   user_set_budget, user_set_effort, effort_note=None):
    """生成参数说明块，用于让用户看到「这次到底用什么参数、从哪来的、建议是什么」。

    MCP 工具参数不会弹窗给用户选择，只能靠 draft_review 的确认清单把参数摊开给用户看。
    """
    notes = []
    if effort_note:
        notes.append(effort_note)
    if provider_name == "zhipu":
        notes.append("GLM 强制思考且无法关闭，思维链会占用 max_tokens，上限设太小会返回空正文")
    if effort in ("medium", "high", "max") and content_chars > LONG_DOC_CHARS:
        notes.append(
            f"思考强度 {effort} 叠加 {content_chars} 字符长文档：耗时与思维链 token 都会明显上升，"
            "正文为空就升 max_tokens，等太久就降为 low"
        )
    elif content_chars > LONG_DOC_CHARS:
        notes.append(f"文档 {content_chars} 字符偏长，GLM 审校可能耗时数分钟")
    if budget < MIN_SAFE_TOKENS:
        notes.append(f"max_tokens={budget} 低于安全线 {MIN_SAFE_TOKENS}，很可能正文为空")

    rec_effort, rec_note = map_effort(provider_name, RECOMMENDED_EFFORT)
    recommended = {"max_tokens": RECOMMENDED_BUDGET, "reasoning_effort": rec_effort}
    if rec_note:
        # 措辞必须点明「这是推荐基线被映射」，否则紧挨着 recommended 容易被误读成
        # 「本次调用传的值被改掉了」。本次调用的映射说明走 advice（effort_note）。
        recommended["note"] = f"推荐基线为 {RECOMMENDED_EFFORT}；{rec_note}"

    block = {
        "max_tokens": budget,
        "reasoning_effort": effort or "（未传，服务端默认）",
        "max_tokens_source": "调用方指定" if user_set_budget else (
            f"env 默认（REVIEWER_MAX_TOKENS={MAX_TOKENS}）"),
        "reasoning_effort_source": "调用方指定" if user_set_effort else (
            f"env 默认（REVIEWER_REASONING_EFFORT={REASONING_EFFORT}）"
            if REASONING_EFFORT else "未设置，服务端默认"),
        "effort_supported_by_provider": list(EFFORT_SUPPORT.get(provider_name, ())),
        "recommended": recommended,
    }
    if budget != RECOMMENDED_BUDGET or effort != rec_effort:
        block["note"] = (
            f"当前参数与推荐值（{RECOMMENDED_BUDGET} / {rec_effort}）不同；"
            "若担心超时或空正文，可让用户改回推荐值"
        )
    block["advice"] = "；".join(notes) if notes else "当前参数适用"
    return block


# ---------------------------------------------------------------- 审校提示词

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

ARBITRATION_PROMPT = """你是审校仲裁专家。下面是两位独立审校者对同一份{kind}的审查意见。
两人来自不同厂商的大模型，彼此不知道对方存在，因此命中集往往互补。

你的任务是**合并去重并给出最终可执行结论**，规则：
- 两人都指出的问题 -> agreement 填 "both"，视为高置信度
- 只有一人指出的问题 -> agreement 填该审校者代号（"A" 或 "B"），保留但视为待复核
- 两人结论直接矛盾（一方认为有问题、另一方明确认为无问题）-> 放进 disputed
- **不要凭空新增**两位审校者都没提到的问题
- 严重级别取两人中较高者

输出严格 JSON（不要输出其他任何内容）：
{{"verdict": "pass 或 fail",
  "issues": [{{"severity": "blocker/major/minor",
               "location": "具体位置",
               "problem": "问题描述",
               "suggestion": "具体修改建议",
               "agreement": "both 或 A 或 B"}}],
  "disputed": [{{"location": "位置", "topic": "分歧点",
                 "views": {{"A": "A 的观点", "B": "B 的观点"}}}}],
  "summary": "一段话说明两份意见的一致程度、最需优先处理的项，以及是否存在系统性分歧"}}

共 {n} 条问题：{severity_note}

审校者 A（{model_a}）的原始意见：
{review_a}

审校者 B（{model_b}）的原始意见：
{review_b}
"""


def build_requirements_block(requirements):
    if not requirements or not requirements.strip():
        return ""
    return (
        "\n任务边界条件与验收标准（这是产物必须满足的要求，逐条核对符合性，"
        "不满足任意一条即为 blocker 问题）：\n" + requirements.strip() + "\n"
    )


def _make_prompt(artifact_type, content, focus, requirements, kind):
    prompt = REVIEW_PROMPT.format(
        kind=kind,
        content=content,
        requirements_block=build_requirements_block(requirements),
    )
    if focus:
        prompt += "\n额外侧重维度：" + focus
    return prompt


def _load_content(content, content_file):
    """content 为空时从 content_file 读取。
    便于送审大文档：把产物落到文件、只传路径，避免内联超长文本。
    """
    if content and str(content).strip():
        return content
    if content_file and str(content_file).strip():
        path = str(content_file).strip().strip('"').strip("'")
        if not os.path.isfile(path):
            raise ValueError(f"content_file 不存在: {path}")
        if os.path.getsize(path) > 8 * 1024 * 1024:
            raise ValueError(f"content_file 超过 8MB，疑似非文本: {path}")
        raw = open(path, "rb").read()
        if b"\x00" in raw[:4096]:
            raise ValueError(f"content_file 疑似二进制文件: {path}")
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"content_file 编码无法识别（试过 utf-8/gbk）: {path}")
    return content


# ---------------------------------------------------------------- 模型调用

def _build_request(pcfg, payload_dict):
    return urllib.request.Request(
        f"{pcfg['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload_dict).encode(),
        headers={
            "Authorization": f"Bearer {pcfg['api_key']}",
            "Content-Type": "application/json",
        },
    )


def _is_retryable(err):
    """网络类失败值得重试；参数类 / 鉴权类错误重试没有意义。"""
    if isinstance(err, urllib.error.HTTPError):
        return err.code in (408, 429, 500, 502, 503, 504)
    text = str(err).lower()
    return any(k in text for k in (
        "timed out", "timeout", "connection reset", "remote end closed",
        "disconnected", "temporarily unavailable", "eof occurred", "broken pipe",
        "connection aborted",
    ))


def _call_stream(pcfg, payload_dict, on_progress, include_usage=True):
    """SSE 流式调用：思考过程持续有字节流动，既避免 socket 空闲超时，又能报进度。

    include_usage 让服务端在收尾的 chunk 里带上 usage，从而保留
    total_tokens / reasoning_tokens 这两个诊断 GLM 额度陷阱的关键指标。
    """
    payload = dict(payload_dict)
    payload["stream"] = True
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    req = _build_request(pcfg, payload)

    parts, c_chars, r_chars, finish, usage = [], 0, 0, None, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=STREAM_IDLE_TIMEOUT) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if chunk.get("usage"):        # 收尾 chunk：choices 为空、只带 usage
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
                    c_chars += len(piece)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    r_chars += len(reasoning)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if on_progress:
                on_progress(c_chars, r_chars)
    return {
        "content": "".join(parts).strip(),
        "finish_reason": finish,
        "usage": usage,
        "streamed": True,
        "content_chars": c_chars,
        "reasoning_chars": r_chars,
        "elapsed": round(time.time() - t0, 1),
    }


def _call_plain(pcfg, payload_dict):
    """非流式调用，作为流式不可用时的回退。"""
    req = _build_request(pcfg, payload_dict)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    return {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage") or {},
        "streamed": False,
        "content_chars": len(content),
        "reasoning_chars": 0,
        "elapsed": round(time.time() - t0, 1),
    }


def _invoke_once(pcfg, payload_dict, on_progress, model_id):
    """流式优先（带 usage）；被 400 拒绝则逐级降级。

    降级次序：带 usage 的流式 -> 不带 usage 的流式 -> 去掉 reasoning_effort 的流式 -> 非流式。
    最后一级是兜底：某些厂商/模型不认 reasoning_effort，去掉它至少还能把审校跑完。
    """
    payloads = [dict(payload_dict)]
    if "reasoning_effort" in payload_dict:
        payloads.append({k: v for k, v in payload_dict.items() if k != "reasoning_effort"})

    for payload in payloads:
        for use_usage in (True, False):
            try:
                return _call_stream(pcfg, payload, on_progress, include_usage=use_usage)
            except urllib.error.HTTPError as err:
                if err.code != 400:
                    raise
                _log_event(
                    f"{model_id} 流式被拒(400)：usage={use_usage}、"
                    f"reasoning_effort={'有' if 'reasoning_effort' in payload else '无'}，降级重试"
                )
    _log_event(f"{model_id} 流式全部被拒，回退非流式")
    return _call_plain(pcfg, payloads[-1])


def _invoke_with_retry(pcfg, payload_dict, on_progress, model_id):
    """网络类失败自动重试，指数退避。"""
    last = None
    for i in range(NET_RETRY + 1):
        try:
            return _invoke_once(pcfg, payload_dict, on_progress, model_id)
        except Exception as err:
            last = err
            if i >= NET_RETRY or not _is_retryable(err):
                raise
            _log_event(f"{model_id} 第 {i + 1} 次调用失败将重试: {str(err)[:180]}")
            if on_progress:
                on_progress(-1, -1)   # -1 表示正在重试
            time.sleep(2 * (i + 1))
    raise last


def call_model(provider_name, model_id, prompt, max_tokens=None,
               reasoning_effort=None, on_progress=None, use_cache=True):
    """按厂商路由调用，返回 (content, meta)；失败抛异常。

    三层韧性 + 缓存：
      · 命中缓存直接复用（不重复计费），meta 标 cached=true；
      · 网络类失败（连接重置 / 5xx / 超时）自动重试 NET_RETRY 次；
      · GLM-5.x 始终思考，思维链会占用 max_tokens。若因思考占满额度而返回空正文
        （finish_reason=length），自动以更高额度再试一次。
    """
    pcfg = PROVIDERS[provider_name]
    if not pcfg["api_key"]:
        hint = "REVIEWER_API_KEY" if provider_name == "deepseek" else "ZHIPU_API_KEY"
        raise RuntimeError(
            f"{pcfg['label']} 未配置 API Key，请在 mcp.json 的 env 中填入 {hint}"
        )

    budget0 = int(max_tokens) if max_tokens else MAX_TOKENS
    effort = (reasoning_effort or REASONING_EFFORT).strip().lower()

    cache_key = _cache_key(provider_name, model_id, prompt, budget0, effort)
    if use_cache:
        hit = _cache_get(cache_key)
        if hit and hit.get("content"):
            _log_event(f"缓存命中 {provider_name}/{model_id} key={cache_key[:8]}")
            cached_meta = dict(hit.get("meta") or {})
            cached_meta.update({
                "cached": True,
                "cache_key": cache_key,
                "cached_at": hit.get("cached_at"),
                "elapsed_s": 0.0,
                "note": "命中结果缓存，未重复调用模型（如需强制重跑请传 use_cache=False）",
            })
            if on_progress:
                on_progress(len(hit["content"]), 0)
            return hit["content"], cached_meta

    budgets = [budget0]
    if budget0 < MAX_ESCALATE_TOKENS:
        budgets.append(min(budget0 * 2, MAX_ESCALATE_TOKENS))

    meta = {}
    for attempt, budget in enumerate(budgets):
        payload_dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": budget,
        }
        if effort in VALID_EFFORTS:
            payload_dict["reasoning_effort"] = effort

        res = _invoke_with_retry(pcfg, payload_dict, on_progress, model_id)
        usage = res["usage"] or {}
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

        meta = {
            "provider": provider_name,
            "provider_label": pcfg["label"],
            "model": model_id,
            "elapsed_s": res["elapsed"],
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "content_chars": res["content_chars"],
            "reasoning_chars": res["reasoning_chars"],
            "finish_reason": res["finish_reason"],
            "max_tokens_used": budget,
            "reasoning_effort_used": effort or "（未传，服务端默认）",
            "streamed": res["streamed"],
            "attempts": attempt + 1,
            "cached": False,
            "cache_key": cache_key,
        }

        if res["content"]:
            if meta["finish_reason"] == "length":
                meta["warning"] = (
                    f"输出被 max_tokens={budget} 截断，JSON 可能不完整；"
                    "请调高 max_tokens 后重试"
                )
            if attempt > 0:
                meta["escalated"] = True
            if use_cache:
                _cache_put(cache_key, res["content"], meta)
            return res["content"], meta

    raise RuntimeError(
        f"{model_id} 返回正文为空（推理 token {meta.get('reasoning_tokens')}，"
        f"finish_reason={meta.get('finish_reason')}）。该模型思考占满了输出额度，"
        f"已自动加倍至 max_tokens={meta.get('max_tokens_used')} 仍不足；"
        f"请显式传入更大的 max_tokens（当前默认 {MAX_TOKENS}）后重试。"
    )


# ---------------------------------------------------------------- 结果解析

def parse_review_json(text):
    """尽力把模型输出解析成 dict（容忍 ```json 包裹与前后废话）。"""
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


def count_severities(issues):
    counts = {"blocker": 0, "major": 0, "minor": 0}
    for it in issues or []:
        sev = str((it or {}).get("severity", "")).strip().lower()
        if sev in counts:
            counts[sev] += 1
    return counts


# ---------------------------------------------------------------- 报告渲染

def _safe_filename(name):
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()[:120]


def _model_rows(meta_list):
    rows = []
    for meta in meta_list:
        if not meta:
            continue
        rows.append(
            f"| {meta.get('provider_label', '')} / {meta.get('model', '')} "
            f"| {meta.get('elapsed_s', '-')}s "
            f"| {meta.get('total_tokens', '-')}（推理 {meta.get('reasoning_tokens', '-')}） "
            f"| {meta.get('reasoning_effort_used', '-')} "
            f"| {'缓存命中' if meta.get('cached') else '实时调用'} |"
        )
    return rows


def _render_issue_table(issues):
    if not issues:
        return "_（无）_\n"
    lines = ["| # | 级别 | 位置 | 问题 | 建议 | 来源 |",
             "|---|---|---|---|---|---|"]
    for i, it in enumerate(issues, 1):
        it = it or {}
        cell = lambda k: str(it.get(k, "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {cell('severity')} | {cell('location')} | {cell('problem')} "
            f"| {cell('suggestion')} | {cell('agreement') or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _render_one_review(title, model_label, meta, review_text):
    """渲染单个审校者的结果。返回 markdown 片段。"""
    out = [f"### {title}\n"]
    if meta:
        out.append(f"- 模型：{model_label}")
        out.append(f"- 耗时：{meta.get('elapsed_s', '-')}s｜"
                   f"tokens：{meta.get('total_tokens', '-')}"
                   f"（推理 {meta.get('reasoning_tokens', '-')}）")
        out.append(f"- 参数：max_tokens={meta.get('max_tokens_used', '-')}、"
                   f"reasoning_effort={meta.get('reasoning_effort_used', '-')}")
        out.append(f"- 来源：{'缓存命中（' + str(meta.get('cached_at', '')) + '）' if meta.get('cached') else '实时调用'}")
        if meta.get("warning"):
            out.append(f"- ⚠️ {meta['warning']}")
    out.append("")

    parsed = parse_review_json(review_text)
    if not parsed:
        out.append("_模型输出无法解析为 JSON，原文见文末。_\n")
    else:
        issues = parsed.get("issues") or []
        counts = count_severities(issues)
        out.append(f"**结论：{parsed.get('verdict', '?')}** — "
                   f"{len(issues)} 条问题"
                   f"（blocker {counts['blocker']} / major {counts['major']} / minor {counts['minor']}）\n")
        out.append(_render_issue_table(issues))
        if parsed.get("summary"):
            out.append(f"\n**评价：**{parsed['summary']}\n")
    return "\n".join(out)


def render_report(task):
    """把任务记录渲染成 Markdown 报告（同时支持单审 kind=single 与双审 kind=dual）。"""
    tid = task.get("task_id", "-")
    kind = task.get("kind", "single")
    L = ["# 跨模型审校报告\n"]

    L.append("## 任务信息\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 任务号 | `{tid}` |")
    L.append(f"| 类型 | {'跨厂商并发双审 + 仲裁' if kind == 'dual' else '单模型审校'} |")
    L.append(f"| 模型 | {task.get('model_label') or task.get('model') or '-'} |")
    L.append(f"| 产物类型 | {'代码' if task.get('artifact_type') == 'code' else '文档'} |")
    if task.get("artifact_chars"):
        L.append(f"| 产物规模 | {task['artifact_chars']} 字符 |")
    L.append(f"| max_tokens | {task.get('max_tokens', '-')} |")
    L.append(f"| reasoning_effort | {task.get('reasoning_effort') or '（未传，服务端默认）'} |")
    L.append(f"| 耗时 | {task.get('elapsed_s', '-')}s |")
    L.append(f"| 发起时间 | {task.get('created_at', '-')} |")
    L.append(f"| 完成时间 | {_fmt_ts(task.get('finished_ts'))} |")
    if task.get("report_of"):
        L.append(f"| 来源任务 | `{task['report_of']}` |")
    L.append("")

    if task.get("requirements"):
        L.append("## 边界条件与验收标准\n")
        L.append(f"> {str(task['requirements']).strip()}\n")

    if kind == "dual":
        bundle = parse_review_json(task.get("review")) or {}
        reviews = bundle.get("reviews") or []
        arb = bundle.get("arbitration")

        L.append("## 各审校者结果\n")
        metas = [r.get("meta") for r in reviews]
        rows = _model_rows(metas)
        if rows:
            L.append("| 模型 | 耗时 | tokens | 思考强度 | 来源 |")
            L.append("|---|---|---|---|---|")
            L.extend(rows)
            L.append("")
        for idx, r in enumerate(reviews):
            tag = chr(ord("A") + idx)
            L.append(_render_one_review(
                f"审校者 {tag}：{r.get('model', '?')}",
                r.get("model", "?"), r.get("meta"), r.get("review"),
            ))
            if r.get("error"):
                L.append(f"> 该审校者失败：{r['error']}\n")

        L.append("## 仲裁合并结果\n")
        if arb:
            L.append(_render_one_review(
                f"仲裁者：{arb.get('model', '?')}",
                arb.get("model", "?"), arb.get("meta"), arb.get("review"),
            ))
            arb_parsed = parse_review_json(arb.get("review"))
            if arb_parsed and arb_parsed.get("disputed"):
                L.append("### 分歧点\n")
                for d in arb_parsed["disputed"]:
                    views = d.get("views") or {}
                    L.append(f"- **{d.get('location', '?')}** — {d.get('topic', '')}")
                    for k, v in views.items():
                        L.append(f"  - {k}：{v}")
                L.append("")
        else:
            L.append("_未启用仲裁（仅并排返回两位审校者的意见）。_\n")

        L.append("## 原始输出\n")
        for idx, r in enumerate(reviews):
            tag = chr(ord("A") + idx)
            L.append(f"<details><summary>审校者 {tag} 原始 JSON</summary>\n")
            L.append(f"```json\n{r.get('review') or ''}\n```\n</details>\n")
        if arb:
            L.append("<details><summary>仲裁者原始 JSON</summary>\n")
            L.append(f"```json\n{arb.get('review') or ''}\n```\n</details>\n")
    else:
        L.append("## 审校结论\n")
        L.append(_render_one_review(
            "结果", task.get("model_label") or task.get("model", "?"),
            task.get("meta"), task.get("review"),
        ))
        L.append("## 原始输出\n")
        L.append(f"```json\n{task.get('review') or ''}\n```\n")

    L.append("\n---\n")
    L.append(f"_由 cross-model-reviewer 生成于 {_now_str()}_\n")
    return "\n".join(L)


def _resolve_report_path(task, report_path):
    raw = str(report_path or "").strip().strip('"').strip("'")
    if not raw:
        name = f"{task.get('task_id', 'review')}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        return os.path.join(REPORT_DIR, _safe_filename(name))
    if os.path.isdir(raw) or raw.endswith(("/", "\\")):
        return os.path.join(raw, _safe_filename(f"{task.get('task_id', 'review')}.md"))
    return raw


def write_report(task, report_path=""):
    """把任务渲染成 markdown 落盘，返回最终路径。"""
    path = _resolve_report_path(task, report_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(task))
    _log_event(f"{task.get('task_id')} 报告落盘 -> {path}")
    return path


# ---------------------------------------------------------------- 草稿池

_drafts = _read_json(DRAFTS_FILE, {})
_draft_seq = max((int(k.split("-")[-1]) for k in _drafts if k.startswith("rev-")), default=0)


def _save_drafts():
    _atomic_write(DRAFTS_FILE, _drafts)


# ---------------------------------------------------------------- 任务池

_tasks = _read_json(TASKS_FILE, {})
_task_seq = max((int(k.split("-")[-1]) for k in _tasks if k.startswith("task-")), default=0)

# 进程重启前遗留的非终态任务，标记为中断（结果已丢失，但状态可查）
for _t in _tasks.values():
    if _t.get("status") in ("queued", "running"):
        _t["status"] = "interrupted"
        _t["error"] = "MCP 进程重启，该任务中断（结果未保存）"


def _save_tasks():
    """落盘前剔除下划线开头的运行时字段（如 _prompt）。"""
    snapshot = {}
    for tid, t in _tasks.items():
        snapshot[tid] = {k: v for k, v in t.items() if not k.startswith("_")}
    _atomic_write(TASKS_FILE, snapshot)


def _prune_tasks():
    if len(_tasks) <= MAX_TASKS:
        return
    finished = [k for k, v in _tasks.items()
                if v.get("status") in ("done", "failed", "interrupted")]
    finished.sort(key=lambda k: _tasks[k].get("finished_ts")
                  or _tasks[k].get("created_ts") or 0)
    for tid in finished[:max(0, len(_tasks) - MAX_TASKS)]:
        _tasks.pop(tid, None)


def _task_elapsed(t):
    if t.get("started_ts"):
        end = t.get("finished_ts") or time.time()
        return round(end - t["started_ts"], 1)
    return 0.0


def _task_status(task_id):
    return (_tasks.get(task_id) or {}).get("status")


def _is_terminal(status):
    return status in ("done", "failed", "interrupted")


def _task_progress_hint(t):
    status = t.get("status")
    if t.get("kind") == "dual":
        kids = t.get("children") or []
        states = [_task_status(c) for c in kids]
        done = sum(1 for s in states if _is_terminal(s))
        arb = t.get("arbiter")
        if arb and _is_terminal(_task_status(arb)):
            return "仲裁已完成" if status == "done" else "仲裁收尾中"
        if status == "running":
            if not kids:
                return ("双审已启动，正在创建两位审校者"
                        f"（已运行 {int(_task_elapsed(t))}s）")
            detail = "、".join(
                f"{((_tasks.get(c) or {}).get('model') or c)}={_task_status(c) or '?'}"
                for c in kids
            )
            base = f"双审进行中：{done}/{len(kids)} 个审校完成"
            if done >= len(kids) and arb:
                base += f"，仲裁者 {_task_status(arb) or '?'} 中"
            return base + (f"（{detail}）" if detail else "")
    if status == "queued":
        return "排队中，后台线程即将开始"
    if status == "running":
        pr = t.get("progress") or {}
        if pr.get("retrying"):
            return "网络波动，正在自动重试"
        return (f"模型思考/生成中：已收正文 {pr.get('content_chars', 0)} 字符、"
                f"思考过程 {pr.get('reasoning_chars', 0)} 字符，已运行 {int(_task_elapsed(t))}s")
    if status == "done":
        return "已完成"
    if status == "failed":
        return f"失败：{str(t.get('error'))[:160]}"
    return status or "unknown"


def _run_task(task_id):
    """后台线程：执行单模型审校并持续更新进度。"""
    with _lock:
        t = _tasks.get(task_id)
        if not t:
            return
        prompt = t.pop("_prompt", "")
        provider, model = t["provider"], t["model"]
        budget, effort = t["max_tokens"], t["reasoning_effort"]
        use_cache = t.get("use_cache", True)
        report_path = t.get("report_path") or ""
        t["status"] = "running"
        t["started_ts"] = time.time()
        t["progress"] = {"content_chars": 0, "reasoning_chars": 0}
    _save_tasks()
    _log_event(f"{task_id} 开始 | {provider}/{model} | max_tokens={budget} | effort={effort or '默认'}")

    last_save = [0.0]

    def on_progress(c_chars, r_chars):
        with _lock:
            cur = _tasks.get(task_id)
            if not cur:
                return
            if c_chars < 0:      # 重试信号
                cur["progress"] = dict(cur.get("progress") or {}, retrying=True)
            else:
                cur["progress"] = {
                    "content_chars": c_chars,
                    "reasoning_chars": r_chars,
                    "retrying": False,
                }
            now = time.time()
            if now - last_save[0] >= 5:      # 落盘节流，避免高频 IO
                last_save[0] = now
                _save_tasks()

    try:
        review, meta = call_model(provider, model, prompt, max_tokens=budget,
                                  reasoning_effort=effort, on_progress=on_progress,
                                  use_cache=use_cache)
        with _lock:
            t = _tasks.get(task_id)
            if t:
                t.update({"status": "done", "finished_ts": time.time(),
                          "review": review, "meta": meta, "error": None})
        _log_event(f"{task_id} 完成 | {meta['elapsed_s']}s | "
                   f"正文 {meta['content_chars']} 字符 | tokens={meta['total_tokens']}"
                   f"{' | 缓存命中' if meta.get('cached') else ''}")
    except Exception as err:
        with _lock:
            t = _tasks.get(task_id)
            if t:
                t.update({"status": "failed", "finished_ts": time.time(),
                          "error": f"{type(err).__name__}: {err}"})
        _log_event(f"{task_id} 失败 | {str(err)[:200]}")
    finally:
        with _lock:
            t = _tasks.get(task_id)
            if t and t.get("started_ts"):
                t["elapsed_s"] = _task_elapsed(t)
            _prune_tasks()
        _save_tasks()

        if report_path:
            with _lock:
                snapshot = dict(_tasks.get(task_id) or {})
            if snapshot.get("status") == "done":
                try:
                    write_report(snapshot, report_path)
                except Exception as e:
                    _log_event(f"{task_id} 报告落盘失败: {e}")


def _dual_child_wait_budget():
    """单个子任务的墙钟等待上限（秒）。call_model 内部已受 TIMEOUT、网络重试与
    额度递增约束，这里再留一段余量，确保任何异常下父任务都不会永远停在 running。"""
    return TIMEOUT * (NET_RETRY + 1) + 300


def _wait_children(child_ids, budget=None):
    """等待所有子任务进入终态。返回 True 表示全部结束，False 表示超时。"""
    deadline = time.time() + (budget if budget is not None else _dual_child_wait_budget())
    while time.time() < deadline:
        if all(_is_terminal(_task_status(c)) for c in child_ids):
            return True
        # 轮询间隔 2s，但不超过剩余预算，保证预算语义准确
        time.sleep(min(2.0, max(deadline - time.time(), 0.05)))
    return False


def _run_dual_task(parent_id, child_specs, arbiter_spec, artifact_type,
                   content_chars, requirements):
    """后台线程：跑 2 个并发审校 -> 等全部完成 -> 可选仲裁合并。"""
    with _lock:
        parent = _tasks.get(parent_id)
        if not parent:
            return
        use_cache = parent.get("use_cache", True)
        report_path = parent.get("report_path") or ""
        parent["status"] = "running"
        parent["started_ts"] = time.time()
        llm_children = parent.pop("_llm_children", [])
    _save_tasks()

    try:
        # 阶段 1：并发启动 2 个审校
        child_ids = []
        for spec in llm_children:
            cid = _start_task(spec["provider"], spec["model"], spec["prompt"],
                              spec["max_tokens"], spec["reasoning_effort"],
                              parent=parent_id, kind="review",
                              use_cache=use_cache,
                              artifact_type=artifact_type,
                              artifact_chars=content_chars,
                              requirements=requirements)
            child_ids.append(cid)
        with _lock:
            parent = _tasks.get(parent_id)
            if parent:
                parent["children"] = child_ids
        _save_tasks()
        _log_event(f"{parent_id} 双审启动：{child_ids}")

        # 阶段 2：等待两个审校结束
        if not _wait_children(child_ids):
            raise RuntimeError(
                f"双审等待超时（{_dual_child_wait_budget()}s），子任务 "
                f"{[c for c in child_ids if not _is_terminal(_task_status(c))]} 仍未结束"
            )

        reviews = []
        for cid in child_ids:
            c = _tasks.get(cid) or {}
            reviews.append({
                "task_id": cid,
                "model": c.get("model_label"),
                "status": c.get("status"),
                "meta": c.get("meta"),
                "review": c.get("review"),
                "error": c.get("error"),
            })

        ok = [r for r in reviews if r.get("status") == "done" and r.get("review")]

        # 阶段 3：仲裁合并（需要两份都成功）
        arbitration = None
        if arbiter_spec and len(ok) == 2:
            kind = "代码" if artifact_type == "code" else "文档"
            sev_note = "（未解析出结构化问题，请直接依据两位审校者的原文合并）"
            try:
                cnts = [count_severities((parse_review_json(r["review"]) or {}).get("issues"))
                        for r in ok]
                sev_note = ("；".join(
                    f"{r['model']} 给出 {sum(c.values())} 条"
                    f"（blocker {c['blocker']}/major {c['major']}/minor {c['minor']}）"
                    for r, c in zip(ok, cnts)
                ) or sev_note)
            except Exception:
                pass

            arb_prompt = ARBITRATION_PROMPT.format(
                kind=kind, n=2, severity_note=sev_note,
                model_a=ok[0]["model"], review_a=ok[0]["review"],
                model_b=ok[1]["model"], review_b=ok[1]["review"],
            )
            if requirements and requirements.strip():
                arb_prompt += ("\n\n评判该产物时的任务边界条件与验收标准：\n"
                               + requirements.strip() + "\n")

            aid = _start_task(arbiter_spec["provider"], arbiter_spec["model"], arb_prompt,
                              arbiter_spec["max_tokens"], arbiter_spec["reasoning_effort"],
                              parent=parent_id, kind="arbiter", use_cache=use_cache,
                              artifact_type=artifact_type,
                              artifact_chars=content_chars,
                              requirements=requirements)
            with _lock:
                p = _tasks.get(parent_id)
                if p:
                    p["arbiter"] = aid
            _save_tasks()

            if not _wait_children([aid]):
                raise RuntimeError(
                    f"仲裁等待超时（{_dual_child_wait_budget()}s），仲裁任务 {aid} 仍未结束"
                )

            a = _tasks.get(aid) or {}
            arbitration = {
                "task_id": aid,
                "model": a.get("model_label"),
                "status": a.get("status"),
                "meta": a.get("meta"),
                "review": a.get("review"),
                "error": a.get("error"),
            }

        payload = {"kind": "dual", "reviewers": len(reviews),
                   "reviews": reviews, "arbitration": arbitration}
        with _lock:
            p = _tasks.get(parent_id)
            if p:
                p.update({"status": "done", "finished_ts": time.time(),
                          "review": json.dumps(payload, ensure_ascii=False),
                          "meta": {"cached": False},
                          "error": None})
        _log_event(f"{parent_id} 双审完成 | 审校 {len(ok)}/2 成功"
                   f"{' | 已仲裁' if arbitration else ''}")
    except Exception as err:
        with _lock:
            p = _tasks.get(parent_id)
            if p:
                p.update({"status": "failed", "finished_ts": time.time(),
                          "error": f"{type(err).__name__}: {err}"})
        _log_event(f"{parent_id} 双审失败 | {str(err)[:200]}")
    finally:
        with _lock:
            p = _tasks.get(parent_id)
            if p and p.get("started_ts"):
                p["elapsed_s"] = _task_elapsed(p)
            _prune_tasks()
        _save_tasks()

        if report_path:
            with _lock:
                snapshot = dict(_tasks.get(parent_id) or {})
            if snapshot.get("status") == "done":
                try:
                    write_report(snapshot, report_path)
                except Exception as e:
                    _log_event(f"{parent_id} 报告落盘失败: {e}")


def _start_task(provider, model, prompt, budget, effort, request_id=None,
                parent=None, kind="single", use_cache=True, report_path="",
                artifact_type="doc", artifact_chars=0, requirements=""):
    global _task_seq
    with _lock:
        _task_seq += 1
        task_id = f"task-{_task_seq:04d}"
        _tasks[task_id] = {
            "task_id": task_id,
            "request_id": request_id,
            "parent": parent,
            "kind": kind,
            "provider": provider,
            "model": model,
            "model_label": f"{PROVIDERS[provider]['label']} / {model}",
            "max_tokens": budget,
            "reasoning_effort": effort,
            "artifact_type": artifact_type,
            "artifact_chars": artifact_chars,
            "requirements": requirements,
            "use_cache": use_cache,
            "report_path": report_path,
            "status": "queued",
            "created_ts": time.time(),
            "created_at": _now_str(),
            "started_ts": None,
            "finished_ts": None,
            "elapsed_s": 0.0,
            "progress": {},
            "review": None,
            "meta": None,
            "error": None,
            "_prompt": prompt,
        }
        _prune_tasks()
    _save_tasks()

    target = _run_dual_task if kind == "dual" else _run_task
    threading.Thread(target=target, args=(task_id,), daemon=True).start()
    return task_id


def _task_payload(task_id, include_review=True):
    t = _tasks.get(task_id)
    if not t:
        return {"error": f"任务 {task_id} 不存在，调用 list_tasks 查看最近任务"}
    out = {
        "task_id": task_id,
        "kind": t.get("kind", "single"),
        "status": t.get("status"),
        "model": t.get("model_label"),
        "max_tokens": t.get("max_tokens"),
        "reasoning_effort": t.get("reasoning_effort") or "（未传，服务端默认）",
        "elapsed_s": _task_elapsed(t) if t.get("status") == "running" else t.get("elapsed_s"),
        "progress": t.get("progress") or {},
        "hint": _task_progress_hint(t),
    }
    if t.get("kind") == "dual":
        out["children"] = t.get("children") or []
        if t.get("arbiter"):
            out["arbiter"] = t["arbiter"]

    if t.get("status") == "done":
        out["meta"] = t.get("meta")
        if include_review:
            out["review"] = t.get("review")
        out["next_step"] = "已完成，无需再轮询。可用 export_review 导出 Markdown 报告"
    elif t.get("status") == "failed":
        out["error"] = t.get("error")
        out["next_step"] = "任务失败，可按 error 调整参数后重新 draft_review / submit_review"
    elif t.get("status") == "running":
        if t.get("kind") == "dual":
            out["next_step"] = ("双审仍在进行，请等待 30~60 秒后再次 check_review(task_id)。"
                                "并发双审 + 仲裁通常需要数分钟。")
        else:
            out["next_step"] = ("任务仍在运行，请等待 30~60 秒后再次 check_review(task_id)。"
                                "GLM 强制思考，长文档审校可能持续数分钟，属正常现象。")
    elif t.get("status") == "interrupted":
        out["error"] = t.get("error")
        out["next_step"] = "进程重启导致中断，请重新提交"
    else:
        out["next_step"] = "稍后再 check_review(task_id)"
    return out


def _wait_task(task_id, wait_seconds):
    """最多等 wait_seconds 秒；返回 (是否完成, 等待秒数)。"""
    wait_seconds = max(0, min(int(wait_seconds or 0), TASK_WAIT_MAX))
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _is_terminal(_task_status(task_id)):
            return True, wait_seconds
        time.sleep(1)
    return False, wait_seconds


# ---------------------------------------------------------------- 双审模型选择

# ---------------------------------------------------------------- 双审耗时估算

# 经验基准：约 1k 字符产物、reasoning_effort=low 时的实测耗时（秒）。
# 数据来自本机实测（DeepSeek v4-flash 8.3s / GLM-5.3-flash 15.0s / v4-pro 仲裁 116.2s
# @237 字符）。仅用于判断量级，不是承诺值。
_TIME_BASE = {
    "deepseek-v4-flash": 12.0,
    "deepseek-v4-flash-vision-exp": 14.0,
    "deepseek-v4-pro": 60.0,
    "glm-5.3-flashx": 10.0,
    "glm-5.3-flash": 18.0,
    "glm-5.3": 60.0,
    "glm-4.6": 32.0,
}
_TIME_EFFORT_MULT = {"low": 1.0, "medium": 1.6, "high": 2.6, "max": 3.6}
# 仲裁者要额外读完两份审校 JSON，输入更长，且合并推理更重
_ARBITER_INPUT_MULT = 1.9


def _time_scale(chars):
    """篇幅放大系数：以 1k 字符为基准，6000 字符约翻倍，封顶 30000。"""
    return 1.0 + min(max(int(chars or 0), 0), 30000) / 6000.0


def _est_one(model_id, effort, chars, extra_mult=1.0):
    base = _TIME_BASE.get(model_id, 20.0)
    mult = _TIME_EFFORT_MULT.get(effort, 1.5)
    return base * _time_scale(chars) * mult * extra_mult


def estimate_dual_time(eff_specs, arb_spec, content_chars):
    """给出双审的粗粒度耗时估算（秒），并说明依据，供用户判断值不值得跑。"""
    rev = [{"model": s["model"],
            "estimated_s": round(_est_one(s["model"], s["reasoning_effort"], content_chars), 1)}
           for s in eff_specs]
    out = {
        "reviewers": rev,
        "reviewer_parallel_s": round(max([x["estimated_s"] for x in rev], default=0.0), 1),
        "arbiter_s": (round(_est_one(arb_spec["model"], arb_spec["reasoning_effort"],
                                     content_chars, _ARBITER_INPUT_MULT), 1)
                      if arb_spec else 0.0),
    }
    total = out["reviewer_parallel_s"] + out["arbiter_s"]
    out["total_s"] = round(total, 1)
    out["total_range_s"] = [round(total * 0.6, 1), round(total * 1.6, 1)]
    out["basis"] = (
        f"经验估算（非承诺值）：基准取自本机实测——约 1k 字符 / low 档时 "
        f"DeepSeek v4-flash 约 12s、GLM-5.3-flash 约 18s；本次产物 {content_chars} 字符，"
        f"两位审校者并发（取较慢者），仲裁者因需读两份意见额外 ×{_ARBITER_INPUT_MULT}。"
        f"思考强度越高、篇幅越长越慢；智谱对 medium 会映射为 high 故偏慢。"
    )
    if total >= 600:
        out["warning"] = (f"估算约 {out['total_s']:.0f}s，耗时较长。"
                          "若只是初筛，建议改单审（draft_review）或把 reasoning_effort 降到 low。")
    return out


def pick_dual_models(spec):
    """决定双审用的 2 个模型 -> ([(provider, model), ...], error)。"""
    raw = (spec or "").strip() or DUAL_MODELS
    if raw:
        specs = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        specs = []
        for pname, pcfg in PROVIDERS.items():
            if not pcfg["api_key"] or not pcfg["models"]:
                continue
            pref = next((m for m in pcfg["models"] if "flash" in m), pcfg["models"][0])
            specs.append(f"{pname}:{pref}")

    if len(specs) < 2:
        return [], (f"双审至少需要 2 个可用模型，当前只有 {len(specs)} 个"
                    "（需两个厂商都配好 API Key，或用 reviewer_models 显式指定）")
    if len(specs) > 2:
        specs = specs[:2]

    out = []
    for s in specs:
        p, m, err = resolve_model(s)
        if err:
            return [], err
        if not PROVIDERS[p]["api_key"]:
            return [], f"{PROVIDERS[p]['label']} 未配置 API Key"
        out.append((p, m))
    if out[0][0] == out[1][0]:
        # 同厂商也能跑，但跨厂商互补性更强，只提示不阻拦
        return out, None
    return out, None


# ---------------------------------------------------------------- 工具

@mcp.tool()
def list_models() -> str:
    """跨模型审校 · 列出当前可用的审校模型、建议参数与缓存状态（按厂商分组）。
    不确定该用哪个模型、该给多大 max_tokens / 什么 reasoning_effort，或想确认
    智谱模型是否接入成功时，先调用它。
    """
    groups = []
    for pname, pcfg in PROVIDERS.items():
        models = [
            {
                "model": f"{pname}:{m}" if m in _ambiguous else m,
                "desc": MODEL_DESC.get(m, ""),
                "advice": MODEL_ADVICE.get(m, ""),
            }
            for m in pcfg["models"]
        ]
        groups.append({
            "provider": pname,
            "label": pcfg["label"],
            "base_url": pcfg["base_url"],
            "api_key_configured": bool(pcfg["api_key"]),
            "models": models,
        })
    dual_models, dual_err = pick_dual_models("")
    return json.dumps({
        "providers": groups,
        "default_model": DEFAULT_MODEL_SPEC,
        "default_max_tokens": MAX_TOKENS,
        "default_reasoning_effort": REASONING_EFFORT or "（未设置，由服务端决定）",
        "param_tips": [
            "max_tokens 与 reasoning_effort 可以每次调用单独指定："
            "draft_review / review_artifact / dual_review 都支持这两个参数。",
            "思考强度档位因厂商而异：DeepSeek 支持 low/medium/high/max；"
            "智谱只支持 low/high/max（传 medium 会 HTTP 400 报 1210），"
            f"所以智谱收到 medium 时会自动映射为 {EFFORT_FALLBACK['zhipu']['medium']}。",
            f"GLM-5.x 强制思考且无法关闭，思维链会占用 max_tokens："
            f"低于 {MIN_SAFE_TOKENS} 基本不可用，长文档建议 16384~32768。",
            "GLM 太慢时把 reasoning_effort 设为 low，可显著缩短耗时。",
            f"流式超时（无数据）{STREAM_IDLE_TIMEOUT}s，非流式总超时 {TIMEOUT}s，"
            f"网络失败自动重试 {NET_RETRY} 次。",
        ],
        "dual_review": {
            "models": [f"{p}:{m}" for p, m in dual_models] if dual_models else [],
            "error": dual_err,
            "arbitrate_model": ARBITRATE_MODEL or "（未启用仲裁）",
            "note": "dual_review 会跨厂商并发审两份 + 第三模型仲裁合并，适合重要产物。",
        },
        "cache": cache_stats(),
        "report_dir": REPORT_DIR,
        "note": "跨厂商审校独立性更强：若产出方是 DeepSeek 系，建议选智谱模型审校，反之亦然。",
    }, ensure_ascii=False)


@mcp.tool()
def draft_review(artifact_type: str, content: str = "", focus: str = "",
                 reviewer_model: str = "", requirements: str = "",
                 content_file: str = "", max_tokens: int = 0,
                 reasoning_effort: str = "", use_cache: bool = True,
                 report_path: str = "") -> str:
    """跨模型审校 · 默认流程第一步：打包审校请求，生成给用户确认的清单（不调用模型、不产生费用）。
    返回 request_id 和 confirm_checklist。请把 confirm_checklist 完整展示给用户：
    用户确认无误后调用 submit_review(request_id)；用户提出修改则按修改意见重新 draft_review。

    **关键**：max_tokens 与 reasoning_effort 不会弹窗让用户选择，只能靠这里同步。
    展示清单时务必显式列出 reviewer_model / max_tokens / reasoning_effort 三项
    （见 confirm_checklist.review_params，含来源与推荐值，recommended 为实测稳妥值），
    让用户有机会改；用户未表态就按当前值执行。

    Args:
        artifact_type: 产物类型，"code"（代码）或 "doc"（文档）
        content: 待审校的完整产物内容（只传产物本身，不要附带作者解释）
        focus: 可选，额外侧重维度，逗号分隔，如 "数值稳定性,边界条件"
        reviewer_model: 可选，审校模型，可选值见 list_models，例如
            deepseek-v4-flash（默认，快）、deepseek-v4-pro（深度推理，复杂数学模型推荐）、
            glm-5.3（智谱旗舰）、glm-5.3-flash / glm-5.3-flashx（智谱快版）、glm-4.6；
            需要消歧时写 "厂商:模型"，如 zhipu:glm-5.3
        requirements: 强烈建议：任务的边界条件与验收标准（发明主题、应用场景、技术路线、
            精度/性能/合规等硬性指标、必须满足的假设）。从用户对话中提炼，送审前须让用户确认。
        content_file: 可选，待审校产物的本地文件绝对路径。**送审长文档时优先用它**：
            把产物先写到文件、这里只传路径，可避免内联超长文本。content 非空时以 content 为准。
        max_tokens: 可选，本次调用的输出上限，0 表示用默认值。
            GLM 强制思考、思维链占用该额度：短文档 8192 起，长文档建议 16384~32768。
        reasoning_effort: 可选，本次调用的思考强度 low/medium/high/max。
            GLM 太慢时设 low 可显著提速；深度审校（复杂数学模型）用 high。
        use_cache: 可选，默认 True。内容+模型+参数完全相同时直接复用上次结果（不重复计费）。
            用户明确要求"重新审一次""别看旧结果"时传 False。
        report_path: 可选，审校完成后自动把报告写成 Markdown 的路径。
            传目录则自动命名 <task_id>_<时间>.md；不传则不落盘（可用 export_review 事后导出）。
    """
    global _draft_seq

    pname, model, err = resolve_model(reviewer_model)
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)
    if not PROVIDERS[pname]["api_key"]:
        return json.dumps(
            {"error": f"{PROVIDERS[pname]['label']} 未配置 API Key，请先在 mcp.json 中填入"},
            ensure_ascii=False,
        )

    budget, effort, perr, effort_note = resolve_params(max_tokens, reasoning_effort, pname)
    if perr:
        return json.dumps({"error": perr}, ensure_ascii=False)

    try:
        content = _load_content(content, content_file)
    except Exception as e:
        return json.dumps({"error": f"读取待审校内容失败: {e}"}, ensure_ascii=False)
    if not content or not str(content).strip():
        return json.dumps(
            {"error": "content 与 content_file 均为空，请提供待审校的产物内容或文件路径"},
            ensure_ascii=False,
        )

    kind = "代码" if artifact_type == "code" else "文档"
    _draft_seq += 1
    rid = f"rev-{_draft_seq:04d}"
    _drafts[rid] = {
        "provider": pname,
        "model": model,
        "max_tokens": budget,
        "reasoning_effort": effort,
        "use_cache": bool(use_cache),
        "report_path": report_path or "",
        "artifact_type": artifact_type,
        "artifact_chars": len(content),
        "requirements": requirements or "",
        "created_at": _now_str(),
        "prompt": _make_prompt(artifact_type, content, focus, requirements, kind),
    }
    while len(_drafts) > MAX_DRAFTS:
        _drafts.pop(next(iter(_drafts)))
    _save_drafts()

    checklist = {
        "reviewer_provider": PROVIDERS[pname]["label"],
        "reviewer_model": model,
        "artifact_type": artifact_type,
        "artifact_chars": len(content),
        "artifact_preview": content[:150] + ("..." if len(content) > 150 else ""),
        "focus": focus or "（无）",
        "requirements": requirements.strip() or "（无——强烈建议补充任务边界条件后重新起草）",
        "review_params": suggest_params(
            pname, model, len(content), budget, effort,
            user_set_budget=bool(max_tokens), user_set_effort=bool(reasoning_effort),
            effort_note=effort_note,
        ),
        "use_cache": bool(use_cache),
        "report_path": report_path or "（不落盘，可用 export_review 事后导出）",
    }
    if pname == "zhipu" and budget < MIN_SAFE_TOKENS:
        checklist["risk_note"] = (
            f"⚠️ GLM 强制思考，max_tokens={budget} 偏低，思维链可能吃光额度导致正文为空，"
            f"建议 >= {MIN_SAFE_TOKENS}"
        )
    return json.dumps({
        "request_id": rid,
        "confirm_checklist": checklist,
        "must_confirm_with_user": ["reviewer_model", "max_tokens", "reasoning_effort"],
        "next_step": (
            "把 confirm_checklist 完整展示给用户，**务必显式列出 reviewer_model / max_tokens / "
            "reasoning_effort 三项**并附上 recommend 值让用户有机会修改；用户确认后 "
            "submit_review(request_id)，有修改则按意见重新 draft_review。"
        ),
    }, ensure_ascii=False)


@mcp.tool()
def submit_review(request_id: str, wait_seconds: int = 0) -> str:
    """跨模型审校 · 默认流程第二步：提交已经用户确认的审校请求（异步执行）。

    立即返回 task_id，审校在后台线程里跑，**不会阻塞、也不会因客户端超时被切断**。
    拿到 task_id 后请调用 check_review(task_id) 查看进度与结果；未完成时每隔 30~60 秒再查一次。

    只提交用户在确认清单中看到过的内容（与 draft_review 的 request_id 一一对应），提交后草稿作废。

    Args:
        request_id: draft_review 返回的草稿号
        wait_seconds: 可选，先同步等待至多这么多秒（上限 300）。默认 0 = 纯异步立即返回。
            短文档可设 60 直接拿结果；长文档建议保持 0，用 check_review 轮询。
    """
    params = _drafts.pop(request_id, None)
    if params is None:
        return json.dumps(
            {"error": f"草稿 {request_id} 不存在或已提交，请重新 draft_review"},
            ensure_ascii=False,
        )
    _save_drafts()

    task_id = _start_task(params["provider"], params["model"], params["prompt"],
                          params["max_tokens"], params["reasoning_effort"],
                          request_id=request_id,
                          kind="single",
                          use_cache=params.get("use_cache", True),
                          report_path=params.get("report_path", ""),
                          artifact_type=params.get("artifact_type", "doc"),
                          artifact_chars=params.get("artifact_chars", 0),
                          requirements=params.get("requirements", ""))

    finished, waited = _wait_task(task_id, wait_seconds)
    payload = _task_payload(task_id)
    payload["request_id"] = request_id
    if not finished:
        payload["waited_s"] = waited
        if waited:
            payload["hint"] = (f"已等待 {waited}s 仍未完成，{payload.get('hint', '')}")
        payload["next_step"] = (
            f"后台执行中，请调用 check_review('{task_id}') 查看进度与结果；"
            f"未完成时每隔 30~60 秒再查一次。"
        )
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def check_review(task_id: str) -> str:
    """跨模型审校 · 查询后台审校任务的进度与结果（秒回，不会阻塞）。

    状态含义：queued（排队）/ running（执行中）/ done（完成，含 review）/
    failed（失败，含 error）/ interrupted（进程重启中断）。
    running 时会给出已运行秒数与已生成的正文字符数、思考字符数；
    双审任务（kind=dual）还会给出各审校者与仲裁者的子任务号。
    """
    return json.dumps(_task_payload(task_id), ensure_ascii=False)


@mcp.tool()
def list_tasks(limit: int = 10) -> str:
    """跨模型审校 · 列出最近的审校任务（含状态、模型、耗时），用于找回忘记的 task_id。"""
    try:
        limit = max(1, min(int(limit or 10), MAX_TASKS))
    except (TypeError, ValueError):
        limit = 10
    ordered = sorted(_tasks.values(), key=lambda t: t.get("created_ts") or 0, reverse=True)
    items = [{
        "task_id": t.get("task_id"),
        "kind": t.get("kind", "single"),
        "status": t.get("status"),
        "model": t.get("model_label"),
        "created_at": t.get("created_at"),
        "elapsed_s": t.get("elapsed_s") or _task_elapsed(t),
        "request_id": t.get("request_id"),
        "cached": bool((t.get("meta") or {}).get("cached")),
    } for t in ordered[:limit]]
    return json.dumps({
        "count": len(items),
        "tasks": items,
        "state_dir": STATE_DIR,
        "cache": cache_stats(),
    }, ensure_ascii=False)


@mcp.tool()
def review_artifact(artifact_type: str, content: str = "", focus: str = "",
                    reviewer_model: str = "", requirements: str = "",
                    content_file: str = "", max_tokens: int = 0,
                    reasoning_effort: str = "", wait_seconds: int = 45,
                    use_cache: bool = True, report_path: str = "") -> str:
    """跨模型审校 · 免确认直通道：跳过用户确认直接送审（异步执行）。
    仅当用户明确说"不用确认直接送审/跳过确认"时使用；默认流程请用 draft_review + submit_review。

    默认先等待 45 秒：完成则直接返回 review；超时则返回 task_id，
    此时请调用 check_review(task_id) 继续轮询。

    Args:
        artifact_type: 产物类型，"code"（代码）或 "doc"（文档）
        content: 待审校的完整产物内容
        focus: 可选，额外侧重维度
        reviewer_model: 可选，审校模型（见 list_models）；消歧时写 "厂商:模型"
        requirements: 强烈建议：任务边界条件与验收标准
        content_file: 可选，待审校产物的本地文件绝对路径（送审长文档时优先用它）
        max_tokens: 可选，本次调用的输出上限，0 表示用默认值
        reasoning_effort: 可选，本次调用的思考强度 low/medium/high/max
        wait_seconds: 可选，先同步等待的秒数，默认 45，0 = 立即返回 task_id
        use_cache: 可选，默认 True；用户要求"重新审一次"时传 False
        report_path: 可选，完成后自动落盘 Markdown 报告的路径（目录或 .md 文件）
    """
    pname, model, err = resolve_model(reviewer_model)
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)
    if not PROVIDERS[pname]["api_key"]:
        return json.dumps(
            {"error": f"{PROVIDERS[pname]['label']} 未配置 API Key，请先在 mcp.json 中填入"},
            ensure_ascii=False,
        )

    budget, effort, perr, effort_note = resolve_params(max_tokens, reasoning_effort, pname)
    if perr:
        return json.dumps({"error": perr}, ensure_ascii=False)

    try:
        content = _load_content(content, content_file)
    except Exception as e:
        return json.dumps({"error": f"读取待审校内容失败: {e}"}, ensure_ascii=False)
    if not content or not str(content).strip():
        return json.dumps(
            {"error": "content 与 content_file 均为空，请提供待审校的产物内容或文件路径"},
            ensure_ascii=False,
        )

    kind = "代码" if artifact_type == "code" else "文档"
    prompt = _make_prompt(artifact_type, content, focus, requirements, kind)
    task_id = _start_task(pname, model, prompt, budget, effort, kind="single",
                          use_cache=use_cache, report_path=report_path or "",
                          artifact_type=artifact_type, artifact_chars=len(content),
                          requirements=requirements or "")

    finished, waited = _wait_task(task_id, wait_seconds)
    payload = _task_payload(task_id)
    if effort_note:
        payload["effort_note"] = effort_note
    if not finished:
        payload["waited_s"] = waited
        payload["next_step"] = (
            f"后台执行中，请调用 check_review('{task_id}') 查看进度与结果；"
            f"未完成时每隔 30~60 秒再查一次。"
        )
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def dual_review(artifact_type: str, content: str = "", content_file: str = "",
                focus: str = "", requirements: str = "", reviewer_models: str = "",
                max_tokens: int = 0, reasoning_effort: str = "",
                arbitrate: str = "", wait_seconds: int = 0,
                use_cache: bool = True, report_path: str = "",
                dry_run: bool = False) -> str:
    """跨模型审校 · 重型通道：**跨厂商并发双审 + 第三模型仲裁合并**。

    两位来自不同厂商的审校模型独立审同一份产物（真正盲审，互不知晓），
    再由第三个模型把两份意见合并去重、标出共识与分歧。适合重要产物：
    实测两家命中集互补度很高，DeepSeek 擅长挑硬错误、GLM 擅长挑符号层完备性。

    异步执行：返回 task_id，用 check_review(task_id) 轮询（双审 + 仲裁通常要数分钟）。

    Args:
        artifact_type: 产物类型，"code"（代码）或 "doc"（文档）
        content: 待审校的完整产物内容
        content_file: 可选，产物文件绝对路径（长文档优先用）
        focus: 可选，额外侧重维度
        requirements: 强烈建议：任务边界条件与验收标准（仲裁者也会拿它对齐）
        reviewer_models: 可选，逗号分隔的 2 个模型；默认取各厂商 flash 档
            （见 list_models.dual_review）。跨厂商组合最有价值。
        max_tokens: 可选，所有审校与仲裁共用的输出上限
        reasoning_effort: 可选，共用的思考强度
        arbitrate: 可选，仲裁模型；默认用 REVIEWER_ARBITRATE_MODEL（deepseek-v4-pro），
            传 "none" 表示不仲裁、只并排返回两份意见
        wait_seconds: 可选，先同步等待的秒数，默认 0（推荐，双审较慢）
        use_cache: 可选，默认 True；要求"重新审"时传 False
        report_path: 可选，完成后自动落盘 Markdown 报告
        dry_run: 设为 True 时只返回本次双审的配置清单（模型、参数、预估耗时与调用次数），
            不启动任何任务、不产生费用。**建议先 dry_run 给用户确认再正式提交。**
    """
    dual_models, dual_err = pick_dual_models(reviewer_models)
    if dual_err:
        return json.dumps({"error": dual_err}, ensure_ascii=False)

    eff_specs = []
    for pname, model in dual_models:
        budget, effort, perr, note = resolve_params(max_tokens, reasoning_effort, pname)
        if perr:
            return json.dumps({"error": f"{model}: {perr}"}, ensure_ascii=False)
        eff_specs.append({"provider": pname, "model": model, "max_tokens": budget,
                          "reasoning_effort": effort, "effort_note": note})

    arb_spec = None
    arb_note = ""
    if (arbitrate or "").strip().lower() not in ("none", "off", "false", "0"):
        arb_name = (arbitrate or "").strip() or ARBITRATE_MODEL
        ap, am, aerr = resolve_model(arb_name)
        if aerr:
            return json.dumps({"error": f"仲裁模型无效: {aerr}"}, ensure_ascii=False)
        if not PROVIDERS[ap]["api_key"]:
            return json.dumps(
                {"error": f"仲裁模型所属的 {PROVIDERS[ap]['label']} 未配置 API Key"},
                ensure_ascii=False)
        ab, ae, aberr, anote = resolve_params(max_tokens, reasoning_effort, ap)
        if aberr:
            return json.dumps({"error": f"仲裁模型参数无效: {aberr}"}, ensure_ascii=False)
        arb_spec = {"provider": ap, "model": am, "max_tokens": ab, "reasoning_effort": ae}
        arb_note = anote or ""

    try:
        content = _load_content(content, content_file)
    except Exception as e:
        return json.dumps({"error": f"读取待审校内容失败: {e}"}, ensure_ascii=False)
    if not content or not str(content).strip():
        return json.dumps(
            {"error": "content 与 content_file 均为空，请提供待审校的产物内容或文件路径"},
            ensure_ascii=False,
        )

    kind = "代码" if artifact_type == "code" else "文档"
    plan = {
        "artifact_type": artifact_type,
        "artifact_chars": len(content),
        "reviewers": [
            {
                "model": f"{PROVIDERS[s['provider']]['label']} / {s['model']}",
                "max_tokens": s["max_tokens"],
                "reasoning_effort": s["reasoning_effort"],
                **({"effort_note": s["effort_note"]} if s["effort_note"] else {}),
            }
            for s in eff_specs
        ],
        "arbitration": ({
            "model": f"{PROVIDERS[arb_spec['provider']]['label']} / {arb_spec['model']}",
            "max_tokens": arb_spec["max_tokens"],
            "reasoning_effort": arb_spec["reasoning_effort"],
            **({"effort_note": arb_note} if arb_note else {}),
        } if arb_spec else {"model": "（不仲裁，仅并排返回两份意见）"}),
        "model_calls": len(eff_specs) + (1 if arb_spec else 0),
        "cross_vendor": len({s["provider"] for s in eff_specs}) > 1,
        "use_cache": bool(use_cache),
        "report_path": report_path or "（不落盘，可用 export_review 事后导出）",
    }
    plan["estimated_time"] = estimate_dual_time(eff_specs, arb_spec, len(content))

    if dry_run:
        return json.dumps({
            "dry_run": True,
            "plan": plan,
            "note": "这是配置预览，未调用任何模型、未产生费用。"
                    "请把 plan 展示给用户确认（尤其是两个审校模型与仲裁模型），"
                    "确认后去掉 dry_run 正式提交。",
        }, ensure_ascii=False)

    global _task_seq
    with _lock:
        _task_seq += 1
        parent_id = f"task-{_task_seq:04d}"
        _tasks[parent_id] = {
            "task_id": parent_id,
            "parent": None,
            "kind": "dual",
            "provider": "+".join(s["provider"] for s in eff_specs),
            "model": " + ".join(s["model"] for s in eff_specs),
            "model_label": "并发双审：" + " + ".join(s["model"] for s in eff_specs),
            "max_tokens": eff_specs[0]["max_tokens"],
            "reasoning_effort": eff_specs[0]["reasoning_effort"],
            "artifact_type": artifact_type,
            "artifact_chars": len(content),
            "requirements": requirements or "",
            "use_cache": bool(use_cache),
            "report_path": report_path or "",
            "status": "queued",
            "created_ts": time.time(),
            "created_at": _now_str(),
            "started_ts": None,
            "finished_ts": None,
            "elapsed_s": 0.0,
            "progress": {},
            "children": [],
            "arbiter": None,
            "review": None,
            "meta": None,
            "error": None,
            "_llm_children": [
                {"provider": s["provider"], "model": s["model"],
                 "prompt": _make_prompt(artifact_type, content, focus, requirements, kind),
                 "max_tokens": s["max_tokens"], "reasoning_effort": s["reasoning_effort"]}
                for s in eff_specs
            ],
        }
        _prune_tasks()
    _save_tasks()

    threading.Thread(target=_run_dual_task,
                     args=(parent_id, [], arb_spec, artifact_type, len(content),
                           requirements or ""),
                     daemon=True).start()

    finished, waited = _wait_task(parent_id, wait_seconds)
    payload = _task_payload(parent_id)
    payload["plan"] = plan
    if not finished:
        payload["waited_s"] = waited
        payload["next_step"] = (
            f"双审已在后台并发执行，请调用 check_review('{parent_id}') 查看进度；"
            f"两个审校者与仲裁者都结束后会一次性返回合并结果。"
        )
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def export_review(task_id: str, output_path: str = "", include_raw: bool = True) -> str:
    """跨模型审校 · 把某个已完成任务的审校结果导出为 Markdown 报告（便于归档/交付）。

    报告含：任务信息、边界条件、结论与分级问题表、分歧点（双审）、原始 JSON。
    单审任务与双审任务都支持。

    Args:
        task_id: check_review / list_tasks 里的任务号
        output_path: 可选，输出路径。传目录则自动命名 <task_id>_<时间>.md；
            传空则用默认目录（见 list_models.report_dir）
        include_raw: 可选，默认 True，是否附上模型原始 JSON
    """
    t = _tasks.get(task_id)
    if not t:
        return json.dumps({"error": f"任务 {task_id} 不存在，调用 list_tasks 查看最近任务"},
                          ensure_ascii=False)
    if t.get("status") != "done":
        return json.dumps(
            {"error": f"任务 {task_id} 状态为 {t.get('status')}，只有已完成的任务才能导出报告",
             "hint": _task_progress_hint(t)},
            ensure_ascii=False)
    try:
        with _lock:
            snapshot = dict(t)
        text = render_report(snapshot)
        if not include_raw:
            # 去掉原始 JSON 区块
            text = re.sub(r"(?s)## 原始输出.*?(?=\n---\n)", "", text)
        path = _resolve_report_path(snapshot, output_path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        _log_event(f"{task_id} 报告导出 -> {path}")
    except Exception as e:
        return json.dumps({"error": f"导出报告失败: {e}"}, ensure_ascii=False)
    return json.dumps({
        "task_id": task_id,
        "report_path": path,
        "chars": len(text),
        "message": "报告已落盘，可直接在编辑器中打开或用 present_files 展示给用户",
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
