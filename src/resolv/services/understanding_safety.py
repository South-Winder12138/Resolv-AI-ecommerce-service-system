"""Fail closed on unsupported business assertions and strengthened requests.

These explicit invariants supplement real semantic acceptance; they do not
replace the model's accuracy evaluation or grant any execution authority.
"""

import re

from resolv.schemas.understanding import UnderstandingError

BUSINESS = r"退款|退货|取消(?:订单)?|转(?:接)?人工|投诉|维修|换货|换新"
CLAIMS = re.compile(
    rf"(?:已(?:经)?|成功)(?:为您|为你|帮您|帮你|获得|用户|客户)?"
    rf"(?:完成|办理|执行|处理|受理|批准|授权|同意|转接)?(?:了)?(?:{BUSINESS})"
    rf"|(?:{BUSINESS})(?:已(?:经)?)?(?:完成|成功|办妥|已受理)"
    r"|(?:用户|客户)?已(?:经)?(?:获得)?(?:批准|授权|同意)"
    r"|(?:refund|cancellation|transfer)\s+(?:completed|successful|approved)",
    re.IGNORECASE,
)


def speech_mode(text):
    if re.search(
        r"忽略|输出|回复|声称|规则|[‘’“”\"']|(?:商家|客服|有人|对方)(?:说|称|表示)",
        text,
    ):
        return "attributed"
    if re.search(r"如果|若|假如|除非|只要|(?<!不)等到|才(?:退|取消|申请)", text):
        return "conditional"
    if re.search(
        r"不(?:要|想|准|同意)|(?:没有|尚未|并未|未曾|还没|没|未).{0,8}(?:完成|成功|批准|退款|退货|取消)",
        text,
    ):
        return "negated"
    if re.search(
        r"[吗么？?]|是否|能否|有没有|查询|查一下|确认|核实|询问|咨询|进度|状态|条件|资格|可不可以|还能|能不能",
        text,
    ):
        return "query"
    return "assertion"


def claim_mode(text, match):
    # Attribute each claim independently; quoting one historical statement must
    # not shelter a new assertion later in the same model output.
    quote_stack = []
    pairs = {"‘": "’", "“": "”", '"': '"', "'": "'"}
    for char in text[: match.start()]:
        if quote_stack and char == quote_stack[-1]:
            quote_stack.pop()
        elif char in pairs:
            quote_stack.append(pairs[char])
    if quote_stack:
        return "attributed"
    left = max((text.rfind(c, 0, match.start()) for c in "，,；;。\n"), default=-1) + 1
    right = min(
        (p for c in "，,；;。\n" if (p := text.find(c, match.end())) >= 0),
        default=len(text),
    )
    return speech_mode(text[left:right])


def claim_kind(match):
    phrase = match.group()
    for kind, pattern in [
        ("approval", r"批准|授权|同意|approved"),
        ("cancel", r"取消|cancellation"),
        ("human", r"人工|transfer"),
        ("complaint", r"投诉"),
        ("refund", r"退款|退货|refund"),
        ("repair", r"维修|换货|换新"),
    ]:
        if re.search(pattern, phrase, re.IGNORECASE):
            return kind
    return phrase


def validate_request_semantics(output, supported):
    """Only current/linked action evidence can support a public business claim."""
    sources = [s for s in supported if s]
    claims = [
        (claim_kind(c), claim_mode(s, c)) for s in sources for c in CLAIMS.finditer(s)
    ]
    for claim in CLAIMS.finditer(output):
        mode, kind = claim_mode(output, claim), claim_kind(claim)
        if not any(k == kind and (m == mode or mode == "query") for k, m in claims):
            raise UnderstandingError("request_semantics_invalid", True)
    if re.search(
        rf"申请(?:{BUSINESS})|办理(?:{BUSINESS})|取消订单|转(?:接)?人工", output
    ):
        mode = speech_mode(output)
        for source in sources:
            if re.search(BUSINESS + r"|退吗", source):
                original_mode = speech_mode(source)
                if (
                    original_mode in {"query", "negated", "conditional", "attributed"}
                    and mode == "assertion"
                ):
                    raise UnderstandingError("request_semantics_invalid", True)
