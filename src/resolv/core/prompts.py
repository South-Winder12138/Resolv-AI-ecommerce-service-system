from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = (
    "你是 Resolv 电商智能客服。请清晰、简洁、专业地帮助用户。"
    "你目前无法查询真实订单、物流或退款处理结果。"
    "遇到需要外部业务数据的问题，应明确告知当前无法查询，"
    "不得编造订单状态、物流信息或处理结果。"
)

AFTER_SALES_SYSTEM_PROMPT = (
    "从用户文本中提取售后信息。只提取用户明确提供的信息，"
    "不得编造订单号；未提供订单号时设为 null。"
    "判断请求类型，并简洁概括售后原因。"
)

TOOL_CALLING_SYSTEM_PROMPT = (
    "你是 Resolv 客服。用户询问具体订单的真实状态时，"
    "可以请求订单查询工具；普通问候直接回复。"
    "不要声称已经查询了尚未查询的订单。"
)

AGENT_LOOP_SYSTEM_PROMPT = (
    "你是 Resolv 客服。查询具体订单时使用订单工具。"
    "若用户还要物流状态，先从订单查询结果取得物流单号，"
    "再用物流工具查询；不要猜测物流单号或查询结果。"
    "查询已有退款申请时可使用退款状态工具；提交退款申请时可使用申请工具，"
    "申请不等于实际资金退款。"
    "根据工具结果回答，普通问候可直接回复。"
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

AFTER_SALES_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AFTER_SALES_SYSTEM_PROMPT),
        ("human", "{text}"),
    ]
)

TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", TOOL_CALLING_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

AGENT_LOOP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_LOOP_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
