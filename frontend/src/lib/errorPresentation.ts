const messages: Record<string, string> = {
  context_count_unsupported: '模型服务暂时无法计算上下文，请稍后继续原运行。',
  runtime_error: '客服暂时遇到问题，请稍后继续原运行。',
  http_error: '服务连接暂时不可用，请稍后重新同步。',
  Failed_to_fetch: '无法连接客服服务，请检查连接后重试。',
}

export function friendlyError(error: string): string {
  if (error === 'Failed to fetch') return messages.Failed_to_fetch
  return messages[error] || (/^[a-z]+(?:_[a-z]+)+$/.test(error) ? '请求暂时无法完成，请稍后重新同步。' : error)
}
