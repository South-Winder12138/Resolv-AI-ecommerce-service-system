import { request } from './client.ts'
import { parseRefundOffer, parseRefundSubmission } from '../lib/replyMetadata.ts'

const base = (c: string) => `/conversations/${encodeURIComponent(c)}`
const offerPath = (c: string, o: string) => `${base(c)}/refund-offers/${encodeURIComponent(o)}`
const commandPath = (c: string, s: string) => `${base(c)}/refund-submissions/${encodeURIComponent(s)}`
export async function getRefundOffer(c: string, o: string) { return parseRefundOffer(await request(offerPath(c, o))) }
export async function submitRefundOffer(c: string, o: string, body: { request_id: string; confirm: true }) { return parseRefundSubmission(await request(offerPath(c, o) + '/submit', { method: 'POST', body: JSON.stringify(body) })) }
export async function getRefundSubmission(c: string, s: string) { return parseRefundSubmission(await request(commandPath(c, s))) }
export async function resumeRefundSubmission(c: string, s: string) { return parseRefundSubmission(await request(commandPath(c, s) + '/resume', { method: 'POST', body: '{}' })) }
