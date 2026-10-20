import { ApiError } from '../api/client.ts'
import type * as RefundApi from '../api/refunds.ts'
import type { RefundOfferView, RefundPublicMoney, RefundSubmissionView } from '../types/agent.ts'
import { parseRefundMoney } from './replyMetadata.ts'

export type RefundCardState = { conversationId: string; offerId: string; generation: number; open: boolean; busy: boolean; offer: RefundOfferView | null; submissionId: string | null; submission: RefundSubmissionView | null; requestId: string | null; error: string | null; retryable: boolean; pollCount: number }
export type RefundCardEvent = { type: 'offer' | 'submission' | 'change'; conversationId: string; offerId: string; generation: number; submissionId?: string; offer?: RefundOfferView; submission?: RefundSubmissionView; patch?: Partial<RefundCardState> }
export function reduceRefundCard(state: RefundCardState, event: RefundCardEvent): RefundCardState {
  if (event.generation !== state.generation || event.conversationId !== state.conversationId || event.offerId !== state.offerId || (event.submissionId && state.submissionId && state.submissionId !== event.submissionId)) return state
  if (event.type === 'offer' && event.offer) {
    if (event.offer.offer_id !== state.offerId) return state
    const submission = event.offer.submission || state.submission
    if (submission && state.submissionId && submission.submission_id !== state.submissionId) return state
    return {...state, offer:event.offer,submission,submissionId:submission?.submission_id || state.submissionId,busy:false,error:null}
  }
  if (event.type === 'submission' && event.submission) {
    if (state.submissionId && state.submissionId !== event.submission.submission_id) return state
    return {...state,submission:event.submission,submissionId:event.submission.submission_id,busy:false,error:event.submission.error_code,retryable:event.submission.retryable}
  }
  return {...state,...event.patch}
}
export function formatRefundMoney(value: RefundPublicMoney): string {
  const money = parseRefundMoney(value), n = BigInt(money.minor_units_decimal), scale = 10n ** BigInt(money.unit_precision)
  return `${money.currency} ${n / scale}${money.unit_precision ? '.' + String(n % scale).padStart(money.unit_precision, '0') : ''}`
}
export function canSubmitRefund(state: RefundCardState) { return !state.busy && state.offer?.validity === 'available' && !state.submissionId && (!state.error || state.error === 'refund_network_unknown') }
export function canResumeRefund(state: RefundCardState) {
  return !state.busy && !!state.submissionId && state.submission?.status !== 'completed' && state.submission?.status !== 'failed' && (!state.submission?.receipt || (state.submission.status === 'interrupted' && state.submission.retryable))
}
type Clock = {newRequestId: () => string; setTimeout: (callback: () => void | Promise<void>, ms: number) => unknown; clearTimeout: (id: unknown) => void}
export const refundClock: Clock = {newRequestId:()=>crypto.randomUUID(),setTimeout:(fn,ms)=>setTimeout(fn,ms),clearTimeout:id=>clearTimeout(id as ReturnType<typeof setTimeout>)}
type Api = Pick<typeof RefundApi,'getRefundOffer' | 'submitRefundOffer' | 'getRefundSubmission' | 'resumeRefundSubmission'>
const empty = (generation = 0): RefundCardState => ({conversationId:'',offerId:'',generation,open:false,busy:false,offer:null,submissionId:null,submission:null,requestId:null,error:null,retryable:false,pollCount:0})

export function createRefundCardController({api,clock = refundClock,onChange}: {api: Api; clock?: Clock; onChange: (state: RefundCardState) => void}) {
  let state = empty(), timer: unknown, disposed = false
  const requests = new Map<string,string>()
  const identity = () => ({generation:state.generation,conversationId:state.conversationId,offerId:state.offerId})
  const nextOperation = () => { state = {...state,generation:state.generation + 1}; return identity() }
  const active = (id: ReturnType<typeof identity>) => !disposed && state.open && id.generation === state.generation && id.conversationId === state.conversationId && id.offerId === state.offerId
  function change(patch: Partial<RefundCardState>, id = identity()) {
    if (!active(id)) return
    state = reduceRefundCard(state,{type:'change',...id,patch}); onChange(state)
  }
  function cancel() { if (timer !== undefined) clock.clearTimeout(timer); timer = undefined }
  function failure(error: unknown, id: ReturnType<typeof identity>) {
    if (!active(id)) return
    if (error instanceof ApiError) change({busy:false,error:error.code,retryable:error.retryable,submissionId:error.submissionId || state.submissionId},id)
    else change({busy:false,error:'refund_network_unknown',retryable:true},id)
  }
  function acceptSubmission(view: RefundSubmissionView, id: ReturnType<typeof identity>) {
    if (!active(id)) return
    state = reduceRefundCard(state,{type:'submission',...id,submissionId:view.submission_id,submission:view}); onChange(state)
    poll(id)
  }
  function poll(id: ReturnType<typeof identity>) {
    cancel()
    if (!active(id) || !state.submissionId || !['created','running'].includes(state.submission?.status || '') || state.error || state.pollCount >= 10) return
    const command = state.submissionId
    timer = clock.setTimeout(async()=>{
      timer = undefined
      if (!active(id) || state.submissionId !== command) return
      change({pollCount:state.pollCount + 1},id)
      try { acceptSubmission(await api.getRefundSubmission(id.conversationId,command),id) }
      catch(error) { failure(error,id) }
    },2000)
  }
  async function refresh() {
    if (!active(identity()) || state.busy) return
    const id = nextOperation()
    cancel(); change({busy:true,error:null,pollCount:0},id)
    try {
      const offer = await api.getRefundOffer(id.conversationId,id.offerId)
      if (!active(id)) return
      state = reduceRefundCard(state,{type:'offer',...id,offer}); onChange(state)
      if (state.submissionId && state.submission?.status !== 'completed') acceptSubmission(await api.getRefundSubmission(id.conversationId,state.submissionId),id)
      else poll(id)
    } catch(error) { failure(error,id) }
  }
  async function submit() {
    if (!active(identity()) || !canSubmitRefund(state)) return
    const id = nextOperation(), key = JSON.stringify([id.conversationId,id.offerId]), requestId = requests.get(key) || clock.newRequestId()
    requests.set(key,requestId); cancel(); change({busy:true,error:null,requestId,pollCount:0},id)
    try { acceptSubmission(await api.submitRefundOffer(id.conversationId,id.offerId,{request_id:requestId,confirm:true}),id) }
    catch(error) { failure(error,id) }
  }
  async function resume() {
    if (!active(identity()) || !canResumeRefund(state)) return
    const id = nextOperation(), command = state.submissionId!
    cancel(); change({busy:true,error:null,pollCount:0},id)
    try { acceptSubmission(await api.resumeRefundSubmission(id.conversationId,command),id) }
    catch(error) { failure(error,id) }
  }
  function close() { cancel(); state = {...state,open:false,busy:false,generation:state.generation + 1}; if (!disposed) onChange(state) }
  async function open(conversationId: string, offerId: string) {
    if (disposed) return
    cancel(); state = {...empty(state.generation + 1),conversationId,offerId,open:true,requestId:requests.get(JSON.stringify([conversationId,offerId])) || null}; onChange(state)
    await refresh()
  }
  return {open,close,refresh,submit,resume,getState:()=>state,dispose(){close();disposed = true;requests.clear()}}
}
