export const money = {currency:'CNY',minor_units_decimal:'9007199254740993',unit_precision:2}
export const offer = {offer_id:'o',order_id:'A',money,reason:'不想要了',created_at:'2026-10-16T04:00:00Z',expires_at:'2026-10-16T04:30:00Z',validity:'available',submission:null}
export const command = {submission_id:'s',status:'running',receipt:null,error_code:null,retryable:false}
export const receipt = {application_id:'app',order_source:'test',order_id:'A',status:'pending',created:true,money,reason:'不想要了',created_at:'2026-10-16T04:01:00Z'}
const window = {window_id:'w',document_id:'doc',section_instance_id:'section',chunk_ids:['a'],center_chunk_ids:['a'],center_candidates:[],rerank_score:1}
export const evidence = {windows:[window],citations:[{...window,citation_number:1,source:'policy.md',title:'退款政策',section_path:'退款',index_version:'v',index_profile_hash:'a'.repeat(64),sigmoid_score:0.8}],chunks:[{chunk:{chunk_id:'a',document_id:'doc',text:'普通商品完好可二售，签收七天内可无理由退货。',answer:'退款规则',source:'policy.md',questions:'退款政策',section_path:'退款',chunk_index:0},section_instance_id:'section'}],serialized:'fixture',input_tokens:50,evidence_hash:'b'.repeat(64)}
export const refund = {operation:'apply',verdict:'eligible',order_id:'A',reason:'符合条件',missing_information:[],unverified_conditions:[],next_action:'show_offer',release_id:'r',index_version:'v',evidence,policy_citations:[{citation_number:1,window_id:'w',quote:'签收七天内可无理由退货'}],offer_id:'o',application:null,status_scope:'local_only'}
export const metadata = {schema_version:3,reply_kind:'answer',understanding_id:'u',primary_task_id:'p',branch:'refund_flow',clarifies_understanding_id:null,suggested_actions:[],knowledge:null,refund}
export function fakeClock() {
  let tasks=[], ids=0
  return {newRequestId:()=>`request-${++ids}`,setTimeout(fn,ms){const t={fn,ms};tasks.push(t);return t},clearTimeout(t){tasks=tasks.filter(x=>x!==t)},get tasks(){return tasks},async tick(){const t=tasks.shift();if(t){await t.fn();await Promise.resolve()}}}
}
