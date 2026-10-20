import assert from 'node:assert/strict'
import test from 'node:test'
import {offer,command,receipt} from './refundFixtures.mjs'
test('real RefundCardView shows immutable details and pending review semantics',async()=>{
  let render
  try{({renderComponent:render}=await import('./helpers/renderComponent.mjs'))}catch(e){assert.fail('actual TSX render helper missing: '+e.message)}
  const props={state:{conversationId:'c',offerId:'o',generation:1,open:true,busy:false,offer,submissionId:'s',submission:{...command,status:'completed',receipt},error:null,pollCount:0},onSubmit(){},onResume(){},onReassess(){},onRefresh(){},onClose(){}}
  const html=await render('../../src/components/chat/RefundCard.tsx','RefundCardView',props)
  for(const text of ['等待审核','CNY 90071992547409.93','不想要了','app','订单'])assert.ok(html.includes(text),text)
  assert.ok(!html.includes('<input'));assert.ok(!html.includes('退款成功'))
  const expired=await render('../../src/components/chat/RefundCard.tsx','RefundCardView',{...props,state:{...props.state,submission:null,submissionId:null,offer:{...offer,validity:'expired'}}})
  assert.ok(expired.includes('已过期'));assert.ok(expired.includes('重新判断'))
})

test('a receipt with retryable interrupted publication keeps the original resume action',async()=>{
  const {renderComponent:render}=await import('./helpers/renderComponent.mjs')
  const html=await render('../../src/components/chat/RefundCard.tsx','RefundCardView',{
    state:{conversationId:'c',offerId:'o',generation:1,open:true,busy:false,offer,submissionId:'s',submission:{...command,status:'interrupted',receipt,error_code:'refund_submission_publish_failed',retryable:true},error:'refund_submission_publish_failed',pollCount:0},
    onSubmit(){},onResume(){},onReassess(){},onRefresh(){},onClose(){},
  })
  assert.ok(html.includes('恢复原提交'))
  assert.ok(!html.includes('确认提交退款申请'))
})

for (const money of [{currency:'CNY',minor_units_decimal:'2300',unit_precision:2}, null]) {
  test(`a competing offer displays the registered receipt content (money ${money ? 'known' : 'unknown'})`,async()=>{
    const {renderComponent:render}=await import('./helpers/renderComponent.mjs')
    const actual={...receipt,created:false,reason:'原入口登记的原因',money}
    const html=await render('../../src/components/chat/RefundCard.tsx','RefundCardView',{
      state:{conversationId:'c',offerId:'o',generation:1,open:true,busy:false,offer,submissionId:'s',submission:{...command,status:'completed',receipt:actual},error:null,pollCount:0},
      onSubmit(){},onResume(){},onReassess(){},onRefresh(){},onClose(){},
    })
    assert.ok(html.includes('原入口登记的原因'))
    assert.ok(!html.includes(offer.reason))
    assert.ok(!html.includes('CNY 90071992547409.93'))
    assert.ok(money ? html.includes('CNY 23.00') : html.includes('申请金额未记录'))
    assert.ok(html.includes('app'))
  })
}
