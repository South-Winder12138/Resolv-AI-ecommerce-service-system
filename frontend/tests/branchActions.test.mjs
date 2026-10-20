import assert from 'node:assert/strict'
import test from 'node:test'

async function api() {
  try { return await import('../src/lib/branchActions.ts') }
  catch (error) { assert.fail('local draft behavior missing: ' + error.message) }
}
const makeAction = type => ({suggestion_id: 's-' + type, type, label: '入口', preview_only: true, draft: {title: '标题', content: '原始问题'}})

test('draft editing has no submission and preserves the original action', async () => {
  const {openDraft, editDraft, closeDraft} = await api()
  for (const type of ['transfer_human', 'create_ticket']) {
    const original = makeAction(type)
    const edited = editDraft(openDraft(original), 'content', '补充说明')
    assert.equal(edited.content, '补充说明')
    assert.equal(original.draft.content, '原始问题')
    assert.equal(closeDraft(), null)
    assert.equal(edited.approval_id, undefined)
  }
})

test('an execution-capable action cannot open a draft', async () => {
  const {openDraft} = await api()
  assert.throws(() => openDraft({...makeAction('transfer_human'), preview_only: false}))
})
