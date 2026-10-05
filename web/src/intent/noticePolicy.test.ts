/**
 * How long a guide notice (replan, revert, lost target, talk replan…) stays in the narration bar.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { NOTICE_TTL_MS, noticeExpired } from './noticePolicy.ts';

test('a notice is cleared once the step changes or 8 s pass, whichever comes first', () => {
  assert.equal(NOTICE_TTL_MS, 8000);
  // Operator screenshot 2026-10-02: "계획이 다시 짜였습니다. 새 1단계부터 안내합니다." still shown at step 2/3.
  const replanned = { text: '계획이 다시 짜였습니다. 새 1단계부터 안내합니다.', stepIndex: 0, at: 1_000 };
  assert.equal(noticeExpired(replanned, 0, 2_000), false);
  assert.equal(noticeExpired(replanned, 1, 2_000), true, 'the step moved on');
  assert.equal(noticeExpired(replanned, 0, 1_000 + NOTICE_TTL_MS), true, '8 s passed on the same step');
  assert.equal(noticeExpired(replanned, 0, 1_000 + NOTICE_TTL_MS - 1), false);
  assert.equal(noticeExpired(null, 3, 99_000), false);
});
