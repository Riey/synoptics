/**
 * Debug-only checkbox: while checked, this page's camera recording and guide state stream into a server
 * bug report. Shown only when the server has a report directory configured.
 */
import { useState } from 'react';

import type { BugReporter } from '../debug/useBugReporter';

export function BugReportToggle(props: { reporter: BugReporter }) {
  const { reporter } = props;
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  if (!reporter.available) return null;
  const sendComment = async () => {
    const text = draft.trim();
    if (!text || sending) return;
    setSending(true);
    try {
      await reporter.comment(text);
      setDraft('');
    } finally {
      setSending(false);
    }
  };
  const status = reporter.status;
  const statusText =
    reporter.mime === null
      ? '이 브라우저는 영상 녹화를 지원하지 않습니다'
      : !reporter.checked
        ? null
        : status?.error
          ? `전송 오류(다시 시도 중): ${status.error}`
          : status?.reportId
            ? `리포트 ${status.reportId} · 영상 ${status.uploaded}개 · ${(status.bytes / 1048576).toFixed(1)}MB` +
              (status.pending > 0 ? ` · 대기 ${status.pending}` : '') +
              (status.comments > 0 ? ` · 코멘트 ${status.comments}개` : '') +
              (status.calls > 0 ? ` · 모델 호출 ${status.calls}건` : '') +
              (status.dropped > 0 ? ` · 버림 ${status.dropped}` : '') +
              (reporter.recording ? ' · 녹화 중' : '')
            : reporter.recording
              ? '녹화 중 — 30초마다 전송합니다'
              : '카메라를 켜면 녹화·전송을 시작합니다';

  return (
    <div className="bug-report" data-testid="bug-report">
      <label className="mirror-toggle-field">
        <input
          type="checkbox"
          data-testid="bug-report-toggle"
          checked={reporter.checked}
          disabled={reporter.mime === null}
          onChange={(event) => reporter.setChecked(event.target.checked)}
        />
        <span>🐞 버그 리포트 전송 — 켜 두면 카메라 영상과 가이드 상태를 디버깅용으로 서버에 저장합니다.</span>
      </label>
      {reporter.checked && reporter.mime !== null && (
        <form
          className="bug-report-comment"
          onSubmit={(event) => {
            event.preventDefault();
            void sendComment();
          }}
        >
          <input
            className="input-text"
            data-testid="bug-report-comment"
            value={draft}
            placeholder="코멘트 — 지금 무엇이 이상한가요?"
            onChange={(event) => setDraft(event.target.value)}
          />
          <button type="submit" className="btn btn-secondary" disabled={sending || draft.trim() === ''}>
            {sending ? '보내는 중…' : '코멘트 보내기'}
          </button>
        </form>
      )}
      {statusText && (
        <span className="text-secondary bug-report-status" role="status">
          {statusText}
        </span>
      )}
    </div>
  );
}
