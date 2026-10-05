/**
 * The bug-report endpoints (`/api/debug/bugreport*`) as a `ReportApi`: create with metadata, one raw `PUT`
 * per video segment, `PUT meta` to refresh `report.json`, then `done`.
 */
import { extensionForMime, type ReportApi } from './autoReporter';

async function expectOk(response: Response, step: string): Promise<Response> {
  if (response.ok) return response;
  let code = `${response.status}`;
  try {
    const body = (await response.json()) as { error?: string };
    if (body.error) code = `${response.status} ${body.error}`;
  } catch {
    // keep the status alone
  }
  throw new Error(`${step} 실패 (${code})`);
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export const bugReportApi: ReportApi = {
  async create(meta) {
    const response = await expectOk(
      await fetch('/api/debug/bugreport', {
        method: 'POST',
        headers: JSON_HEADERS,
        credentials: 'same-origin',
        body: JSON.stringify(meta),
      }),
      '리포트 생성'
    );
    return ((await response.json()) as { report_id: string }).report_id;
  },
  async putSegment(reportId, position, segment) {
    await expectOk(
      await fetch(`/api/debug/bugreport/${reportId}/video/${position}?ext=${extensionForMime(segment.mime)}`, {
        method: 'PUT',
        headers: { 'Content-Type': segment.mime },
        credentials: 'same-origin',
        body: segment.blob,
      }),
      `영상 ${position + 1} 업로드`
    );
  },
  async putFile(reportId, name, blob) {
    await expectOk(
      await fetch(`/api/debug/bugreport/${reportId}/file/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': blob.type || 'application/octet-stream' },
        credentials: 'same-origin',
        body: blob,
      }),
      `${name} 업로드`
    );
  },
  async putMeta(reportId, meta) {
    await expectOk(
      await fetch(`/api/debug/bugreport/${reportId}/meta`, {
        method: 'PUT',
        headers: JSON_HEADERS,
        credentials: 'same-origin',
        body: JSON.stringify(meta),
      }),
      '기록 갱신'
    );
  },
  async finish(reportId, keepalive) {
    await expectOk(
      await fetch(`/api/debug/bugreport/${reportId}/done`, { method: 'POST', credentials: 'same-origin', keepalive }),
      '리포트 마무리'
    );
  },
};

export async function bugReportEnabled(): Promise<boolean> {
  try {
    const response = await fetch('/api/debug/bugreport/config', { credentials: 'same-origin' });
    return response.ok && ((await response.json()) as { enabled?: boolean }).enabled === true;
  } catch {
    return false;
  }
}
