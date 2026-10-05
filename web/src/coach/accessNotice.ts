/**
 * The sentence shown in place of the access-code field when `/api/health` says no code is required.
 *
 * The server names its mode (`access_mode`); the browser never infers it from its own address. An
 * open-access deployment (`AISW_OPEN_ACCESS=1`) is served inside a tailnet, so the network is the
 * boundary; a local-only deployment (`AISW_LOCAL_ONLY=1`) is reached over loopback. An older server that
 * does not report `access_mode` only ever had the local code-less mode, so it keeps the local wording.
 */
import type { GuideHealthResponse } from '../generated/api.generated';

export type AccessModeName = GuideHealthResponse['access_mode'];

export const OPEN_ACCESS_NOTICE = '테일넷 내부 연결이라 접근 코드 없이 사용합니다.';
export const LOCAL_ACCESS_NOTICE =
  '로컬 전용 연결이라 접근 코드 없이 사용합니다. 네트워크에 직접 공개하는 배포는 접근 코드로 보호해야 합니다.';

export function codeLessAccessNotice(mode: AccessModeName): string {
  return mode === 'open' ? OPEN_ACCESS_NOTICE : LOCAL_ACCESS_NOTICE;
}

/** The suffix on the health status line (`준비 완료 (deepseek) · …`); empty when a code is required. */
export function accessModeStatusSuffix(accessCodeRequired: boolean, mode: AccessModeName): string {
  if (accessCodeRequired) return '';
  return mode === 'open' ? ' · 테일넷 모드' : ' · 로컬 모드';
}
