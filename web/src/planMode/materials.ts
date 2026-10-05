/**
 * Reference materials for the standalone React demo's grounded guidance.
 *
 * These are the shared API's `MaterialInput` values plus the client-side rules for the import panel. The
 * texts are page-lifetime UI state only (never localStorage): the parent holds the accepted list, the
 * panel edits local drafts, and a guide run freezes its own copy when it starts. This module owns the
 * bounds that mirror the server contract, the file gate, the whitespace-normalized "quote is an exact
 * excerpt of the material" test, and the Korean error text for `POST /api/plan/manual`.
 *
 * This is the demo's reference-based guidance flow, not the Live PWA's Plan review/approval state machine.
 */
import type { MaterialInput } from '../generated/api.generated';

/** Mirrors the server's `MaterialInput` bounds (trimmed 1..80 title, 0..32 version, 1..4000 text). */
export const MATERIAL_TITLE_MAX = 80;
export const MATERIAL_VERSION_MAX = 32;
export const MATERIAL_TEXT_MAX = 4000;
/** A plan step's evidence quote (`Evidence.quote`). */
export const MATERIAL_QUOTE_MAX = 120;
/** The plan request's material ceiling (m1..m4, in request order). */
export const MATERIALS_MAX = 4;
/** The server's source ceiling: a 5 MiB file becomes a ~6.7 MiB base64 body. */
export const MATERIAL_FILE_MAX_BYTES = 5 * 1024 * 1024;

const SUPPORTED_EXTENSIONS = ['.txt', '.md', '.docx', '.pdf'];

/** Identity of one material's contents, for "did this actually change?" comparisons. */
export function materialSignature(material: MaterialInput): string {
  return `${material.title}\u0000${material.version ?? ''}\u0000${material.text}`;
}

/** Identity of a whole material list (empty and null compare equal). */
export function materialsSignature(materials: readonly MaterialInput[] | null): string {
  return (materials ?? []).map(materialSignature).join('\u0001');
}

/**
 * Trimmed-or-null: one material the server contract will accept, or null. The panel emits only valid
 * materials, so a draft that is empty, has no title, or exceeds a bound can never leave a stale source
 * behind. Over-limit drafts are refused, never silently sliced.
 */
export function validMaterial(title: string, version: string, text: string): MaterialInput | null {
  const trimmedTitle = title.trim();
  const trimmedVersion = version.trim();
  const trimmedText = text.trim();
  if (trimmedTitle.length === 0 || trimmedTitle.length > MATERIAL_TITLE_MAX) return null;
  if (trimmedVersion.length > MATERIAL_VERSION_MAX) return null;
  if (trimmedText.length === 0 || trimmedText.length > MATERIAL_TEXT_MAX) return null;
  return trimmedVersion.length > 0
    ? { title: trimmedTitle, version: trimmedVersion, text: trimmedText }
    : { title: trimmedTitle, text: trimmedText };
}

export function sameMaterial(a: MaterialInput, b: MaterialInput): boolean {
  return a.title === b.title && (a.version ?? null) === (b.version ?? null) && a.text === b.text;
}

export function sameMaterials(a: readonly MaterialInput[] | null, b: readonly MaterialInput[] | null): boolean {
  const left = a ?? [];
  const right = b ?? [];
  return left.length === right.length && left.every((material, index) => sameMaterial(material, right[index]));
}

/** Collapse every whitespace run: the server's own "quote is an excerpt" test is whitespace-normalized. */
export function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

/** Whether `quote` is a whitespace-normalized excerpt of `materialText` (and within the quote bound). */
export function quoteInMaterial(quote: string, materialText: string): boolean {
  const needle = normalizeWhitespace(quote);
  return (
    needle.length > 0 &&
    needle.length <= MATERIAL_QUOTE_MAX &&
    normalizeWhitespace(materialText).includes(needle)
  );
}

/** The text of the material an evidence id names (`m1`..`m4`), or null when the id is not in the list. */
export function materialTextById(materials: readonly MaterialInput[] | null, materialId: string): string | null {
  const index = /^m([1-9]|1[0-2])$/.exec(materialId);
  if (!index || !materials) return null;
  return materials[Number(index[1]) - 1]?.text ?? null;
}

/** The title of the material an evidence id names, or null when the id is not in the list. */
export function materialTitleById(materials: readonly MaterialInput[] | null, materialId: string): string | null {
  const index = /^m([1-9]|1[0-2])$/.exec(materialId);
  if (!index || !materials) return null;
  return materials[Number(index[1]) - 1]?.title ?? null;
}

/** Cheap client gate before transfer; the server remains the authority on the actual content. */
export function isSupportedMaterialFile(file: Pick<File, 'name' | 'type'>): boolean {
  const dot = file.name.lastIndexOf('.');
  return dot >= 0 && SUPPORTED_EXTENSIONS.includes(file.name.slice(dot).toLowerCase());
}

/** A `POST /api/plan/manual` failure that keeps the server's closed-set code (never prose). */
export class MaterialImportError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string) {
    super(code);
    this.name = 'MaterialImportError';
    this.status = status;
    this.code = code;
  }
}

const MATERIAL_ERROR_TEXT: Record<string, string> = {
  manual_too_large: '파일 또는 압축 해제한 내용이 처리 한도를 넘습니다. 파일을 나누거나 필요한 글자를 붙여넣어 주세요.',
  manual_too_long: '자료 내용이 4,000자를 넘습니다. 필요한 부분만 남겨 주세요.',
  manual_empty: '글자를 읽을 수 없는 파일이나 PDF 페이지가 있습니다. 이미지·빈 페이지를 제외하거나 텍스트로 붙여넣어 주세요.',
  manual_invalid: '파일을 읽지 못했습니다. 손상되었거나 암호가 걸린 문서일 수 있습니다.',
  manual_unsupported: '지원하지 않는 형식입니다. .txt · .md · .docx · 텍스트 PDF만 올릴 수 있습니다.',
  invalid_file_name: '파일 이름을 확인할 수 없습니다. 파일 이름을 바꾼 뒤 다시 시도해 주세요.',
  payload_too_large: '파일이 서버 전송 한도를 넘습니다. 파일을 나눠 주세요.',
  body_too_large: '파일이 서버 전송 한도를 넘습니다. 파일을 나눠 주세요.',
  session_required: '세션이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.',
  session_mismatch: '세션이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.',
  session_expired: '세션이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.',
  origin_required: '보안을 위해 앱을 연 주소에서만 업로드할 수 있습니다. 주소를 확인해 주세요.',
  invalid_origin: '보안을 위해 앱을 연 주소에서만 업로드할 수 있습니다. 주소를 확인해 주세요.',
  invalid_request: '요청 형식이 올바르지 않습니다. 파일을 다시 선택해 주세요.',
  request_failed: '서버가 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.',
};

/** Plain, actionable Korean for every failure the import route can produce. */
export function describeMaterialError(error: unknown): string {
  if (error instanceof MaterialImportError) {
    return MATERIAL_ERROR_TEXT[error.code] ?? `자료를 불러오지 못했습니다(${error.status} ${error.code}).`;
  }
  if (error instanceof TypeError) return '서버에 연결하지 못했습니다. 네트워크 연결을 확인해 주세요.';
  if (error instanceof Error && error.message) return error.message;
  return '자료를 불러오지 못했습니다.';
}

/** Parse the standard `{ error, detail?, reason? }` failure body into a coded error. */
export async function readMaterialError(response: Response): Promise<MaterialImportError> {
  let code = 'request_failed';
  try {
    const payload: unknown = await response.json();
    if (
      payload !== null &&
      typeof payload === 'object' &&
      'error' in payload &&
      typeof (payload as { error?: unknown }).error === 'string'
    ) {
      code = (payload as { error: string }).error;
    }
  } catch {
    // keep the fallback code
  }
  return new MaterialImportError(response.status, code);
}

/** One uploaded file as raw base64 (no `data:` prefix), for the JSON import body. */
export function readFileBase64(file: File): Promise<string> {
  const { promise, resolve, reject } = Promise.withResolvers<string>();
  const reader = new FileReader();
  reader.onerror = () => reject(new Error('파일을 읽지 못했습니다.'));
  reader.onload = () => {
    const result = reader.result;
    if (typeof result !== 'string') {
      reject(new Error('파일을 읽지 못했습니다.'));
      return;
    }
    const comma = result.indexOf(',');
    resolve(comma >= 0 ? result.slice(comma + 1) : result);
  };
  reader.readAsDataURL(file);
  return promise;
}
