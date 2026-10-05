/**
 * Reference-material input for the standalone React demo: file import (extracted on the app server)
 * plus paste/edit rows, up to the shared contract's `materials` ceiling.
 *
 * The parent owns the accepted list; this panel owns only the drafts. Every change is validated locally
 * and `onChange` is called with the valid materials (never a stale source while a draft is invalid) — so
 * the parent's task fence retires an old guide rather than mis-attributing it, and an incomplete edit is
 * reported through `onDraftStateChange` instead of being silently dropped or sliced. The raw material
 * text is never written to localStorage.
 *
 * This is the demo's reference-based guidance flow. The Live PWA is where a plan is reviewed and
 * approved/revised; this panel only feeds reference documents to the demo's plan request.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import type { MaterialInput } from '../generated/api.generated';
import {
  MATERIALS_MAX,
  MATERIAL_FILE_MAX_BYTES,
  MATERIAL_TEXT_MAX,
  MATERIAL_TITLE_MAX,
  MATERIAL_VERSION_MAX,
  MaterialImportError,
  describeMaterialError,
  isSupportedMaterialFile,
  materialsSignature,
  readFileBase64,
  readMaterialError,
  validMaterial,
} from './materials';
import './plan-mode.css';

export interface MaterialDraftState {
  /** Any row holds typed/pasted content (a source the user is working on). */
  hasDraft: boolean;
  /** Every row that holds content is a valid material (nothing over-limit or missing a title). */
  complete: boolean;
}

interface MaterialDraft {
  key: number;
  title: string;
  version: string;
  text: string;
}

function draftsFrom(materials: readonly MaterialInput[], firstKey: number): MaterialDraft[] {
  if (materials.length === 0) return [{ key: firstKey, title: '', version: '', text: '' }];
  return materials.map((material, index) => ({
    key: firstKey + index,
    title: material.title,
    version: material.version ?? '',
    text: material.text,
  }));
}

export function PlanMaterialPanel({
  materials,
  onChange,
  onDraftStateChange,
  ensureSession,
  disabled,
}: {
  materials: readonly MaterialInput[];
  onChange: (materials: MaterialInput[]) => void;
  onDraftStateChange?: (state: MaterialDraftState) => void;
  ensureSession: () => Promise<string>;
  disabled: boolean;
}) {
  const nextKeyRef = useRef(0);
  const [drafts, setDrafts] = useState<MaterialDraft[]>(() => draftsFrom(materials, 0));
  nextKeyRef.current = drafts.reduce((max, draft) => Math.max(max, draft.key + 1), nextKeyRef.current);

  const [importingKey, setImportingKey] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileNote, setFileNote] = useState<{ key: number; text: string } | null>(null);

  const lastEmittedRef = useRef<string>(materialsSignature(materials));
  /** Bumped on every import; a result whose sequence is stale may not touch the drafts. */
  const importSeqRef = useRef(0);
  const controllersRef = useRef(new Map<number, AbortController>());
  const fileInputsRef = useRef(new Map<number, HTMLInputElement>());

  // One extraction at a time: a new import or any edit supersedes the in-flight one (its result can never apply).
  const cancelImports = useCallback(() => {
    importSeqRef.current += 1;
    for (const controller of controllersRef.current.values()) controller.abort();
    controllersRef.current.clear();
    setImportingKey(null);
  }, []);

  const editDraft = useCallback(
    (key: number, field: 'title' | 'version' | 'text', value: string) => {
      // Editing supersedes a file extraction just as selecting another file does.
      cancelImports();
      setFileNote(null);
      setError(null);
      setDrafts((current) => current.map((draft) => (draft.key === key ? { ...draft, [field]: value } : draft)));
    },
    [cancelImports]
  );

  // One reconciliation: adopt an external change (the parent cleared or replaced the list) without
  // clobbering a draft with the value this panel itself just emitted, and otherwise publish the valid
  // materials — the single source of truth for the parent — whenever a draft changes.
  useEffect(() => {
    const propSignature = materialsSignature(materials);
    if (propSignature !== lastEmittedRef.current) {
      lastEmittedRef.current = propSignature;
      cancelImports();
      setDrafts(draftsFrom(materials, nextKeyRef.current));
      setFileNote(null);
      setError(null);
      return;
    }
    const parsed: MaterialInput[] = [];
    for (const draft of drafts) {
      const material = validMaterial(draft.title, draft.version, draft.text);
      if (material) parsed.push(material);
    }
    const draftSignature = materialsSignature(parsed);
    if (draftSignature === lastEmittedRef.current) return;
    lastEmittedRef.current = draftSignature;
    onChange(parsed);
  }, [drafts, materials, onChange, cancelImports]);

  // Whether the user is mid-edit on an incomplete source; the parent blocks Start on this instead of
  // silently sending a partial (or slicing it).
  useEffect(() => {
    let hasDraft = false;
    let incomplete = false;
    for (const draft of drafts) {
      const holdsContent = draft.title.trim() !== '' || draft.version.trim() !== '' || draft.text.trim() !== '';
      if (!holdsContent) continue;
      hasDraft = true;
      if (validMaterial(draft.title, draft.version, draft.text) === null) incomplete = true;
    }
    onDraftStateChange?.({ hasDraft, complete: !incomplete });
  }, [drafts, onDraftStateChange]);

  // Leaving the flow (or the session ending) cancels in-flight extractions; their results can never apply.
  useEffect(() => {
    if (!disabled) return;
    cancelImports();
  }, [disabled, cancelImports]);

  useEffect(
    () => () => {
      // Unmount: abort in-flight extractions without a state update (the component is going away).
      importSeqRef.current += 1;
      for (const controller of controllersRef.current.values()) controller.abort();
      controllersRef.current.clear();
    },
    []
  );

  const cancelImport = useCallback(() => {
    cancelImports();
    setFileNote({ key: -1, text: '가져오기를 취소했습니다.' });
  }, [cancelImports]);

  const addRow = useCallback(() => {
    setDrafts((current) =>
      current.length >= MATERIALS_MAX
        ? current
        : [...current, { key: nextKeyRef.current++, title: '', version: '', text: '' }]
    );
  }, []);

  const removeRow = useCallback(
    (key: number) => {
      cancelImports();
      setDrafts((current) => {
        const remaining = current.filter((draft) => draft.key !== key);
        return remaining.length > 0 ? remaining : [{ key: nextKeyRef.current++, title: '', version: '', text: '' }];
      });
    },
    [cancelImports]
  );

  const handleFile = useCallback(
    async (key: number, event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null;
      // Reset first so picking the same file again still fires a change event.
      event.target.value = '';
      if (!file) return;

      cancelImports();
      setError(null);
      setFileNote(null);

      if (file.size > MATERIAL_FILE_MAX_BYTES) {
        setError(describeMaterialError(new MaterialImportError(413, 'manual_too_large')));
        return;
      }
      if (!isSupportedMaterialFile(file)) {
        setError(describeMaterialError(new MaterialImportError(415, 'manual_unsupported')));
        return;
      }

      const seq = importSeqRef.current;
      const controller = new AbortController();
      controllersRef.current.set(key, controller);
      setImportingKey(key);
      try {
        const sessionId = await ensureSession();
        const contentBase64 = await readFileBase64(file);
        if (seq !== importSeqRef.current || controller.signal.aborted) return;
        const response = await fetch('/api/plan/manual', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ session_id: sessionId, filename: file.name, content_base64: contentBase64 }),
          signal: controller.signal,
        });
        if (seq !== importSeqRef.current || controller.signal.aborted) return;
        if (!response.ok) throw await readMaterialError(response);
        const source = (await response.json()) as MaterialInput;
        if (seq !== importSeqRef.current || controller.signal.aborted) return;
        setError(null);
        setDrafts((current) =>
          current.map((draft) => (draft.key === key ? { ...draft, title: source.title, text: source.text } : draft))
        );
        setFileNote({
          key,
          text: `${source.title}에서 ${source.text.length.toLocaleString('ko-KR')}자를 가져왔습니다. 내용을 확인한 뒤 시작하세요.`,
        });
      } catch (caught) {
        if (seq !== importSeqRef.current || controller.signal.aborted) return;
        setError(describeMaterialError(caught));
      } finally {
        if (seq === importSeqRef.current) {
          controllersRef.current.delete(key);
          setImportingKey(null);
        }
      }
    },
    [ensureSession]
  );

  return (
    <section className="card plan-mode-panel" data-testid="plan-material-panel" data-count={drafts.length}>
      <h2 className="card-title">참고자료 (선택)</h2>
      <p className="card-description">
        작업서·설명서를 최대 {MATERIALS_MAX}개까지 올리거나 붙여넣으세요. 예: <strong>짐 싸기</strong> — 여행 가방에
        품목 목록대로 담기 · <strong>장비 설치</strong> — 삼각대와 조명을 매뉴얼 순서대로 세팅.
      </p>
      <p className="card-description" data-testid="plan-material-transfer">
        올린 파일은 글자 추출을 위해 <strong>이 앱 서버</strong>로 전송됩니다. 참고자료 기반 안내를 시작하면 자료와 현재
        화면이 AI 제공자의 <strong>계획·완료 확인·대화 요청</strong>에 사용됩니다.
      </p>
      <p className="card-description text-secondary">
        지원 형식: .txt · .md · .docx · 텍스트 PDF (최대 5MiB, PDF 100쪽). 이미지·도표는 읽지 않습니다.
        스캔·빈 페이지가 포함된 PDF와 암호 문서는 글자를 직접 붙여넣어 주세요.
      </p>
      <p className="card-description text-secondary" data-testid="plan-material-demo-notice">
        이 화면은 별도의 참고자료 데모입니다. 실제 검토 후 승인·수정하는 Plan 흐름은 Live PWA에서 진행합니다.
      </p>

      {drafts.map((draft, index) => {
        const trimmedTitle = draft.title.trim();
        const trimmedVersion = draft.version.trim();
        const trimmedText = draft.text.trim();
        const titleOver = trimmedTitle.length > MATERIAL_TITLE_MAX;
        const versionOver = trimmedVersion.length > MATERIAL_VERSION_MAX;
        const textOver = trimmedText.length > MATERIAL_TEXT_MAX;
        const importing = importingKey === draft.key;
        return (
          <div className="plan-mode-material" key={draft.key} data-testid="plan-material-row">
            <div className="plan-mode-material__head">
              <span className="form-label">자료 {index + 1}</span>
              {drafts.length > 1 && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-testid="plan-material-remove"
                  aria-label={`자료 ${index + 1} 삭제`}
                  disabled={disabled}
                  onClick={() => removeRow(draft.key)}
                >
                  삭제
                </button>
              )}
            </div>
            <div className="plan-mode-row">
              <div className="plan-mode-field">
              <label className="form-label" htmlFor={`material-title-${draft.key}`}>자료 {index + 1} 제목 (필수)</label>
              <input
                id={`material-title-${draft.key}`}
                className="input-text"
                type="text"
                value={draft.title}
                disabled={disabled}
                autoComplete="off"
                placeholder="자료 제목 (필수, 1~80자)"
                data-testid="plan-material-title"
                onChange={(event) => editDraft(draft.key, 'title', event.target.value)}
              />
              </div>
              <div className="plan-mode-field">
              <label className="form-label" htmlFor={`material-version-${draft.key}`}>자료 {index + 1} 버전 (선택)</label>
              <input
                id={`material-version-${draft.key}`}
                className="input-text"
                type="text"
                value={draft.version}
                disabled={disabled}
                autoComplete="off"
                placeholder="버전 (선택, 최대 32자)"
                data-testid="plan-material-version"
                onChange={(event) => editDraft(draft.key, 'version', event.target.value)}
              />
              </div>
            </div>
            <div className="plan-mode-row">
              <button
                type="button"
                className="btn btn-secondary"
                data-testid="plan-material-file"
                aria-label={`자료 ${index + 1} 파일 선택`}
                disabled={disabled || importing}
                onClick={() => fileInputsRef.current.get(draft.key)?.click()}
              >
                파일 선택
              </button>
              {importing && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-testid="plan-material-cancel"
                  onClick={cancelImport}
                >
                  가져오기 취소
                </button>
              )}
              <input
                ref={(node) => {
                  if (node) fileInputsRef.current.set(draft.key, node);
                  else fileInputsRef.current.delete(draft.key);
                }}
                className="visually-hidden"
                type="file"
                tabIndex={-1}
                aria-hidden="true"
                accept=".txt,.md,.docx,.pdf,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                disabled={disabled || importing}
                onChange={(event) => void handleFile(draft.key, event)}
              />
              <span className={`plan-mode-count${titleOver ? ' plan-mode-count--over' : ''}`}>
                제목 {trimmedTitle.length} / {MATERIAL_TITLE_MAX}자
              </span>
              {trimmedVersion.length > 0 && (
                <span className={`plan-mode-count${versionOver ? ' plan-mode-count--over' : ''}`}>
                  버전 {trimmedVersion.length} / {MATERIAL_VERSION_MAX}자
                </span>
              )}
            </div>
            {importing && (
              <p className="plan-mode-status" role="status" data-testid="plan-material-importing">
                파일에서 글자를 추출하는 중입니다…
              </p>
            )}
            {fileNote?.key === draft.key && (
              <p className="plan-mode-status" role="status" data-testid="plan-material-file-note">
                {fileNote.text}
              </p>
            )}
            <label className="form-label" htmlFor={`material-text-${draft.key}`}>자료 {index + 1} 내용</label>
            <textarea
              id={`material-text-${draft.key}`}
              className="input-text plan-mode-textarea"
              rows={6}
              value={draft.text}
              disabled={disabled}
              placeholder="자료 내용을 붙여넣으세요. 순서·수량·주의사항이 담긴 원문이 좋습니다."
              data-testid="plan-material-text"
              onChange={(event) => editDraft(draft.key, 'text', event.target.value)}
            />
            <span className={`plan-mode-count${textOver ? ' plan-mode-count--over' : ''}`}>
              내용 {trimmedText.length.toLocaleString('ko-KR')} / {MATERIAL_TEXT_MAX.toLocaleString('ko-KR')}자
            </span>
            {(titleOver || versionOver || textOver) && (
              <p className="form-feedback form-feedback-error" role="alert" data-testid="plan-material-over">
                {textOver
                  ? `자료 내용이 너무 깁니다(최대 ${MATERIAL_TEXT_MAX.toLocaleString('ko-KR')}자). 필요한 부분만 남겨 주세요.`
                  : titleOver
                    ? `자료 제목이 너무 깁니다(최대 ${MATERIAL_TITLE_MAX}자).`
                    : `버전이 너무 깁니다(최대 ${MATERIAL_VERSION_MAX}자).`}
              </p>
            )}
          </div>
        );
      })}

      {drafts.length < MATERIALS_MAX && (
        <div className="plan-mode-row">
          <button type="button" className="btn btn-secondary" data-testid="plan-material-add" disabled={disabled} onClick={addRow}>
            자료 추가 ({drafts.length}/{MATERIALS_MAX})
          </button>
        </div>
      )}

      {error && (
        <p className="form-feedback form-feedback-error" role="alert" data-testid="plan-material-error">
          {error}
        </p>
      )}

      <p className="card-description text-secondary">
        한 작업은 최대 16단계로 안내합니다. 필수 절차가 더 많으면 작업 범위를 나눠 주세요.
        AI가 절차를 빠뜨리거나 잘못 해석할 수 있으므로 시작 후 전체 단계와 원문을 비교해 주세요.
      </p>
    </section>
  );
}
