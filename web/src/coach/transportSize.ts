/**
 * The transport bounds every camera upload is held to, shared by the main thread and the capture workers
 * (pure: no DOM, so a worker can import it without pulling in the main-thread capture code).
 */

/** The human-facing noun used in bounds errors, so camera frames never read as "사진". */
export type SceneSubject = '사진' | '카메라 화면';

const MAX_DIM = 1920;
const MAX_TOTAL_PIXELS = 2073600;
const MIN_WIDTH = 320;
const MIN_HEIGHT = 240;
export const MAX_DECODED_BYTES = 1500000; // 1.5MB server limit
export const MAX_BASE64_LENGTH = 2000000; // Base64 representation of <= 1.5MB

/** The quality the JPEG search starts at on both encode paths. */
export const JPEG_START_QUALITY = 0.88;

/**
 * The uniform-scale transport size for a source, with the established bounds errors. Shared by the
 * main-thread encoder and the off-thread ones, so the same frame is never sized differently by path.
 * `maxLongSide` lowers the size cap (the adaptive tracking upload, `trackScale.ts`); never above `MAX_DIM`.
 */
export function computeTransportSize(
  width: number,
  height: number,
  subject: SceneSubject,
  maxLongSide: number = MAX_DIM
): { targetWidth: number; targetHeight: number } {
  if (width < MIN_WIDTH || height < MIN_HEIGHT) {
    throw new Error(
      `${subject} 해상도가 너무 작습니다 (${width}x${height}). 최소 ${MIN_WIDTH}x${MIN_HEIGHT} 이상이어야 합니다.`
    );
  }

  const aspect = width / height;
  if (aspect > 4.0 || aspect < 0.25) {
    throw new Error(
      `${subject}의 가로세로 비율이 너무 왜곡되었습니다. 대상이 전체적으로 보이도록 일반적인 각도에서 다시 시도해주세요.`
    );
  }

  // Uniform scaling maintaining aspect ratio strictly
  let scale = 1.0;
  const longSide = Math.min(MAX_DIM, maxLongSide);
  if (width > longSide || height > longSide) {
    scale = Math.min(longSide / width, longSide / height);
  }

  let targetWidth = Math.round(width * scale);
  let targetHeight = Math.round(height * scale);
  if (targetWidth * targetHeight > MAX_TOTAL_PIXELS) {
    const pixelScale = Math.sqrt(MAX_TOTAL_PIXELS / (targetWidth * targetHeight));
    targetWidth = Math.floor(targetWidth * pixelScale);
    targetHeight = Math.floor(targetHeight * pixelScale);
  }

  // Double check minimums and strict pixel limit
  if (targetWidth < MIN_WIDTH || targetHeight < MIN_HEIGHT) {
    const upScale = Math.max(MIN_WIDTH / targetWidth, MIN_HEIGHT / targetHeight);
    targetWidth = Math.round(targetWidth * upScale);
    targetHeight = Math.round(targetHeight * upScale);
  }

  // Final single-pixel adjustment if rounding pushed product slightly over limit
  while (targetWidth * targetHeight > MAX_TOTAL_PIXELS) {
    if (targetWidth > targetHeight) {
      targetWidth -= 1;
    } else {
      targetHeight -= 1;
    }
  }

  return { targetWidth, targetHeight };
}
