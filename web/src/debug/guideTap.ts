/**
 * Debug tap on the guide lane's calls (`/api/guide/*`): while a bug report is being sent, every request and
 * its answer (or error) is handed to the subscriber. Off (`null`) by default, so the normal path costs one
 * null check per call.
 */

export interface GuideCallEvent {
  path: string;
  payload: unknown;
  response?: unknown;
  error?: string;
  /** Epoch ms. */
  startedAt: number;
  endedAt: number;
}

export interface StrippedImage {
  /** Dotted path of the field inside the request, e.g. `scene.image_base64`. */
  field: string;
  base64: string;
  mime: string;
}

let tap: ((event: GuideCallEvent) => void) | null = null;

export function setGuideTap(next: ((event: GuideCallEvent) => void) | null): void {
  tap = next;
}

/** Runs one guide call and reports it to the tap, if any; the call's own result or error passes through. */
export async function tapGuideCall<T>(path: string, payload: unknown, run: () => Promise<T>): Promise<T> {
  const listener = tap;
  if (!listener) return run();
  const startedAt = Date.now();
  try {
    const response = await run();
    listener({ path, payload, response, startedAt, endedAt: Date.now() });
    return response;
  } catch (error) {
    listener({ path, payload, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error), startedAt, endedAt: Date.now() });
    throw error;
  }
}

const DATA_URL = /^data:([^;,]+);base64,/;

/** A copy of `value` with every `image_base64` string replaced by `<image:field>`, plus the images. */
export function stripImages(value: unknown): { request: unknown; images: StrippedImage[] } {
  const images: StrippedImage[] = [];
  const walk = (node: unknown, path: string): unknown => {
    if (Array.isArray(node)) return node.map((item, index) => walk(item, `${path}[${index}]`));
    if (node === null || typeof node !== 'object') return node;
    const copy: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(node)) {
      const field = path ? `${path}.${key}` : key;
      if (key === 'image_base64' && typeof child === 'string') {
        const match = DATA_URL.exec(child);
        images.push({ field, base64: match ? child.slice(match[0].length) : child, mime: match ? match[1] : 'image/jpeg' });
        copy[key] = `<image:${field}>`;
      } else {
        copy[key] = walk(child, field);
      }
    }
    return copy;
  };
  return { request: walk(value, ''), images };
}

/** File-name-safe form of an image field: `before_scene.image_base64` → `before_scene`. */
export function imageFileTag(field: string): string {
  return field.replace(/\.image_base64$/, '').replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '').toLowerCase() || 'image';
}
