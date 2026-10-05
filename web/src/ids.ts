/**
 * Opaque scene and asset identifiers.
 *
 * `crypto.randomUUID` is only guaranteed in secure contexts and is missing from older Safari, so ids
 * fall back to `getRandomValues` and then to `Math.random`. What matters contractually is that no id is
 * ever reused across camera sessions or scene sources: that is what makes a late guidance result
 * fenceable by frame identity alone.
 */
export function newOpaqueId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  const bytes = new Uint8Array(16);
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  let hex = '';
  for (const byte of bytes) hex += byte.toString(16).padStart(2, '0');
  return `${prefix}-${hex}`;
}
