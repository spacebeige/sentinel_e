/**
 * Robust clipboard copy helper for formatted model responses.
 * - Strips hidden reasoning tags
 * - Removes simple markdown emphasis artifacts for plain-text copy
 * - Uses Clipboard API first, then textarea execCommand fallback
 */
export function sanitizeForClipboard(rawContent) {
  const source = String(rawContent ?? '');
  return source
    .replace(/<(think|thinking|analysis|reasoning|reflection|internal|system_note|meta)[^>]*>[\s\S]*?<\/\1>/gi, '')
    .replace(/\[(INTERNAL|DEBUG|SYSTEM)\][\s\S]*?\[\/\1\]/gi, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export async function copyToClipboard(rawContent) {
  const clean = sanitizeForClipboard(rawContent);
  if (!clean) return false;

  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(clean);
      return true;
    } catch {
      // fall through to textarea fallback
    }
  }

  try {
    const textarea = document.createElement('textarea');
    textarea.value = clean;
    textarea.style.cssText = 'position:fixed;opacity:0;pointer-events:none;left:-9999px;top:-9999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}
