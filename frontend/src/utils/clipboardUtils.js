/**
 * ============================================================
 * Clipboard Utilities
 * ============================================================
 *
 * Handles:
 * - Cross-browser clipboard operations
 * - Fallback for older browsers
 * - Error handling
 * - Toast notifications
 */

/**
 * Copy text to clipboard using modern Clipboard API
 * with fallback for older browsers
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} - Success status
 */
export async function copyToClipboard(text) {
  try {
    // Try modern Clipboard API first (Chrome, Firefox, Edge, Safari 13.1+)
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    // Fallback for older browsers
    return fallbackCopyToClipboard(text);
  } catch (error) {
    console.error('Clipboard copy failed:', error);
    return fallbackCopyToClipboard(text);
  }
}

/**
 * Fallback method for copying to clipboard
 * Uses deprecated execCommand but works in older browsers
 * @param {string} text - Text to copy
 * @returns {boolean} - Success status
 */
function fallbackCopyToClipboard(text) {
  try {
    // Create temporary textarea element
    const textarea = document.createElement('textarea');
    textarea.value = text;

    // Make it invisible and position it off-screen
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    textarea.style.top = '-10000px';
    textarea.style.left = '-10000px';

    // Prevent layout shift
    textarea.style.width = '2em';
    textarea.style.height = '2em';

    document.body.appendChild(textarea);

    // Select and copy
    textarea.select();
    textarea.setSelectionRange(0, text.length);

    // Try to copy
    const successful = document.execCommand('copy');

    // Clean up
    document.body.removeChild(textarea);

    return successful;
  } catch (error) {
    console.error('Fallback clipboard copy failed:', error);
    return false;
  }
}

/**
 * Copy machine code to clipboard
 * @param {string} code - Machine/system code
 * @param {function} onSuccess - Success callback
 * @param {function} onError - Error callback
 */
export async function copyCode(code, onSuccess, onError) {
  const success = await copyToClipboard(code);

  if (success) {
    if (onSuccess) onSuccess('Code copied to clipboard!');
  } else {
    if (onError) onError('Failed to copy code. Please try again.');
  }

  return success;
}

/**
 * Copy user query to clipboard
 * @param {string} query - User query text
 * @param {function} onSuccess - Success callback
 * @param {function} onError - Error callback
 */
export async function copyQuery(query, onSuccess, onError) {
  const success = await copyToClipboard(query);

  if (success) {
    if (onSuccess) onSuccess('Query copied to clipboard!');
  } else {
    if (onError) onError('Failed to copy query. Please try again.');
  }

  return success;
}

/**
 * Copy entire message (both query and response)
 * @param {string} userMessage - User message
 * @param {string} assistantMessage - Assistant response
 * @param {function} onSuccess - Success callback
 * @param {function} onError - Error callback
 */
export async function copyMessage(userMessage, assistantMessage, onSuccess, onError) {
  const fullMessage = `User: ${userMessage}\n\nAssistant: ${assistantMessage}`;
  const success = await copyToClipboard(fullMessage);

  if (success) {
    if (onSuccess) onSuccess('Message copied to clipboard!');
  } else {
    if (onError) onError('Failed to copy message. Please try again.');
  }

  return success;
}

/**
 * Copy JSON data to clipboard (pretty-printed)
 * @param {object} data - JSON data to copy
 * @param {function} onSuccess - Success callback
 * @param {function} onError - Error callback
 */
export async function copyJSON(data, onSuccess, onError) {
  try {
    const jsonString = JSON.stringify(data, null, 2);
    const success = await copyToClipboard(jsonString);

    if (success) {
      if (onSuccess) onSuccess('JSON copied to clipboard!');
    } else {
      if (onError) onError('Failed to copy JSON. Please try again.');
    }

    return success;
  } catch (error) {
    console.error('Failed to copy JSON:', error);
    if (onError) onError('Invalid JSON data.');
    return false;
  }
}

/**
 * Read text from clipboard
 * @returns {Promise<string|null>} - Clipboard text or null if failed
 */
export async function readFromClipboard() {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      const text = await navigator.clipboard.readText();
      return text;
    }

    console.warn('Clipboard read not supported');
    return null;
  } catch (error) {
    console.error('Failed to read from clipboard:', error);
    return null;
  }
}

/**
 * Check if clipboard API is available
 * @returns {boolean} - True if clipboard operations are supported
 */
export function isClipboardSupported() {
  return (navigator.clipboard && window.isSecureContext) || document.queryCommandSupported('copy');
}

const clipboardUtils = {
  copyToClipboard,
  copyCode,
  copyQuery,
  copyMessage,
  copyJSON,
  readFromClipboard,
  isClipboardSupported,
};

export default clipboardUtils;
