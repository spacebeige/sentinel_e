/**
 * validation.js — Response Shape Verification
 * 
 * Ensures API responses match expected schemas before they propagate
 * through the application. Logs mismatches for observability.
 */

/**
 * Validates an object against a schema of required keys.
 * @param {Object} data   — The data to validate
 * @param {string[]} keys — List of required keys
 * @param {string} source — Name of the API/Function for logging
 * @returns {boolean}     — True if valid
 */
export function validateResponseShape(data, keys, source = 'Unknown API') {
  if (!data || typeof data !== 'object') {
    console.warn(`[Validation] ${source}: Response is not an object.`, data);
    return false;
  }

  const missing = keys.filter(key => !(key in data));
  if (missing.length > 0) {
    console.error(`[Validation Error] ${source}: Missing required keys: ${missing.join(', ')}`, {
      received: Object.keys(data),
      fullData: data
    });
    return false;
  }

  return true;
}

/**
 * Standard schemas for common API responses
 */
export const Schemas = {
  CHAT_RUN: ['chat_id', 'formatted_output'],
  MESSAGES_LIST: (data) => Array.isArray(data),
  HISTORY: ['chats', 'total'],
  SESSION_STATE: ['chat_id', 'chat_name'],
};
