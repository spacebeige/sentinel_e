import React from 'react';
import { AlertCircle, FileQuestion, RefreshCcw } from 'lucide-react';

/**
 * DataFallback — A visible fallback UI for missing or invalid data.
 * Replaces silent 'return null' to provide better user feedback.
 */
export const DataFallback = ({ 
  message = "Data unavailable", 
  type = "missing", // "missing" | "error" | "invalid"
  onRetry = null,
  className = "" 
}) => {
  const isError = type === 'error';
  const isInvalid = type === 'invalid';

  return (
    <div className={`flex flex-col items-center justify-center p-6 rounded-2xl border border-dashed text-center ${
      isError ? 'bg-red-50/50 border-red-200 text-red-600' : 
      isInvalid ? 'bg-orange-50/50 border-orange-200 text-orange-600' :
      'bg-gray-50/50 border-gray-200 text-gray-500'
    } ${className}`}>
      {isError ? <AlertCircle className="w-8 h-8 mb-2 opacity-80" /> : 
       isInvalid ? <AlertCircle className="w-8 h-8 mb-2 opacity-80" /> :
       <FileQuestion className="w-8 h-8 mb-2 opacity-80" />}
      
      <p className="text-sm font-medium mb-3">
        {message}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-current text-xs font-semibold hover:bg-gray-50 transition-colors"
        >
          <RefreshCcw className="w-3 h-3" />
          Retry
        </button>
      )}
    </div>
  );
};

export default DataFallback;
