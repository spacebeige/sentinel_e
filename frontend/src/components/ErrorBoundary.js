/**
 * ErrorBoundary — Graceful Error Handling for Admin Dashboard
 * Catches errors and displays fallback UI
 */
import React from 'react';
import { AlertCircle, RefreshCw, Home } from 'lucide-react';

const FONT = "'Inter', -apple-system, sans-serif";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1,
    }));
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-[#f5f5f7]">
          <div className="max-w-2xl w-full mx-auto px-6">
            <div className="bg-white rounded-2xl p-12 border border-red-200 shadow-lg">
              {/* Error Icon */}
              <div className="flex justify-center mb-6">
                <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center">
                  <AlertCircle className="w-8 h-8 text-red-600" />
                </div>
              </div>

              {/* Error Title */}
              <h1
                className="text-3xl font-bold text-center text-[#1d1d1f] mb-2"
                style={{ fontFamily: FONT }}
              >
                Something Went Wrong
              </h1>
              <p className="text-center text-[#6e6e73] mb-6">
                The admin dashboard encountered an unexpected error.
              </p>

              {/* Error Details (Development) */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="mb-8 p-4 bg-red-50 rounded-lg border border-red-200">
                  <p className="text-xs font-mono text-red-700 break-words">
                    {this.state.error.toString()}
                  </p>
                  {this.state.errorInfo && (
                    <details className="mt-2 text-xs text-red-600">
                      <summary className="cursor-pointer font-semibold mb-2">
                        Stack Trace
                      </summary>
                      <pre className="overflow-auto whitespace-pre-wrap text-xs">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </details>
                  )}
                </div>
              )}

              {/* Error Count Warning */}
              {this.state.errorCount > 3 && (
                <div className="mb-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <p className="text-sm text-yellow-800 font-medium">
                    ⚠️ Multiple errors detected ({this.state.errorCount}). 
                    Consider refreshing the page or checking your connection.
                  </p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 justify-center">
                <button
                  onClick={this.handleReset}
                  className="flex items-center gap-2 px-6 py-3 rounded-lg bg-[#3b82f6] text-white font-medium hover:opacity-90 transition-all"
                  style={{ fontFamily: FONT }}
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </button>
                <a
                  href="/"
                  className="flex items-center gap-2 px-6 py-3 rounded-lg bg-[#f5f5f7] text-[#1d1d1f] font-medium hover:bg-[#e8e8ed] transition-all"
                  style={{ fontFamily: FONT }}
                >
                  <Home className="w-4 h-4" />
                  Go Home
                </a>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
