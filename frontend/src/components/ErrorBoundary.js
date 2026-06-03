import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      const isDev = process.env.NODE_ENV === 'development';
      
      return (
        <div style={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          textAlign: 'center',
          backgroundColor: '#0f172a',
          color: '#f8fafc',
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
        }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '16px',
            backgroundColor: '#ef444420',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '24px'
          }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          </div>

          <h1 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '12px', color: '#f8fafc' }}>
            Application Error
          </h1>
          
          <p style={{ fontSize: '15px', marginBottom: '32px', maxWidth: '480px', opacity: 0.7, lineHeight: '1.6' }}>
            {isDev 
              ? "A runtime error occurred during development. Check the debug info below." 
              : "We've encountered an unexpected issue. Please try reloading the application."}
          </p>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              onClick={() => window.location.reload()}
              style={{
                padding: '12px 24px',
                backgroundColor: '#38bdf8',
                color: '#0f172a',
                border: 'none',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              Reload App
            </button>
            <button 
              onClick={() => window.location.href = '/'}
              style={{
                padding: '12px 24px',
                backgroundColor: 'transparent',
                color: '#f8fafc',
                border: '1px solid #ffffff20',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Back to Home
            </button>
          </div>

          {isDev && this.state.error && (
            <div style={{
              marginTop: '40px',
              width: '100%',
              maxWidth: '800px',
              textAlign: 'left'
            }}>
              <p style={{ fontSize: '12px', fontWeight: '600', color: '#ef4444', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Debug Information (Development Only)
              </p>
              <div style={{
                padding: '20px',
                backgroundColor: '#1e293b',
                borderRadius: '12px',
                overflow: 'auto',
                maxHeight: '300px',
                border: '1px solid #ffffff10'
              }}>
                <pre style={{ margin: 0, fontSize: '13px', color: '#f1f5f9', lineHeight: '1.5' }}>
                  {this.state.error.stack || this.state.error.toString()}
                </pre>
              </div>
            </div>
          )}
        </div>
      );
    }

    return this.props.children; 
  }
}

export default ErrorBoundary;
