import React, { useEffect, useState } from 'react';
import { Loader2, X } from 'lucide-react';
import { authenticateWithProvider } from '../services/firebaseAuth';
import '../styles/LoginModal.css';

const TABS = [
  { id: 'login', label: 'Login', title: 'Welcome back', subtitle: 'Continue with your preferred provider to access chat and models.' },
  { id: 'signup', label: 'Sign Up', title: 'Create your account', subtitle: 'Start with Google or GitHub and your session will stay synced across refreshes.' },
];

function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="auth-provider-icon">
      <path fill="#EA4335" d="M12 10.2v3.9h5.4c-.2 1.3-1.7 3.9-5.4 3.9-3.3 0-6-2.7-6-6s2.7-6 6-6c1.9 0 3.2.8 3.9 1.5l2.7-2.6C17 3.3 14.7 2.2 12 2.2 6.6 2.2 2.2 6.6 2.2 12S6.6 21.8 12 21.8c6.9 0 9.4-4.8 9.4-7.3 0-.5-.1-.9-.1-1.3H12Z" />
    </svg>
  );
}

function GithubMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="auth-provider-icon auth-provider-icon--github">
      <path fill="currentColor" d="M12 2.2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.8c-2.9.6-3.5-1.2-3.5-1.2-.5-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 .1 1.6 1 1.6 1 .9 1.6 2.5 1.1 3.1.9.1-.7.4-1.1.7-1.4-2.3-.3-4.8-1.2-4.8-5.3 0-1.2.4-2.2 1-3-.1-.3-.4-1.3.1-2.8 0 0 .9-.3 3 .9a10.2 10.2 0 0 1 5.4 0c2.1-1.2 3-.9 3-.9.6 1.5.2 2.5.1 2.8.6.8 1 1.8 1 3 0 4.1-2.5 5-4.9 5.3.4.3.8 1 .8 2v2.9c0 .3.2.6.7.5A10 10 0 0 0 12 2.2Z" />
    </svg>
  );
}

export default function LoginModal({
  isOpen,
  initialError = '',
  onClose,
  onLoginSuccess,
  returnTo = '/chat',
}) {
  const [mode, setMode] = useState('login');
  const [loadingProvider, setLoadingProvider] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) {
      setLoadingProvider(null);
      setError('');
      setMode('login');
      return;
    }

    setError(initialError || '');
  }, [initialError, isOpen]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const onKeyDown = (event) => {
      if (event.key === 'Escape' && !loadingProvider) {
        onClose?.();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, loadingProvider, onClose]);

  if (!isOpen) return null;

  const activeTab = TABS.find((tab) => tab.id === mode) || TABS[0];

  const handleProviderAuth = async (provider) => {
    setError('');
    setLoadingProvider(provider);

    try {
      const user = await authenticateWithProvider(provider, { returnTo });
      onLoginSuccess?.(user);
      onClose?.();
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : 'Unable to start authentication.');
    } finally {
      setLoadingProvider(null);
    }
  };

  return (
    <div
      className="login-modal-overlay"
      onClick={() => {
        if (!loadingProvider) onClose?.();
      }}
    >
      <div
        className="login-modal-container"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="login-modal-backdrop" />

        <div className="login-modal-content">
          <div className="login-modal-header">
            <div>
              <div className="login-modal-kicker">Secure Access</div>
              <h2>{activeTab.title}</h2>
              <p>{activeTab.subtitle}</p>
            </div>
            <button
              className="login-modal-close"
              onClick={() => onClose?.()}
              disabled={Boolean(loadingProvider)}
              aria-label="Close authentication modal"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="login-modal-tabs" role="tablist" aria-label="Authentication mode">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={mode === tab.id}
                className={`login-modal-tab ${mode === tab.id ? 'login-modal-tab--active' : ''}`}
                onClick={() => {
                  setMode(tab.id);
                  setError('');
                }}
                disabled={Boolean(loadingProvider)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="login-modal-body">
            <button
              type="button"
              className="auth-provider-button"
              onClick={() => handleProviderAuth('google')}
              disabled={Boolean(loadingProvider)}
            >
              <span className="auth-provider-mark">
                {loadingProvider === 'google' ? <Loader2 className="w-4 h-4 animate-spin" /> : <GoogleMark />}
              </span>
              <span>Continue with Google</span>
            </button>

            <button
              type="button"
              className="auth-provider-button"
              onClick={() => handleProviderAuth('github')}
              disabled={Boolean(loadingProvider)}
            >
              <span className="auth-provider-mark auth-provider-mark--github">
                {loadingProvider === 'github' ? <Loader2 className="w-4 h-4 animate-spin" /> : <GithubMark />}
              </span>
              <span>Continue with GitHub</span>
            </button>

            {error && <div className="error-message">{error}</div>}

            <div className="login-modal-footer">
              <span>Sessions use secure httpOnly cookies.</span>
              <span>No internal backend errors are exposed in the UI.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
