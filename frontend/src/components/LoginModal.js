/**
 * ============================================================
 * Login Modal Component
 * ============================================================
 *
 * Handles:
 * - User login/signup flows
 * - Form validation
 * - Error handling
 * - Role selection for new accounts
 * - Toggle between login and signup modes
 */

import React, { useState } from 'react';
import {
  signInUser,
  createUser,
  USER_ROLES,
} from '../services/firebaseAuth';
import '../styles/LoginModal.css';

const LoginModal = ({ isOpen, onClose, onLoginSuccess }) => {
  const [mode, setMode] = useState('login'); // 'login' or 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [selectedRole, setSelectedRole] = useState(USER_ROLES.USER);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Form validation
  const validateForm = () => {
    if (!email.trim()) {
      setError('Email is required');
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address');
      return false;
    }

    if (!password || password.length < 6) {
      setError('Password must be at least 6 characters');
      return false;
    }

    if (mode === 'signup' && !displayName.trim()) {
      setError('Display name is required');
      return false;
    }

    return true;
  };

  // Handle login
  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateForm()) return;

    setLoading(true);
    const result = await signInUser(email, password);

    if (result.success) {
      onLoginSuccess(result.user);
      resetForm();
      onClose();
    } else {
      setError(result.error || 'Failed to sign in');
    }

    setLoading(false);
  };

  // Handle signup
  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateForm()) return;

    setLoading(true);
    const result = await createUser(email, password, selectedRole, displayName);

    if (result.success) {
      // Auto-login after signup
      const loginResult = await signInUser(email, password);
      if (loginResult.success) {
        onLoginSuccess(loginResult.user);
        resetForm();
        onClose();
      } else {
        setError('Account created but auto-login failed. Please login manually.');
      }
    } else {
      setError(result.error || 'Failed to create account');
    }

    setLoading(false);
  };

  const resetForm = () => {
    setEmail('');
    setPassword('');
    setDisplayName('');
    setSelectedRole(USER_ROLES.USER);
    setError('');
    setShowPassword(false);
  };

  const toggleMode = () => {
    resetForm();
    setMode(mode === 'login' ? 'signup' : 'login');
  };

  if (!isOpen) return null;

  return (
    <div className="login-modal-overlay">
      <div className="login-modal-container">
        {/* Header */}
        <div className="login-modal-header">
          <h2>Sentinel-E Authentication</h2>
          <button className="login-modal-close" onClick={() => {
            resetForm();
            onClose();
          }}>
            ✕
          </button>
        </div>

        {/* Form */}
        <form onSubmit={mode === 'login' ? handleLogin : handleSignup} className="login-form">
          {/* Email */}
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              disabled={loading}
            />
          </div>

          {/* Display Name (Signup only) */}
          {mode === 'signup' && (
            <div className="form-group">
              <label htmlFor="displayName">Full Name</label>
              <input
                type="text"
                id="displayName"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="John Doe"
                disabled={loading}
              />
            </div>
          )}

          {/* Password */}
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="password-input-wrapper">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                disabled={loading}
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                disabled={loading}
              >
                {showPassword ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          {/* Role Selection (Signup only) */}
          {mode === 'signup' && (
            <div className="form-group">
              <label>Account Type</label>
              <div className="role-selector">
                <label className="role-option">
                  <input
                    type="radio"
                    value={USER_ROLES.USER}
                    checked={selectedRole === USER_ROLES.USER}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    disabled={loading}
                  />
                  <span>Regular User</span>
                  <small>Standard access to chat interface</small>
                </label>
                <label className="role-option">
                  <input
                    type="radio"
                    value={USER_ROLES.ADMIN}
                    checked={selectedRole === USER_ROLES.ADMIN}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    disabled={loading}
                  />
                  <span>Admin</span>
                  <small>Full access including monitoring dashboard</small>
                </label>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && <div className="error-message">{error}</div>}

          {/* Submit Button */}
          <button
            type="submit"
            className="submit-button"
            disabled={loading}
          >
            {loading ? 'Processing...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        {/* Toggle Mode */}
        <div className="login-modal-footer">
          <p>
            {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
            <button
              type="button"
              className="toggle-mode-button"
              onClick={toggleMode}
              disabled={loading}
            >
              {mode === 'login' ? 'Sign Up' : 'Sign In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginModal;
