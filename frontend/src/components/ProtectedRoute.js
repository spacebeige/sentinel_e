/**
 * ProtectedRoute — Admin Role Verification Component
 * Redirects non-admin users to /chat
 */
import React from 'react';
import { Navigate } from 'react-router-dom';
import jwt_decode from 'jwt-decode';

export function ProtectedRoute({ children }) {
  const token = localStorage.getItem('access_token');

  // No token or invalid token → redirect to chat
  if (!token) {
    return <Navigate to="/chat" replace />;
  }

  try {
    const decoded = jwt_decode(token);
    const role = decoded.role || 'user';

    // Not admin → redirect to chat
    if (role !== 'admin') {
      return <Navigate to="/chat" replace />;
    }

    // Admin → render component
    return children;
  } catch (error) {
    console.error('Token decode error:', error);
    return <Navigate to="/chat" replace />;
  }
}

export default ProtectedRoute;
