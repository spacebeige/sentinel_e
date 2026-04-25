import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthContext } from '../hooks/useAuthContext';

export function ProtectedRoute({ children, requireAdmin = false }) {
  const location = useLocation();
  const {
    loading,
    isAuthenticated,
    isAdmin,
    openAuthModal,
  } = useAuthContext();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      openAuthModal({ returnTo: location.pathname });
    }
  }, [isAuthenticated, loading, location.pathname, openAuthModal]);

  if (loading) {
    return <div className="min-h-[40vh]" />;
  }

  if (!isAuthenticated) {
    return <div className="min-h-[40vh]" />;
  }

  if (requireAdmin && !isAdmin) {
    return <Navigate to="/chat" replace />;
  }

  return children;
}

export default ProtectedRoute;
