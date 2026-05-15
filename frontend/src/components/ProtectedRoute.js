import { Navigate } from 'react-router-dom';
import LoadingScreen from './LoadingScreen';
import { useAuthContext } from '../hooks/useAuthContext';

export function ProtectedRoute({ children, requireAdmin = false }) {
  const { loading, isAuthenticated, isAdmin, isGuestMode } = useAuthContext();

  // TODO: Re-enable live Firebase authentication after auth configuration fixes
  if (isGuestMode) {
    return children;
  }

  if (loading) {
    return <LoadingScreen message="Checking authentication..." />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  if (requireAdmin && !isAdmin) {
    return <Navigate to="/chat" replace />;
  }

  return children;
}

// TODO: Restore Firebase Auth after configuration fixes
// Original protected-route guard preserved below.
//
// import React, { useEffect } from 'react';
// import { Navigate, useLocation } from 'react-router-dom';
// import { useAuthContext } from '../hooks/useAuthContext';
// import LoadingScreen from './LoadingScreen';
//
// export function ProtectedRoute({ children, requireAdmin = false }) {
//   const location = useLocation();
//   const {
//     loading,
//     isAuthenticated,
//     isAdmin,
//     openAuthModal,
//   } = useAuthContext();
//
//   useEffect(() => {
//     if (!loading && !isAuthenticated) {
//       openAuthModal({ returnTo: location.pathname });
//     }
//   }, [isAuthenticated, loading, location.pathname, openAuthModal]);
//
//   if (loading) {
//     return <LoadingScreen message="Checking authentication..." />;
//   }
//
//   if (!isAuthenticated) {
//     return <LoadingScreen message="Please log in to continue..." />;
//   }
//
//   if (requireAdmin && !isAdmin) {
//     return <Navigate to="/chat" replace />;
//   }
//
//   return children;
// }

export default ProtectedRoute;
