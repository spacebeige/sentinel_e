// TODO: Remove deprecated Firebase auth flow after Supabase stabilization
import { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import LoadingScreen from './LoadingScreen';
import { useAuthContext } from '../hooks/useAuthContext';
import useStore from '../stores/useStore';

export function ProtectedRoute({ children, requireAdmin = false }) {
  const location = useLocation();
  // authResolved: auth check finished (not still loading)
  // sessionReady: authenticated + auth resolved — safe to render protected content
  const { loading, authResolved, isAuthenticated, isAdmin, openAuthModal } = useAuthContext();
  // hasHydrated: localStorage rehydration complete — prevents blank flash before store loads
  const hasHydrated = useStore(state => state.hasHydrated);

  useEffect(() => {
    if (authResolved && !isAuthenticated) {
      openAuthModal({ returnTo: location.pathname });
    }
  }, [isAuthenticated, authResolved, location.pathname, openAuthModal]);

  // Wait for both auth resolution AND store hydration to avoid blank/flash UI
  if (loading || !hasHydrated) {
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
