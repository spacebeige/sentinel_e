import { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import LoadingScreen from './LoadingScreen';
import { useAuthContext } from '../hooks/useAuthContext';
import useStore from '../stores/useStore';

/**
 * ProtectedRoute — Supabase Auth Gate
 *
 * Dual guard pattern:
 *   - loading: auth check still in-flight (Supabase session restore)
 *   - hasHydrated: Zustand localStorage rehydration not yet complete
 *
 * Both must be resolved before rendering protected content or redirecting
 * to prevent blank flashes and race conditions on page reload.
 *
 * requireAdmin: additionally requires the user to have role='admin'.
 * Only oomkaragarkhed0710@gmail.com is granted runtime admin status.
 */
export function ProtectedRoute({ children, requireAdmin = false }) {
  const location = useLocation();
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

export default ProtectedRoute;
