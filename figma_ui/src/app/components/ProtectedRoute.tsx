import React from 'react';
import { Navigate, useLocation } from 'react-router';
import { useAuthContext } from '../providers/AuthProvider';

export const ProtectedRoute: React.FC<{ children: React.ReactNode; requireAdmin?: boolean; requireOwner?: boolean }> = ({ children, requireAdmin = false, requireOwner = false }) => {
  const { loading, isAuthenticated, isAdmin, user } = useAuthContext();
  const location = useLocation();
  const renderCount = React.useRef(0);

  renderCount.current += 1;
  console.log(`[PROTECTED_ROUTE] Render ${renderCount.current} | loading=${loading} | auth=${isAuthenticated} | user=${user?.id} | path=${location.pathname}`);

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#09090b]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-[#8b5cf6]"></div>
          <span className="text-sm font-medium text-zinc-400">Authenticating Sentinel-E...</span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    // Save the location they were trying to go to
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Owner concept not supported in AuthProvider currently, mapping to admin temporarily
  if (requireOwner && !isAdmin) {
    return <Navigate to="/chat" replace />;
  }

  if (requireAdmin && !isAdmin) {
    return <Navigate to="/chat" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
