/**
 * useAdminRole — Check if current user has admin role
 * Decodes JWT token to get role claim
 */
import { useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';

export function useAdminRole() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setIsAdmin(false);
        setLoading(false);
        return;
      }

      try {
        const decoded = jwtDecode(token);
        // Role is populated by backend on auth
        const role = decoded.role || 'user';
        setIsAdmin(role === 'admin');
      } catch (e) {
        console.error('Failed to decode token:', e);
        setIsAdmin(false);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  return { isAdmin, loading };
}

export default useAdminRole;
