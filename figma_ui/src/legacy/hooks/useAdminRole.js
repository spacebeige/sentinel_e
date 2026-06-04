import { useAuthContext } from './useAuthContext';

export function useAdminRole() {
  const { isAdmin, loading } = useAuthContext();
  return { isAdmin, loading };
}

export default useAdminRole;
