# Legacy Brain Enforcement Report

## Objective
Verify that the "Old Brain + New Face" architecture is strictly enforced across the entire repository.

## Constraint Verification

### 1. Supabase Singleton
`grep -R "createClient(" figma_ui/src`
- Result: ONLY `figma_ui/src/legacy/lib/supabase.js`
- Status: **PASSED**

### 2. Axios Singleton
`grep -R "axios.create" figma_ui/src`
- Result: ONLY `figma_ui/src/legacy/services/api.js`
- Status: **PASSED**

### 3. MCO Execution
`grep -R "sendMCOQuery(" figma_ui/src`
- Result: Defined in `legacy/services/api.js` and called purely by UI components like `ChatPage.tsx`.
- Status: **PASSED**

### 4. Store Authority
All UI components pull states explicitly via `@stores/useStore` (which aliases to `legacy/stores/useStore.js`).
- Status: **PASSED**

### 5. Auth Authority
All auth hooks (login, logout, admin checks) operate explicitly via `@hooks/useSupabaseAuth` (which aliases to `legacy/hooks/useSupabaseAuth.js`).
- Status: **PASSED**

## Conclusion
The legacy architecture is fully isolated and acts as the strict sole authority for the application. No duplicate layers exist.
