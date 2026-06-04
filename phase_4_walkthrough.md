# Sentinel-E EVO — Figma UI Integration: Auth, Profile, & Settings

This walkthrough summarizes the successful execution of **Phase 4** of the Figma UI Conformance Migration, linking the frontend UI components (`LoginPage`, `SignupPage`, `ProfilePage`, `SettingsPage`, and Navigation) to the authoritative legacy backend services.

## Overview of Changes

We strictly enforced the architectural mandate to **adapt the new UI to the old system**, relying exclusively on `api.js` and `useSupabaseAuth()`. No new stores or legacy mock implementations were added.

### 1. Authentication Integration (`LoginPage.tsx`, `SignupPage.tsx`)
- **Removed Guest/Anonymous Logic:** Eradicated "Continue as Guest" and "Guest Mode" options.
- **Removed Local State Mocks:** Deleted deprecated implementations like `setLoginMode`, `useAuthContext()` (mock version), and `submitAdminRequest()`.
- **Bound to Authoritative Hook:** Fully integrated `useSupabaseAuth()` to manage `session`, `signInWithEmail`, `signUpWithEmail`, and `signInWithGoogle`. 
- **Fixed Signature Bug:** Updated the `signUpWithEmail` payload signature to match the backend expectations `({ email, password, options: { data: { name } } })`.
- **Role Enforcement:** Navbar and routing logic now check `session` and resolve `role` via `user.user_metadata.role`, correctly routing unauthenticated users to `/login`.

### 2. Profile Page Consolidation (`ProfilePage.tsx`)
- **Bound to Authoritative Service:** Replaced direct Supabase client calls (`import { supabase }`) with `api.get('/api/user')` from `@services/api`.
- **Disabled Mock Data Modifications:** Display operations for Avatar uploads and Profile Name editing have been correctly disabled/mocked as "Read-only" with user alerts, adhering to the rule of not creating new API endpoints where none exist.

### 3. Settings UI Conformance (`SettingsPage.tsx`)
- **Data Model Alignment:** Bound `fetchSettings` to `api.get('/api/user/settings')`, properly mapping properties such as `defaultMode`, `defaultModel`, `responseStyle`, and `debateDepth` directly to the `SETTINGS_SCHEMA`.
- **Removal of Unsupported Fields:** Completely removed the "Privacy & Security" section (telemetry, analytics, and feedback toggles) from the UI as they were not backed by the API.
- **Secured Data Controls:** Replaced mock functions for "Export Data" and "Delete Account" with alert placeholders to signify the current lack of backend support, preventing destructive mock side effects.
- **State Updates:** Patched `savePreferences` to use `api.put('/api/user/settings')` for all validated changes.

### 4. Navigation & Layout Conformance (`Navbar.tsx`)
- **Guest Mode Extermination:** Validated the removal of all guest mode UI strings and conditional references in Navigation.
- **Admin Role Synchronization:** Fixed the destructured `useAuthContext()` call to properly omit the non-existent `isAdmin` flag and rely directly on the generic `role` property (`role === 'admin' || role === 'owner'`).

## Validation
- Successfully executed `npm run build` on `figma_ui` leveraging Vite without any TypeScript errors, assuring API typings and dependencies are correctly mapped.
- Verified hot-reloading stability of `npm run dev`.

The user interface is now firmly coupled with the frozen core Sentinel-E services architecture.
