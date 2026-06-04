# Auth, Profile, and Settings Integration

## 1. Login Mapping
- **UI Source**: `figma_ui/src/app/components/LoginPage.tsx`
- **Data Source**: `@hooks/useSupabaseAuth`
- **Mechanism**: Use existing `signInWithEmail` and `signInWithGoogle` methods from the hook. The login form's `onSubmit` logic will be replaced to pipe through `useSupabaseAuth.signInWithEmail`.
- **Removed Features**: Guest Login, Anonymous Session Generation, and corresponding "Continue as Guest" buttons are stripped per requirements.

## 2. Signup Mapping
- **UI Source**: `figma_ui/src/app/components/SignupPage.tsx`
- **Data Source**: `@hooks/useSupabaseAuth`
- **Mechanism**: The UI triggers `signUpWithEmail`. The existing `ensure_user_exists` flow natively handles creation inside the backend. No backend modifications required.

## 3. Profile Mapping
- **UI Source**: `figma_ui/src/app/components/ProfilePage.tsx`
- **Data Source**: `api.get('/api/user')` & `useSupabaseAuth().user`
- **Mechanism**: We will replace direct Supabase fetches in `ProfilePage` with `api.get('/api/user')`. 
- **Mapping**:
  - `user.email` -> Email field
  - `user.user_metadata?.name` -> Display Name field
  - `stats.chat_count` -> Conversations Stat
  - `stats.message_count` -> Messages Stat

## 4. Settings Mapping
- **UI Source**: `figma_ui/src/app/components/SettingsPage.tsx`
- **Data Source**: `api.get('/api/user/settings')` & `api.put('/api/user/settings')`
- **Mechanism**: The local settings states will be synced directly with the backend `/api/user/settings` response payload.

## 5. Role Mapping
- **Data Source**: `useAdminRole` (legacy frontend hook)
- **Mechanism**: If `useAdminRole()` resolves to true, the Admin menu item is displayed in the UI sidebar/nav. If false, it remains hidden. This prevents exposing Admin features to standard users without relying on Figma's local role checks.

## 6. Unsupported Fields & Disabled Controls
- **Telemetry Opt-In**: Not in legacy schema. (Control will be visually hidden or disabled with a "Coming Soon" tooltip).
- **Analytics Opt-In**: Not in legacy schema. (Will be hidden).
- **Feedback Opt-In**: Not in legacy schema. (Will be hidden).
- **Avatar Upload**: Backend uses Gravatar or OAuth avatars, no `/api/user/avatar` endpoint exists. (Upload button will be disabled/mocked as read-only).
- **Data Controls (Export/Delete)**: Not currently exported as REST endpoints in `api.js`. (Will trigger `alert("Not implemented")` or be disabled).
