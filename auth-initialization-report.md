# Authentication Initialization Fix Report

## Overview
The `figma_ui` application was logging warnings indicating `"Supabase auth is not configured"` and `"Session restore safety timeout hit"`, which prevented authentication from initializing and kicked valid users to guest mode or caused `/login` loops.

## Root Cause
The legacy architecture logic responsible for spinning up the Supabase Client natively relies on `process.env.REACT_APP_SUPABASE_URL`. When Vercel compiles the Vite project via `npm run build`, `process.env.REACT_APP` strings are silently omitted unless explicitly exposed. This resulted in undefined URLs and keys being injected into the auth client. 

## Files Modified
1. `figma_ui/src/legacy/lib/supabase.js`

## Resolution Details
- Directly migrated `supabaseUrl` and `supabaseAnonKey` getters to consume `import.meta.env.VITE_SUPABASE_URL` and `import.meta.env.VITE_SUPABASE_ANON_KEY`.
- Updated initialization error warnings to instruct developers to configure `VITE_` prefixed environment properties rather than `REACT_APP`.
- Removed all legacy React environment footprint. 

The underlying session and Supabase client structures remain identical; only the configuration pipeline was patched.
