# Frontend Duplication Audit

## Scope: `figma_ui/src`

### `createClient(`
| File | Count | Status |
|------|-------|--------|
| `legacy/lib/supabase.js:11` | 1 | ✅ SOLE instance |

**No duplicates.** One Supabase client, correctly located in the legacy layer.

---

### `axios.create(`
| File | Count | Status |
|------|-------|--------|
| `legacy/services/api.js:28` | 1 | ✅ SOLE instance |

**No duplicates.** One Axios instance, correctly located in the legacy layer.

---

### `sendMCOQuery(`
| File | Count | Status |
|------|-------|--------|
| `legacy/services/api.js` | Defined | ✅ Authoritative |
| `app/components/ChatPage.tsx:47` | Imported via `@services/api` | ✅ Correct |

**No duplicates.** Alias `@services/api` → `legacy/services/api.js` confirmed in `vite.config.ts`.

---

### `getChatMessages(`
| File | Count | Status |
|------|-------|--------|
| `legacy/services/api.js` | Defined | ✅ Authoritative |
| `app/components/ChatPage.tsx` | Imported via `@services/api` | ✅ Correct |

**No duplicates.**

---

### `getChatHistory(`
Not present in codebase. History loading uses `getHistory()` from `legacy/services/api.js`.

---

### `useSessionPersistence(`
Not present in codebase. Session state is managed by `useStore.js` Zustand persist middleware.

---

### `useSupabaseAuth(`
| File | Usage |
|------|-------|
| `legacy/hooks/useSupabaseAuth.js` | DEFINED (authoritative) |
| `app/providers/AuthProvider.tsx` | Imported via `@hooks/useSupabaseAuth` ✅ |
| `app/components/ChatPage.tsx` | Imported via `@hooks/useSupabaseAuth` ✅ |
| `app/components/LoginPage.tsx` | Imported via `@hooks/useSupabaseAuth` ✅ |
| `app/components/Navbar.tsx` | Imported via `@hooks/useSupabaseAuth` ✅ |
| (and other page components) | All via `@hooks/useSupabaseAuth` ✅ |

**No duplicates.** All consumers use the canonical alias.

---

### `useStore(`
| File | Usage |
|------|-------|
| `legacy/stores/useStore.js` | DEFINED (authoritative) |
| `app/components/ChatPage.tsx` | Imported via `@stores/useStore` ✅ |

**No duplicates.**

---

## Summary
| Layer | createClient | axios.create | sendMCOQuery | useStore | useSupabaseAuth |
|-------|:---:|:---:|:---:|:---:|:---:|
| `legacy/` | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 |
| `app/` | 0 | 0 | 0 | 0 | 0 |

**RESULT: ZERO DUPLICATES.** All consumers correctly import from the single legacy layer via Vite path aliases.
