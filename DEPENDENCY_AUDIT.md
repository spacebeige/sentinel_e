# Dependency Audit & Deprecation Notes

## Issue
Vercel build showing deprecation warnings from transitive npm dependencies.

## Root Cause
These deprecated packages are pulled in by **react-scripts 5.0.1**, which is the latest (but aging) version of Create React App. CRA has built-in support for PWA (via Workbox), testing (via JSdom), and other utilities that depend on these packages.

## Deprecated Packages Explanation

### 1. **stable@0.1.8** (Array.sort polyfill)
- **Status**: Deprecated - Modern JS guarantees stable sort
- **Why it exists**: Pulled by workbox-core or testing dependencies
- **Risk**: None - it's just a polyfill that doesn't affect modern browsers
- **Action**: Suppressed via overrides

### 2. **q@1.5.1** (Promise library)
- **Status**: Deprecated - Use native Promises
- **Why it exists**: Legacy callback-based code may still use it
- **Risk**: None - works fine, just not maintained
- **Action**: Removed from overrides to use transitive versions

### 3. **workbox packages** (6.6.0)
- `workbox-cacheable-response`
- `workbox-google-analytics`
- **Status**: Maintenance mode but stable
- **Why it exists**: CRA uses Workbox for PWA support (service workers, caching)
- **Risk**: Low - Workbox is still widely used and supported
- **Action**: Keep as-is; no breaking changes

### 4. **whatwg-encoding@1.0.5**
- **Status**: Deprecated in favor of native methods
- **Why it exists**: JSdom dependency for testing
- **Risk**: None - provides TextEncoder/TextDecoder polyfills
- **Action**: Override to prevent if possible, else accept

### 5. **abab@2.0.6** (Base64 codec)
- **Status**: Deprecated - use native `atob()` and `btoa()`
- **Why it exists**: JSdom polyfill for older Node environments
- **Risk**: None - native methods available
- **Action**: Accept; only used in test environment

### 6. **w3c-hr-time@1.0.2** (High Resolution Timer)
- **Status**: Deprecated - use native `performance.now()`
- **Why it exists**: JSdom polyfill for testing
- **Risk**: None - testing-only dependency
- **Action**: Accept; only used in test environment

### 7. **domexception@2.0.1**
- **Status**: Deprecated - use native `DOMException`
- **Why it exists**: JSdom polyfill
- **Risk**: None - testing-only dependency
- **Action**: Accept; only used in test environment

## Solutions Applied

### ✅ Changes Made
1. **Removed `q` from overrides** - Allow npm to use transitive version
2. **Upgraded direct dependencies**:
   - `axios`: ^1.6.0 → ^1.7.7
   - `lucide-react`: ^0.300.0 → ^0.400.0
3. **Added Node.js version requirement**: engines field requires Node >=20.0.0
4. **Added engine locking**: `.nvmrc` = 20
5. **Created `.npmrc`** for consistent npm configuration

### ⚠️ Warnings to Accept
These warnings are **not actionable problems**:
- Workbox packages (maintained but old)
- JSdom/testing polyfills (only in dev, not shipped)
- These don't affect production or security

## Migration Path (Future)

### Option 1: Vite + React (Recommended)
- Replace Create React App with Vite
- Eliminates Workbox/JSdom dependencies
- 10x faster builds
- Requires: Migrate away from react-scripts

### Option 2: Upgrade to React 19 + Next.js
- Modern alternative to CRA
- Better performance
- More maintained dependencies
- Requires: Significant refactoring

### Option 3: Create React App 6.0 (experimental)
- When released officially
- Addresses many deprecations
- Requires: Waiting for release

## Current Recommendation
**Build as-is.** These warnings are primarily informational and don't impact:
- Production code
- Security
- Performance
- The Vercel build

Track for future migration to Vite when time permits.

## Testing Deprecations

Run locally to verify:
```bash
cd frontend
npm ci
npm run build
npm run test  # Will show test-related warnings
```

## Notes
- Vercel build is **green** ✅
- No actual errors, only deprecation notices
- Can be suppressed if needed via .npmrc
- Recommend addressing in Q3 2026 with Vite migration
