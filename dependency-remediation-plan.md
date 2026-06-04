# Dependency Remediation Plan

This report answers the core questions surrounding the Vercel build failure on `Rollup failed to resolve import "zustand"` and provides the minimum required mitigation strategy to resolve these architectural pipeline issues without modifying the legacy code itself.

## Build Pipeline Analysis

### Q1: What `package.json` is Vercel actually building?
Vercel is building **`figma_ui/package.json`**.
**Evidence:** The Vercel Rollup build failed exclusively inside the Vite compilation step targeting the `figma_ui` framework configuration (`vite build`). Root dependencies (`/package.json`) are completely bypassed because Vercel treats `figma_ui` as the "Root Directory" for the deployment pipeline, ensuring that `npm install` runs strictly inside `figma_ui`.

### Q3: Why did the build fail on `zustand` specifically?
Even though `zustand: "latest"` exists in `figma_ui/package.json`, the Rollup bundler failed to resolve it.
- **Import Origin:** `frontend/src/stores/useStore.js`
- **Alias Chain:** `import { useStore } from "@stores/useStore"` -> points strictly to `../frontend/src/stores/useStore.js`
- **Root Cause (Node Resolution Scope):** Node module resolution climbs up the directory tree strictly relative to the importing file. Because `useStore.js` sits in `/frontend/src/stores/`, Rollup checks `/frontend/src/node_modules`, `/frontend/node_modules`, and `/node_modules`. It **never checks** `/figma_ui/node_modules` because `figma_ui` is a sibling, not an ancestor. Since Vercel only executes `npm install` inside `figma_ui`, the ancestor paths lack `zustand`, leading to the resolution crash.

## Minimum Required Additions & Fixes
To unblock Vercel without altering the legacy files or indiscriminately copying `/frontend/package.json`, the following two-part remediation plan must be executed.

### Part 1: Dependency Additions (`figma_ui/package.json`)
The `figma_ui` package must declare the implicit dependencies required by the legacy architecture.

| Package | Reason Required | Version Source |
|---------|-----------------|----------------|
| `axios` | Critical dependency for `@services/api` | `^1.7.7` (Match legacy frontend config) |
| `react-router-dom` | Critical dependency for `@hooks/useAuthContext` | `^7.13.0` (Match current figma_ui router version) |

*Note: `zustand` and `react` are already present in `figma_ui/package.json`.*

### Part 2: Vite Resolution Alias Fix
Merely adding the dependencies to `package.json` will not solve the Node module resolution bug out-of-the-box (as proven by `zustand`). Vite must be forcefully instructed to resolve these specific packages using the local `figma_ui/node_modules` path, bypassing the default upstream folder crawl.

This requires adding resolution overrides directly to `figma_ui/vite.config.ts`:
```ts
resolve: {
  alias: {
    // Component Aliases
    "@services": path.resolve(__dirname, "../frontend/src/services"),
    "@stores": path.resolve(__dirname, "../frontend/src/stores"),
    "@hooks": path.resolve(__dirname, "../frontend/src/hooks"),
    "@utils": path.resolve(__dirname, "../frontend/src/utils"),
    
    // Dependency Resolution Overrides
    "zustand": path.resolve(__dirname, "./node_modules/zustand"),
    "axios": path.resolve(__dirname, "./node_modules/axios"),
    "react-router-dom": path.resolve(__dirname, "./node_modules/react-router-dom"),
  }
}
```

*This strategy allows legacy files to resolve dependencies successfully against the `figma_ui` execution context without polluting or rewriting the legacy configurations.*
