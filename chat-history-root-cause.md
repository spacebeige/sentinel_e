# Chat History Root Cause

## The Problem
The primary issue causing the "ReferenceError: isWebSearchEnabled is not defined" crash during message sends in `ChatPage.tsx` was traced to an undeclared React state variable. 

When the Figma UI was ported, `isWebSearchEnabled` was passed into `sendMCOQuery(..., { force_retrieval: isWebSearchEnabled })` but the `useState` definition for it was missing or accidentally stripped. 

## The Fix
Added the missing state declaration at the top of the component:
```tsx
const [isWebSearchEnabled, setIsWebSearchEnabled] = useState(false);
```
This guarantees that `force_retrieval` falls back to `false` cleanly instead of throwing a catastrophic `ReferenceError` during the render and execution cycle.
