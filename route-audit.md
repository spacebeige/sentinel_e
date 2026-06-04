# Vercel SPA Routing Audit

## Objective
To investigate and remediate the 404 behavior triggered during direct navigation to routes like `/login`, `/chat`, or `/auth/callback` in the production environment.

## Root Cause
Single Page Applications (SPAs) built with React Router and Vite serve all content from a single `index.html` file. Local Vite development servers natively intercept direct requests and forward them to the `index.html` host to handle client-side routing. However, when hosted on Vercel natively via static deployment, direct route hits expect a physical `.html` file at the specified path (e.g., `login.html`). Without configuration, this throws a standard 404 HTTP Error.

## Validation
Verified the following routes are defined correctly within the Figma UI React Router setup (in `main.tsx` and related top-level route files):
- `/`
- `/chat`
- `/pricing`
- `/models`
- `/admin`

*Note: Auth routes such as `/login`, `/signup`, and `/auth/callback` are handled via modals and global authentication hooks or specifically directed components.*

## Resolution
A dedicated `vercel.json` configuration file was created inside the `figma_ui` root directory:
```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```
This forces Vercel's Edge Network to rewrite all non-file route parameters silently to the root SPA, restoring functionality for all Sentinel-E pages.
