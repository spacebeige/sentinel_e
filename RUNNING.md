# Sentinel-E Cinematic OS — Running the System

This guide outlines the steps to run and manage the Sentinel-E Cinematic Operating System workspace (including the Vite Cinematic Frontend and the Python Backend).

---

## 1. Cinematic Frontend (`figma_ui`)
The cinematic frontend is built with React, Vite, and tailwind/motion elements, and is located in the `figma_ui` directory.

### Commands:
```bash
# Navigate to the frontend directory
cd figma_ui

# Install dependencies (if not already installed)
npm install

# Start the local development server (Vite)
npm run dev

# Build the production bundle
npm run build
```

* By default, Vite runs on port **5173** (or falls back to **5174/5175** if ports are in use). 
* Check your terminal output for the active URL, e.g., `http://localhost:5175/`.

---

## 2. Python Backend (`backend`)
The backend is a FastAPI application running with Uvicorn, located in the `backend` directory.

### Commands:
```bash
# Activate the virtual environment
source .venv/bin/activate

# Start the FastAPI backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* The backend API server will run at `https://sentinel-e-evo.onrender.com`.

---

## 3. Environment Variables (.env)
Ensure your `.env` files are configured under their respective directories:
* **Root/Backend**: `.env` (contains API keys, service roles, etc.)
* **Frontend**: `figma_ui/.env.development` or `.env`

---

## 4. Verification & Testing
To ensure the cinematic UI theme, routes, and hooks are fully functional without any crash triggers:
* Run the dev server (`npm run dev`) and navigate through:
  * `/` (Home/Atmospheric Landing)
  * `/chat` (Cinematic Chat & Orchestration)
  * `/pricing` (Resource Packages)
  * `/engines` (Model Hub)
* Verify that the light/dark mode switch changes themes smoothly and consistently using the centralized theme provider.
