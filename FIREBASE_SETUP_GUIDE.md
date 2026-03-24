# Firebase Setup Guide for Sentinel-E

## Overview
Firebase provides authentication, real-time database, and session management for Sentinel-E. This guide walks through setting up your Firebase project and configuring credentials.

---

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click **"Add project"**
3. Enter project name: `sentinel-e` (or your preferred name)
4. Enable Google Analytics (optional)
5. Click **"Create project"**

---

## Step 2: Set Up Authentication

1. In Firebase Console, navigate to **Authentication** (left sidebar)
2. Click **"Get started"**
3. Enable **Email/Password** authentication:
   - Select "Email/Password" provider
   - Enable both "Email/Password" and "Email link (passwordless sign-in)"
   - Click **"Save"**
4. Optional: Enable **Google Sign-In**
   - Select "Google" provider
   - Add your support email
   - Click **"Save"**

---

## Step 3: Create Firestore Database

1. Navigate to **Firestore Database** (left sidebar)
2. Click **"Create database"**
3. Start in **Production mode** (or Development for testing)
4. Choose your preferred region (e.g., `us-east1`)
5. Click **"Create"**

### Firestore Security Rules

Replace default rules with:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users collection — owners can read/write their own docs
    match /users/{userId} {
      allow read, write: if request.auth.uid == userId;
      allow read: if request.auth.uid != null && 
                     (resource.data.role == 'admin' || 
                      resource.data.public == true);
    }
    
    // Sessions collection — users can read/write their own sessions
    match /sessions/{sessionId} {
      allow read, write: if request.auth.uid == resource.data.userId;
      allow list: if request.auth.uid != null;
    }
  }
}
```

---

## Step 4: Get Frontend Credentials (Web App Config)

1. In Firebase Console, go to **Project Settings** (gear icon, top-left)
2. Click **"Your apps"** tab
3. Under "Web apps", click the web app icon (</> )
4. If no web app exists, click **"Add app"** and select **"Web"**
5. Copy the Firebase configuration object:

```javascript
const firebaseConfig = {
  apiKey: "AIzaSy...",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123def456"
};
```

6. Add these to **`frontend/.env.local`**:

```
REACT_APP_FIREBASE_API_KEY=AIzaSy...
REACT_APP_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your-project-id
REACT_APP_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=123456789
REACT_APP_FIREBASE_APP_ID=1:123456789:web:abc123def456
```

---

## Step 5: Get Backend Credentials (Service Account)

1. In **Project Settings**, click **"Service Accounts"** tab
2. Click **"Generate New Private Key"**
3. A JSON file will download. Keep it **secure**!
4. Extract these values and add to **`backend/.env`**:

```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=key-id
FIREBASE_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n
FIREBASE_CLIENT_EMAIL=firebase-adminsdk@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://accounts.google.com/o/oauth2/token
FIREBASE_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
```

⚠️ **Never commit the private key to version control!** Use `.env` (gitignored) only.

---

## Step 6: Configure Firestore Collections

Create the following collection structures:

### Users Collection
```
/users/{uid}
├── uid: string
├── email: string
├── displayName: string
├── role: "admin" | "user"
├── createdAt: timestamp
├── isActive: boolean
└── preferences: {
    theme: string,
    language: string,
    phoneticPreference: string
}
```

### Sessions Collection
```
/sessions/{sessionId}
├── userId: string
├── title: string
├── messages: array
├── mode: string
├── createdAt: timestamp
├── updatedAt: timestamp
├── metadata: {
    messageCount: number,
    avgLatency: number,
    feedbackScore: number
}
```

---

## Step 7: Test the Setup

### Backend Test
```python
from firebase_admin import credentials, initialize_app, firestore

cred = credentials.Certificate('/path/to/serviceAccountKey.json')
initialize_app(cred)
db = firestore.client()

# Test write
db.collection('users').document('test-user').set({'email': 'test@example.com'})
print("Firebase backend connected!")
```

### Frontend Test
```javascript
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  // Your config from .env.local
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
console.log("Firebase frontend connected!");
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Permission denied" on Firestore | Check security rules match user auth status |
| "Invalid API Key" | Verify REACT_APP_FIREBASE_API_KEY matches Firebase console |
| "Failed to get document" | Ensure Firestore database exists and is initialized |
| Private key format error | Ensure `\n` characters are preserved in multiline keys |
| Auth not persisting | Check `browserLocalPersistence` is set in `firebaseAuth.js` |

---

## Security Best Practices

✅ **DO:**
- Store sensitive credentials in `.env` files (gitignored)
- Use Firebase Security Rules to restrict data access
- Enable Multi-Factor Authentication (MFA) for admin accounts
- Regularly rotate service account keys
- Use different projects for development/staging/production

❌ **DON'T:**
- Commit `.env` files to version control
- Expose API keys in client-side code
- Use production credentials in development
- Share private keys via email/chat
- Use overly permissive Firestore rules

---

## Environment Setup Scripts

Run these commands to verify your setup:

```bash
# Backend
cd backend
source .venv/bin/activate
python -c "from firebase_admin import credentials; print('✓ Firebase imports OK')"

# Frontend
cd frontend
npm install
echo "REACT_APP_FIREBASE_API_KEY is set to: $REACT_APP_FIREBASE_API_KEY" | grep -q "REACT_APP" && echo "✓ Frontend env vars loaded"
```

---

## Next Steps

1. ✅ Create Firebase project
2. ✅ Add credentials to `.env` files
3. ✅ Test connections from backend and frontend
4. ✅ Deploy authentication service (`firebaseAuth.js`)
5. ✅ Deploy session manager (`sessionManager.js`)
6. ✅ Enable admin dashboard access

For additional help, see:
- [Firebase Web Setup](https://firebase.google.com/docs/web/setup)
- [Firebase Authentication](https://firebase.google.com/docs/auth)
- [Firestore Documentation](https://firebase.google.com/docs/firestore)
