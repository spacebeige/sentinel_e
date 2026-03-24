# Phase 2: Firebase & Session Management - Quick Reference

## 🚀 Quick Start (5 Steps)

### Step 1: Install Firebase
```bash
cd frontend
npm install firebase
```

### Step 2: Create .env.local
```env
REACT_APP_FIREBASE_API_KEY=your_api_key
REACT_APP_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your_project_id
REACT_APP_FIREBASE_STORAGE_BUCKET=your_bucket.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
REACT_APP_FIREBASE_APP_ID=your_app_id
```

### Step 3: Wrap App with AuthProvider
```javascript
// App.js
import { AuthProvider } from './hooks/useAuthContext';

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
```

### Step 4: Use In Components
```javascript
import { useAuthContext } from '../hooks/useAuthContext';

const MyComponent = () => {
  const { user, isAdmin, signOut } = useAuthContext();
  // Use user data...
};
```

### Step 5: Deploy Firestore Rules
In Firebase Console > Firestore > Rules:
```firestore
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth.uid == userId;
      allow read: if request.auth.token.role == 'admin';
    }
    match /sessions/{sessionId} {
      allow read, write: if request.auth.uid == resource.data.userId;
      allow read: if request.auth.token.role == 'admin';
    }
  }
}
```

---

## 📋 File Reference

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `firebaseAuth.js` | Authentication service | 318 | ✅ |
| `sessionManager.js` | Session & message tracking | 371 | ✅ |
| `LoginModal.js` | Auth UI component | 170 | ✅ |
| `AdminDashboard.js` | Admin monitoring dashboard | 220 | ✅ |
| `useAuthContext.js` | Global auth state hook | 70 | ✅ |
| `clipboardUtils.js` | Cross-browser clipboard | 191 | ✅ |
| `LoginModal.css` | Auth modal styling | 350+ | ✅ |
| `AdminDashboard.css` | Dashboard styling | 480+ | ✅ |

---

## 🔑 Core APIs

### Authentication
```javascript
import { 
  signInUser, 
  createUser, 
  getCurrentUser, 
  isUserAdmin 
} from '../services/firebaseAuth';

// Login
const result = await signInUser('user@email.com', 'password');
if (result.success) console.log(result.user);

// Check if admin
const admin = await isUserAdmin(userId);
```

### Sessions
```javascript
import sessionManager from '../services/sessionManager';

// Create session
const { sessionId } = await sessionManager.createSession(
  userId, 
  'query', // mode
  '', // subMode
  userPreferences
);

// Add message with latency
await sessionManager.addMessageToSession(
  userId,
  sessionId,
  'user',
  'Hello!',
  125 // latency in ms
);

// Get all sessions
const sessions = await sessionManager.getUserSessions(userId);
```

### Clipboard
```javascript
import { copyToClipboard, copyMessage } from '../utils/clipboardUtils';

// Copy any text
await copyToClipboard('text to copy');

// Copy conversation
await copyMessage(
  'user message',
  'assistant response',
  () => console.log('Success'),
  () => console.log('Failed')
);
```

---

## 🎯 Common Patterns

### Check if User is Admin
```javascript
const { isAdmin } = useAuthContext();

if (isAdmin) {
  return <AdminDashboard />;
} else {
  return <ChatInterface />;
}
```

### Track Session Messages
```javascript
const { user } = useAuthContext();
const [sessionId, setSessionId] = useState(null);

useEffect(() => {
  const initSession = async () => {
    const result = await sessionManager.createSession(user.uid);
    setSessionId(result.sessionId);
  };
  initSession();
}, [user.uid]);

const handleSendMessage = async (msg) => {
  const start = Date.now();
  const response = await getResponse(msg);
  const latency = Date.now() - start;
  
  await sessionManager.addMessageToSession(
    user.uid, sessionId, 'user', msg
  );
  await sessionManager.addMessageToSession(
    user.uid, sessionId, 'assistant', response, latency
  );
};
```

### Add Copy Button
```javascript
import { copyMessage } from '../utils/clipboardUtils';

<button onClick={async () => {
  const success = await copyMessage(userMsg, assistantMsg);
  if (success) showToast('Copied!');
}}>
  📋 Copy
</button>
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "Firebase not initialized" | Check `.env.local` variables |
| "Permission denied" | Update Firestore rules (Step 5) |
| "Copy not working" | Use HTTPS; fallback handled automatically |
| "No admin dashboard" | Verify user.role === 'admin' in Firestore |
| "Sessions not loading" | Check userId matches auth.uid |

---

## 🔍 Debug Commands

```javascript
// In browser console:

// Check auth context
import { useAuthContext } from './hooks/useAuthContext';
const { user, isAdmin } = useAuthContext();
console.log('User:', user);
console.log('Is Admin:', isAdmin);

// Check Firestore connection
import { db } from './services/firebaseAuth';
import { getDoc, doc } from 'firebase/firestore';
const userDoc = await getDoc(doc(db, 'users', userId));
console.log('User in Firestore:', userDoc.data());

// Test session creation
import sessionManager from './services/sessionManager';
const result = await sessionManager.createSession(userId, 'test');
console.log('Session created:', result);

// Test clipboard
import { copyToClipboard } from './utils/clipboardUtils';
await copyToClipboard('test');
```

---

## 📊 Admin Dashboard Metrics

**Overview Tab**
- Total Users (breakdown by role)
- Total Sessions (with message count)
- Average Latency (ms)
- Feedback Score (1-5)
- Sessions with Feedback (percentage)

**Users Tab**
- Email, Display Name, Role, Status
- Account Creation Date
- Last Login Date

**Analytics Tab**
- Performance: Latency, sessions, messages, avg/session
- Feedback: Avg score, rating percentage, negative alerts

---

## 🔐 Firestore Structure

```
users/
  {userId}/
    uid: string
    email: string
    displayName: string
    role: 'admin' | 'user'
    isActive: boolean
    createdAt: timestamp
    preferences: {
      theme: 'dark' | 'light'
      language: 'en' | 'hi' | ...
      phoneticPreference: 'english' | 'phonetic_roman' | 'native_script'
    }
    metadata: {
      lastLogin: timestamp
      loginCount: number
    }

sessions/
  {sessionId}/
    userId: string
    title: string
    messages: [{role, content, timestamp, latency}]
    mode: string
    subMode: string
    detectedLanguage: string
    phoneticPreference: string
    createdAt: timestamp
    updatedAt: timestamp
    metadata: {
      messageCount: number
      avgLatency: number
      userFeedback: string
      feedbackScore: 1-5
    }
```

---

## 🎬 User Flows

### Login Flow
```
User visits app
  ↓
Not authenticated? → Show LoginModal
  ↓
User enters email/password
  ↓
Firebase authenticates
  ↓
User profile loaded from Firestore
  ↓
AuthContext updated
  ↓
Redirect to Chat (user) or Admin Dashboard (admin)
```

### Message Tracking Flow
```
User types message → Click Send
  ↓
Record timestamp (latency start)
  ↓
Send to AI backend
  ↓
Receive response → Record latency
  ↓
sessionManager.addMessageToSession(
  - userId, sessionId, 'user', message
  - userId, sessionId, 'assistant', response, latency
)
  ↓
Store in Firestore
  ↓
Display copy button
  ↓
User can provide star rating
```

### Admin Monitoring Flow
```
Admin logs in
  ↓
isAdmin check → Show AdminDashboard
  ↓
Load metrics from Firestore
  ↓
Display real-time stats
  ↓
View all users & sessions
  ↓
Monitor latency & feedback trends
  ↓
Auto-refresh every 30s
```

---

## 📞 Support Links

- Firebase Docs: https://firebase.google.com/docs
- Firestore: https://firebase.google.com/docs/firestore
- Authentication: https://firebase.google.com/docs/auth
- Security Rules: https://firebase.google.com/docs/firestore/security/

---

## ✅ Pre-Deployment Checklist

- [ ] Firebase SDK installed (`npm install firebase`)
- [ ] `.env.local` file created with all variables
- [ ] Firebase project created at console.firebase.google.com
- [ ] Firestore database enabled
- [ ] Authentication methods enabled
- [ ] Security rules deployed
- [ ] Components imported in App.js
- [ ] AuthProvider wraps entire app
- [ ] Admin account created
- [ ] All features tested locally
- [ ] No console errors
- [ ] Ready for production

---

**Quick Setup Time**: ~15 minutes  
**Components**: 6 services/components  
**Total Code**: ~2,250 lines  
**Documentation**: 4 guides + this reference  
**Status**: ✅ Production Ready

