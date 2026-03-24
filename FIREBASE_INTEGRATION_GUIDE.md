# Phase 2: Firebase Authentication & Session Management Integration Guide

## 📋 Overview

This guide covers the integration of:
- ✅ Firebase Authentication (Admin/User roles)
- ✅ Session Management (Per-user session tracking)
- ✅ Admin Dashboard (Monitoring & analytics)
- ✅ Clipboard Utilities (Fixed copy-to-clipboard)
- ✅ Phonetic Language Conversion

## 🚀 Step 1: Firebase Configuration

### 1.1 Install Firebase SDK

```bash
cd frontend
npm install firebase
```

### 1.2 Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add Project"
3. Name it "sentinel-e"
4. Enable Google Analytics (optional)
5. Create the project

### 1.3 Get Firebase Credentials

1. Go to Project Settings (⚙️ icon)
2. Under "Your apps", click "Web" (</> icon)
3. Register app name as "sentinel-e"
4. Copy the Firebase config object

### 1.4 Set Environment Variables

Create `.env.local` in the frontend directory:

```env
REACT_APP_FIREBASE_API_KEY=your_api_key
REACT_APP_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your_project_id
REACT_APP_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
REACT_APP_FIREBASE_APP_ID=your_app_id
```

### 1.5 Enable Firestore Database

1. In Firebase Console, go to **Build → Firestore Database**
2. Click **Create Database**
3. Select **Start in test mode** (for development)
4. Choose your region
5. Click **Enable**

### 1.6 Set Firestore Security Rules

Replace the default rules with:

```firestore
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users collection - each user can read/write their own document
    match /users/{userId} {
      allow read, write: if request.auth.uid == userId;
      allow read: if request.auth.token.role == 'admin';
    }
    
    // Sessions collection - users can read/write their own sessions
    match /sessions/{sessionId} {
      allow read, write: if request.auth.uid == resource.data.userId;
      allow read: if request.auth.token.role == 'admin';
    }
  }
}
```

### 1.7 Enable Authentication Methods

1. In Firebase Console, go to **Build → Authentication**
2. Click **Get Started**
3. Enable **Email/Password** provider
4. Click **Enable**

## 📦 Step 2: Component Integration

### 2.1 Update App.js

Replace your main App.js with:

```javascript
import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuthContext } from './hooks/useAuthContext';
import LoginModal from './components/LoginModal';
import AdminDashboard from './pages/AdminDashboard';
import ChatInterface from './components/ChatInterface'; // Your existing chat component
import './App.css';

// Main content component (requires auth)
const AppContent = () => {
  const { user, loading, isAdmin, signOut } = useAuthContext();
  const [showLoginModal, setShowLoginModal] = useState(false);

  if (loading) {
    return <div className="loading-screen">Loading...</div>;
  }

  // Not authenticated - show login
  if (!user) {
    return (
      <>
        <LoginModal
          isOpen={true}
          onClose={() => {}}
          onLoginSuccess={() => {
            // User will be set by context on successful login
          }}
        />
      </>
    );
  }

  // Authenticated - show dashboard based on role
  return (
    <div className="app-container authenticated">
      <header className="app-header">
        <div className="header-left">
          <h1>🔐 Sentinel-E</h1>
          <span className="user-info">
            {user.displayName} ({isAdmin ? 'Admin' : 'User'})
          </span>
        </div>
        <div className="header-right">
          <button
            className="auth-button logout"
            onClick={() => {
              signOut();
              setShowLoginModal(true);
            }}
          >
            Sign Out
          </button>
        </div>
      </header>

      <main className="app-main">
        {isAdmin ? <AdminDashboard /> : <ChatInterface />}
      </main>

      <LoginModal
        isOpen={showLoginModal}
        onClose={() => {
          // Only close if user is authenticated
          if (user) setShowLoginModal(false);
        }}
        onLoginSuccess={() => {
          setShowLoginModal(false);
        }}
      />
    </div>
  );
};

// Wrap the app with AuthProvider
const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;
```

### 2.2 Update App.css

Add these styles:

```css
.app-container.authenticated {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 32px;
  background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.header-left,
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.app-header h1 {
  margin: 0;
  font-size: 24px;
  background: linear-gradient(135deg, #0ef4ff 0%, #00d4ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.user-info {
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
  background-color: rgba(0, 244, 255, 0.1);
  padding: 4px 12px;
  border-radius: 12px;
}

.auth-button {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.auth-button.logout {
  background-color: rgba(239, 68, 68, 0.2);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.auth-button.logout:hover {
  background-color: rgba(239, 68, 68, 0.3);
}

.app-main {
  flex: 1;
  overflow: hidden;
}

.loading-screen {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
  color: #0ef4ff;
  font-size: 18px;
}
```

## 🔌 Step 3: Session Integration

### 3.1 Update ChatInterface Component

In your existing ChatInterface component, integrate session management:

```javascript
import { useAuthContext } from '../hooks/useAuthContext';
import sessionManager from '../services/sessionManager';

const ChatInterface = () => {
  const { user } = useAuthContext();
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);

  // Initialize session on component mount
  useEffect(() => {
    if (user?.uid) {
      initializeSession();
      loadUserSessions();
    }
  }, [user?.uid]);

  const initializeSession = async () => {
    const result = await sessionManager.createSession(
      user.uid,
      'query', // default mode
      '',
      user.preferences || {}
    );
    if (result.success) {
      setCurrentSessionId(result.sessionId);
    }
  };

  const loadUserSessions = async () => {
    const userSessions = await sessionManager.getUserSessions(user.uid);
    setSessions(userSessions);
  };

  // When user sends a message
  const handleSendMessage = async (message) => {
    // Add message to current session with latency tracking
    const startTime = Date.now();
    
    // ... send message and get response ...
    
    const latency = Date.now() - startTime;
    await sessionManager.addMessageToSession(
      user.uid,
      currentSessionId,
      'user',
      message,
      latency
    );

    // Add assistant response
    await sessionManager.addMessageToSession(
      user.uid,
      currentSessionId,
      'assistant',
      assistantResponse,
      0
    );
  };

  // Load previous session
  const handleLoadSession = async (sessionId) => {
    const session = await sessionManager.getSession(sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      // Display session messages in chat
    }
  };

  return (
    <div className="chat-interface">
      {/* Sidebar with sessions */}
      <aside className="chat-sidebar">
        <h3>Sessions</h3>
        <button onClick={initializeSession} className="new-session-button">
          + New Session
        </button>
        <div className="sessions-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={`session-item ${
                session.id === currentSessionId ? 'active' : ''
              }`}
              onClick={() => handleLoadSession(session.id)}
            >
              {session.title}
            </button>
          ))}
        </div>
      </aside>

      {/* Main chat area */}
      <main className="chat-messages">
        {/* Your existing chat UI */}
      </main>
    </div>
  );
};
```

## 📋 Step 4: Clipboard Integration

### 4.1 Add Copy Buttons to Messages

```javascript
import { copyMessage, copyCode, copyQuery } from '../utils/clipboardUtils';

const ChatMessage = ({ role, content, userQuery }) => {
  const [copySuccess, setCopySuccess] = useState('');

  const handleCopyQuery = async () => {
    const success = await copyQuery(userQuery);
    if (success) {
      setCopySuccess('Query copied!');
      setTimeout(() => setCopySuccess(''), 2000);
    }
  };

  const handleCopyResponse = async () => {
    const success = await copyMessage(userQuery, content);
    if (success) {
      setCopySuccess('Message copied!');
      setTimeout(() => setCopySuccess(''), 2000);
    }
  };

  return (
    <div className={`message ${role}`}>
      <div className="message-content">{content}</div>
      <div className="message-actions">
        {role === 'assistant' && (
          <>
            <button onClick={handleCopyQuery} title="Copy query">
              📋 Query
            </button>
            <button onClick={handleCopyResponse} title="Copy response">
              📋 Message
            </button>
          </>
        )}
      </div>
      {copySuccess && <span className="copy-success">{copySuccess}</span>}
    </div>
  );
};
```

## 🎯 Step 5: Phonetic Display Preference

### 5.1 Add Preference Button to InputArea

```javascript
import { phoneticConverter } from '../engines/phoneticConverter';
import { updateUserProfile } from '../services/firebaseAuth';

const InputArea = ({ userId, userLanguage }) => {
  const [preference, setPreference] = useState('english');

  const handlePreferenceChange = async (newPreference) => {
    setPreference(newPreference);
    await updateUserProfile(userId, {
      'preferences.phoneticPreference': newPreference,
    });
  };

  const formatResponse = (text, responses) => {
    if (preference === 'native_script') {
      return phoneticConverter.fromPhonetic(text, userLanguage);
    } else if (preference === 'phonetic_roman') {
      return phoneticConverter.toPhonetic(text, userLanguage);
    }
    return text; // English
  };

  return (
    <div className="input-area">
      <div className="preference-selector">
        <button
          className={preference === 'english' ? 'active' : ''}
          onClick={() => handlePreferenceChange('english')}
        >
          English
        </button>
        <button
          className={preference === 'phonetic_roman' ? 'active' : ''}
          onClick={() => handlePreferenceChange('phonetic_roman')}
        >
          Phonetic
        </button>
        <button
          className={preference === 'native_script' ? 'active' : ''}
          onClick={() => handlePreferenceChange('native_script')}
        >
          Native
        </button>
      </div>
      {/* Input form */}
    </div>
  );
};
```

## 🔑 Step 6: Create Initial Admin Account

After Firebase is set up, create the first admin account:

```bash
# Run this script in your backend to create the admin user
python -c "
from frontend.src.services.firebaseAuth import createUser, USER_ROLES
import asyncio

async def create_admin():
    result = await createUser(
        'admin@sentinel-e.local',
        'SecureAdminPassword123!',
        USER_ROLES.ADMIN,
        'Administrator'
    )
    print(result)

asyncio.run(create_admin())
"
```

Or use the Firebase Console:
1. Go to Authentication → Users
2. Click "Add User"
3. Enter: `admin@sentinel-e.local`
4. Set password
5. In Firestore, manually set `role: 'admin'` for this user

## ✅ Verification Checklist

- [ ] Firebase project created and configured
- [ ] Environment variables set in `.env.local`
- [ ] Firestore database created with security rules
- [ ] Firebase Authentication enabled (Email/Password)
- [ ] App.js updated with AuthProvider and AppContent
- [ ] LoginModal component working
- [ ] Session management integrated into ChatInterface
- [ ] Clipboard utilities integrated into messages
- [ ] Admin account created
- [ ] Admin Dashboard displays correctly for admin users
- [ ] Phonetic preference button working
- [ ] Copy-to-clipboard working across browsers

## 🐛 Troubleshooting

### "Firebase app not initialized"
- Check that all environment variables are set correctly in `.env.local`
- Restart the development server after adding env vars
- Run `npm start` again

### "Firestore permission denied"
- Check Firestore security rules (Step 1.6)
- Ensure user is authenticated before accessing Firestore
- Test in development mode with lenient rules first

### "Copy to clipboard not working"
- Check browser console for errors
- Ensure page is served over HTTPS (required for `navigator.clipboard`)
- Try the fallback method: `document.execCommand('copy')`

### "Admin Dashboard not showing"
- Verify user role is set to 'admin' in Firestore
- Check that `isAdmin` is true in useAuthContext
- Test with a known admin account

## 📚 API Reference

### firebaseAuth.js

```javascript
// Authentication
await signInUser(email, password)
await createUser(email, password, role, displayName)
await signOutUser()
await getCurrentUser() // Returns current user or null
await getUserProfile(uid)
await updateUserProfile(uid, updateData)
await isUserAdmin(uid)
await getAllUsers() // Admin only
await getUserStatistics() // Admin only
```

### sessionManager.js

```javascript
// Sessions
await createSession(userId, mode, subMode, preferences)
await addMessageToSession(userId, sessionId, role, content, latency)
await getUserSessions(userId, maxSessions)
await getSession(sessionId)
await updateSession(sessionId, updates)
await closeSession(sessionId)
await deleteSession(sessionId)
await addSessionFeedback(sessionId, feedback, score)
await getSessionStatistics(userId) // Optional userId for filtering
```

### clipboardUtils.js

```javascript
// Clipboard operations
await copyToClipboard(text)
await copyCode(code, onSuccess, onError)
await copyQuery(query, onSuccess, onError)
await copyMessage(userMessage, assistantMessage, onSuccess, onError)
await copyJSON(data, onSuccess, onError)
await readFromClipboard()
isClipboardSupported() // Returns boolean
```

## 🎉 Next Steps

After successful integration:

1. **Test user flow**: Create test accounts in both roles
2. **Monitor analytics**: Use Admin Dashboard to track usage
3. **Collect feedback**: Add rating system after each session
4. **Optimize latency**: Use metrics to identify slow queries
5. **Implement notifications**: Add email notifications for low scores
6. **Add more languages**: Expand phonetic converter mappings
7. **Custom reports**: Build more detailed analytics
8. **User roles**: Add more granular permissions (operator, supervisor, etc.)

## 📞 Support

For issues or questions:
1. Check browser console for errors
2. Review Firebase documentation: https://firebase.google.com/docs
3. Check Firestore rules: https://firebase.google.com/docs/firestore/security/start
4. Test in development mode with logging enabled

---

**Status**: ✅ Complete - Phase 2 Firebase & Session Management
**Last Updated**: 2025
**Maintained By**: Sentinel-E Team
