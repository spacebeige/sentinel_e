/**
 * ============================================================
 * Firebase Authentication Configuration & Setup
 * ============================================================
 *
 * Handles:
 * - User authentication (email/password)
 * - Role-based access control (Admin / User)
 * - Session management
 * - User profile management
 * - Token refresh
 */

import { initializeApp } from 'firebase/app';
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  setPersistence,
  browserLocalPersistence,
} from 'firebase/auth';
import {
  getFirestore,
  collection,
  doc,
  setDoc,
  getDoc,
  updateDoc,
  getDocs,
} from 'firebase/firestore';

// Firebase configuration
const FIREBASE_CONFIG = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY || 'AIzaSyDummyKeyForDevelopment',
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || 'sentinel-e.firebaseapp.com',
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || 'sentinel-e-project',
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || 'sentinel-e.appspot.com',
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || '123456789',
  appId: process.env.REACT_APP_FIREBASE_APP_ID || '1:123456789:web:abc123def456',
};

// Initialize Firebase
let app;
let auth;
let db;

try {
  app = initializeApp(FIREBASE_CONFIG);
  auth = getAuth(app);
  db = getFirestore(app);

  // Set persistence to LOCAL (survives browser restarts)
  setPersistence(auth, browserLocalPersistence).catch((error) => {
    console.error('Failed to set Firebase persistence:', error);
  });
} catch (error) {
  console.error('Failed to initialize Firebase:', error);
}

// User roles
export const USER_ROLES = {
  ADMIN: 'admin',
  USER: 'user',
};

/**
 * Create a new user (Admin only)
 * @param {string} email - User email
 * @param {string} password - User password
 * @param {string} role - User role (admin/user)
 * @param {string} displayName - User display name
 * @returns {Promise} - User creation result
 */
export async function createUser(email, password, role = USER_ROLES.USER, displayName = '') {
  try {
    // Create Firebase auth user
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // Create user profile in Firestore
    await setDoc(doc(db, 'users', user.uid), {
      uid: user.uid,
      email: email,
      displayName: displayName || email.split('@')[0],
      role: role,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isActive: true,
      preferences: {
        theme: 'dark',
        language: 'en',
        phoneticPreference: 'phonetic_roman',
      },
      metadata: {
        lastLogin: null,
        loginCount: 0,
        deviceInfo: {},
      },
    });

    return {
      success: true,
      user: {
        uid: user.uid,
        email: user.email,
        role: role,
      },
    };
  } catch (error) {
    console.error('Error creating user:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Sign in user
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise} - Sign in result
 */
export async function signInUser(email, password) {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // Fetch user profile from Firestore
    const userDoc = await getDoc(doc(db, 'users', user.uid));
    const userData = userDoc.data();

    // Update last login
    await updateDoc(doc(db, 'users', user.uid), {
      'metadata.lastLogin': new Date().toISOString(),
      'metadata.loginCount': (userData?.metadata?.loginCount || 0) + 1,
    });

    return {
      success: true,
      user: {
        uid: user.uid,
        email: user.email,
        ...userData,
      },
    };
  } catch (error) {
    console.error('Error signing in:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Sign out user
 * @returns {Promise} - Sign out result
 */
export async function signOutUser() {
  try {
    await signOut(auth);
    return { success: true };
  } catch (error) {
    console.error('Error signing out:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Get current authenticated user
 * @returns {Promise} - Current user data or null
 */
export async function getCurrentUser() {
  return new Promise((resolve) => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));
        const userData = userDoc.data();
        resolve({
          uid: firebaseUser.uid,
          email: firebaseUser.email,
          ...userData,
        });
      } else {
        resolve(null);
      }
      unsubscribe();
    });
  });
}

/**
 * Get user profile by ID
 * @param {string} uid - User ID
 * @returns {Promise} - User profile data
 */
export async function getUserProfile(uid) {
  try {
    const userDoc = await getDoc(doc(db, 'users', uid));
    return userDoc.data() || null;
  } catch (error) {
    console.error('Error fetching user profile:', error);
    return null;
  }
}

/**
 * Update user profile
 * @param {string} uid - User ID
 * @param {object} updateData - Data to update
 * @returns {Promise} - Update result
 */
export async function updateUserProfile(uid, updateData) {
  try {
    await updateDoc(doc(db, 'users', uid), {
      ...updateData,
      updatedAt: new Date().toISOString(),
    });
    return { success: true };
  } catch (error) {
    console.error('Error updating user profile:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Check if user is admin
 * @param {string} uid - User ID
 * @returns {Promise} - Boolean indicating admin status
 */
export async function isUserAdmin(uid) {
  try {
    const userDoc = await getDoc(doc(db, 'users', uid));
    const userData = userDoc.data();
    return userData?.role === USER_ROLES.ADMIN;
  } catch (error) {
    console.error('Error checking admin status:', error);
    return false;
  }
}

/**
 * Get all users (Admin only)
 * @returns {Promise} - Array of user profiles
 */
export async function getAllUsers() {
  try {
    const usersRef = collection(db, 'users');
    const querySnapshot = await getDocs(usersRef);
    const users = [];
    querySnapshot.forEach((doc) => {
      users.push(doc.data());
    });
    return users;
  } catch (error) {
    console.error('Error fetching all users:', error);
    return [];
  }
}

/**
 * Get user statistics (Admin dashboard)
 * @returns {Promise} - User stats
 */
export async function getUserStatistics() {
  try {
    const users = await getAllUsers();

    return {
      totalUsers: users.length,
      adminCount: users.filter((u) => u.role === USER_ROLES.ADMIN).length,
      regularUserCount: users.filter((u) => u.role === USER_ROLES.USER).length,
      activeUsers: users.filter((u) => u.isActive).length,
      lastUserCreated: users[users.length - 1]?.createdAt || null,
    };
  } catch (error) {
    console.error('Error fetching user statistics:', error);
    return null;
  }
}

const firebaseAuthService = {
  auth,
  db,
  createUser,
  signInUser,
  signOutUser,
  getCurrentUser,
  getUserProfile,
  updateUserProfile,
  isUserAdmin,
  getAllUsers,
  getUserStatistics,
  USER_ROLES,
};

export default firebaseAuthService;
