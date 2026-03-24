"""
============================================================
Firebase Admin SDK Service
============================================================
Initializes and manages Firebase Admin SDK for:
- User authentication verification
- Firestore session & user management
- JWT token validation
"""

import os
import json
import logging
from typing import Optional, Dict, Any

try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    FIREBASE_ADMIN_AVAILABLE = False

logger = logging.getLogger("FirebaseService")


class FirebaseService:
    """Firebase Admin SDK wrapper for backend operations."""

    def __init__(self):
        self.app = None
        self.db = None
        self.enabled = False
        self._initialize()

    def _initialize(self):
        """Initialize Firebase Admin SDK from environment variables."""
        if not FIREBASE_ADMIN_AVAILABLE:
            logger.warning("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
            self.enabled = False
            return

        try:
            # Try to build credentials from env variables
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            private_key_id = os.getenv("FIREBASE_PRIVATE_KEY_ID")
            private_key = os.getenv("FIREBASE_PRIVATE_KEY")
            client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
            client_id = os.getenv("FIREBASE_CLIENT_ID")

            if not all([project_id, private_key_id, private_key, client_email, client_id]):
                logger.warning("Firebase credentials incomplete in environment variables")
                self.enabled = False
                return

            # Build service account config
            service_account_info = {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": private_key_id,
                "private_key": private_key,
                "client_email": client_email,
                "client_id": client_id,
                "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv(
                    "FIREBASE_AUTH_PROVIDER_X509_CERT_URL",
                    "https://www.googleapis.com/oauth2/v1/certs"
                ),
            }

            # Initialize Firebase
            cred = credentials.Certificate(service_account_info)
            self.app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()

            logger.info(f"Firebase initialized for project: {project_id}")
            self.enabled = True

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.enabled = False

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a Firebase JWT token.

        Args:
            token: Firebase ID token

        Returns:
            Decoded token with user claims, or None if invalid
        """
        if not self.enabled:
            logger.debug("Firebase not enabled, skipping token verification")
            return None

        try:
            decoded_token = auth.verify_id_token(token)
            return decoded_token
        except auth.InvalidIdTokenError:
            logger.warning(f"Invalid token format")
            return None
        except auth.ExpiredIdTokenError:
            logger.warning("Token expired")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user profile from Firestore."""
        if not self.enabled or not self.db:
            logger.debug("Firebase not enabled, cannot fetch user profile")
            return None

        try:
            doc = self.db.collection("users").document(user_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch session from Firestore."""
        if not self.enabled or not self.db:
            logger.debug("Firebase not enabled, cannot fetch session")
            return None

        try:
            doc = self.db.collection("sessions").document(session_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Error fetching session: {e}")
            return None

    def create_session(self, session_id: str, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Create a new session in Firestore."""
        if not self.enabled or not self.db:
            logger.debug("Firebase not enabled, cannot create session")
            return False

        try:
            session_data["userId"] = user_id
            session_data["createdAt"] = firestore.SERVER_TIMESTAMP
            session_data["updatedAt"] = firestore.SERVER_TIMESTAMP

            self.db.collection("sessions").document(session_id).set(session_data)
            logger.debug(f"Session created: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session in Firestore."""
        if not self.enabled or not self.db:
            logger.debug("Firebase not enabled, cannot update session")
            return False

        try:
            updates["updatedAt"] = firestore.SERVER_TIMESTAMP
            self.db.collection("sessions").document(session_id).update(updates)
            logger.debug(f"Session updated: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False

    def get_user_sessions(self, user_id: str, limit: int = 50) -> list:
        """Fetch all sessions for a user."""
        if not self.enabled or not self.db:
            logger.debug("Firebase not enabled, cannot fetch sessions")
            return []

        try:
            docs = (
                self.db.collection("sessions")
                .where("userId", "==", user_id)
                .order_by("createdAt", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            return [doc.to_dict() for doc in docs]

        except Exception as e:
            logger.error(f"Error fetching user sessions: {e}")
            return []

    def health_check(self) -> bool:
        """Check if Firebase is accessible."""
        if not self.enabled:
            return False

        try:
            # Simple check: try to read a non-existent doc (should fail gracefully)
            self.db.collection("_health").document("_check").get()
            logger.debug("Firebase health check passed")
            return True
        except Exception as e:
            logger.error(f"Firebase health check failed: {e}")
            return False


# Global instance
_firebase_service: Optional[FirebaseService] = None


def get_firebase_service() -> FirebaseService:
    """Get or initialize the global Firebase service instance."""
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service


def firebase_is_enabled() -> bool:
    """Check if Firebase is enabled and initialized."""
    return get_firebase_service().enabled
