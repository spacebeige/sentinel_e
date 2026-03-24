"""
Session Ownership Validation Middleware
Fixes: Session hijacking by ensuring per-request ownership verification
"""

from fastapi import Request, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from backend.database.models import User, Session as SessionModel
from backend.gateway.auth import decode_token, get_current_user
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def verify_session_ownership(
    request: Request,
    user_id: str,
    session_id: str,
    db: DBSession,
    cache_kernel: Optional[object] = None
) -> bool:
    """
    Verify that a session belongs to the requesting user.
    Called on EVERY session access, not just restore.
    
    Fixes: Session hijacking gap
    
    Args:
        request: FastAPI request
        user_id: ID of user making request
        session_id: ID of session being accessed
        db: Database session
        cache_kernel: Cached kernel object (if any)
    
    Returns:
        True if valid, raises HTTPException if not
    
    Raises:
        HTTPException(403) if session doesn't belong to user
    """
    
    # Check database ownership
    db_session = db.query(SessionModel).filter(
        SessionModel.id == session_id,
        SessionModel._owner_user_id == user_id
    ).first()
    
    if not db_session:
        logger.warning(
            f"⚠️ SESSION HIJACK ATTEMPT: User {user_id} tried to access session {session_id}",
            extra={"user_id": user_id, "session_id": session_id, "remote_addr": request.client.host}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: session does not belong to this user"
        )
    
    # CRITICAL: Also verify cached kernel ownership (not just DB)
    if cache_kernel and hasattr(cache_kernel, '_owner_user_id'):
        if cache_kernel._owner_user_id != user_id:
            logger.critical(
                f"🔴 KERNEL CACHE MISMATCH: Cached kernel {cache_kernel._owner_user_id} "
                f"doesn't match requesting user {user_id}!",
                extra={"session_id": session_id, "cached_owner": cache_kernel._owner_user_id}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session validation failed: kernel ownership mismatch"
            )
    
    logger.debug(f"✓ Session {session_id} ownership verified for user {user_id}")
    return True


async def validate_kernel_before_use(
    kernel: object,
    user_id: str,
    session_id: str
) -> None:
    """
    Validate kernel object structure before using it.
    Prevents silent failures from corrupted cached kernels.
    
    Args:
        kernel: Kernel object to validate
        user_id: Owner user ID
        session_id: Session ID
    
    Raises:
        ValueError if kernel is corrupted
    """
    
    # Required attributes
    required_attrs = ['_owner_user_id', '_session_id', 'run', 'reset']
    
    for attr in required_attrs:
        if not hasattr(kernel, attr):
            logger.error(
                f"❌ Corrupted kernel: missing '{attr}' attribute",
                extra={"session_id": session_id, "user_id": user_id}
            )
            raise ValueError(f"Kernel validation failed: missing '{attr}'")
    
    # Verify ownership
    if kernel._owner_user_id != user_id:
        logger.error(
            f"🔴 Kernel ownership mismatch: {kernel._owner_user_id} vs {user_id}",
            extra={"session_id": session_id}
        )
        raise ValueError(f"Kernel ownership validation failed")
    
    if kernel._session_id != session_id:
        logger.error(
            f"🔴 Kernel session mismatch: {kernel._session_id} vs {session_id}",
            extra={"user_id": user_id}
        )
        raise ValueError(f"Kernel session validation failed")
    
    logger.debug(f"✓ Kernel validated for user {user_id}, session {session_id}")


def wrap_kernel_access(func):
    """
    Decorator to add session ownership verification to all kernel access.
    
    Usage:
        @wrap_kernel_access
        async def use_kernel(session_id, kernel, user_id):
            return await kernel.run(...)
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(session_id: str, kernel: object, user_id: str, *args, **kwargs):
        # Always validate before use
        await validate_kernel_before_use(kernel, user_id, session_id)
        
        # Call original function
        try:
            result = await func(session_id, kernel, user_id, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(
                f"Error during kernel execution: {str(e)}",
                exc_info=True,
                extra={"session_id": session_id, "user_id": user_id}
            )
            raise
    
    return wrapper
