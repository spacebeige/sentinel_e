"""
Exception handling utilities
Fixes: Silent exception swallowing, missing logging, impossible debugging
"""

import logging
import functools
from typing import Callable, Any, Optional, Type, Tuple
import traceback
import sys

logger = logging.getLogger(__name__)


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    exception_types: Tuple[Type[Exception], ...] = (Exception,),
    log_level: str = "error",
    context_info: str = "",
    reraise: bool = False,
    **kwargs
) -> Any:
    """
    Safely execute a function with comprehensive error logging.
    
    Fixes: Bare except: pass blocks that hide errors
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Value to return on exception (if not reraising)
        exception_types: Tuple of exceptions to catch (default: all)
        log_level: Logging level (error, warning, debug)
        context_info: Additional context for the error message
        reraise: If True, rethrow after logging
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_func = getattr(logger, log_level.lower(), logger.error)
        
        # Format comprehensive error message
        error_msg = f"Exception in {func.__name__}"
        if context_info:
            error_msg += f" ({context_info})"
        error_msg += f": {type(e).__name__}: {str(e)}"
        
        # Log with full traceback
        log_func(error_msg, exc_info=True)
        
        if reraise:
            raise
        
        return default_return


def handle_exceptions(*exceptions: Type[Exception]) -> Callable:
    """
    Decorator for explicit exception handling with logging.
    
    Fixes: Implicit bare except blocks
    
    Usage:
        @handle_exceptions(ValueError, KeyError)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(
                    f"Error in {func.__name__}: {type(e).__name__}: {str(e)}",
                    exc_info=True,
                    extra={"function": func.__name__, "exception_type": type(e).__name__}
                )
                raise
        return wrapper
    return decorator


def log_unhandled_exceptions():
    """
    Install global exception hook to catch unhandled exceptions.
    Call once during app startup.
    
    Fixes: Critical errors disappearing from logs
    """
    def excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            f"UNHANDLED EXCEPTION: {exc_type.__name__}: {str(exc_value)}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = excepthook
    logger.info("✓ Global exception hook installed")


def validate_response(response: Any, expected_type: Type = None, context: str = "") -> bool:
    """
    Validate response structure to catch silent failures.
    
    Fixes: Garbled responses from sub-mode failures
    
    Args:
        response: Response to validate
        expected_type: Expected type (if any)
        context: Context for error messages
    
    Returns:
        True if valid, False otherwise
    """
    if response is None:
        logger.warning(f"Response is None {context}")
        return False
    
    if expected_type and not isinstance(response, expected_type):
        logger.warning(
            f"Response type mismatch {context}: expected {expected_type.__name__}, "
            f"got {type(response).__name__}"
        )
        return False
    
    if isinstance(response, dict) and "error" in response:
        logger.warning(f"Response contains error {context}: {response.get('error')}")
        return False
    
    return True
