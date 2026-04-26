from typing import Any, Optional, Dict
from fastapi.responses import JSONResponse

def api_response(
    success: bool = True,
    data: Any = None,
    error: Optional[str] = None,
    status_code: int = 200
) -> JSONResponse:
    """
    Standardized JSON response format:
    {
      "success": bool,
      "data": <object|array>,
      "error": <string|null>
    }
    """
    content = {
        "success": success,
        "data": data if data is not None else ([] if isinstance(data, list) else {}),
        "error": error
    }
    
    # If data is None and success is true, ensure it's an empty dict or list
    if data is None:
        if success:
            content["data"] = {}
        else:
            content["data"] = None

    return JSONResponse(content=content, status_code=status_code)

def api_error(message: str, status_code: int = 400) -> JSONResponse:
    """Shorthand for error response."""
    return api_response(success=False, error=message, status_code=status_code)

def api_success(data: Any = None, status_code: int = 200) -> JSONResponse:
    """Shorthand for success response."""
    return api_response(success=True, data=data, status_code=status_code)
