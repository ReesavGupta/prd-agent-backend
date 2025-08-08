"""
Base schemas and response utilities.
"""

from typing import Any, Dict


class BaseResponse:
    """Base response format for consistent API responses."""

    @staticmethod
    def success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Create a success response."""
        return {
            "success": True,
            "message": message,
            "data": data
        }

    @staticmethod
    def error(message: str, details: Any = None, status_code: int = 400) -> Dict[str, Any]:
        """Create an error response."""
        response = {
            "success": False,
            "message": message,
            "status_code": status_code
        }
        if details:
            response["details"] = details
        return response

    @staticmethod
    def paginated(
        items: list, 
        total: int, 
        limit: int, 
        offset: int, 
        message: str = "Success"
    ) -> Dict[str, Any]:
        """Create a paginated response."""
        return {
            "success": True,
            "message": message,
            "data": {
                "items": items,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + len(items) < total
                }
            }
        }