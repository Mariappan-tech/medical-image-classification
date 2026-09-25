"""
audit.py — Audit Logging Middleware & Router
Provides:
  - FastAPI middleware that auto-logs every request/response
  - /audit/* endpoints for querying logs (admin-only)
  - Helper decorator @audit_action for manual action logging
"""

from __future__ import annotations

import os
import sys
import time
import json
import functools
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import insert_audit_log, list_audit_logs, get_audit_stats
from backend.auth import require_role, get_current_user

# ─── Router ───────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/audit", tags=["Audit"])


# ─── Middleware: Auto-Audit Every Request ─────────────────────────────────────
class AuditMiddleware:
    """
    Starlette middleware that records every HTTP request and response
    into the audit_logs table automatically.
    """

    # Paths to skip (health checks, docs, static files)
    SKIP_PATHS = {
        "/health", "/docs", "/redoc", "/openapi.json",
        "/favicon.ico", "/static",
    }

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request   = Request(scope, receive)
        path      = request.url.path
        method    = request.method

        # Skip boring paths
        if any(path.startswith(skip) for skip in self.SKIP_PATHS):
            await self.app(scope, receive, send)
            return

        ip_address = _get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        start_time = time.perf_counter()
        status_code = 200

        # Intercept the response status
        async def _send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        # Extract user from Authorization header (best-effort, no raise)
        user_id  = None
        username = None
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                from backend.auth import decode_token, _hash_token
                from backend.database import get_session_by_token
                raw_token  = auth_header.split(" ", 1)[1]
                token_hash = _hash_token(raw_token)
                session    = get_session_by_token(token_hash)
                if session:
                    user_id  = session.get("user_id")
                    username = session.get("username")
        except Exception:
            pass

        await self.app(scope, receive, _send_wrapper)
        duration_ms = (time.perf_counter() - start_time) * 1000

        action = _path_to_action(method, path)
        audit_status = "SUCCESS" if status_code < 400 else "FAILURE"

        try:
            insert_audit_log(
                action=action,
                status=audit_status,
                user_id=user_id,
                username=username,
                resource=path,
                detail={"method": method, "status_code": status_code},
                ip_address=ip_address,
                user_agent=user_agent,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as e:
            print(f"[AUDIT] Failed to write audit log: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _path_to_action(method: str, path: str) -> str:
    """Convert HTTP method + path to a human-readable audit action name."""
    mapping = {
        ("POST", "/predict"):               "PREDICT",
        ("POST", "/predict/uncertainty"):   "PREDICT_UNCERTAINTY",
        ("POST", "/evaluate"):              "EVALUATE",
        ("GET",  "/evaluate/results"):      "GET_RESULTS",
        ("POST", "/auth/login"):            "LOGIN",
        ("POST", "/auth/register"):         "REGISTER",
        ("POST", "/auth/logout"):           "LOGOUT",
        ("POST", "/auth/logout-all"):       "LOGOUT_ALL",
        ("GET",  "/auth/me"):               "GET_PROFILE",
        ("GET",  "/audit/logs"):            "AUDIT_QUERY",
        ("GET",  "/audit/stats"):           "AUDIT_STATS",
        ("GET",  "/predictions"):           "LIST_PREDICTIONS",
        ("GET",  "/model/info"):            "MODEL_INFO",
    }
    for (m, p), action in mapping.items():
        if method == m and (path == p or path.startswith(p + "/")):
            return action
    return f"{method}:{path}"


# ─── Decorator: Manual Action Audit ───────────────────────────────────────────
def audit_action(action: str, resource: str = None):
    """
    Decorator to manually log a specific action.

    Usage:
        @audit_action("TRAIN_MODEL", resource="/internal/train")
        def start_training(user: dict = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user = None
            for v in kwargs.values():
                if isinstance(v, dict) and "user_id" in v:
                    user = v
                    break
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                insert_audit_log(
                    action=action,
                    status="SUCCESS",
                    user_id=user.get("user_id") if user else None,
                    username=user.get("username") if user else None,
                    resource=resource,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                return result
            except HTTPException as e:
                insert_audit_log(
                    action=action,
                    status="FAILURE",
                    user_id=user.get("user_id") if user else None,
                    username=user.get("username") if user else None,
                    resource=resource,
                    detail={"error": e.detail},
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                raise
        return wrapper
    return decorator


# ─── Audit Router Endpoints ───────────────────────────────────────────────────
@router.get("/logs", summary="List audit logs (admin only)")
async def get_audit_logs(
    action:  Optional[str] = Query(None, description="Filter by action e.g. LOGIN"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status:  Optional[str] = Query(None, description="Filter by status: SUCCESS | FAILURE"),
    limit:   int           = Query(100, ge=1, le=500),
    offset:  int           = Query(0,   ge=0),
    _admin:  dict          = Depends(require_role("admin")),
):
    """Return audit logs. Accessible only by admins."""
    logs = list_audit_logs(action=action, user_id=user_id,
                           status=status, limit=limit, offset=offset)
    return {"count": len(logs), "logs": logs}


@router.get("/stats", summary="Audit statistics (admin only)")
async def get_audit_stats_endpoint(
    _admin: dict = Depends(require_role("admin")),
):
    """Return aggregated audit statistics."""
    return get_audit_stats()


@router.get("/my-activity", summary="Current user's own activity log")
async def get_my_activity(
    limit:        int  = Query(50,  ge=1, le=200),
    offset:       int  = Query(0,   ge=0),
    current_user: dict = Depends(get_current_user),
):
    """Return the current user's own audit trail."""
    logs = list_audit_logs(user_id=current_user["user_id"], limit=limit, offset=offset)
    return {"count": len(logs), "logs": logs}
