"""Execution-scoped project identity for Director caches and UI events."""

from __future__ import annotations

import inspect
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

current_project_id = ContextVar("minimax_director_project_id", default="")


def project_id_from_timeline(timeline_data) -> str:
    if isinstance(timeline_data, str):
        try:
            timeline_data = json.loads(timeline_data or "{}")
        except (TypeError, ValueError):
            return ""
    if not isinstance(timeline_data, dict):
        return ""
    project_id = timeline_data.get("projectId", timeline_data.get("project_id", ""))
    if not project_id:
        return ""
    if not isinstance(project_id, str) or not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise ValueError("Invalid Director project id; reopen the workflow in the Director UI.")
    return project_id


@contextmanager
def project_scope(timeline_data):
    token = current_project_id.set(project_id_from_timeline(timeline_data))
    try:
        yield
    finally:
        current_project_id.reset(token)


def project_scoped(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        with project_scope(arguments.get("timeline_data")):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                from .progress import report_director_error

                report_director_error(arguments.get("unique_id"), str(exc))
                raise

    return wrapped
