"""
Embedded browser cognition runtime.

This package is intentionally small and workspace-native. It does not import
the dumped Browser-Use framework; it integrates the required browser
capabilities as an internal execution subsystem.
"""

from agents.browser_runtime.browser_observer import BrowserObserver
from agents.browser_runtime.browser_actions import BrowserActionExecutor
from agents.browser_runtime.planner import BrowserPlanner
from agents.browser_runtime.reflection import BrowserReflection
from agents.browser_runtime.tab_manager import BrowserTabManager

__all__ = [
    "BrowserObserver",
    "BrowserActionExecutor",
    "BrowserPlanner",
    "BrowserReflection",
    "BrowserTabManager",
]
