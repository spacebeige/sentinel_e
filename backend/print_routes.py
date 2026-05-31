import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from main import app

for route in app.routes:
    if hasattr(route, "methods"):
        print(route.methods, route.path)
    else:
        print(route.path)
