"""
agents/dashboard.py
===================
Wrapper to allow running either:
    python -m streamlit run dashboard.py
or:
    python -m streamlit run agents/dashboard.py
"""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

target_dashboard = os.path.join(root_dir, "dashboard.py")

with open(target_dashboard, "r", encoding="utf-8") as _f:
    _code = compile(_f.read(), target_dashboard, "exec")
    exec(_code, globals())
