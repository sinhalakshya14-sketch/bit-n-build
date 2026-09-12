import sys
import sqlite3
from pathlib import Path
from streamlit.testing.v1 import AppTest

# Initialize AppTest from dashboard.py
at = AppTest.from_file("dashboard.py", default_timeout=120)
print("Running initial dashboard...")
at.run()
print(f"Initial run done. Exception: {at.exception}")

# Find analyst name text input
print(f"Text inputs: {[ti.key for ti in at.text_input]}")
for ti in at.text_input:
    if "analyst" in (ti.key or ""):
        print(f"Found analyst text input: {ti.key}")
        ti.input("Jordan Chen").run()
        break

print(f"After entering name. Exception: {at.exception}")
print(f"Buttons found: {[b.key for b in at.button]}")

# Look for watchlist button
add_btn = None
for b in at.button:
    if b.key and b.key.startswith("wl_add_"):
        add_btn = b
        print(f"Found add button: {b.key} with label {b.label!r}".encode('ascii', 'backslashreplace').decode('ascii'))
        break

if add_btn:
    print(f"Clicking button {add_btn.key}...")
    add_btn.click().run()
    print(f"After click. Exception: {at.exception}")
    print(f"Buttons after click: {[b.key for b in at.button]}")
else:
    print("No wl_add_ button found!")

# Look for remove button
rm_btn = None
for b in at.button:
    if b.key == "wl_rm_V0725":
        rm_btn = b
        print(f"Found remove button: {b.key}")
        break

if rm_btn:
    print(f"Clicking remove button {rm_btn.key}...")
    rm_btn.click().run()
    print(f"After remove click. Exception: {at.exception}")
    print(f"Buttons after remove click: {[b.key for b in at.button]}")

# Check database again
if db_path.exists():
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM watchlist WHERE analyst_name = 'Jordan Chen'").fetchall()
    print(f"DB rows for Jordan Chen after remove: {rows}")
    conn.close()
