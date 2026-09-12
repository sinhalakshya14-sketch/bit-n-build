import sqlite3
import time
from pathlib import Path
from streamlit.testing.v1 import AppTest

print("=== STARTING END-TO-END WATCHLIST FLOW TEST ===")
t0 = time.perf_counter()
at = AppTest.from_file("dashboard.py", default_timeout=120)
at.run()
init_time = time.perf_counter() - t0
print(f"[1] App initialized in {init_time:.2f}s, Exceptions: {len(at.exception)}")

# Step 1: Sign in as 'Elena Rostova'
analyst = "Elena Rostova"
t_sign = time.perf_counter()
for ti in at.text_input:
    if "analyst" in (ti.key or ""):
        ti.input(analyst).run()
        break
sign_time = time.perf_counter() - t_sign
print(f"[2] Signed in as '{analyst}' in {sign_time:.2f}s, Exceptions: {len(at.exception)}")

# Check success banner
success_msgs = [s.value for s in at.success]
print(f"    Success messages: {success_msgs}")
assert any(f"Signed in as {analyst}" in s for s in success_msgs), "Sign-in banner not found!"

# Check empty watchlist caption
captions = [c.value for c in at.caption]
has_empty_cap = any("No vessels on your watchlist yet" in c for c in captions)
print(f"    Empty watchlist caption present: {has_empty_cap}")

# Step 2: Find a flagged vessel to add
target_btn = None
for b in at.button:
    if b.key and b.key.startswith("wl_add_"):
        target_btn = b
        break

assert target_btn is not None, "No flagged vessel add button found!"
target_vid = target_btn.key.replace("wl_add_", "")
print(f"[3] Target flagged vessel to add: {target_vid} (button key: {target_btn.key})")

# Step 3: Click 'Add to watchlist'
t_add = time.perf_counter()
target_btn.click().run()
add_duration = time.perf_counter() - t_add
print(f"[4] Add click completed in {add_duration:.2f}s (fast fragment rerun), Exceptions: {len(at.exception)}")
assert len(at.exception) == 0, f"Exceptions after add click: {at.exception}"

# Step 4: Verify button state visibly changed
button_keys = [b.key for b in at.button if b.key]
print(f"    Button keys after add: {button_keys}")
assert f"wl_on_{target_vid}" in button_keys, f"Expected button 'wl_on_{target_vid}' to be present!"
assert f"wl_rm_{target_vid}" in button_keys, f"Expected remove button 'wl_rm_{target_vid}' in My Watchlist!"

# Step 5: Check database
conn = sqlite3.connect("watchlist.db")
rows = conn.execute("SELECT * FROM watchlist WHERE analyst_name = ?", (analyst,)).fetchall()
conn.close()
print(f"[5] Database rows in watchlist.db for '{analyst}': {rows}")
assert any(r[0] == analyst and r[1] == target_vid for r in rows), f"Row for {target_vid} not in DB!"

# Step 6: Test Remove from watchlist
rm_btn = None
for b in at.button:
    if b.key == f"wl_rm_{target_vid}":
        rm_btn = b
        break

assert rm_btn is not None, "Remove button not found!"
t_rm = time.perf_counter()
rm_btn.click().run()
rm_duration = time.perf_counter() - t_rm
print(f"[6] Remove click completed in {rm_duration:.2f}s, Exceptions: {len(at.exception)}")
assert len(at.exception) == 0, f"Exceptions after remove click: {at.exception}"

button_keys_after_rm = [b.key for b in at.button if b.key]
assert f"wl_add_{target_vid}" in button_keys_after_rm, f"Expected 'wl_add_{target_vid}' to return!"
assert f"wl_rm_{target_vid}" not in button_keys_after_rm, f"Expected 'wl_rm_{target_vid}' to be removed!"

conn = sqlite3.connect("watchlist.db")
rows_after = conn.execute("SELECT * FROM watchlist WHERE analyst_name = ?", (analyst,)).fetchall()
conn.close()
print(f"[7] Database rows after removal: {rows_after}")
assert len(rows_after) == 0, "Expected empty watchlist for analyst after removal!"

print("\nALL VERIFICATION CHECKS PASSED PERFECTLY!")
