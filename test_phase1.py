import sys
import os
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from orchestrator import initialise, get_state
from agents.investigator import generate_vessel_brief

def main():
    print("Initializing state...")
    initialise()
    state = get_state()
    
    # Test 1: Flagged vessel (e.g. V0004 or similar, based on test run)
    vessel_1 = "V0004"
    print(f"\n--- Testing Brief for {vessel_1} ---")
    brief_1 = generate_vessel_brief(vessel_1, state)
    print(json.dumps(brief_1, indent=2))
    
    # Test 2: Another vessel (e.g. V0001)
    vessel_2 = "V0001"
    print(f"\n--- Testing Brief for {vessel_2} ---")
    brief_2 = generate_vessel_brief(vessel_2, state)
    print(json.dumps(brief_2, indent=2))

if __name__ == "__main__":
    main()
