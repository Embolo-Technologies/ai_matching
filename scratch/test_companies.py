import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_engine.fast_server import companies_compatible, clean_company_name

test_cases = [
    # (Query Company, Candidate Company, Expected Result, Description)
    ("DR REDDY", "Hetero", False, "Mismatched companies"),
    ("DR REDDY", "DR. REDDYS LABORATORIES", True, "Reddy suffix and lab word matching"),
    ("DR.REDDY'S", "DR REDDY", True, "Punctuation tolerance"),
    ("CADILA", "ZYDUS CADILA HEALTHCARE", True, "Substring Cadila"),
    ("ZYDUS CADILA", "CADILA PHARMACEUTICALS", True, "Cross zydus/pharma matching via cadila word"),
    ("ABBOTT INDIA LTD", "ABBOTT", True, "Stopword removal"),
    ("HETERO HEALTHCARE", "HETERO", True, "Stopword removal (Healthcare)"),
    ("WALLACE", "WALLACE PHARMA", True, "Wallace phama matching"),
    ("", "Hetero", True, "Blank query company compatible with anything"),
    ("DR REDDY", "", True, "Blank candidate company compatible with anything"),
    ("--", "WALLACE", True, "Placeholder query company compatible with anything"),
    ("SERDIA", "SERDIA PHARMACEUTICALS", True, "Serdia matching"),
    ("DR MOREPEN", "DR REDDY", False, "Should ignore DR overlap and return False"),
]

print("=" * 80)
print("TESTING COMPANY COMPATIBILITY")
print("=" * 80)

all_passed = True
for q_comp, c_comp, expected, desc in test_cases:
    res = companies_compatible(q_comp, c_comp)
    status = "✅ PASSED" if res == expected else "❌ FAILED"
    if res != expected:
        all_passed = False
    print(f"  {status} | Query: '{q_comp:<20}' | Candidate: '{c_comp:<20}' | Expected: {expected} | Got: {res} | ({desc})")

print("=" * 80)
if all_passed:
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
else:
    print("❌ SOME TESTS FAILED!")
print("=" * 80)
