import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_engine.matcher import HybridMatcher

csv_path = "/Users/admin/Downloads/item_export_2026-05-30_09-08-22.csv"
if not os.path.exists(csv_path):
    csv_path = "/Users/admin/Downloads/Item_export_2026-05-29_17-33-30.csv"

print("Initializing Matcher...")
matcher = HybridMatcher(master_csv_path=csv_path, model_key="qwen3")
matcher.load_catalog()

# Run query that should go to LLM
query = "TAXIM INJ"
print(f"\n--- Matching '{query}' ---")
res = matcher.match(query, top_k=5)
print("\nMATCH RESULT:", res)
