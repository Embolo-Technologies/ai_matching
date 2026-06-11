"""
Test script for AI Matching Engine improvements
================================================
Tests speed improvements, accuracy fixes, and new features.
Run from the /Volumes/ssd embolo/embolo/ai directory.
"""
import sys
import os
import time

# Ensure our package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_engine.pharma_data import (
    detect_formulation, formulations_compatible, 
    are_brands_equivalent, find_brand_aliases
)

# ─── Test 1: Pharma Data Module ──────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Pharma Data Module")
print("=" * 70)

# Formulation detection
tests_formulation = [
    ("PAN 40 TAB", "solid_oral"),
    ("AUGMENTIN INJ 1.2GM", "injectable"),
    ("CALPOL SYRUP 60ML", "liquid_oral"),
    ("BETADINE CREAM 20GM", "topical"),
    ("ASTHALIN INHALER", "inhaler_respiratory"),
    ("CIPRODEX EAR DROPS", "eye_ear_nasal"),
    ("PAN 40", ""),  # No formulation keyword
]

print("\n--- Formulation Detection ---")
for text, expected in tests_formulation:
    result = detect_formulation(text)
    status = "✅" if result == expected else "❌"
    print(f"  {status} '{text}' → '{result}' (expected: '{expected}')")

# Formulation compatibility
tests_compat = [
    ("PAN 40 TAB", "PAN 40 TAB", True),       # Same
    ("PAN 40 TAB", "PAN INJ", False),          # Tab vs Inj
    ("CALPOL TAB", "CALPOL SYRUP", False),     # Tab vs Syrup (NEW!)
    ("BETADINE CREAM", "BETADINE INJ", False), # Cream vs Inj (NEW!)
    ("ASTHALIN INH", "ASTHALIN TAB", False),   # Inhaler vs Tab (NEW!)
    ("PAN 40", "PAN 40 TAB", True),            # No formulation → assume ok
    ("PAN 40 TAB", "PAN 40 TABLET", True),     # Same category
]

print("\n--- Formulation Compatibility ---")
for text1, text2, expected in tests_compat:
    result = formulations_compatible(text1, text2)
    status = "✅" if result == expected else "❌"
    print(f"  {status} '{text1}' ↔ '{text2}' → {result} (expected: {expected})")

# Brand equivalence
tests_brand = [
    ("augmentin", "amoxyclav", True),
    ("crocin", "paracetamol", True),
    ("crocin", "dolo", True),       # Transitive: both are paracetamol
    ("pan", "pantop", True),
    ("pan", "omez", False),         # Different drugs entirely
    ("ecosprin", "aspirin", True),
    ("azithral", "azee", True),
    ("randomdrug", "otherdrug", False),
]

print("\n--- Brand Equivalence ---")
for brand1, brand2, expected in tests_brand:
    result = are_brands_equivalent(brand1, brand2)
    status = "✅" if result == expected else "❌"
    print(f"  {status} '{brand1}' ↔ '{brand2}' → {result} (expected: {expected})")

# ─── Test 2: Decimal Strength Regex ──────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: Decimal Strength Regex Fix")
print("=" * 70)

import re

# Old regex (broken)
old_pattern = r'\d+'
# New regex (fixed)
new_pattern = r'\d+\.?\d*'

tests_strength = [
    "DEXONA 0.5MG TAB",
    "WYSOLONE 2.5MG",
    "CLONAZEPAM 0.25",
    "METOPROLOL 47.5MG",
    "PAN 40MG",
    "AMOXYCLAV 625",
]

print(f"\n  {'Input':<30} {'Old regex':<20} {'New regex (fixed)':<20}")
print(f"  {'-'*30} {'-'*20} {'-'*20}")
for text in tests_strength:
    old_result = re.findall(old_pattern, text)
    new_result = re.findall(new_pattern, text)
    fix_marker = " ✅ FIXED" if old_result != new_result else ""
    print(f"  {text:<30} {str(old_result):<20} {str(new_result):<20}{fix_marker}")

# ─── Test 3: Searcher Speed Benchmark ────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: Searcher Speed Benchmark")
print("=" * 70)

# Try to find a CSV file
csv_candidates = [
    "/Users/admin/Downloads/item_export_2026-05-30_09-08-22.csv",
    "/Users/admin/Downloads/Item_export_2026-05-29_17-33-30.csv",
]

csv_path = None
for p in csv_candidates:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path:
    print(f"\n  Loading CSV: {csv_path}")
    from qwen3_engine.searcher import FuzzySearcher
    
    searcher = FuzzySearcher(csv_path=csv_path)
    
    t0 = time.time()
    searcher.load()
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s ({len(searcher.catalog):,} items)")
    print(f"  Prefix index size: {len(searcher._prefix_index)} buckets")
    print(f"  Company index size: {len(searcher._company_prefix_index)} buckets")
    
    # Benchmark queries
    test_queries = [
        ("PAN 40 TAB", ""),
        ("AUGMENTIN 625 DUO TAB", ""),
        ("DOLO 650", ""),
        ("CROCIN ADVANCE", "15 TAB"),
        ("AMOXYCLAV 625", ""),
        ("DEXONA 0.5", ""),
        ("SHELCAL 500", ""),
        ("METROGYL 400", ""),
        ("RANTAC 150", ""),
        ("ECOSPRIN 75", ""),
    ]
    
    print(f"\n  {'Query':<35} {'Time (ms)':<12} {'Top Result':<40} {'Score'}")
    print(f"  {'-'*35} {'-'*12} {'-'*40} {'-'*8}")
    
    total_time = 0
    for name, pack in test_queries:
        t0 = time.time()
        results, elapsed_ms = searcher.search(name, pack=pack, top_k=3)
        actual_ms = (time.time() - t0) * 1000
        total_time += actual_ms
        
        top_name = results[0]["name"] if results else "NO MATCH"
        top_score = results[0]["final_score"] if results else 0
        print(f"  {name:<35} {actual_ms:>8.2f}ms   {top_name:<40} {top_score:.1f}")
    
    avg_time = total_time / len(test_queries)
    print(f"\n  Average query time: {avg_time:.2f}ms")
    print(f"  Estimated time for 30K vendor items: {avg_time * 30000 / 1000 / 60:.1f} min")
    
    # ─── Test 4: FastMatcher with improvements ───────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: FastMatcher Accuracy (with brand aliases)")
    print("=" * 70)
    
    from qwen3_engine.fast_server import FastMatcher
    
    matcher = FastMatcher(searcher)
    matcher.load_catalog()
    
    test_matches = [
        ("PAN 40 TAB", "", ""),
        ("AUGMENTIN 625 DUO TAB", "", ""),
        ("DOLO 650 TAB", "", ""),
        ("CROCIN ADVANCE TAB", "CROCIN ADVANCE", "15 TAB"),
        ("DEXONA 0.5MG TAB", "DEXONA 0.5MG TAB", ""),
        ("METROGYL 400 TAB", "", ""),
        ("ECOSPRIN 75 TAB", "", ""),
        ("RANTAC 150 TAB", "", ""),
        # Formulation mismatch tests (should return None)
        ("PAN 40 INJ", "PAN 40 INJ", ""),
        # Messy human / vendor prefix junk codes
        ("RS-PAN 40 TAB", "RS-PAN 40 TAB", ""),      # Prefix code
        ("XY PNTOP 40 TAB", "XY PNTOP 40 TAB", ""),    # Typo + prefix code
        ("PAN 40/TAB", "PAN 40/TAB", ""),              # Slashes
        ("DOLO 650? TAB", "DOLO 650? TAB", ""),        # Symbols
    ]
    
    print(f"\n  {'Query':<35} {'Match':<40} {'Status'}")
    print(f"  {'-'*35} {'-'*40} {'-'*8}")
    
    for query, name, pack in test_matches:
        result = matcher.match(query, name=name, pack=pack)
        match_name = result["name"] if result else "NO MATCH"
        status = "✅" if result else "⚠️ None"
        print(f"  {query:<35} {match_name:<40} {status}")

else:
    print("\n  ⚠️ No CSV file found. Skipping searcher/matcher benchmarks.")
    print("  Place a CSV file at one of these locations:")
    for p in csv_candidates:
        print(f"    - {p}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
