#!/usr/bin/env python3
"""
Full production match: Input (chemist) vs Master (wholesaler)
Outputs Excel with confidence scores for human review.
"""
import os, sys, time, re, argparse
import pandas as pd
import numpy as np

# Make the python module search path dynamic to support the local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_engine.searcher import FuzzySearcher
from qwen3_engine.fast_server import FastMatcher
from rapidfuzz import fuzz


def extract_strength(name_str):
    if not isinstance(name_str, str):
        return ""
    # Compound doses like "500/125MG" — return full slash expression so both
    # components are available during strength matching.
    compound = re.search(r'\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*(?:mg|ml|gm|g|mcg)?\b', name_str, re.IGNORECASE)
    if compound:
        return compound.group(0)
    match = re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|gm|g|mcg|cap|tab)\b', name_str, re.IGNORECASE)
    if match:
        return match.group(0)
    match_num = re.search(r'\b\d+(?:\.\d+)?\b', name_str)
    if match_num:
        val = match_num.group(0)
        try:
            if float(val) <= 10.0 and '.' not in val:
                return ""
        except ValueError:
            pass
        return val
    return ""


def compute_confidence(query_name, matched_name, matched_brand, query_pack, matched_pack):
    """Compute a human-readable confidence score (0-100) for a match."""
    if not matched_name:
        return 0

    q = query_name.lower().strip()
    m = matched_name.lower().strip()

    # 1. Brand similarity (most important) — first word comparison
    q_words = re.sub(r'[^a-z0-9\s]', '', q).split()
    m_words = re.sub(r'[^a-z0-9\s]', '', m).split()

    q_brand = q_words[0] if q_words else ""
    m_brand = m_words[0] if m_words else ""

    # Skip short vendor prefix codes
    if len(q_brand) <= 2 and len(q_words) > 1:
        q_brand = q_words[1]

    brand_score = fuzz.ratio(q_brand, m_brand)

    # 2. Full name similarity
    full_score = fuzz.token_sort_ratio(q, m)

    # 3. Strength match bonus
    q_nums = set(re.findall(r'\d+\.?\d*', q))
    m_nums = set(re.findall(r'\d+\.?\d*', m))
    # Remove small pack counts
    PACK_NUMS = {'1','2','3','4','5','6','7','8','9','10'}
    q_strengths = q_nums - PACK_NUMS
    m_strengths = m_nums - PACK_NUMS
    
    strength_bonus = 0
    if q_strengths and m_strengths:
        if q_strengths & m_strengths:  # intersection = matching strengths
            strength_bonus = 10
        else:
            strength_bonus = -15  # strength mismatch penalty

    # Weighted final score
    confidence = (brand_score * 0.5) + (full_score * 0.4) + strength_bonus
    return max(0, min(100, round(confidence)))


def classify_confidence(score):
    """Human-readable status based on confidence score."""
    if score >= 90:
        return "✅ High Confidence"
    elif score >= 75:
        return "🔶 Medium - Review"
    elif score >= 60:
        return "⚠️ Low - Check"
    else:
        return "❌ Very Low - Likely Wrong"


def main():
    parser = argparse.ArgumentParser(description="Full production matching run")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0=all)")
    parser.add_argument("--no-ai", action="store_true", default=False, help="Disable AI (heuristic only, faster but less accurate)")
    parser.add_argument("--master-xlsx", default="/Volumes/ssd embolo/Games/masterdata.xlsx")
    parser.add_argument("--input-xlsx", default="/Volumes/ssd embolo/Games/input.xlsx")
    parser.add_argument("--output-xlsx", default="/Volumes/ssd embolo/Games/full_matching_results.xlsx")
    args = parser.parse_args()

    print("=" * 80)
    print("FULL PRODUCTION MATCHING RUN")
    print(f"AI Matching: {'OFF (heuristic only)' if args.no_ai else 'ON — low confidence items will be AI matched'}")
    print("=" * 80)

    # 1. Load master catalog
    print(f"\n[1/4] Loading master catalog...")
    t0 = time.time()
    if args.master_xlsx.endswith('.csv'):
        df_master = pd.read_csv(args.master_xlsx)
    else:
        df_master = pd.read_excel(args.master_xlsx)

    # Standardize column names for master database
    rename_map = {
        'Material Description': 'name',
        'Brands': 'Compname',
        'Pack Size Per Strip': 'Pack'
    }
    for old_col, new_col in rename_map.items():
        if old_col in df_master.columns and new_col not in df_master.columns:
            df_master = df_master.rename(columns={old_col: new_col})

    # Ensure required columns exist
    for col in ['name', 'Compname', 'Pack']:
        if col not in df_master.columns:
            df_master[col] = ""

    df_master = df_master.dropna(subset=['name'])
    df_master = df_master[~df_master['name'].astype(str).str.contains(r'^(name|----)$', case=False, na=False)]
    df_master['code'] = [f"M_{i}" for i in range(1, len(df_master) + 1)]
    df_master['strength'] = df_master['name'].apply(extract_strength)
    df_master['Compname'] = df_master['Compname'].fillna("").astype(str).str.strip()
    df_master['Pack'] = df_master['Pack'].fillna("").astype(str).str.strip()
    df_master['name'] = df_master['name'].astype(str).str.strip()

    input_base = os.path.splitext(os.path.basename(args.input_xlsx))[0]
    temp_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"masterdata_converted_{input_base}.csv")
    df_master[['code', 'name', 'Compname', 'Pack', 'strength']].to_csv(temp_csv, index=False, encoding='utf-8-sig')
    print(f"   ✓ {len(df_master):,} master items loaded in {time.time()-t0:.1f}s")

    # 2. Init matching engine
    print(f"\n[2/4] Initializing matching engine...")
    searcher = FuzzySearcher(temp_csv)
    searcher.load()
    matcher = FastMatcher(searcher, model_key="gemma4_2b")
    matcher.load_catalog()
    print(f"   ✓ Engine ready")

    # 3. Load input
    print(f"\n[3/4] Loading chemist input...")
    if args.input_xlsx.endswith('.csv'):
        df_input = pd.read_csv(args.input_xlsx)
    else:
        df_input = pd.read_excel(args.input_xlsx)
    df_input = df_input.dropna(subset=['name'])
    df_input = df_input[~df_input['name'].astype(str).str.contains(r'^(name|----)$', case=False, na=False)]
    
    if args.limit > 0:
        df_input = df_input.head(args.limit)
    print(f"   ✓ {len(df_input):,} input items loaded")

    # 4. Run matching
    print(f"\n[4/4] Running matches on {len(df_input):,} items...")
    results = []
    matched_count = 0
    t_start = time.time()
    total = len(df_input)

    for i, (idx, row) in enumerate(df_input.iterrows()):
        q_name = str(row['name']).strip()
        q_pack = str(row.get('Pack', '')).strip()
        q_comp = str(row.get('Compname', '')).strip()
        q_code = str(row.get('code', '')).strip()
        
        if q_pack in ["nan", "None", "NaN"]: q_pack = ""
        if q_comp in ["nan", "None", "NaN"]: q_comp = ""
        
        query_str = f"{q_name} {q_pack}".strip()

        t0q = time.time()
        candidates_res, _ = searcher.search(q_name, pack=q_pack, compname=q_comp, top_k=5)
        candidates = [
            {"code": r["code"], "name": r["name"], "brand": r["compname"],
             "pack": r["pack"], "strength": r["strength"]}
            for r in candidates_res
        ]

        heur_result, heur_confidence = matcher.heuristic_match(query_str, candidates, name=q_name, compname=q_comp)
        
        if heur_result and heur_confidence >= 95:
            # HIGH confidence — accept directly
            matched_item = heur_result
            match_method = "Heuristic"
        elif heur_result and heur_confidence >= 75:
            # MEDIUM confidence — accept heuristic (good enough)
            matched_item = heur_result
            match_method = "Heuristic"
        elif not args.no_ai and candidates:
            # Pre-filter candidates by loose validation check to skip calling AI on complete mismatches
            valid_candidates = []
            for cand in candidates:
                if matcher.validate_match(query_str, cand, name=q_name, compname=q_comp, strict=False):
                    valid_candidates.append(cand)
            
            if not valid_candidates:
                matched_item = None
                match_method = "No Match"
            else:
                if heur_result:
                    print(f"   [AI Match] '{q_name}' heuristic confidence={heur_confidence}% too low, sending to AI...")
                else:
                    print(f"   [AI Match] '{q_name}' no heuristic match, sending to AI...")
                ai_result = matcher.llm_rerank(query_str, valid_candidates, name=q_name, compname=q_comp)
                if ai_result:
                    matched_item = ai_result
                    match_method = "AI Match"
                else:
                    matched_item = None
                    match_method = "No Match"
        elif heur_result:
            # AI disabled but we have a low-confidence heuristic — accept with warning
            matched_item = heur_result
            match_method = "Heuristic (Low)"
        else:
            matched_item = None
            match_method = "No Match"

        latency = (time.time() - t0q) * 1000

        if matched_item:
            matched_count += 1
            confidence = compute_confidence(q_name, matched_item['name'], matched_item.get('brand',''), q_pack, matched_item.get('pack',''))
            status = classify_confidence(confidence)
            
            # Company match status
            comp_status = "—"
            if q_comp and q_comp.strip() and q_comp.strip() != '--':
                m_comp = matched_item.get('brand', '') or ''
                if m_comp:
                    from rapidfuzz import fuzz as _fuzz
                    comp_sim = _fuzz.partial_ratio(q_comp.lower(), m_comp.lower())
                    if comp_sim >= 60:
                        comp_status = "✅ Match"
                    elif comp_sim >= 35:
                        comp_status = "⚠️ Partial"
                    else:
                        comp_status = "❌ Mismatch"
                else:
                    comp_status = "— No Master Co."
            
            results.append({
                "Input Code": q_code,
                "Input Name": q_name,
                "Input Pack": q_pack,
                "Input Company": q_comp,
                "": "",  # spacer column
                "Matched Name": matched_item['name'],
                "Matched Pack": matched_item.get('pack', ''),
                "Matched Company": matched_item.get('brand', ''),
                "Matched Code": matched_item['code'],
                " ": "",  # spacer column
                "Confidence %": confidence,
                "Status": status,
                "Company Match": comp_status,
                "Method": match_method,
                "Time (ms)": round(latency, 1),
            })
        else:
            results.append({
                "Input Code": q_code,
                "Input Name": q_name,
                "Input Pack": q_pack,
                "Input Company": q_comp,
                "": "",
                "Matched Name": "",
                "Matched Pack": "",
                "Matched Company": "",
                "Matched Code": "",
                " ": "",
                "Confidence %": 0,
                "Status": "— No Match Found",
                "Company Match": "—",
                "Method": match_method,
                "Time (ms)": round(latency, 1),
            })

        # Progress report every 500 items
        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            match_pct = matched_count / (i + 1) * 100
            print(f"   [{i+1:,}/{total:,}] {match_pct:.1f}% matched | {rate:.0f} items/sec | ETA: {eta:.0f}s")

    elapsed_total = time.time() - t_start
    match_rate = matched_count / total * 100 if total > 0 else 0

    # 5. Create Excel output
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Items:      {total:,}")
    print(f"Matched:          {matched_count:,} ({match_rate:.1f}%)")
    print(f"Not Matched:      {total - matched_count:,} ({100-match_rate:.1f}%)")
    print(f"Total Time:       {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Avg Speed:        {total/elapsed_total:.0f} items/sec")

    # Create DataFrame and sort: matched items by confidence (low first for review), then unmatched
    df_out = pd.DataFrame(results)
    
    # Split into matched and unmatched
    df_matched = df_out[df_out['Confidence %'] > 0].sort_values('Confidence %', ascending=True)
    df_unmatched = df_out[df_out['Confidence %'] == 0]
    
    # Confidence distribution
    if len(df_matched) > 0:
        high = len(df_matched[df_matched['Confidence %'] >= 90])
        medium = len(df_matched[(df_matched['Confidence %'] >= 75) & (df_matched['Confidence %'] < 90)])
        low = len(df_matched[df_matched['Confidence %'] < 75])
        print(f"\nConfidence Breakdown (matched items):")
        print(f"  ✅ High (≥90%):    {high:,} — auto-accept")
        print(f"  🔶 Medium (75-89%): {medium:,} — quick review")
        print(f"  ⚠️  Low (<75%):     {low:,} — needs manual check")

    # Write to Excel with multiple sheets
    print(f"\nSaving to: {args.output_xlsx}")
    with pd.ExcelWriter(args.output_xlsx, engine='openpyxl') as writer:
        # Sheet 1: All results sorted by confidence (low first = needs review first)
        df_all_sorted = pd.concat([df_matched, df_unmatched])
        df_all_sorted.to_excel(writer, sheet_name='All Results', index=False)
        
        # Sheet 2: Only matched, sorted by confidence ascending (review worst first)
        if len(df_matched) > 0:
            df_matched.to_excel(writer, sheet_name='Matched (Review)', index=False)
        
        # Sheet 3: Only unmatched
        if len(df_unmatched) > 0:
            df_unmatched.to_excel(writer, sheet_name='Not Matched', index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if cell.value:
                            max_len = max(max_len, len(str(cell.value)))
                    except:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 2, 40)

    # Clean up unique temp CSV
    try:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    except:
        pass
    print(f"✓ Excel saved successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
