#!/usr/bin/env python3
"""
Utility to merge partial matching results from multiple PCs back into one consolidated file.
"""
import os
import glob
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge partial Excel result files")
    parser.add_argument("--input-dir", default="/Volumes/ssd embolo/Games/split_results", help="Directory containing partial Excel result files")
    parser.add_argument("--output-xlsx", default="/Volumes/ssd embolo/Games/full_matching_results.xlsx", help="Path to save merged master Excel file")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Results directory not found at: {args.input_dir}")
        return

    # Find all result files (e.g., matching_results_part_*.xlsx)
    pattern = os.path.join(args.input_dir, "*.xlsx")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No Excel files found in {args.input_dir}")
        return

    print(f"Found {len(files)} result files to merge:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    all_dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            all_dfs.append(df)
            print(f"   Loaded {len(df):,} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"   ❌ Error loading {os.path.basename(f)}: {e}")

    if not all_dfs:
        print("No valid data loaded. Exiting.")
        return

    # Concatenate all parts
    df_merged = pd.concat(all_dfs, ignore_index=True)
    total_rows = len(df_merged)
    print(f"\nMerged data contains {total_rows:,} rows total.")

    # Split into matched and unmatched for sorting (maintaining same output structure as original script)
    df_matched = df_merged[df_merged['Confidence %'] > 0].sort_values('Confidence %', ascending=True)
    df_unmatched = df_merged[df_merged['Confidence %'] == 0]

    # Recombine sorted data
    df_final = pd.concat([df_matched, df_unmatched])

    print(f"Saving merged output to: {args.output_xlsx}")
    with pd.ExcelWriter(args.output_xlsx, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='All Results', index=False)
        if len(df_matched) > 0:
            df_matched.to_excel(writer, sheet_name='Matched (Review)', index=False)
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

    print(f"✓ Consolidated results saved successfully to {args.output_xlsx}!")

if __name__ == "__main__":
    main()
