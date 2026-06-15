#!/usr/bin/env python3
"""
Utility to split input.xlsx into N equal parts for parallel processing across multiple PCs.
"""
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Split input Excel file into N parts")
    parser.add_argument("--input-xlsx", default="/Volumes/ssd embolo/Games/input.xlsx", help="Path to input Excel file")
    parser.add_argument("--parts", type=int, default=10, help="Number of parts to split into")
    parser.add_argument("--output-dir", default="/Volumes/ssd embolo/Games/split_inputs", help="Directory to save split parts")
    args = parser.parse_args()

    if not os.path.exists(args.input_xlsx):
        print(f"Error: Input file not found at: {args.input_xlsx}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading {args.input_xlsx}...")
    df = pd.read_excel(args.input_xlsx)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows.")

    # Split rows
    chunk_size = (total_rows + args.parts - 1) // args.parts
    print(f"Splitting into {args.parts} parts (~{chunk_size:,} rows each)...")

    for i in range(args.parts):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]
        
        part_num = i + 1
        output_filename = f"input_part_{part_num:02d}.xlsx"
        output_path = os.path.join(args.output_dir, output_filename)
        
        df_chunk.to_excel(output_path, index=False)
        print(f"  ✓ Saved Part {part_num:02d} ({len(df_chunk):,} rows) -> {output_path}")

    print("\nDone! Copy each part to a separate PC along with the masterdata.xlsx and matching engine code.")

if __name__ == "__main__":
    main()
