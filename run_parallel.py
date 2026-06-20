#!/usr/bin/env python3
"""
Parallel matching runner: splits the input dataset, runs multiple instances
in parallel on the GPU, and merges the results back into a consolidated file.
"""
import os
import sys
import time
import subprocess
import argparse
import shutil
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Parallel Matching Engine Runner")
    parser.add_argument("--input-xlsx", default="input.xlsx", help="Path to input Excel file")
    parser.add_argument("--master-xlsx", default="masterdata.xlsx", help="Path to master Excel file")
    parser.add_argument("--output-xlsx", default="full_matching_results.xlsx", help="Path to save final merged results")
    parser.add_argument("--parts", type=int, default=5, help="Number of parallel processes to run")
    parser.add_argument("--shutdown", action="store_true", default=False, help="Shut down the system after completion")
    args = parser.parse_args()

    t_start = time.time()

    # 1. Verification
    if not os.path.exists(args.input_xlsx):
        print(f"Error: Input file not found: {args.input_xlsx}")
        sys.exit(1)

    master_path = args.master_xlsx
    if not os.path.exists(master_path):
        print(f"Error: Master file not found: {master_path}")
        sys.exit(1)

    print("=" * 80)
    print("PARALLEL MATCHING RUNNER")
    print(f"Input file:     {args.input_xlsx}")
    print(f"Master file:    {master_path}")
    print(f"Output file:    {args.output_xlsx}")
    print(f"Parallel parts: {args.parts}")
    print("=" * 80)

    # 2. Setup temp directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    split_in_dir = os.path.join(base_dir, "split_inputs")
    split_out_dir = os.path.join(base_dir, "split_results")

    # Clean existing
    for d in [split_in_dir, split_out_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 3. Load & Split Input Dataset
    print("\n[1/5] Loading and splitting input dataset...")
    if args.input_xlsx.endswith('.csv'):
        df_input = pd.read_csv(args.input_xlsx)
    else:
        df_input = pd.read_excel(args.input_xlsx)
    total_rows = len(df_input)
    print(f"   ✓ Loaded {total_rows:,} total input rows.")

    chunk_size = (total_rows + args.parts - 1) // args.parts
    print(f"   ✓ Splitting into {args.parts} chunks (~{chunk_size:,} rows each)...")

    chunk_files = []
    result_files = []
    for i in range(args.parts):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        df_chunk = df_input.iloc[start_idx:end_idx]

        chunk_path = os.path.join(split_in_dir, f"input_part_{i+1:02d}.xlsx")
        result_path = os.path.join(split_out_dir, f"result_part_{i+1:02d}.xlsx")

        df_chunk.to_excel(chunk_path, index=False)
        chunk_files.append(chunk_path)
        result_files.append(result_path)

    print(f"   ✓ Generated {len(chunk_files)} split input files.")

    # 4. Launch Subprocesses
    print(f"\n[2/5] Launching {args.parts} parallel matching processes...")
    processes = []
    python_exe = sys.executable or "python3"
    
    # We want to use the virtual environment's python if available
    venv_python = os.path.join(base_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        python_exe = venv_python

    for i in range(args.parts):
        cmd = [
            python_exe,
            os.path.join(base_dir, "run_batch_match.py"),
            "--master-xlsx", master_path,
            "--input-xlsx", chunk_files[i],
            "--output-xlsx", result_files[i]
        ]
        
        # Open separate log files for each process
        log_f = open(os.path.join(base_dir, f"matching_worker_{i+1:02d}.log"), "w")
        print(f"   → Starting Worker {i+1:02d} (PID will be assigned)... logging to matching_worker_{i+1:02d}.log")
        
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((p, log_f))

    # 5. Monitor Processes
    print("\n[3/5] Processing items... Monitoring workers:")
    active_workers = list(range(args.parts))
    while active_workers:
        time.sleep(5)
        for idx in list(active_workers):
            p, log_f = processes[idx]
            status = p.poll()
            if status is not None:
                # Process finished
                log_f.close()
                active_workers.remove(idx)
                if status == 0:
                    print(f"   ✓ Worker {idx+1:02d} completed successfully.")
                else:
                    print(f"   ❌ Worker {idx+1:02d} failed with exit code {status}. Check matching_worker_{idx+1:02d}.log")

    # 6. Merge Results
    print("\n[4/5] Merging results...")
    all_dfs = []
    for r_file in result_files:
        if os.path.exists(r_file):
            try:
                df = pd.read_excel(r_file)
                all_dfs.append(df)
            except Exception as e:
                print(f"   ❌ Error loading result chunk {r_file}: {e}")

    if not all_dfs:
        print("Error: No result files found to merge!")
        sys.exit(1)

    df_merged = pd.concat(all_dfs, ignore_index=True)
    
    # Separate matched and unmatched, sort matched by confidence ascending
    df_matched = df_merged[df_merged['Confidence %'] > 0].sort_values('Confidence %', ascending=True)
    df_unmatched = df_merged[df_merged['Confidence %'] == 0]
    df_final = pd.concat([df_matched, df_unmatched])

    print(f"   ✓ Merged {len(df_final):,} rows.")
    print(f"   ✓ Saving consolidated output to: {args.output_xlsx}")

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

    print("   ✓ Final consolidated Excel saved.")

    # 7. Cleanup
    print("\n[5/5] Cleaning up temporary files...")
    shutil.rmtree(split_in_dir)
    shutil.rmtree(split_out_dir)
    for i in range(args.parts):
        log_path = os.path.join(base_dir, f"matching_worker_{i+1:02d}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
    print("   ✓ Cleanup complete.")

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"TOTAL EXECUTION TIME: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print("=" * 80)

    # 8. Optional Shutdown
    if args.shutdown:
        print("\n!!! Shutdown requested. Turning off VM now !!!")
        subprocess.run(["sudo", "shutdown", "-h", "now"])

if __name__ == "__main__":
    main()
