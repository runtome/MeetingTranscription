"""
Prepare LLaMA Factory dataset from ASR output and ground truth CSVs.
Joins on the 'path' column and outputs alpaca-format JSON for fine-tuning.

Usage:
    python prepare_dataset.py --asr_csv asr_results.csv --gt_csv ground_truth.csv --output data/asr_correction_train.json
"""

import argparse
import csv
import json


SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยแก้ไขข้อความภาษาไทยจากระบบรู้จำเสียง (ASR) "
    "กรุณาแก้ไขข้อความต่อไปนี้ให้ถูกต้อง โดยรักษาความหมายเดิมไว้"
)


def read_csv(path):
    """Read CSV with (path, sentence) columns and return dict keyed by path."""
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                rows[row[0]] = row[1]
    print(f"Read {len(rows)} rows from {path} (columns: {header})")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare LLaMA Factory dataset from ASR + ground truth CSVs")
    parser.add_argument("--asr_csv", required=True, help="CSV with ASR output (path,sentence)")
    parser.add_argument("--gt_csv", required=True, help="CSV with ground truth (path,sentence)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    asr_rows = read_csv(args.asr_csv)
    gt_rows = read_csv(args.gt_csv)

    # Inner join on path
    asr_keys = set(asr_rows.keys())
    gt_keys = set(gt_rows.keys())
    common_keys = asr_keys & gt_keys

    asr_only = asr_keys - gt_keys
    gt_only = gt_keys - asr_keys

    if asr_only:
        print(f"WARNING: {len(asr_only)} paths in ASR CSV but not in ground truth")
        for p in sorted(asr_only)[:5]:
            print(f"  {p}")
        if len(asr_only) > 5:
            print(f"  ... and {len(asr_only) - 5} more")

    if gt_only:
        print(f"WARNING: {len(gt_only)} paths in ground truth but not in ASR CSV")
        for p in sorted(gt_only)[:5]:
            print(f"  {p}")
        if len(gt_only) > 5:
            print(f"  ... and {len(gt_only) - 5} more")

    # Build alpaca-format dataset
    dataset = []
    for path in sorted(common_keys):
        asr_text = asr_rows[path].strip()
        gt_text = gt_rows[path].strip()

        # Skip empty pairs
        if not asr_text or not gt_text:
            continue

        dataset.append({
            "instruction": SYSTEM_PROMPT,
            "input": asr_text,
            "output": gt_text,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nCreated {len(dataset)} training examples -> {args.output}")


if __name__ == "__main__":
    main()
