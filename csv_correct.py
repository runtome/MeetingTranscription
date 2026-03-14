"""
Fix submission CSV to match Kaggle expected format.
Aligns paths with sample_submission.csv, fixes empty sentences, and ensures LF line endings.

Usage:
    python csv_correct.py --input results.csv --output fixed.csv --sample sample_submission.csv
    python csv_correct.py --input results.csv --output fixed.csv --placeholder "."
    python csv_correct.py --input submission.csv --output fixed.csv --sample test.csv
"""

import argparse
import csv
import io
import re


def main():
    parser = argparse.ArgumentParser(description="Fix submission CSV for Kaggle")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--sample", default=None, help="Test CSV or sample submission from Kaggle (to align paths and row count)")
    parser.add_argument("--placeholder", default=".", help="Replacement for empty sentences (default: '.')")
    args = parser.parse_args()

    # Read input CSV
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        input_header = next(reader)
        input_rows = {row[0]: row[1] for row in reader}

    print(f"Input: {len(input_rows)} rows, columns: {input_header}")

    if args.sample:
        # Read sample submission to get expected paths, column names, and order
        with open(args.sample, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            sample_header = next(reader)
            sample_paths = [row[0] for row in reader]

        print(f"Sample: {len(sample_paths)} rows, columns: {sample_header}")

        # Use input's column names (sample may only have 'path')
        out_header = input_header

        # Try to match input paths to sample paths
        # Build lookup: extract number from path for flexible matching
        def extract_num(path):
            m = re.search(r"(\d+)", path.replace(".mp3", "").split("_")[-1])
            return int(m.group(1)) if m else None

        input_by_num = {}
        for path, sentence in input_rows.items():
            num = extract_num(path)
            if num is not None:
                input_by_num[num] = sentence
            input_by_num[path] = sentence  # also keep exact path

        out_rows = []
        matched = 0
        filled = 0
        for sample_path in sample_paths:
            # Try exact match first
            if sample_path in input_rows:
                sentence = input_rows[sample_path]
                matched += 1
            else:
                # Try number-based match
                num = extract_num(sample_path)
                if num is not None and num in input_by_num:
                    sentence = input_by_num[num]
                    matched += 1
                else:
                    sentence = args.placeholder
                    filled += 1

            # Fix empty sentences
            if not sentence or not sentence.strip():
                sentence = args.placeholder
                filled += 1

            out_rows.append([sample_path, sentence])

        dropped = len(input_rows) - matched
        print(f"Matched: {matched}, Filled with placeholder: {filled}, Dropped extra: {dropped}")
        if dropped > 0:
            extra = set(input_rows.keys()) - set(sample_paths)
            for p in sorted(extra):
                print(f"  Dropped: {p}")
    else:
        # No sample - just fix empty sentences
        out_header = input_header
        out_rows = []
        filled = 0
        for path, sentence in input_rows.items():
            if not sentence or not sentence.strip():
                sentence = args.placeholder
                filled += 1
            out_rows.append([path, sentence])
        print(f"Fixed empty sentences: {filled}")

    # Write output with LF line endings (Linux-compatible for Kaggle)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(out_header)
        writer.writerows(out_rows)

    print(f"Output: {len(out_rows)} rows -> {args.output}")
    print(f"Line endings: LF (Linux-compatible)")


if __name__ == "__main__":
    main()
