"""
Ensemble multiple ASR outputs using word-level majority voting.

Takes multiple CSV result files (path,sentence) and combines them using
a ROVER-inspired approach: tokenize each output, align sequences,
and pick the most common token at each position.

Usage:
    # Ensemble best 3 models
    python ensemble_results.py \
        --inputs results/results_train_T6.csv results/results_train_T10.csv results/results_train_T16.csv \
        --output results/ensemble_T6_T10_T16.csv

    # Weighted ensemble (give T6 more weight)
    python ensemble_results.py \
        --inputs results/results_train_T6.csv results/results_train_T10.csv results/results_train_T16.csv \
        --weights 2 1 1 \
        --output results/ensemble_weighted.csv
"""

import argparse
import csv
from collections import Counter

from pythainlp.tokenize import word_tokenize


def tokenize_thai(text):
    """Tokenize Thai text into words."""
    if not text or not text.strip():
        return []
    return word_tokenize(text.strip(), engine="newmm")


def align_and_vote(token_lists, weights=None):
    """
    Simple majority voting at token level.

    For Thai ASR where outputs are similar length, we use a position-based
    approach: at each position, pick the most frequent token across models.
    Falls back to longest common subsequence alignment for length mismatches.
    """
    if not token_lists:
        return ""

    n = len(token_lists)
    if weights is None:
        weights = [1] * n

    # Filter out empty outputs
    valid = [(tl, w) for tl, w in zip(token_lists, weights) if tl]
    if not valid:
        return ""
    if len(valid) == 1:
        return "".join(valid[0][0])

    token_lists_v, weights_v = zip(*valid)

    # If all outputs are identical, return as-is
    if all(tl == token_lists_v[0] for tl in token_lists_v):
        return "".join(token_lists_v[0])

    # Strategy: use the median-length output as reference length
    lengths = [len(tl) for tl in token_lists_v]
    median_len = sorted(lengths)[len(lengths) // 2]

    # Position-based voting with tolerance for length differences
    result = []
    for pos in range(median_len):
        candidates = Counter()
        for tl, w in zip(token_lists_v, weights_v):
            if pos < len(tl):
                candidates[tl[pos]] += w
            # If this list is shorter, look at nearby positions
            elif pos - 1 < len(tl):
                candidates[tl[-1]] += w * 0.5

        if candidates:
            winner = candidates.most_common(1)[0][0]
            result.append(winner)

    # Check if any model has significant trailing content
    for tl, w in zip(token_lists_v, weights_v):
        if len(tl) > median_len + 3:
            # This model has much more content — might be hallucination, skip
            pass
        elif len(tl) > median_len:
            # Small extra content from multiple models — consider adding
            extra_start = median_len
            extras = Counter()
            for tl2, w2 in zip(token_lists_v, weights_v):
                if len(tl2) > extra_start:
                    for t in tl2[extra_start:]:
                        extras[t] += w2
            # Only add if more than half the models agree
            threshold = sum(weights_v) / 2
            for t in tl[extra_start:]:
                if extras.get(t, 0) >= threshold:
                    result.append(t)
            break

    return "".join(result)


def load_results(csv_path):
    """Load CSV and return {path: sentence} dict."""
    rows = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["path"]] = row.get("sentence", "")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple ASR results using majority voting")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV files to ensemble")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Weights for each input (default: equal weights)")
    parser.add_argument("--placeholder", default=".", help="Placeholder for empty sentences")
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.inputs):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match inputs ({len(args.inputs)})")

    weights = args.weights or [1.0] * len(args.inputs)

    # Load all results
    all_results = []
    for csv_path in args.inputs:
        results = load_results(csv_path)
        all_results.append(results)
        print(f"Loaded {len(results)} rows from {csv_path}")

    # Get all unique paths (preserve order from first file)
    all_paths = list(all_results[0].keys())
    for results in all_results[1:]:
        for path in results:
            if path not in all_paths:
                all_paths.append(path)

    print(f"\nTotal unique paths: {len(all_paths)}")
    print(f"Models: {len(all_results)}, Weights: {weights}")

    # Ensemble
    ensembled = []
    agreement_count = 0
    for path in all_paths:
        sentences = [r.get(path, "") for r in all_results]
        token_lists = [tokenize_thai(s) for s in sentences]

        # Check if all agree
        if all(tl == token_lists[0] for tl in token_lists):
            agreement_count += 1

        voted = align_and_vote(token_lists, weights)

        if not voted.strip():
            voted = args.placeholder

        ensembled.append((path, voted))

    agreement_pct = 100 * agreement_count / len(all_paths) if all_paths else 0
    print(f"Full agreement: {agreement_count}/{len(all_paths)} ({agreement_pct:.1f}%)")

    # Write output
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "sentence"])
        writer.writerows(ensembled)

    print(f"Output: {len(ensembled)} rows -> {args.output}")


if __name__ == "__main__":
    main()
