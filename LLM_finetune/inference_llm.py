"""
Run inference with a fine-tuned LLM to correct ASR output.
Reads ASR CSV, applies the correction model, and outputs corrected CSV.

Usage:
    python inference_llm.py --model ./qwen3-8b-asr-corrector --input_csv asr_results.csv --output_csv corrected.csv
"""

import argparse
import csv
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยแก้ไขข้อความภาษาไทยจากระบบรู้จำเสียง (ASR) "
    "กรุณาแก้ไขข้อความต่อไปนี้ให้ถูกต้อง โดยรักษาความหมายเดิมไว้"
)


def strip_think_blocks(text):
    """Remove <think>...</think> blocks from Qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser(description="Correct ASR output using fine-tuned LLM")
    parser.add_argument("--model", required=True, help="Path to merged model")
    parser.add_argument("--input_csv", required=True, help="Input CSV with ASR output (path,sentence)")
    parser.add_argument("--output_csv", required=True, help="Output CSV with corrected text (path,sentence)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Using device: {device}")

    # Read input CSV
    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    print(f"Read {len(rows)} rows from {args.input_csv}")

    # Process and write output
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])

        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]

            for path, asr_text in batch:
                idx = i + batch.index((path, asr_text)) + 1

                if not asr_text.strip():
                    writer.writerow([path, ""])
                    print(f"[{idx}/{len(rows)}] {path} -> (empty, skipped)")
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": asr_text},
                ]

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                # Decode only the generated tokens (exclude input)
                generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                corrected = tokenizer.decode(generated_ids, skip_special_tokens=True)
                corrected = strip_think_blocks(corrected).strip()

                writer.writerow([path, corrected])
                print(f"[{idx}/{len(rows)}] {path}")
                print(f"  ASR:       {asr_text}")
                print(f"  Corrected: {corrected}")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
