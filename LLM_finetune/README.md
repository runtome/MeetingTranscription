# LLM Fine-tuning Pipeline for ASR Error Correction

Fine-tune Qwen3-8B using LLaMA Factory to correct Thai ASR (Whisper) output.

**Pipeline**: ASR output (hallucinated) → Fine-tuned LLM → Corrected text

## Prerequisites

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) installed in a conda environment
- Qwen3-8B model weights downloaded to `${PROJECT_PATH}/model/qwen3-8b`
- Two CSVs per split: ASR output and ground truth, each with `path,sentence` columns

## Quick Start

### 1. Prepare Dataset

Convert CSV pairs to LLaMA Factory alpaca JSON format:

```bash
python prepare_dataset.py \
    --asr_csv ../train/asr_output.csv \
    --gt_csv ../train/train.csv \
    --output data/asr_correction_train.json

python prepare_dataset.py \
    --asr_csv ../val/asr_output.csv \
    --gt_csv ../val/val.csv \
    --output data/asr_correction_val.json
```

### 2. Tokenize Dataset

```bash
export PROJECT_PATH=/project/lt200239-thaig/MeetingTranscription
llamafactory-cli train yaml/1_data_process.config.yaml
```

### 3. Train (LoRA SFT)

```bash
FORCE_TORCHRUN=1 llamafactory-cli train yaml/2_lora_sft.config.yaml
```

### 4. Merge Adapter

```bash
llamafactory-cli export yaml/3_merge_adapter.config.yaml
```

### 5. Run Inference

```bash
python inference_llm.py \
    --model ${PROJECT_PATH}/trained_model/qwen3-8b-asr-corrector \
    --input_csv asr_results.csv \
    --output_csv corrected.csv
```

## Running on Lanta (SLURM)

Submit jobs sequentially:

```bash
# Step 1: Prepare data + tokenize
sbatch slurm/1_prepare_data.sh

# Step 2: Train (after step 1 completes)
sbatch slurm/2_train_sft.sh

# Step 3: Merge adapter (after step 2 completes)
sbatch slurm/3_merge_adapter.sh
```

Update `PROJECT_PATH` in each SLURM script before submitting.

## Swapping Models

To use a different model (e.g., Qwen3-4B, Llama-3.1-8B):

1. Change `model_name_or_path` in all 3 YAML configs
2. Change `template` to match (e.g., `llama3` for Llama models)
3. Update folder names in `output_dir` / `export_dir` as desired

## Directory Structure

```
LLM_finetune/
├── prepare_dataset.py           # CSV pair -> LLaMA Factory alpaca JSON
├── dataset_info.json            # LLaMA Factory dataset registry
├── inference_llm.py             # Run inference with fine-tuned model
├── README.md
├── data/                        # Generated JSON datasets (gitignored)
├── yaml/
│   ├── 1_data_process.config.yaml
│   ├── 2_lora_sft.config.yaml
│   └── 3_merge_adapter.config.yaml
├── deepspeed/
│   └── ds_z2_config.json
└── slurm/
    ├── 1_prepare_data.sh
    ├── 2_train_sft.sh
    └── 3_merge_adapter.sh
```
