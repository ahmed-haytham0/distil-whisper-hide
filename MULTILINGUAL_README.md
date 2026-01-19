# Multilingual Distil-Whisper Training Guide 🌍

This guide explains how to train a Distil-Whisper model on a **multilingual dataset** (mixed languages).

This is designed for ease of use. Follow these 2 steps.

## Prerequisites

Ensure you have your environment set up (see main `README.md`).
Your dataset must have a column (default: `language`) specifying the language for each audio file (e.g., "ar", "en", "fr").

---

## Step 1: Pseudo-Labelling

We first generate "pseudo-labels" (transcriptions) using a large teacher model.

**Key Feature**: This script now supports **dynamic languages**. It looks at the `language` column in your dataset and forces the model to use that language for transcription.

### Command

```bash
accelerate launch training/run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "your_org/your_dataset" \
  --dataset_split_name "train" \  # <--- Specifying split is safer if you don't have validation/test
  --language_column_name "language" \  # <--- IMPORTANT: Column name for language codes
  --output_dir "./pseudo_labels" \
  --per_device_eval_batch_size 8 \
  --max_label_length 256 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 4 \
  --report_to "wandb" \
  --wandb_project "distil-whisper-multilingual" \
  --push_to_hub
```

**What this does:**
1.  Loads your dataset.
2.  For each sample, reads the language (e.g., "ar").
3.  Generates a transcription using the teacher model (forced to "ar").
4.  Saves a dataset with a format like: `<|startoftranscript|><|ar|><|transcribe|><|notimestamps|> Your text here...`

---

## Step 2: Training (Distillation)

Now we train the student model to mimic the teacher.

**Encoder Learning**: By default, the Encoder is **NOT frozen**, meaning it will learn and adapt to your data. This is usually what you want for best performance.

### Command

```bash
accelerate launch training/run_distillation.py \
  --model_name_or_path "./my-student-model-init" \
  --teacher_model_name_or_path "openai/whisper-large-v3" \
  --train_dataset_name "your_org/pseudo_labels_from_step_1" \
  --train_dataset_config_name "default" \
  --output_dir "./distil-whisper-multilingual-trained" \
  --per_device_train_batch_size 32 \
  --learning_rate 0.0001 \
  --warmup_steps 50 \
  --max_steps 5000 \
  --freeze_encoder False \  # <--- ENSURE THIS IS FALSE (Default) to update the encoder!
  --freeze_decoder False \  # <--- Default is False
  --gradient_checkpointing \
  --fp16 \
  --push_to_hub
```

**Important Notes:**
*   **Do NOT set `--language "en"`**. Leave it empty. The model will pick up the language from the special tokens in your data (generated in Step 1).
*   **Encoder Training**: If you want to freeze the encoder (to save memory/speed, but learning less), set `--freeze_encoder True`. But for best results, leave it as `False`.

---

## Troubleshooting

*   **Repository Not Found**: Make sure you have created the target repository on GitHub before pushing.
*   **Language Codes**: Ensure your dataset's language column uses standard Whisper codes (e.g., "ar", "en", "de", "es").

**Happy Training! 🚀**
