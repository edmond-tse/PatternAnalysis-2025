"""
calculate_rouge.py - Two-phase pipeline for generation (GPU) and evaluation (CPU)
"""
import torch
import time
from modules import load_model, get_device
from dataset import BioLaySumm
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load

# Configuration
CHECKPOINT_PATH = './saved_model'
BATCH_SIZE = 16
MAX_LENGTH = 512

print("=" * 60)
print("ROUGE Score Calculation (Two-phase: GPU → CPU)")
print("=" * 60)

# ---------------------- #
# Phase 1: Generation
# ---------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model, tokenizer = load_model(CHECKPOINT_PATH, device)
model.eval()

# Load dataset
data = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
test_dataset = BioLaySumm(data["validation"], tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTest samples: {len(test_dataset)}")

# Start timing
t_start = time.time()
predictions, references = [], []

print("\n[Phase 1] Generating summaries on GPU...\n")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} / {len(test_dataset)} samples...")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        # Decode and collect
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(preds)
        references.extend(refs)

t_gen_end = time.time()
gen_time_min = (t_gen_end - t_start) / 60
print(f"\nGeneration complete ({len(predictions)} samples)")
print(f"Generation time: {gen_time_min:.2f} min")

del model
torch.cuda.empty_cache()

# ---------------------- #
# Phase 2: ROUGE Eval
# ---------------------- #
print("\n[Phase 2] Computing ROUGE on CPU...\n")

rouge = load("rouge")

# Move data to CPU explicitly
predictions = [p.strip() for p in predictions]
references = [r.strip() for r in references]

t_eval_start = time.time()
result = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,
    use_aggregator=True
)
t_eval_end = time.time()
eval_time_min = (t_eval_end - t_eval_start) / 60

# Summary of Results
print("=" * 60)
print("ROUGE Scores (evaluated on CPU):")
print("=" * 60)
print(f"  ROUGE-1: {result['rouge1']:.4f}")
print(f"  ROUGE-2: {result['rouge2']:.4f}")
print(f"  ROUGE-L: {result['rougeL']:.4f}")
print(f"  ROUGE-Lsum: {result['rougeLsum']:.4f}")
print("=" * 60)
print(f"Generation time: {gen_time_min:.2f} min")
print(f"ROUGE eval time: {eval_time_min:.2f} min")
print(f"Total runtime: {gen_time_min + eval_time_min:.2f} min")
print("=" * 60)
