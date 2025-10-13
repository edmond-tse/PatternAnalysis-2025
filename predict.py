"""
predict.py - Inference script for BioLaySumm
Shows example usage of the trained model with sample predictions
"""

import torch
from modules import load_model, get_device
from datasets import load_dataset
from evaluate import load

# Setup
device = get_device()
print(f"Using device: {device}")

# Load trained model
print("\nLoading trained model...")
model, tokenizer = load_model('./saved_model', device)
print("Model loaded successfully!")

print("\nLoading validation dataset...")
data = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
test_data = data["validation"]

rouge_metric = load("rouge")

print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)

example_indices = [0, 10, 50, 100, 200]

model.eval()
all_predictions = []
all_references = []

for i, idx in enumerate(example_indices, 1):
    example = test_data[idx]

    prompt = "Simplify this medical report for a patient. Replace technical terms with everyday words and explain what they mean: "
    input_text = prompt + example['radiology_report']

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=512,
        truncation=True
    ).to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    # Decode
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reference = example['layman_report']

    # Calculate ROUGE score
    rouge_scores = rouge_metric.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )

    # Store for overall metrics
    all_predictions.append(prediction)
    all_references.append(reference)

    # Print example
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i}")
    print(f"{'='*80}")
    print(f"\nInput (Radiology Report):")
    print(f"{example['radiology_report']}")
    print(f"\nReference (Ground Truth Layman Report):")
    print(f"{reference}")
    print(f"\nPrediction (Model Generated):")
    print(f"{prediction}")
    print(f"\nROUGE Scores for this example:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

# Calculate average ROUGE across all examples
print(f"\n{'='*80}")
print("OVERALL METRICS (5 examples)")
print(f"{'='*80}")

overall_rouge = rouge_metric.compute(
    predictions=all_predictions,
    references=all_references,
    use_stemmer=True
)

print(f"\nAverage ROUGE Scores:")
print(f"  ROUGE-1: {overall_rouge['rouge1']:.4f}")
print(f"  ROUGE-2: {overall_rouge['rouge2']:.4f}")
print(f"  ROUGE-L: {overall_rouge['rougeL']:.4f}")
print(f"  ROUGE-Lsum: {overall_rouge['rougeLsum']:.4f}")
