"""
train.py - Training script for BioLaySumm
Contains the code for training, validating, testing, and saving
"""
import torch
import matplotlib.pyplot as plt
import os
import time
from torch.cuda.amp import autocast, GradScaler
from modules import ModelWrapper, get_device, get_tokenizer, save_model
from dataset import BioLaySumm
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load

torch.backends.cudnn.benchmark = True
# Configuration
USE_LORA = True
LORA_R = 16
BATCH_SIZE = 16
MAX_LENGTH = 512

# Setup device
device = get_device()
print(f"Using device: {device}")

# Load model
print("Loading model...")
model_wrapper = ModelWrapper(
    model_name="google/flan-t5-base",
    use_lora=USE_LORA,
    lora_r=LORA_R
)
model = model_wrapper.get_model()
model.to(device)
model.compile()
print(f"Model loaded. LoRA: {model_wrapper.is_using_lora()}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer()

# Load dataset
print("Loading dataset...")
data = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

# Create datasets
train_dataset = BioLaySumm(data["train"].select(range(75000)), tokenizer, MAX_LENGTH)
val_dataset = BioLaySumm(data["validation"], tokenizer, MAX_LENGTH)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Setup complete!")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

EPOCHS = 5
LEARNING_RATE = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()


def train(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader, start=1):
        start = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        end = time.time()

        if (i + 1) % 50 == 0:
            print(f"Batch {i}/{len(loader)} time {end - start}")

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# Main training loop
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)

start_time = time.time()
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    # Train
    train_loss = train(model, train_loader, optimizer, device, scaler)
    train_losses.append(train_loss)
    print(f"  Train Loss: {train_loss:.4f}")

    # Validate
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)
    print(f"  Val Loss:   {val_loss:.4f}")

end_time = time.time()
training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Val Loss:   {val_losses[-1]:.4f}")
print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({training_time:.2f} seconds)")

# Evaluate on test set
print("\n" + "=" * 60)
print("Evaluating on Test Set")
print("=" * 60)

# Create test dataset and loader
test_dataset = BioLaySumm(data["test"], tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

# Calculate test loss
test_loss = validate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")

# Calculate ROUGE scores
print("\nCalculating ROUGE scores...")
rouge = load("rouge")

model.eval()
predictions = []
references = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 100:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate predictions
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4
        )

        # Decode predictions and references
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        predictions.extend(batch_predictions)
        references.extend(batch_references)

# Compute ROUGE scores
result = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,
    use_aggregator=True
)

print("\nROUGE Scores:")
print(f"  ROUGE-1: {result['rouge1']:.4f}")
print(f"  ROUGE-2: {result['rouge2']:.4f}")
print(f"  ROUGE-L: {result['rougeL']:.4f}")
print(f"  ROUGE-Lsum: {result['rougeLsum']:.4f}")

# Plot training curves
print("\n" + "=" * 60)
print("Plotting Training Curves")
print("=" * 60)

plt.figure(figsize=(10, 6))
epochs_range = range(1, EPOCHS + 1)

plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/training_losses.png', dpi=300, bbox_inches='tight')
print("Training loss plot saved to ./plots/training_losses.png")
plt.close()

# Save trained model
print("\n" + "=" * 60)
print("Saving Model")
print("=" * 60)

os.makedirs('./saved_model', exist_ok=True)
save_model(model, tokenizer, './saved_model')
print("Model and tokenizer saved to ./saved_model/")

print("\n" + "=" * 60)
print("Finished")
print("=" * 60)

