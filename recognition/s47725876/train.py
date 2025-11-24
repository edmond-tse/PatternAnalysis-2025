"""
train.py - Training script for BioLaySumm
Contains the code for training, validating, testing, and saving
"""
import torch
import matplotlib.pyplot as plt
import os
import time
from modules import ModelWrapper, get_device, get_tokenizer, save_model
from dataset import BioLaySumm
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load
import math

# Performance optimizations
# torch.backends.cudnn.benchmark = True enables cuDNN autotuner for faster training
# but results may vary slightly between runs (~1-2% ROUGE variance)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# For exact reproducibility, uncomment below and comment above:
# import random
# import numpy as np
# random.seed(3710)
# np.random.seed(3710)
# torch.manual_seed(3710)
# torch.cuda.manual_seed_all(3710)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Configuration
USE_LORA = True
LORA_R = 16
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 5
LEARNING_RATE = 5e-5


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader, start=1):
        start = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        end = time.time()

        if i % 50 == 0:
            print(f"Batch {i}/{len(loader)} - Time: {end - start:.4f}s - Loss: {loss.item():.4f}")

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


if __name__ == '__main__':
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
    train_dataset = BioLaySumm(data["train"].select(range(math.floor(len(data["train"]) * 0.5))), tokenizer, MAX_LENGTH)
    val_dataset = BioLaySumm(data["validation"], tokenizer, MAX_LENGTH)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)

    print("Setup complete!")

    # Show trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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
        epoch_start = time.time()
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"  Train Loss: {train_loss:.4f} - Epoch Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"  Val Loss:   {val_loss:.4f}")

        # checkpoint
        checkpoint_dir = f'./checkpoint_epoch_{epoch + 1}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_model(model, tokenizer, checkpoint_dir)
        print(f"  Checkpoint saved to {checkpoint_dir}/")

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
    test_dataset = BioLaySumm(data["validation"], tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    print(f"Test samples: {len(test_dataset)}")

    # Calculate test loss
    test_loss = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print("\nNote: For ROUGE scores, run: python calculate_rouge.py")

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
