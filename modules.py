"""
modules.py - Model components and helper functions
Contains reusable classes and functions for the BioLaySumm project
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class ModelWrapper:
    """Wrapper class for FLAN-T5 model with optional LoRA"""

    def __init__(self, model_name="google/flan-t5-base", use_lora=False, lora_r=8):
        """
        Initialize the model wrapper

        Args:
            model_name: HuggingFace model name
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA rank parameter
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if use_lora:
            self.model = self._apply_lora(lora_r)

    def _apply_lora(self, lora_r):
        """Apply LoRA to the model"""
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()
        return model

    def get_model(self):
        """Return the model"""
        return self.model

    def is_using_lora(self):
        """Return boolean indicating if LoRA is used"""
        return self.use_lora


def get_tokenizer(model_name="google/flan-t5-base"):
    """
    Load and return tokenizer

    Args:
        model_name: HuggingFace model name

    Returns:
        tokenizer: AutoTokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_device():
    """
    Get available device (MPS for Mac, CUDA for GPU, CPU otherwise)

    Returns:
        device: torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Evaluator:
    """Class for evaluation metrics"""

    def __init__(self, tokenizer):
        """
        Initialize evaluator

        Args:
            tokenizer: tokenizer for decoding
        """
        self.tokenizer = tokenizer

    def compute_metrics(self, model, dataloader, device):
        """
        Compute validation loss

        Args:
            model: the model to evaluate
            dataloader: validation dataloader
            device: torch device

        Returns:
            avg_loss: average validation loss
        """
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def generate_sample(self, model, input_text, device, max_length=512):
        """
        Generate sample output for given input

        Args:
            model: the model
            input_text: input text string
            device: torch device
            max_length: max generation length

        Returns:
            generated_text: generated output string
        """
        model.eval()
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def save_model(model, tokenizer, save_path):
    """
    Save model and tokenizer

    Args:
        model: the model to save
        tokenizer: the tokenizer
        save_path: directory path to save
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def load_model(load_path, device):
    """
    Load saved model

    Args:
        load_path: path to saved model
        device: torch device

    Returns:
        model: loaded model
        tokenizer: loaded tokenizer
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model.to(device)
    print(f"Model loaded from {load_path}")
    return model, tokenizer
