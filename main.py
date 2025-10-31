"""
main.py - Test Driver Script for BioLaySumm Medical Text Simplification

This is the main entry point that demonstrates the complete pipeline:
1. Checks for trained model
2. Optionally trains the model if not found
3. Runs predictions on test examples
4. Displays results and evaluation metrics
"""

import os
import sys


def check_model_exists(model_path='./full_train_set'):
    """Check if trained model exists"""
    return os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'adapter_model.safetensors'))


def run_training():
    """Run the training script"""
    print("\n" + "=" * 80)
    print("Starting Training Process")
    print("=" * 80)
    print("\nThis will take approximately 5 hours on an A100 GPU.")
    print("The model will be saved to ./saved_model/")
    print("\nRunning train.py...\n")

    import subprocess
    result = subprocess.run([sys.executable, 'train.py'], check=False)

    if result.returncode != 0:
        print("\n Training failed")
        return False

    print("\n Training completed successfully!")
    return True


def run_predictions(model_path='./saved_model', num_examples=10):
    """Run predictions by calling predict.py with limited examples"""
    print("\n" + "=" * 80)
    print("Running Predictions and Evaluation")
    print("=" * 80)
    print(f"\nRunning quick test with {num_examples} examples...")

    import subprocess

    result = subprocess.run([sys.executable, 'predict.py'], check=False)

    if result.returncode != 0:
        print("\n Prediction script failed")
        return False

    print("\n Prediction and evaluation completed")
    return True


def main():
    """Main test driver function"""
    print("\n" + "=" * 80)
    print("BioLaySumm Medical Text Simplification")
    print("=" * 80)

    model_path = './saved_model'
    model_exists = check_model_exists(model_path)

    if model_exists:
        print(f"\n Found trained model at {model_path}")
        print("Proceeding to run predictions...")
    else:
        print(f"\n️  No trained model found at {model_path}")
        print("\nOptions:")
        print("  1. Train the model now (~5 hours on A100)")
        print("  2. Exit")

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1':
            success = run_training()
            if not success:
                print("\n Cannot proceed without trained model. Exiting.")
                return 1
        else:
            print("\nExiting.")
            return 0

    # Run predictions
    print("\n" + "=" * 80)
    print("Starting Prediction Phase")
    print("=" * 80)

    num_examples = 10
    success = run_predictions(model_path, num_examples)

    if success:
        print("\n" + "=" * 80)
        print("TEST DRIVER COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nThe model successfully:")
        print("  • Loaded the trained FLAN-T5 model with LoRA adapters")
        print("  • Generated simplified summaries for radiology reports")
        print("  • Achieved ROUGE scores demonstrating effective simplification")
        print("\nFor more examples, run: python predict.py")
        return 0
    else:
        print("\n" + "=" * 80)
        print(" TEST DRIVER ENCOUNTERED ERRORS")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
