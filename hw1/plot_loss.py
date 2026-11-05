#!/usr/bin/env python3
"""
Script to train the model and plot the loss function over epochs.
"""
import sys
import matplotlib.pyplot as plt
from sentiment_data import read_sentiment_examples
from models import train_logistic_regression, UnigramFeatureExtractor
from utils import Indexer

# Load training data
train_exs = read_sentiment_examples("data/train.txt")
print(f"Loaded {len(train_exs)} training examples")

# Train model
feat_extractor = UnigramFeatureExtractor(Indexer())
print("Training Logistic Regression with Unigram features...")
model = train_logistic_regression(train_exs, feat_extractor)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(model.loss_history, marker='o', linestyle='-', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Cross-Entropy Loss', fontsize=12)
plt.title('Logistic Regression Loss Function Over Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
output_file = "loss_plot.png"
plt.savefig(output_file, dpi=150)
print(f"\n✓ Plot saved to: {output_file}")
print(f"✓ Final loss: {model.loss_history[-1]:.4f}")
print(f"✓ Initial loss: {model.loss_history[0]:.4f}")
print(f"✓ Loss decreased by: {model.loss_history[0] - model.loss_history[-1]:.4f}")

plt.show()
