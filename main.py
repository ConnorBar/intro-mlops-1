import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Using Path objects makes it really easy to modify paths when we refactor
# Normally we would put these in a gloabl file and import them but we will not do that for this project (but feel free too!)
root_directory_path = Path("~/project/intro-mlops-1/").expanduser() 

data_path =  root_directory_path / "data.csv"

print("Loading dataset...")
data = pd.read_csv(data_path)
print(f"Dataset shape: {data.shape}")

print("Preprocessing data...")
# Remove any missing values
data = data.dropna()

# Splitting data into features and target
feature_cols = data.columns[:-1].tolist()
target_col = data.columns[-1]

X = data[feature_cols].values
y = data[target_col].values

# Predicting string labels so encode them
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data 80/20
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Convert to PyTorch tensors (required)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Simple NN declaration
class SimpleNN(nn.Module):
  def __init__(self, input_size, num_classes=3):
    super(SimpleNN, self).__init__()
    self.layer1 = nn.Linear(input_size, 64)
    self.layer2 = nn.Linear(64, 32)
    self.layer3 = nn.Linear(32, num_classes)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    
  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.dropout(x)
    x = self.relu(self.layer2(x))
    x = self.dropout(x)
    x = self.layer3(x)
    return x

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = len(np.unique(y))

model = SimpleNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop!
print("Starting training...")
epochs = 50
train_losses = [] # these are for metric tracking
test_losses = []
accuracies = []

for epoch in range(epochs):
  # Train one epoch
  model.train()
  total_loss = 0
  for batch_X, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  avg_train_loss = total_loss / len(train_loader)

  train_losses.append(avg_train_loss)
  
  # Evaluate on test set
  model.eval()
  with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    # Calculate Accuracy
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    test_losses.append(test_loss.item())
    accuracies.append(accuracy)
  
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}')

print("Training completed!")

plot_directory = root_directory_path / ""

# Save Training/Test Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.savefig(plot_directory / 'training_loss_plot.png')
plt.close() 

# Save Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Time')
plt.legend()
plt.savefig(plot_directory / 'accuracy_plot.png')
plt.close() 

# Save some results to a file
log_dir = root_directory_path / ""
accuracy = accuracies[-1]
with open(log_dir / 'results.txt', 'w') as f:
    f.write(f"Final Test Accuracy: {accuracy}\n")
    f.write(f"Training epochs: {epochs}\n")
    f.write(f"Model architecture: SimpleNN with {input_size} input features\n")

print("All done!") 

