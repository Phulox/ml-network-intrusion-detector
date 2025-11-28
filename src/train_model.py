import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load in dataset

df = pd.read_csv("data/processed/train.csv")

X = df.drop(columns=["label"]).values.astype("float32")
y = df["label"].values.astype("float32")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Normalize input values for consistency and accuracy
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.fit_transform(X_test).astype("float32")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # output probability (0 to 1)
        )
    def forward(self, x):
        return self.net(x)
    
input_dim = X_train.shape[1]
model = MLP(input_dim)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20 

for epoch in range(epochs):
    #prepares model for training (good practice)
    model.train()
    
    #resets gradient of each node to zero 
    optimizer.zero_grad()

    #runs X_train and removes last useless dimension to have outputs.shape() == y_train.shape()
    outputs = model(X_train).squeeze()
    
    #Calculates loss on each output
    loss = criterion(outputs, y_train)

    #BACKPROPAGATION to calculate gradient of each node for corresponding loss
    loss.backward()

    #uses the gradient calculated to re-caliberate each weight for optimal output
    optimizer.step()

    # Accuracy on the test set
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
        test_pred_labels = (test_pred >= 0.5).float()
        acc = (test_pred_labels == y_test).float().mean().item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")


torch.save(model.state_dict(), "model/model.pt")
np.save("model/scaler_mean.npy", scaler.mean_)
np.save("model/scaler_scale.npy", scaler.scale_)
    

with torch.no_grad():
    final_pred = model(X_test).squeeze()
    final_pred_labels = (final_pred >= 0.5).float()
    final_acc = (final_pred_labels == y_test).float().mean().item()

print("Final Test Accuracy:", final_acc)

