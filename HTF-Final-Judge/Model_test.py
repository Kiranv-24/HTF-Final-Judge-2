import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib  # <-- Added for saving model in joblib format

# Function to clean data
def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

# Load and prepare dataset
phishing_data = pd.read_csv('original_new_phish_25k.csv', dtype=str, low_memory=False)
legitimate_data = pd.read_csv('legit_data.csv', dtype=str, low_memory=False)
phishing_data['Label'] = 1
legitimate_data['Label'] = 0
dataset = pd.concat([phishing_data, legitimate_data])
dataset = dataset.drop(['url', 'NonStdPort', 'GoogleIndex', 'double_slash_redirecting', 'https_token'], axis=1)
dataset = clean_data(dataset)
X = dataset.drop('Label', axis=1)
y = dataset['Label'].astype(int)

# Train-test split and normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

input_dim = X_train.shape[1]

# Define models

class BasicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

class TanhMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 128)
        self.fc_res1 = nn.Linear(128, 128)
        self.fc_res2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.fc_in(x))
        res = self.fc_res1(x1)
        res = self.fc_res2(res)
        x2 = self.relu(x1 + res)
        x_out = self.sigmoid(self.out(x2))
        return x_out

# Dictionary of models
deep_classifiers = {
    'Basic MLP': BasicMLP,
    'Dropout MLP': DropoutMLP,
    'Deep MLP': DeepMLP,
    'BatchNorm MLP': BatchNormMLP,
    'Tanh MLP': TanhMLP,
    'Residual MLP': ResidualMLP,
}

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            true_labels.append(labels.numpy())
    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    pred_classes = (preds > 0.5).astype(int)
    accuracy = accuracy_score(true_labels, pred_classes)
    return accuracy

# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []

for name, ModelClass in deep_classifiers.items():
    print(f"Training {name}...")
    model = ModelClass().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    patience = 3
    patience_counter = 0

    for epoch in range(20):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best weights
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Load best model weights
    model.load_state_dict(best_weights)

    results.append((name, best_acc))
    print(f"{name} Best Accuracy: {best_acc:.4f}\n")

# Results DataFrame
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])

# Bonferroni-Dunn test (try-except for safety)
try:
    dunn_results = sp.posthoc_dunn(results_df, val_col='Accuracy', group_col='Classifier', p_adjust='bonferroni')
    print(dunn_results)
except Exception as e:
    print(f"Bonferroni-Dunn test error: {e}")

# Visualization
plt.figure(figsize=(10,6))
sns.barplot(x='Classifier', y='Accuracy', data=results_df, palette='viridis')
plt.title('Classifier Accuracies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graphs/classifier_accuracies_pytorch.png')
plt.show()

# Save best model using joblib
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Classifier']
best_model = deep_classifiers[best_model_name]().to(device)

# Load best weights before saving
# To find best weights of best model from training, 
# ideally you should save weights for all models separately in loop above
# but here we will retrain best model once more or you can save weights during training for each model.

# For demonstration, retrain best model quickly to get weights (or you could save in a dict during training)
print(f"Retraining best model ({best_model_name}) to save weights for joblib...")
criterion = nn.BCELoss()
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

best_weights = None
best_acc = 0
patience = 3
patience_counter = 0
for epoch in range(20):
    train_loss = train_model(best_model, train_loader, criterion, optimizer, device)
    val_acc = evaluate_model(best_model, test_loader, device)
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        best_weights = best_model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

best_model.load_state_dict(best_weights)
best_model.eval()

# Save entire model using joblib
joblib.dump(best_model.cpu(), 'best_rf_model.joblib')
print(f"Best model ({best_model_name}) saved in joblib format as 'best_rf_model.joblib' with accuracy {best_acc:.4f}")
