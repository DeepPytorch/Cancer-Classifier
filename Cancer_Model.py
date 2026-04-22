import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub

# ── Dataset ───────────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("erdemtaha/cancer-data")
print("Path to dataset files:", path)

csv_path = os.path.join(path, "cancer_data.csv")
data = pd.read_csv(csv_path)
print(data.head())

# Encode target column if it's categorical (e.g. 'M'/'B' → 1/0)
target_col = data.columns[-1]
if data[target_col].dtype == object:
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col])
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Drop non-numeric columns (e.g. id)
data = data.select_dtypes(include=["number"])

# ── Splits ────────────────────────────────────────────────────────────────────
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

X_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, -1].values,  dtype=torch.float32)
X_test  = torch.tensor(test_data.iloc[:, :-1].values,  dtype=torch.float32)
y_test  = torch.tensor(test_data.iloc[:, -1].values,   dtype=torch.float32)

input_dim = X_train.shape[1]
print(f"Input features: {input_dim}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class CancerModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

