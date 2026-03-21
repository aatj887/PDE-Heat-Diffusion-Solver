import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE   = "heat_data.csv"
MODEL_FILE  = "heat_nn.pt"

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC = 0.15 (implicit remainder)

BATCH_SIZE  = 64
LR          = 1e-3
MAX_EPOCHS  = 300
PATIENCE    = 25   # early-stopping patience (epochs without val improvement)

RANDOM_SEED = 0
NX          = 51   # spatial points (must match generate_data.py)

# ── 1. Load & filter data ─────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_FILE)

# The CSV stores every snapshot; we only want the *final* one per simulation.
# The last snapshot corresponds to the maximum recorded time for each sample_id.
last_t = df.groupby("sample_id")["t"].transform("max")
df_final = df[df["t"] == last_t].reset_index(drop=True)

print(f"  Total rows in CSV : {len(df):,}")
print(f"  Rows after filter : {len(df_final):,}  (one per simulation, final snapshot)")
print(f"  Final snapshot at : t = {df_final['t'].iloc[0]:.2f} s\n")

# ── 2. Build input (X) and target (y) matrices ────────────────────────────────
u0_cols  = [f"u0_{i}" for i in range(NX)]
u_t_cols = [f"u_{i}"  for i in range(NX)]

# Input: log10(alpha) + u0 profile  →  shape (N, 1 + NX)
# We log-transform alpha because it spans 3 orders of magnitude (1e-7 … 1e-4).
log_alpha = np.log10(df_final["alpha"].values).reshape(-1, 1)
u0        = df_final[u0_cols].values                    # (N, 51)

X = np.hstack([log_alpha, u0]).astype(np.float32)      # (N, 52)
y = df_final[u_t_cols].values.astype(np.float32)       # (N, 51)

N = len(X)
print(f"Input  shape : {X.shape}  [log10(alpha), u0_0 … u0_50]")
print(f"Target shape : {y.shape}  [u_0 … u_50 at final time]\n")

# ── 3. Train / val / test split ───────────────────────────────────────────────
# We shuffle by sample index (each row is an independent simulation).
rng   = np.random.default_rng(RANDOM_SEED)
idx   = rng.permutation(N)

n_train = int(TRAIN_FRAC * N)
n_val   = int(VAL_FRAC   * N)

idx_train = idx[:n_train]
idx_val   = idx[n_train : n_train + n_val]
idx_test  = idx[n_train + n_val:]

print(f"Split  →  train: {len(idx_train)}  |  val: {len(idx_val)}  |  test: {len(idx_test)}")

# ── 4. Normalisation (fit on train only, apply to all) ────────────────────────
# Standard scaling: z = (x − μ) / σ  per feature.
X_train_raw = X[idx_train]
y_train_raw = y[idx_train]

X_mean, X_std = X_train_raw.mean(0), X_train_raw.std(0) + 1e-8
y_mean, y_std = y_train_raw.mean(0), y_train_raw.std(0) + 1e-8

def normalise_X(arr): return (arr - X_mean) / X_std
def normalise_y(arr): return (arr - y_mean) / y_std
def denormalise_y(arr): return arr * y_std + y_mean

X_tr = normalise_X(X[idx_train])
X_va = normalise_X(X[idx_val])
X_te = normalise_X(X[idx_test])

y_tr = normalise_y(y[idx_train])
y_va = normalise_y(y[idx_val])
y_te = normalise_y(y[idx_test])

# ── 5. PyTorch DataLoaders ────────────────────────────────────────────────────
def make_loader(Xn, yn, shuffle):
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr, y_tr, shuffle=True)
val_loader   = make_loader(X_va, y_va, shuffle=False)

# ── 6. Model ──────────────────────────────────────────────────────────────────
# Architecture: 4-hidden-layer MLP with 256 units per layer.
#
# Why this shape?
#   • The IC-to-final-state mapping is spatially non-local (heat diffuses
#     across the whole rod), so we need depth to capture those interactions.
#   • 4 × 256 ≈ 200 K parameters; with 3 500 training samples the ratio is
#     ~17 samples/parameter — acceptable but tight, so we add BatchNorm and
#     a small Dropout (p=0.1) to regularise.
#   • Going wider (512) or deeper (6 layers) risks overfitting at this data
#     scale; going shallower loses expressivity.
#   • Output layer is linear — temperatures are unbounded.

class HeatMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = n_in
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


n_in  = X_tr.shape[1]   # 52  (1 log-alpha + 51 u0 values)
n_out = y_tr.shape[1]   # 51  (final temperature profile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = HeatMLP(n_in, n_out).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: 4 × 256 MLP  |  {total_params:,} trainable parameters")
print(f"Device: {device}\n")

# ── 7. Training ───────────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=10
)

best_val_loss  = float("inf")
epochs_no_impr = 0
train_losses, val_losses = [], []

print("Training …")
for epoch in range(1, MAX_EPOCHS + 1):
    # ── train ──
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        running += loss.item() * len(xb)
    train_loss = running / len(idx_train)

    # ── validate ──
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.from_numpy(X_va).to(device))
        val_loss  = criterion(val_preds, torch.from_numpy(y_va).to(device)).item()

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        epochs_no_impr = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "X_mean": X_mean, "X_std": X_std,
            "y_mean": y_mean, "y_std": y_std,
            "n_in": n_in, "n_out": n_out,
        }, MODEL_FILE)
    else:
        epochs_no_impr += 1

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}  train MSE={train_loss:.5f}  val MSE={val_loss:.5f}"
              f"  lr={optimiser.param_groups[0]['lr']:.2e}")

    if epochs_no_impr >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

# ── 8. Evaluation on held-out test set ───────────────────────────────────────
checkpoint = torch.load(MODEL_FILE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    y_te_pred_norm = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

y_te_pred = denormalise_y(y_te_pred_norm)
y_te_true = y[idx_test]

mse_test  = float(np.mean((y_te_pred - y_te_true) ** 2))
rmse_test = float(np.sqrt(mse_test))
mae_test  = float(np.mean(np.abs(y_te_pred - y_te_true)))

print(f"\nTest set results ({len(idx_test)} samples)")
print(f"  MSE  : {mse_test:.4f}  (°C²)")
print(f"  RMSE : {rmse_test:.4f} (°C)")
print(f"  MAE  : {mae_test:.4f}  (°C)")
print(f"\nBest model saved to '{MODEL_FILE}'")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
x_grid = np.linspace(0, 1, NX)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(train_losses, label="Train MSE")
axes[0].plot(val_losses,   label="Val MSE")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE (normalised)")
axes[0].set_title("Training curve")
axes[0].legend()
axes[0].grid(True)

# A few prediction vs ground-truth comparisons
n_examples = 4
for k in range(n_examples):
    axes[1].plot(x_grid, y_te_true[k],  color=f"C{k}", lw=1.5, label=f"True {k}")
    axes[1].plot(x_grid, y_te_pred[k],  color=f"C{k}", lw=1.5, ls="--", label=f"Pred {k}")

axes[1].set_xlabel("Position x (m)")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Prediction vs Ground Truth (test samples)")
axes[1].legend(fontsize=7)
axes[1].grid(True)

plt.tight_layout()
plt.show()
