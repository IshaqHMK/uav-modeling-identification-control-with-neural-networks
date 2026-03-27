# ================================================
# Quadcopter Attitude Identification with an MLP
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.patches as patches

# Standard figure size for all plots
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C1_"  # prefix added to all saved figures

# Set experimental data .mat file path
mat_path = r'I:\My Drive\nn_quad_identfication\Python\nn_quad_codes\quad_AGD__01_05_25_11_06_38.mat'
if not os.path.isfile(mat_path):
    raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust mat_path to your .mat file")

print(f"Loading MAT file: {mat_path}")
data = sio.loadmat(mat_path)

# Extract and preprocess data
# original ctrl_full has columns [U1, U2, U3, U4]
ctrl_full   = data['control_input_data']    # shape (27000,4)
ctrl        = ctrl_full[:, 1:4]             # keep U2,U3,U4 only → shape (27000,3)
att_rad     = data['attitude_data']          # (27000,3) in radians
time_vec    = data['sim_times'].flatten()    # (27000,)
# convert to degrees
att_deg  = np.rad2deg(att_rad)            # (27000,3)

print(f"ctrl (U2-U4) shape: {ctrl.shape}")
print(f"att_deg       shape: {att_deg.shape}")

# 2. Build one-step supervised dataset
# X_t = [u2, u3, u4, roll, pitch, yaw]_t   → 6 features
# Y_t = [roll, pitch, yaw]_{t+1}          → 3 targets
X = np.hstack([ctrl[:-1, :], att_deg[:-1, :]])  # → (N-1, 6)
Y = att_deg[1:, :]                              # → (N-1, 3)
time_Y = time_vec[1:]                           # align time for Y

# 3. Train/test split + scaling
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False)

scaler_X = StandardScaler().fit(X_train)
scaler_Y = StandardScaler().fit(Y_train)

X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)
Y_train_s = scaler_Y.transform(Y_train)
Y_test_s  = scaler_Y.transform(Y_test)

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t  = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Y_train_t  = torch.tensor(Y_train_s, dtype=torch.float32, device=device)
X_test_t   = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
Y_test_t   = torch.tensor(Y_test_s,  dtype=torch.float32, device=device)

# 4. Define the MLP
class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, depth=2, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth-1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

torch.manual_seed(0)
model = MLP(inp_dim=6, hid_dim=8, out_dim=3, depth=2, dropout=0.2).to(device)

print(model)

# 5. Train with standard settings (baseline)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # keep as reference (Adam)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # stochastic Gradient Descent (SGD)

epochs = 100 # fixed number of epochs

train_hist, test_hist = [], []

for ep in range(1, epochs + 1):
    # Training step
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss   = criterion(y_pred, Y_train_t)
    loss.backward()
    optimizer.step()

    # Evaluation step
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_t)
        test_loss   = criterion(y_test_pred, Y_test_t)

    # Store losses
    train_hist.append(loss.item())
    test_hist.append(test_loss.item())
    print(f"Epoch {ep:3d}  Train MSE {loss.item():.4e}  Test MSE {test_loss.item():.4e}")


# 6. Final evaluation
model.eval()
with torch.no_grad():
    Y_pred_s = model(X_test_t).cpu().numpy()

Y_pred = scaler_Y.inverse_transform(Y_pred_s)
Y_true = Y_test

rmse = np.sqrt(mean_squared_error(Y_true, Y_pred, multioutput='raw_values'))
print(f"Final RMSE [deg]: roll={rmse[0]:.3f} pitch={rmse[1]:.3f} yaw={rmse[2]:.3f}")

# 6b. Save trained direct model (weights + scalers + config)
# ----------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
ckpt_path = os.path.join("models", f"{SAVE_PREFIX}mlp_direct_model.pt")
checkpoint = {
    "model_class": "MLP",
    "model_kwargs": {
        "inp_dim": 6,
        "hid_dim": 8,
        "out_dim": 3,
        "depth": 2,
        "dropout": 0.2,
    },
    "state_dict": model.state_dict(),
    "scaler_X": scaler_X,
    "scaler_Y": scaler_Y,
    "feature_names": ["u2", "u3", "u4", "roll_deg", "pitch_deg", "yaw_deg"],
    "target_names": ["roll_deg", "pitch_deg", "yaw_deg"],
    "train_history": train_hist,
    "test_history": test_hist,
    "rmse_deg": rmse.tolist() if hasattr(rmse, "tolist") else rmse,
    "pytorch_version": torch.__version__,
}
torch.save(checkpoint, ckpt_path)
print(f"Saved trained model checkpoint to: {ckpt_path}")

# --------------------------------------
#                Plots
# --------------------------------------

# 
# time axis for all test points
t_full = time_Y[len(X_train):]   # length = len(Y_test)

# predict full sequence
X_all_s       = scaler_X.transform(X)                          # X is from step 2, shape (N-1,6)
X_all_t       = torch.tensor(X_all_s, dtype=torch.float32, device=device)
model.eval()
with torch.no_grad():
    Y_all_pred_s = model(X_all_t).cpu().numpy()

# back to degrees
Y_all_pred = scaler_Y.inverse_transform(Y_all_pred_s)
Y_all_true = Y                                         # shape (N-1,3)
time_full  = time_Y                                    # sim_times[1:], shape (N-1,)

def draw_mlp_structure(ax, input_size, hidden_size1, hidden_size2, output_size):
    ax.clear()
    ax.set_xlim(0, 4)
    max_neurons = max(input_size, hidden_size1, hidden_size2, output_size)
    ax.set_ylim(0, max_neurons + 1)
    ax.axis('off')

    # Helper to get vertical positions for neurons
    def get_positions(n, x):
        spacing = (ax.get_ylim()[1] - 1) / (n + 1)
        return [(x, (i + 1) * spacing) for i in range(n)]

    # Layers positions
    input_pos = get_positions(input_size, 0.5)
    hidden1_pos = get_positions(hidden_size1, 1.5)
    hidden2_pos = get_positions(hidden_size2, 2.5)
    output_pos = get_positions(output_size, 3.5)

    # Draw neurons for each layer
    for (x, y) in input_pos:
        ax.add_patch(patches.Circle((x, y), 0.1, color='skyblue'))
    for (x, y) in hidden1_pos:
        ax.add_patch(patches.Circle((x, y), 0.1, color='orange'))
    for (x, y) in hidden2_pos:
        ax.add_patch(patches.Circle((x, y), 0.1, color='orange'))
    for (x, y) in output_pos:
        ax.add_patch(patches.Circle((x, y), 0.1, color='limegreen'))

    # Draw connections
    for (x1, y1) in input_pos:
        for (x2, y2) in hidden1_pos:
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)
    for (x1, y1) in hidden1_pos:
        for (x2, y2) in hidden2_pos:
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)
    for (x1, y1) in hidden2_pos:
        for (x2, y2) in output_pos:
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)


# 1) Test-set prediction vs true (degrees)
fig1 = plt.figure(num="Test-set: True vs Pred", figsize=PLOT_FIGSIZE)
for i, label in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
    plt.subplot(3, 1, i+1)
    plt.plot(t_full, Y_true[:, i],  label='True', linewidth=1)
    plt.plot(t_full, Y_pred[:, i],  '--', label='Pred', linewidth=1)
    plt.ylabel(label)
    plt.grid(alpha=0.3)
plt.legend(loc='upper right')
plt.xlabel('Time [s]')
plt.tight_layout()
fig1.savefig(f"{SAVE_PREFIX}fig_mlp_test_predictions.png", dpi=300)

# 2) Training and test loss curves
fig2 = plt.figure(num="Learning Curves", figsize=PLOT_FIGSIZE)
plt.plot(train_hist, label='train')
plt.plot(test_hist, label='test')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.grid(alpha=0.3)
plt.legend()
plt.title('Learning Curves')
plt.tight_layout()
fig2.savefig(f"{SAVE_PREFIX}fig_mlp_learning_curves.png", dpi=300)

# 3) Full sequence prediction vs true
fig3 = plt.figure(num="Full Sequence: True vs Pred", figsize=PLOT_FIGSIZE)
for i, label in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
    plt.subplot(3,1,i+1)
    plt.plot(time_full, Y_all_true[:,i],  label='True', linewidth=1)
    plt.plot(time_full, Y_all_pred[:,i], '--', label='Pred', linewidth=1)
    plt.ylabel(label)
    plt.grid(alpha=0.3)
plt.legend(loc='upper right')
plt.xlabel('Time [s]')
plt.tight_layout()
fig3.savefig(f"{SAVE_PREFIX}fig_mlp_full_sequence.png", dpi=300)

# 4) MLP architecture diagram
fig4, ax = plt.subplots(num="MLP Architecture", figsize=PLOT_FIGSIZE)
draw_mlp_structure(ax, input_size=6, hidden_size1=8, hidden_size2=8, output_size=3)
plt.title("MLP Architecture for Quadcopter Attitude Modeling", pad=20)
fig4.tight_layout()
fig4.savefig(f"{SAVE_PREFIX}mlp_quadcopter_architecture.png", dpi=300)

# Show all open figures at once (blocks until windows are closed)
plt.show()
