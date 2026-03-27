# ================================================
# C14: MLP Direct Model (rad units) with PSO init
# - Same as C10 (direct one-step model in rad / rad/s)
# - Adds a lightweight PSO stage to initialize ONLY the final layer
# - Then trains full model with Adam as before
# - Plots in degrees for readability; metrics in rad / rad/s
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


# ----------------------------
# Config
# ----------------------------
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C14_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'

# PSO settings (kept conservative — small and fast)
USE_PSO_INIT = True
PSO_PARTICLES = 24
PSO_ITERS = 40
PSO_SUBSET = 2000   # number of train samples to evaluate PSO objective
PSO_W = 0.7         # inertia
PSO_C1 = 1.5        # cognitive
PSO_C2 = 1.5        # social


def load_and_build_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")
    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    # Controls [U1,U2,U3,U4] -> keep U2..U4
    ctrl_full = data['control_input_data']    # (N,4)
    ctrl      = ctrl_full[:, 1:4]             # (N,3)

    # Attitude and rates already in radians / rad/s
    att_rad   = data['attitude_data']         # (N,3) [phi,theta,psi]
    if 'gyro_data' not in data:
        raise KeyError("MAT does not contain 'gyro_data' (expected p,q,r in rad/s)")
    pqr_rad_s = data['gyro_data']             # (N,3) [p,q,r]

    time_vec = data['sim_times'].ravel()

    # One-step dataset
    X = np.hstack([ctrl[:-1, :], att_rad[:-1, :], pqr_rad_s[:-1, :]])  # (N-1,9)
    Y = np.hstack([att_rad[1:, :],  pqr_rad_s[1:, :]])                 # (N-1,6)
    time_Y = time_vec[1:]
    return X, Y, time_Y


class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, depth=2, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
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


def get_last_linear(model: nn.Module) -> nn.Linear:
    last_lin = None
    for m in reversed(list(model.net)):
        if isinstance(m, nn.Linear):
            last_lin = m
            break
    if last_lin is None:
        raise RuntimeError("No Linear layer found in model.net")
    return last_lin


def layer_to_vector(W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([W.flatten(), b.flatten()])


def vector_to_layer(vec: torch.Tensor, W_shape, b_shape):
    w_num = np.prod(W_shape)
    W = vec[:w_num].view(W_shape)
    b = vec[w_num:w_num + b_shape[0]].view(b_shape)
    return W, b


def pso_initialize_last_layer(model: nn.Module,
                              X_sub_t: torch.Tensor,
                              Y_sub_t: torch.Tensor,
                              device: torch.device,
                              iters=PSO_ITERS, particles=PSO_PARTICLES,
                              w=PSO_W, c1=PSO_C1, c2=PSO_C2):
    model.eval()
    last = get_last_linear(model)

    # Base params and bounds from current init
    with torch.no_grad():
        base_W = last.weight.detach().clone()
        base_b = last.bias.detach().clone()

    base_vec = layer_to_vector(base_W.view(-1), base_b.view(-1))
    dim = base_vec.numel()

    # Approx bound based on weight std
    w_std = float(base_W.std().cpu()) if base_W.numel() > 1 else 0.1
    b_std = float(base_b.std().cpu()) if base_b.numel() > 1 else 0.1
    std = max(w_std, b_std, 0.05)
    low = (base_vec - std).cpu().numpy()
    high = (base_vec + std).cpu().numpy()

    # Init swarm
    rng = np.random.RandomState(0)
    pos = rng.uniform(low, high, size=(particles, dim))
    vel = np.zeros_like(pos)

    def eval_f(pos_vec: np.ndarray) -> float:
        v = torch.tensor(pos_vec, dtype=torch.float32, device=device)
        with torch.no_grad():
            W_new, b_new = vector_to_layer(v, last.weight.shape, last.bias.shape)
            last.weight.copy_(W_new)
            last.bias.copy_(b_new)
            y_pred = model(X_sub_t)
            loss = nn.MSELoss()(y_pred, Y_sub_t)
        return float(loss.item())

    pbest = pos.copy()
    pbest_val = np.array([eval_f(p) for p in pbest])
    g_idx = int(np.argmin(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    for it in range(iters):
        r1 = rng.rand(*pos.shape)
        r2 = rng.rand(*pos.shape)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = pos + vel
        pos = np.minimum(np.maximum(pos, low), high)

        vals = np.array([eval_f(p) for p in pos])
        improved = vals < pbest_val
        pbest[improved] = pos[improved]
        pbest_val[improved] = vals[improved]
        if float(vals.min()) < gbest_val:
            g_idx = int(np.argmin(vals))
            gbest = pos[g_idx].copy()
            gbest_val = float(vals[g_idx])
        if (it + 1) % 10 == 0 or it == 0:
            print(f"PSO iter {it+1}/{iters}  best MSE {gbest_val:.4e}")

    # Set best found
    with torch.no_grad():
        v = torch.tensor(gbest, dtype=torch.float32, device=device)
        W_best, b_best = vector_to_layer(v, last.weight.shape, last.bias.shape)
        last.weight.copy_(W_best)
        last.bias.copy_(b_best)


def main():
    # 1) Data
    X, Y, time_Y = load_and_build_dataset(MAT_PATH)
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")

    # 2) Split and scale
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_test_s  = scaler_Y.transform(Y_test)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
    Y_test_t  = torch.tensor(Y_test_s,  dtype=torch.float32, device=device)

    # 3) Model
    torch.manual_seed(0)
    model = MLP(inp_dim=9, hid_dim=512, out_dim=6, depth=2, dropout=0.2).to(device)
    print(model)

    # 3.1) Optional PSO init (last layer only, on a small train subset)
    if USE_PSO_INIT:
        n = X_train_t.shape[0]
        k = min(PSO_SUBSET, n)
        rng_idx = np.random.RandomState(0).choice(n, size=k, replace=False)
        X_sub = X_train_t[rng_idx]
        Y_sub = Y_train_t[rng_idx]
        print(f"Running PSO init on subset: {k} samples, {PSO_PARTICLES} particles x {PSO_ITERS} iters")
        pso_initialize_last_layer(model, X_sub, Y_sub, device)

    # 4) Train with Adam
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 250
    train_hist, test_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss   = criterion(y_pred, Y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_t)
            test_loss   = criterion(y_test_pred, Y_test_t)

        train_hist.append(loss.item())
        test_hist.append(test_loss.item())
        if ep % 50 == 0 or ep == 1:
            print(f"Epoch {ep:3d}  Train MSE {loss.item():.4e}  Test MSE {test_loss.item():.4e}")

    # 5) Evaluation (MSE)
    model.eval()
    with torch.no_grad():
        Y_pred_test_s = model(X_test_t).cpu().numpy()
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_s)
    Y_true_test = Y_test

    # Split angles vs rates
    A_true = Y_true_test[:, :3];  A_pred = Y_pred_test[:, :3]
    R_true = Y_true_test[:, 3:];  R_pred = Y_pred_test[:, 3:]
    mse_angles = mean_squared_error(A_true, A_pred, multioutput='raw_values')
    mse_rates  = mean_squared_error(R_true, R_pred, multioutput='raw_values')
    print(f"MSE angles [rad^2]:  phi={mse_angles[0]:.4e}  theta={mse_angles[1]:.4e}  psi={mse_angles[2]:.4e}")
    print(f"MSE rates  [(rad/s)^2]: p={mse_rates[0]:.4e}  q={mse_rates[1]:.4e}  r={mse_rates[2]:.4e}")

    # 6) Predict full sequence for plotting
    X_all_s  = scaler_X.transform(X)
    X_all_t  = torch.tensor(X_all_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_all_pred_s = model(X_all_t).cpu().numpy()
    Y_all_pred = scaler_Y.inverse_transform(Y_all_pred_s)
    Y_all_true = Y
    t_full     = time_Y

    # Convert to degrees only for plotting
    A_true_deg = np.rad2deg(Y_all_true[:, :3])
    A_pred_deg = np.rad2deg(Y_all_pred[:, :3])
    R_true_deg = np.rad2deg(Y_all_true[:, 3:])
    R_pred_deg = np.rad2deg(Y_all_pred[:, 3:])

    # 7) Plots (in degrees)
    plt.close('all')

    # Angles - test set
    t_test = t_full[len(X_train):]
    A_true_test_deg = np.rad2deg(A_true)
    A_pred_test_deg = np.rad2deg(A_pred)
    fig1 = plt.figure(num="C14: Test-set Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_test, A_true_test_deg[:, i],  label='True', linewidth=1)
        plt.plot(t_test, A_pred_test_deg[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_angles_true_vs_pred.png", dpi=300)

    # Rates - test set
    R_true_test_deg = np.rad2deg(R_true)
    R_pred_test_deg = np.rad2deg(R_pred)
    fig2 = plt.figure(num="C14: Test-set Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_test, R_true_test_deg[:, i],  label='True', linewidth=1)
        plt.plot(t_test, R_pred_test_deg[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}test_rates_true_vs_pred.png", dpi=300)

    # Angles - full sequence
    fig3 = plt.figure(num="C14: Full Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, A_true_deg[:, i],  label='True', linewidth=1)
        plt.plot(t_full, A_pred_deg[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    # Rates - full sequence
    fig4 = plt.figure(num="C14: Full Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, R_true_deg[:, i],  label='True', linewidth=1)
        plt.plot(t_full, R_pred_deg[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    # Learning curves
    fig5 = plt.figure(num="C14: Learning Curves", figsize=(6,4))
    plt.plot(train_hist, label='train'); plt.plot(test_hist, label='test')
    plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('MSE loss')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig5.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)

    plt.show()

    # 8) Save checkpoint
    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}mlp_direct_model_angles_rates.pt")
    ckpt = {
        "model_class": "MLP",
        "model_kwargs": {"inp_dim": 9, "hid_dim": 512, "out_dim": 6, "depth": 2, "dropout": 0.2},
        "state_dict": model.state_dict(),
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "feature_names": [
            "u2","u3","u4","phi_rad","theta_rad","psi_rad","p_rad_s","q_rad_s","r_rad_s"
        ],
        "target_names":  [
            "phi_rad","theta_rad","psi_rad","p_rad_s","q_rad_s","r_rad_s"
        ],
        "train_history": train_hist,
        "test_history": test_hist,
        "eval_metric": "MSE",
        "units": {"angles": "rad", "rates": "rad/s"},
        "pso": {
            "used": USE_PSO_INIT,
            "particles": PSO_PARTICLES,
            "iters": PSO_ITERS,
            "subset": PSO_SUBSET,
            "w": PSO_W,
            "c1": PSO_C1,
            "c2": PSO_C2,
        }
    }
    torch.save(ckpt, ckpt_path)
    print(f"Saved trained model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()

