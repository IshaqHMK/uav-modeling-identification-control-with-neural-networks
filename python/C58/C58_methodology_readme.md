# C58 Methodology

## 1) Objective
Apply closed-loop adaptive control with:
\[
u_k = u_k^{base} + \Delta u_k
\]
where adaptation is driven by model mismatch between measured and estimated states.

## 2) Signals
- Measured state: \(x_k = [z_k, \dot z_k]\)
- Estimated state: \(\hat{x}_k = [\hat{z}_k, \hat{\dot{z}}_k]\)
- Reference: \(r_k\)
- Tracking error: \(e_k = r_k - z_k\)
- Model mismatch:
\[
e_k^m = x_k - \hat{x}_k
\]

## 3) Base control law
\[
u_k^{base} = \pi_\theta(z_k, e_k, \dot{e}_k, \int e)
\]

## 4) Adaptive correction law
Feature vector:
\[
\phi_k = [e_{z,k}^m,\ \alpha e_{\dot{z},k}^m,\ 1]
\]
Correction:
\[
\Delta u_k = \mathrm{clip}(w_k^T \phi_k,\ \pm \Delta u_{max})
\]
Applied control in adaptive mode:
\[
u_k = u_k^{base} + \Delta u_k
\]

## 5) Online adaptation rule
When adaptation is enabled:
\[
w_{k+1} = w_k - \eta \nabla_w \mathcal{L}_k
\]
where \(\mathcal{L}_k\) is formed from model mismatch terms \((e_{z,k}^m, e_{\dot{z},k}^m)\).

## 6) Safety and fallback
Mismatch norm:
\[
\|e_k^m\| = \sqrt{(e_{z,k}^m)^2 + (\alpha e_{\dot{z},k}^m)^2}
\]
Rules:
1. Adapt only if \(\|e_k^m\| < \varepsilon_{adapt}\).
2. If \(\|e_k^m\| > \varepsilon_{safety}\), freeze adaptation and apply PID fallback for a hold window:
\[
u_k = u_k^{pid}
\]

## 7) Per-step algorithm
1. Measure \(x_k\), read \(r_k\).
2. Compute \(u_k^{base}\).
3. Compute mismatch \(e_k^m = x_k - \hat{x}_k\).
4. Compute \(\Delta u_k\) from \(w_k\) and \(\phi_k\).
5. Apply control:
   - adaptive mode: \(u_k = u_k^{base} + \Delta u_k\)
   - fallback mode: \(u_k = u_k^{pid}\)
6. Update plant and estimator states.
7. If enabled, update \(w_k\).

## 8) Parameter roles (method-only)
- \(\eta\): adaptation step size
- \(\alpha\): relative weighting of \(\dot{z}\)-mismatch
- \(\Delta u_{max}\): correction saturation limit
- \(\varepsilon_{adapt}\): adaptation enable threshold
- \(\varepsilon_{safety}\): safety/fallback trigger threshold

## 9) Fixed vs adapted
- Fixed: base controller parameters \(\theta\)
- Fixed: direct-model parameters
- Adapted online: correction weights \(w_k\)
