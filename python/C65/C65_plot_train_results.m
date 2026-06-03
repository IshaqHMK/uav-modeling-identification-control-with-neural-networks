% C65 training MAT plotter (for SLURM headless runs)
% Loads C65_train_results.mat and regenerates C65 figures in MATLAB.

clear; clc;

base_dir = fileparts(mfilename('fullpath'));
mat_path = fullfile(base_dir, 'mat_results', 'C65_train_results.mat');
out_dir = fullfile(base_dir, 'c65_train_figures_matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

S = load(mat_path);
labels = local_to_label_cell(S.dataset_labels);

% Step 2 learning curve
if isfield(S, 'step2_train_loss') && isfield(S, 'step2_validation_loss')
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 420]);
    plot(S.step2_train_loss, 'LineWidth', 1.2); hold on;
    plot(S.step2_validation_loss, 'LineWidth', 1.2);
    xlabel('Epoch'); ylabel('MSE'); title('C65 Step 2 Pitch Learning Curve');
    grid on; legend({'Train', 'Validation'}, 'Location', 'best');
    saveas(f, fullfile(out_dir, 'C65_step2_pitch_learning_curve.png'));
    close(f);
end

for i = 1:numel(labels)
    lbl = labels{i};
    tag = local_sanitize(lbl);

    % ---------------- Step 1 ----------------
    f_time = sprintf('step1_%s_time', tag);
    if isfield(S, f_time)
        t = S.(f_time);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1150 820]);
        tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t, S.(sprintf('step1_%s_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_z_ref', tag)), 'r--', 'LineWidth', 1);
        title(sprintf('%s: Altitude', lbl)); xlabel('Time (s)'); ylabel('z (m)'); grid on; legend('z','z_{ref}', 'Location', 'best');

        nexttile;
        plot(t, S.(sprintf('step1_%s_u1', tag)), 'g', 'LineWidth', 1);
        title('U1 (PID)'); xlabel('Time (s)'); ylabel('N'); grid on;

        nexttile;
        plot(t, S.(sprintf('step1_%s_theta', tag)), 'c', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_theta_ref', tag)), 'k--', 'LineWidth', 1);
        title('Pitch'); xlabel('Time (s)'); ylabel('rad'); grid on; legend('\theta','\theta_{ref}', 'Location', 'best');

        nexttile;
        plot(t, S.(sprintf('step1_%s_tau_y', tag)), 'LineWidth', 1);
        title('tau_y (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        nexttile;
        plot(t, S.(sprintf('step1_%s_phi', tag)), 'm', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_phi_ref', tag)), 'k--', 'LineWidth', 1);
        plot(t, S.(sprintf('step1_%s_psi', tag)), 'Color', [0.9 0.4 0.1], 'LineWidth', 1);
        plot(t, S.(sprintf('step1_%s_psi_ref', tag)), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
        title('Roll/Yaw'); xlabel('Time (s)'); ylabel('rad'); grid on;
        legend('\phi','\phi_{ref}','\psi','\psi_{ref}', 'Location', 'best');

        nexttile;
        plot(t, S.(sprintf('step1_%s_wind', tag)), 'r', 'LineWidth', 1);
        title('Wind disturbance'); xlabel('Time (s)'); ylabel('N'); grid on;

        saveas(f, fullfile(out_dir, sprintf('C65_%s_step1_pid_dataset.png', lbl)));
        close(f);
    end

    % ---------------- Step 2 ----------------
    f_t2 = sprintf('step2_%s_time', tag);
    if isfield(S, f_t2)
        t2 = S.(f_t2);
        y_true = S.(sprintf('step2_%s_tau_y_true', tag));
        y_pred = S.(sprintf('step2_%s_tau_y_pred', tag));
        e2 = S.(sprintf('step2_%s_pitch_error', tag));
        tr_end = S.(sprintf('step2_%s_split_train_end', tag));
        val_end = S.(sprintf('step2_%s_split_val_end', tag));

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 950 420]);
        plot(t2, y_true, 'b', 'LineWidth', 1); hold on;
        plot(t2, y_pred, 'r', 'LineWidth', 1);
        xline(t2(tr_end + 1), 'k--', 'LineWidth', 1);
        xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
        title(sprintf('%s: Step 2 Pitch Control', lbl));
        xlabel('Time (s)'); ylabel('tau_y (N*m)'); grid on;
        legend('True', 'Pred', 'Train/Val', 'Val/Test', 'Location', 'best');
        saveas(f, fullfile(out_dir, sprintf('C65_%s_step2_pitch_controls.png', lbl)));
        close(f);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 950 420]);
        plot(t2, e2, 'k', 'LineWidth', 1); hold on;
        xline(t2(tr_end + 1), 'k--', 'LineWidth', 1);
        xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
        title(sprintf('%s: Step 2 Pitch Error Input', lbl));
        xlabel('Time (s)'); ylabel('pitch error (rad)'); grid on;
        legend('error', 'Train/Val', 'Val/Test', 'Location', 'best');
        saveas(f, fullfile(out_dir, sprintf('C65_%s_step2_pitch_error_inputs.png', lbl)));
        close(f);
    end

    % ---------------- Step 3 ----------------
    f_t3 = sprintf('step3_%s_time', tag);
    if isfield(S, f_t3)
        t3 = S.(f_t3);
        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 700]);
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_theta', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_theta', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_theta_ref', tag)), 'r--', 'LineWidth', 1);
        title(sprintf('%s: Pitch tracking', lbl));
        xlabel('Time (s)'); ylabel('theta (rad)'); grid on;
        legend('PID', 'GRU', 'ref', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_tau_y', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_tau_y', tag)), 'g', 'LineWidth', 1);
        title('Pitch control'); xlabel('Time (s)'); ylabel('tau_y (N*m)'); grid on;
        legend('PID', 'GRU', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_z', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_z_ref', tag)), 'r--', 'LineWidth', 1);
        title('Altitude context'); xlabel('Time (s)'); ylabel('z (m)'); grid on;
        legend('PID', 'GRU-mode', 'z_{ref}', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_u1', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_u1', tag)), 'g', 'LineWidth', 1);
        title('Z control context'); xlabel('Time (s)'); ylabel('u1 (N)'); grid on;
        legend('PID', 'GRU-mode', 'Location', 'best');

        saveas(f, fullfile(out_dir, sprintf('C65_%s_step3_pid_vs_model.png', lbl)));
        close(f);
    end
end

disp(['Saved MATLAB figures to: ' out_dir]);


function labels = local_to_label_cell(raw)
    if iscell(raw)
        labels = cell(size(raw));
        for k = 1:numel(raw)
            labels{k} = char(string(raw{k}));
        end
    elseif isstring(raw)
        labels = cellstr(raw);
    elseif ischar(raw)
        labels = {char(raw)};
    else
        labels = cellstr(string(raw(:)));
    end
end

function tag = local_sanitize(lbl)
    tag = strrep(lbl, '-', 'm');
    tag = strrep(tag, '.', 'p');
end
