% C61 test MAT plotter (for SLURM headless runs)
% Loads C61_test_results.mat and regenerates C59-style test figures in MATLAB.

clear; clc;

base_dir = fileparts(mfilename('fullpath'));
mat_path = fullfile(base_dir, 'mat_results', 'C61_test_results.mat');
out_dir = fullfile(base_dir, 'c61_test_figures_matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

S = load(mat_path);
labels = local_to_label_cell(S.dataset_labels);

for i = 1:numel(labels)
    lbl = labels{i};
    tag = local_sanitize(lbl);

    % ---------------- Step 1 ----------------
    f_time = sprintf('step1_%s_time', tag);
    if isfield(S, f_time)
        t = S.(f_time);
        z = S.(sprintf('step1_%s_z', tag));
        z_ref = S.(sprintf('step1_%s_z_ref', tag));
        u1 = S.(sprintf('step1_%s_u1', tag));
        phi = S.(sprintf('step1_%s_phi', tag));
        phi_ref = S.(sprintf('step1_%s_phi_ref', tag));
        theta = S.(sprintf('step1_%s_theta', tag));
        theta_ref = S.(sprintf('step1_%s_theta_ref', tag));
        psi = S.(sprintf('step1_%s_psi', tag));
        psi_ref = S.(sprintf('step1_%s_psi_ref', tag));

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 800]);
        tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t, z, 'b', 'LineWidth', 1); hold on;
        plot(t, z_ref, 'r--', 'LineWidth', 1);
        title(sprintf('%s: Z Tracking', lbl));
        xlabel('Time (s)'); ylabel('Altitude (m)'); grid on; legend('z', 'z\_ref', 'Location', 'best');

        nexttile;
        plot(t, u1, 'g', 'LineWidth', 1);
        title('Control Input U1');
        xlabel('Time (s)'); ylabel('U1 (N)'); grid on;

        nexttile;
        plot(t, phi, 'm', 'LineWidth', 1); hold on;
        plot(t, phi_ref, 'k--', 'LineWidth', 1);
        title('Roll');
        xlabel('Time (s)'); ylabel('\phi (rad)'); grid on; legend('\phi', '\phi\_ref', 'Location', 'best');

        nexttile;
        plot(t, theta, 'c', 'LineWidth', 1); hold on;
        plot(t, theta_ref, 'k--', 'LineWidth', 1);
        title('Pitch');
        xlabel('Time (s)'); ylabel('\theta (rad)'); grid on; legend('\theta', '\theta\_ref', 'Location', 'best');

        nexttile;
        plot(t, psi, 'Color', [0.9 0.4 0.1], 'LineWidth', 1); hold on;
        plot(t, psi_ref, 'k--', 'LineWidth', 1);
        title('Yaw');
        xlabel('Time (s)'); ylabel('\psi (rad)'); grid on; legend('\psi', '\psi\_ref', 'Location', 'best');

        nexttile; axis off;

        saveas(f, fullfile(out_dir, sprintf('C61_%s_step1_z_tracking.png', lbl)));
        close(f);
    end

    % ---------------- Step 2 ----------------
    f_t2 = sprintf('step2_%s_time', tag);
    if isfield(S, f_t2)
        t2 = S.(f_t2);
        u_true = S.(sprintf('step2_%s_u_true', tag));
        u_pred = S.(sprintf('step2_%s_u_pred', tag));
        e2 = S.(sprintf('step2_%s_error', tag));
        tr_end = S.(sprintf('step2_%s_split_train_end', tag));
        val_end = S.(sprintf('step2_%s_split_val_end', tag));

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        ctrl_names = {'u1', 'tau_x', 'tau_y', 'tau_z'};
        for k = 1:4
            nexttile;
            plot(t2, u_true(:, k), 'b', 'LineWidth', 1); hold on;
            plot(t2, u_pred(:, k), 'r--', 'LineWidth', 1);
            xline(t2(tr_end + 1), 'k:', 'LineWidth', 1);
            xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
            title(ctrl_names{k});
            xlabel('Time (s)'); grid on;
        end
        saveas(f, fullfile(out_dir, sprintf('C61_%s_step2_controls.png', lbl)));
        close(f);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        err_names = {'z error', 'phi error', 'theta error', 'psi error'};
        for k = 1:4
            nexttile;
            plot(t2, e2(:, k), 'k', 'LineWidth', 1); hold on;
            xline(t2(tr_end + 1), 'k:', 'LineWidth', 1);
            xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
            title(err_names{k});
            xlabel('Time (s)'); grid on;
        end
        saveas(f, fullfile(out_dir, sprintf('C61_%s_step2_error_inputs.png', lbl)));
        close(f);
    end

    % ---------------- Step 3 ----------------
    f_t3 = sprintf('step3_%s_time', tag);
    if isfield(S, f_t3)
        t3 = S.(f_t3);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_z', tag)), 'r', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_z_ref', tag)), 'k--', 'LineWidth', 1);
        title('z'); xlabel('Time (s)'); grid on; legend('PID', 'Model', 'Ref', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_phi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_phi', tag)), 'r', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_phi_ref', tag)), 'k--', 'LineWidth', 1);
        title('phi'); xlabel('Time (s)'); grid on;

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_theta', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_theta', tag)), 'r', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_theta_ref', tag)), 'k--', 'LineWidth', 1);
        title('theta'); xlabel('Time (s)'); grid on;

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_psi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_model_psi', tag)), 'r', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_psi_ref', tag)), 'k--', 'LineWidth', 1);
        title('psi'); xlabel('Time (s)'); grid on;

        saveas(f, fullfile(out_dir, sprintf('C61_%s_step3_tracking.png', lbl)));
        close(f);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        ctrl_names = {'u1', 'tau_x', 'tau_y', 'tau_z'};
        for k = 1:4
            nexttile;
            pid_key = sprintf('step3_%s_pid_%s', tag, ctrl_names{k});
            mdl_key = sprintf('step3_%s_model_%s', tag, ctrl_names{k});
            plot(t3, S.(pid_key), 'b', 'LineWidth', 1); hold on;
            plot(t3, S.(mdl_key), 'r', 'LineWidth', 1);
            title(ctrl_names{k});
            xlabel('Time (s)'); grid on;
        end
        saveas(f, fullfile(out_dir, sprintf('C61_%s_step3_controls.png', lbl)));
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
