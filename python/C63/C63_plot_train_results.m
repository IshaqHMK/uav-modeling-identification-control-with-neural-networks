
% C63 training MAT plotter (for SLURM headless runs)
% Loads C63_train_results.mat and regenerates C63 figures in MATLAB.

clear; clc;

base_dir = fileparts(mfilename('fullpath'));
mat_path = fullfile(base_dir, 'mat_results', 'C63_train_results.mat');
out_dir = fullfile(base_dir, 'c63_train_figures_matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

S = load(mat_path);
labels = local_to_label_cell(S.dataset_labels);
axes_list = {'z','roll','pitch','yaw'};

% Step 2 learning curves for all axes
f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 700]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:numel(axes_list)
    ax = axes_list{i};
    tr_key = sprintf('step2_%s_train_loss', ax);
    va_key = sprintf('step2_%s_validation_loss', ax);
    nexttile;
    if isfield(S, tr_key) && isfield(S, va_key)
        plot(S.(tr_key), 'LineWidth', 1.2); hold on;
        plot(S.(va_key), 'LineWidth', 1.2);
        title(sprintf('%s learning curve', upper(ax)));
        xlabel('Epoch'); ylabel('MSE');
        grid on; legend({'Train', 'Validation'}, 'Location', 'best');
    else
        axis off;
        text(0.1, 0.5, sprintf('No loss for %s', ax));
    end
end
saveas(f, fullfile(out_dir, 'C63_step2_learning_curves.png'));
close(f);

for i = 1:numel(labels)
    lbl = labels{i};
    tag = local_sanitize(lbl);

    % ---------------- Step 1 ----------------
    f_time = sprintf('step1_%s_time', tag);
    if isfield(S, f_time)
        t = S.(f_time);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 900]);
        tiledlayout(4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile; plot(t, S.(sprintf('step1_%s_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_z_ref', tag)), 'r--', 'LineWidth', 1);
        title(sprintf('%s: Altitude', lbl)); xlabel('Time (s)'); ylabel('z (m)'); grid on; legend('z', 'z_{ref}', 'Location', 'best');

        nexttile; plot(t, S.(sprintf('step1_%s_u1', tag)), 'g', 'LineWidth', 1);
        title('U1 (PID)'); xlabel('Time (s)'); ylabel('N'); grid on;

        nexttile; plot(t, S.(sprintf('step1_%s_phi', tag)), 'm', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_phi_ref', tag)), 'k--', 'LineWidth', 1);
        title('Roll'); xlabel('Time (s)'); ylabel('rad'); grid on; legend('\phi','\phi_{ref}', 'Location', 'best');

        nexttile; plot(t, S.(sprintf('step1_%s_tau_x', tag)), 'LineWidth', 1);
        title('tau_x (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        nexttile; plot(t, S.(sprintf('step1_%s_theta', tag)), 'c', 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_theta_ref', tag)), 'k--', 'LineWidth', 1);
        title('Pitch'); xlabel('Time (s)'); ylabel('rad'); grid on; legend('\theta','\theta_{ref}', 'Location', 'best');

        nexttile; plot(t, S.(sprintf('step1_%s_tau_y', tag)), 'LineWidth', 1);
        title('tau_y (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        nexttile; plot(t, S.(sprintf('step1_%s_psi', tag)), 'Color', [0.9 0.4 0.1], 'LineWidth', 1); hold on;
        plot(t, S.(sprintf('step1_%s_psi_ref', tag)), 'k--', 'LineWidth', 1);
        title('Yaw'); xlabel('Time (s)'); ylabel('rad'); grid on; legend('\psi','\psi_{ref}', 'Location', 'best');

        nexttile; plot(t, S.(sprintf('step1_%s_tau_z', tag)), 'LineWidth', 1);
        title('tau_z (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        saveas(f, fullfile(out_dir, sprintf('C63_%s_step1_pid_dataset.png', lbl)));
        close(f);
    end

    % ---------------- Step 2 ----------------
    f_t2 = sprintf('step2_%s_time', tag);
    if isfield(S, f_t2)
        t2 = S.(f_t2);
        tr_end = S.(sprintf('step2_%s_split_train_end', tag));
        val_end = S.(sprintf('step2_%s_split_val_end', tag));

        for j = 1:numel(axes_list)
            ax = axes_list{j};
            key_true = sprintf('step2_%s_%s_true', tag, ax);
            key_pred = sprintf('step2_%s_%s_pred', tag, ax);
            key_err = sprintf('step2_%s_%s_error', tag, ax);
            if isfield(S, key_true) && isfield(S, key_pred)
                f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 950 420]);
                plot(t2, S.(key_true), 'b', 'LineWidth', 1); hold on;
                plot(t2, S.(key_pred), 'r', 'LineWidth', 1);
                xline(t2(tr_end + 1), 'k--', 'LineWidth', 1);
                xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
                title(sprintf('%s: Step 2 %s Control', lbl, upper(ax)));
                xlabel('Time (s)'); ylabel('Control'); grid on;
                legend('True', 'Pred', 'Train/Val', 'Val/Test', 'Location', 'best');
                saveas(f, fullfile(out_dir, sprintf('C63_%s_step2_%s_controls.png', lbl, ax)));
                close(f);
            end

            if isfield(S, key_err)
                f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 950 420]);
                plot(t2, S.(key_err), 'k', 'LineWidth', 1); hold on;
                xline(t2(tr_end + 1), 'k--', 'LineWidth', 1);
                xline(t2(val_end + 1), 'k--', 'LineWidth', 1);
                title(sprintf('%s: Step 2 %s Error Input', lbl, upper(ax)));
                xlabel('Time (s)'); ylabel('Error'); grid on;
                legend('error', 'Train/Val', 'Val/Test', 'Location', 'best');
                saveas(f, fullfile(out_dir, sprintf('C63_%s_step2_%s_error_inputs.png', lbl, ax)));
                close(f);
            end
        end
    end

    % ---------------- Step 3 ----------------
    f_t3 = sprintf('step3_%s_time', tag);
    if isfield(S, f_t3)
        t3 = S.(f_t3);
        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 900]);
        tiledlayout(4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        for j = 1:numel(axes_list)
            ax = axes_list{j};
            ref_key = sprintf('step3_%s_%s_ref', tag, ax);
            pid_s_key = sprintf('step3_%s_%s_pid_state', tag, ax);
            mdl_s_key = sprintf('step3_%s_%s_model_state', tag, ax);
            pid_c_key = sprintf('step3_%s_%s_pid_ctrl', tag, ax);
            mdl_c_key = sprintf('step3_%s_%s_model_ctrl', tag, ax);

            nexttile;
            if isfield(S, ref_key) && isfield(S, pid_s_key) && isfield(S, mdl_s_key)
                plot(t3, S.(pid_s_key), 'b', 'LineWidth', 1); hold on;
                plot(t3, S.(mdl_s_key), 'g', 'LineWidth', 1);
                plot(t3, S.(ref_key), 'r--', 'LineWidth', 1);
                title(sprintf('%s: %s state', lbl, upper(ax)));
                xlabel('Time (s)'); ylabel('state'); grid on;
                legend('PID', 'GRU', 'ref', 'Location', 'best');
            else
                axis off;
            end

            nexttile;
            if isfield(S, pid_c_key) && isfield(S, mdl_c_key)
                plot(t3, S.(pid_c_key), 'b', 'LineWidth', 1); hold on;
                plot(t3, S.(mdl_c_key), 'g', 'LineWidth', 1);
                title(sprintf('%s: %s control', lbl, upper(ax)));
                xlabel('Time (s)'); ylabel('control'); grid on;
                legend('PID', 'GRU', 'Location', 'best');
            else
                axis off;
            end
        end

        saveas(f, fullfile(out_dir, sprintf('C63_%s_step3_pid_vs_gru_all_axes.png', lbl)));
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
