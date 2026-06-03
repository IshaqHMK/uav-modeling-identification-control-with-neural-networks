% C67 test MAT plotter (for headless/HPC runs)
% Loads C67_test_results.mat and regenerates C67 comparison figures in MATLAB.

clear; clc;

base_dir = fileparts(mfilename('fullpath'));
mat_path = fullfile(base_dir, 'mat_results', 'C67_test_results.mat');
out_dir = fullfile(base_dir, 'c67_test_figures_matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

S = load(mat_path);
labels = local_to_label_cell(S.dataset_labels);

for i = 1:numel(labels)
    lbl = labels{i};
    tag = local_sanitize(lbl);

    % ---------------- Step 1 (PID baseline dataset) ----------------
    f_t1 = sprintf('step1_%s_time', tag);
    if isfield(S, f_t1)
        t1 = S.(f_t1);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1150 900]);
        tiledlayout(4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t1, S.(sprintf('step1_%s_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t1, S.(sprintf('step1_%s_z_ref', tag)), 'r--', 'LineWidth', 1);
        title(sprintf('%s: Altitude (PID)', lbl));
        xlabel('Time (s)'); ylabel('z (m)'); grid on; legend('z', 'z_{ref}', 'Location', 'best');

        nexttile;
        plot(t1, S.(sprintf('step1_%s_u1', tag)), 'LineWidth', 1);
        title('u1 (PID)'); xlabel('Time (s)'); ylabel('N'); grid on;

        nexttile;
        plot(t1, S.(sprintf('step1_%s_phi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t1, S.(sprintf('step1_%s_phi_ref', tag)), 'r--', 'LineWidth', 1);
        title('Roll (PID)'); xlabel('Time (s)'); ylabel('\phi (rad)'); grid on; legend('\phi', '\phi_{ref}', 'Location', 'best');

        nexttile;
        plot(t1, S.(sprintf('step1_%s_tau_x', tag)), 'LineWidth', 1);
        title('\tau_x (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        nexttile;
        plot(t1, S.(sprintf('step1_%s_theta', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t1, S.(sprintf('step1_%s_theta_ref', tag)), 'r--', 'LineWidth', 1);
        title('Pitch (PID)'); xlabel('Time (s)'); ylabel('\theta (rad)'); grid on; legend('\theta', '\theta_{ref}', 'Location', 'best');

        nexttile;
        plot(t1, S.(sprintf('step1_%s_tau_y', tag)), 'LineWidth', 1);
        title('\tau_y (PID)'); xlabel('Time (s)'); ylabel('N*m'); grid on;

        nexttile;
        plot(t1, S.(sprintf('step1_%s_psi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t1, S.(sprintf('step1_%s_psi_ref', tag)), 'r--', 'LineWidth', 1);
        title('Yaw (PID)'); xlabel('Time (s)'); ylabel('\psi (rad)'); grid on; legend('\psi', '\psi_{ref}', 'Location', 'best');

        nexttile;
        plot(t1, S.(sprintf('step1_%s_tau_z', tag)), 'LineWidth', 1); hold on;
        plot(t1, S.(sprintf('step1_%s_wind', tag)), '--', 'LineWidth', 1);
        title('\tau_z and wind (PID)'); xlabel('Time (s)'); ylabel('N*m / N'); grid on; legend('\tau_z', 'wind', 'Location', 'best');

        saveas(f, fullfile(out_dir, sprintf('C67_%s_step1_pid_dataset.png', lbl)));
        close(f);
    end

    % ---------------- Step 3 (PID vs 4-GRU) ----------------
    f_t3 = sprintf('step3_%s_pid_time', tag);
    if isfield(S, f_t3)
        t3 = S.(f_t3);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1150 900]);
        tiledlayout(4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_z', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_z_ref', tag)), 'r--', 'LineWidth', 1);
        title(sprintf('%s: Altitude PID vs 4-GRU', lbl));
        xlabel('Time (s)'); ylabel('z (m)'); grid on; legend('PID', '4-GRU', 'z_{ref}', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_u1', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_u1', tag)), 'g', 'LineWidth', 1);
        title('u1'); xlabel('Time (s)'); ylabel('N'); grid on; legend('PID', '4-GRU', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_phi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_phi', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_phi_ref', tag)), 'r--', 'LineWidth', 1);
        title('Roll'); xlabel('Time (s)'); ylabel('\phi (rad)'); grid on; legend('PID', '4-GRU', '\phi_{ref}', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_tau_x', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_tau_x', tag)), 'g', 'LineWidth', 1);
        title('\tau_x'); xlabel('Time (s)'); ylabel('N*m'); grid on; legend('PID', '4-GRU', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_theta', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_theta', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_theta_ref', tag)), 'r--', 'LineWidth', 1);
        title('Pitch'); xlabel('Time (s)'); ylabel('\theta (rad)'); grid on; legend('PID', '4-GRU', '\theta_{ref}', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_tau_y', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_tau_y', tag)), 'g', 'LineWidth', 1);
        title('\tau_y'); xlabel('Time (s)'); ylabel('N*m'); grid on; legend('PID', '4-GRU', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_psi', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_psi', tag)), 'g', 'LineWidth', 1);
        plot(t3, S.(sprintf('step3_%s_pid_psi_ref', tag)), 'r--', 'LineWidth', 1);
        title('Yaw'); xlabel('Time (s)'); ylabel('\psi (rad)'); grid on; legend('PID', '4-GRU', '\psi_{ref}', 'Location', 'best');

        nexttile;
        plot(t3, S.(sprintf('step3_%s_pid_tau_z', tag)), 'b', 'LineWidth', 1); hold on;
        plot(t3, S.(sprintf('step3_%s_gru_tau_z', tag)), 'g', 'LineWidth', 1);
        title('\tau_z'); xlabel('Time (s)'); ylabel('N*m'); grid on; legend('PID', '4-GRU', 'Location', 'best');

        saveas(f, fullfile(out_dir, sprintf('C67_%s_step3_pid_vs_4gru.png', lbl)));
        close(f);

        % RMS summary figure
        pid_rms_z = S.(sprintf('step3_%s_pid_rms_z', tag)); pid_rms_z = pid_rms_z(1);
        pid_rms_roll = S.(sprintf('step3_%s_pid_rms_roll', tag)); pid_rms_roll = pid_rms_roll(1);
        pid_rms_pitch = S.(sprintf('step3_%s_pid_rms_pitch', tag)); pid_rms_pitch = pid_rms_pitch(1);
        pid_rms_yaw = S.(sprintf('step3_%s_pid_rms_yaw', tag)); pid_rms_yaw = pid_rms_yaw(1);
        gru_rms_z = S.(sprintf('step3_%s_gru_rms_z', tag)); gru_rms_z = gru_rms_z(1);
        gru_rms_roll = S.(sprintf('step3_%s_gru_rms_roll', tag)); gru_rms_roll = gru_rms_roll(1);
        gru_rms_pitch = S.(sprintf('step3_%s_gru_rms_pitch', tag)); gru_rms_pitch = gru_rms_pitch(1);
        gru_rms_yaw = S.(sprintf('step3_%s_gru_rms_yaw', tag)); gru_rms_yaw = gru_rms_yaw(1);

        pid_vals = [pid_rms_z, pid_rms_roll, pid_rms_pitch, pid_rms_yaw];
        gru_vals = [gru_rms_z, gru_rms_roll, gru_rms_pitch, gru_rms_yaw];

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 850 420]);
        b = bar([pid_vals(:), gru_vals(:)], 'grouped');
        b(1).FaceColor = [0.2 0.4 0.9];
        b(2).FaceColor = [0.2 0.7 0.3];
        set(gca, 'XTickLabel', {'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]'});
        ylabel('RMS error');
        title(sprintf('%s: RMS comparison PID vs 4-GRU', lbl));
        grid on;
        legend('PID', '4-GRU', 'Location', 'best');
        saveas(f, fullfile(out_dir, sprintf('C67_%s_step3_rms_comparison.png', lbl)));
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
