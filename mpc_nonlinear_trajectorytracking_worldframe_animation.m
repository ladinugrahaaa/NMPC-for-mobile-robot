clear;
clc;

%% MODEL PARAMETERS
d = 0.1; r = 0.035; l = 19; Kvt = 0.0059; R = 1.69;
La = 0.00011; M = 10; J = 0.025; Bv = 0.94; Bvn = 0.96;
Bw = 0.01; Cv = 2.2; Cvn = 1.5; Cw = 0.099;

%% STATE-SPACE MODEL
A11 = -3 * Kvt^2 * l^2 / (2 * r^2 * R * M) - Bv / M;
A22 = -3 * Kvt^2 * l^2 / (2 * r^2 * R * M) - Bvn / M;
A33 = -3 * d^2 * Kvt^2 * l^2 / (r^2 * R * J) - Bw / J;
A_mdl = [A11 0 0; 0 A22 0; 0 0 A33];

B_mdl = l * Kvt / (r * R) * [0 sqrt(3)/(2*M), -sqrt(3)/(2*M);
                             -1/M, 1/(2*M), 1/(2*M);
                              d/J, d/J, d/J];
K_mdl = [-Cv/M 0 0; 0 -Cvn/M 0; 0 0 -Cw/J];

%% TIME
dt = 0.01; t = 0:dt:100; steps = length(t);
% RECTANGULAR TRAJECTORY PARAMETERS
length_rect = 10; % Panjang persegi panjang
width_rect = 5; % Lebar persegi panjang
period = 80; % Waktu untuk menyelesaikan seluruh lintasan (dalam detik)
side_time = period / 4; % Waktu untuk setiap sisi lintasan
total_time = 4 * side_time; % Total waktu lintasan

%% INITIALIZATION
x_rw = zeros(3, steps); % State in world frame
x_rw(:, 1) = [0; 0; 0]; % Initial position in world frame
x_rw_dot = zeros(3, steps); % Velocity in world frame
x_rw_dot(:, 1) = [0; 0; 0];

x = zeros(3, steps); % State in robot frame
x(:, 1) = x_rw(:, 1); % Initial state assumed same as world frame
x_dot = zeros(3, steps); % Velocity in robot frame

u = zeros(3, steps); % Control input
xsp_rw = zeros(3, steps); % Setpoint in world frame
xsp_rw_dot = zeros(3, steps); % Velocity setpoint in world frame

%% MPC PARAMETERS
N = 50; % Prediction horizon
Q = diag([1000, 1000, 1000]); % State tracking weights
R = diag([0.1, 0.1, 0.1]); % Control effort weights
u_min = [-6; -6; -6];
u_max = [6; 6; 6];

for k = 1:steps-1
    % Hentikan jika lintasan selesai
    if t(k) > total_time
        disp('Lintasan persegi selesai!');
        break;
    end

    % Waktu saat ini
    current_time = t(k);

    % Hitung lintasan persegi panjang berdasarkan sisi lintasan
    if current_time <= side_time
        % Sisi bawah (bergerak ke kanan)
        xsp_rw(:, k) = [length_rect * current_time / side_time; 0; 0];
        xsp_rw_dot(:, k) = [length_rect / side_time; 0; 0];
    elseif current_time <= 2 * side_time
        % Sisi kanan (bergerak ke atas)
        xsp_rw(:, k) = [length_rect; width_rect * (current_time - side_time) / side_time; 0];
        xsp_rw_dot(:, k) = [0; width_rect / side_time; 0];
    elseif current_time <= 3 * side_time
        % Sisi atas (bergerak ke kiri)
        xsp_rw(:, k) = [length_rect - length_rect * (current_time - 2 * side_time) / side_time; width_rect; 0];
        xsp_rw_dot(:, k) = [-length_rect / side_time; 0; 0];
    else
        % Sisi kiri (bergerak ke bawah)
        xsp_rw(:, k) = [0; width_rect - width_rect * (current_time - 3 * side_time) / side_time; 0];
        xsp_rw_dot(:, k) = [0; -width_rect / side_time; 0];
    end

    % MPC Optimization
    u_guess = reshape(repmat(u(:, k), 1, N), [], 1); % Initial guess (repeat current input)
    options = optimset('Display', 'off'); % Suppress output

    % Solve for optimal control sequence
    u_opt= fmincon(@(u_seq) computeCost(x_rw(:, k), x_rw_dot(:, k), ...
                    reshape(u_seq, 3, N), xsp_rw(:, min(k:k+N-1, end)), ...
                    Q, R, A_mdl, B_mdl, K_mdl, N, dt), ...
                    u_guess, [], [], [], [], ...
                    repmat(u_min, N, 1), repmat(u_max, N, 1), [], options);

    % Extract the first control input
    u(:, k) = u_opt(1:3);

    % Dynamics in robot frame
    x_dot_dot = A_mdl * x_dot(:, k) + K_mdl * sign(x_dot(:, k)) + B_mdl * u(:, k);
    x_dot(:, k+1) = x_dot(:, k) + dt * x_dot_dot;
    x(:, k+1) = x(:, k) + dt * x_dot(:, k);

    % Transform back to world frame
    % Update posisi dan kecepatan di world frame
    theta_next = x_rw(3, k) + dt * x_rw_dot(3, k); % Orientasi berikutnya
    T_robot_to_rw = [cos(theta_next), -sin(theta_next), 0;
                 sin(theta_next),  cos(theta_next), 0;
                 0,                0,               1];
    x_rw_dot(:, k+1) = T_robot_to_rw * x_dot(:, k+1);
    x_rw(:, k+1) = x_rw(:, k) + dt * x_rw_dot(:, k+1);


    % Plotting the animation
    if mod(k, 10) == 0
        figure(1);
        clf;
        hold on;
        plot(xsp_rw(1, 1:k), xsp_rw(2, 1:k), 'k--', 'LineWidth', 1.5); % Reference trajectory
        plot(x_rw(1, 1:k), x_rw(2, 1:k), 'b', 'LineWidth', 1.5); % Actual trajectory
        plot(x_rw(1, k), x_rw(2, k), 'ro', 'MarkerFaceColor', 'r'); % Robot position
        axis equal;
        title('2D Trajectory of the Robot (World Frame)');
        xlabel('x [m]');
        ylabel('y [m]');
        legend('Reference Trajectory', 'Actual Trajectory', 'Robot Position');
        grid on;
        drawnow;
    end
end

%% ERROR PLOTS DAN RMSE
% Hitung error
error_x = x_rw(1, 1:steps) - xsp_rw(1, 1:steps);
error_y = x_rw(2, 1:steps) - xsp_rw(2, 1:steps);
error_theta = x_rw(3, 1:steps) - xsp_rw(3, 1:steps);

% Root Mean Square Error (RMSE)
rmse_x = sqrt(mean(error_x.^2));
rmse_y = sqrt(mean(error_y.^2));
rmse_theta = sqrt(mean(error_theta.^2));

% Tampilkan RMSE di Command Window
disp('==== RMSE ====');
fprintf('RMSE x: %.4f m\n', rmse_x);
fprintf('RMSE y: %.4f m\n', rmse_y);
fprintf('RMSE theta: %.4f rad\n', rmse_theta);
%% PLOTS
subplot(3,3,1)
plot(t(1:end-1), x_rw_dot(1,1:end-1), 'r', 'LineWidth', 1)
title('v(t) vs t')
xlabel('Time [s]')
ylabel('v(t) [m/s]')
legend('v(t)')
grid on
hold off

subplot(3,3,2)
plot(t(1:end-1),x_rw_dot(2,1:end-1),'g','LineWidth',1)
title('vn(t) vs t')
xlabel('Time [s]')
ylabel('vn(t) [m/s]')
legend('vn(t)')
grid on
hold off

subplot(3,3,3)
plot(t(1:end-1),x_rw_dot(3,1:end-1),'b','LineWidth',1)
title('w(t) vs t')
xlabel('Time [s]')
ylabel('w(t) [m/s]')
legend('w(t)')
grid on
hold off

subplot(3,3,4)
plot(t(1:end-1),x_rw(1,1:end-1),'r','LineWidth',1)
hold on
plot(t(1:end-1),xsp_rw(1,1:end-1),'k--','LineWidth',0.5)
title('x(t) vs t')
xlabel('Time [s]')
ylabel('x(t) [m]')
legend('x','x setpoint')
grid on
hold off

subplot(3,3,5)
plot(t(1:end-1),x_rw(2,1:end-1),'g','LineWidth',1)
hold on
plot(t(1:end-1),xsp_rw(2,1:end-1),'k--','LineWidth',0.5)
title('y(t) vs t')
xlabel('Time [s]')
ylabel('y(t) [m]')
legend('y','y setpoint')
grid on
hold off

subplot(3,3,6)
plot(t(1:end-1),x_rw(3,1:end-1),'b','LineWidth',1)
hold on
plot(t(1:end-1),xsp_rw(3,1:end-1),'k--','LineWidth',0.5)
title('\theta(t) vs t')
xlabel('Time [s]')
ylabel('\theta(t) [rad]')
legend('\theta','\theta setpoint')
grid on
hold off

subplot(3,3,7)
plot(x_rw(1,1:end-1),x_rw(2,1:end-1),'r','LineWidth',1)
hold on
plot(xsp_rw(1,1:end-1),xsp_rw(2,1:end-1),'k--','LineWidth',0.5)
title('XY Plot')
xlabel('x [m]')
ylabel('y [m]')
legend('XY robot','XY setpoint')
grid on
xlim([-2, 12]) % Adjust as per your data range
ylim([-2, 10]) % Adjust as per your data range
hold off


subplot(3,3,8)
plot(t(1:end-1),u(1,1:end-1),'r','LineWidth',1)
hold on
plot(t(1:end-1),u(2,1:end-1),'g','LineWidth',1)
plot(t(1:end-1),u(3,1:end-1),'b','LineWidth',1)
title('Control Signal u(t) vs t')
xlabel('Time [s]')
ylabel('u(t) [V]')
legend('u1(t)','u2(t)','u3(t)')
grid on
hold off

subplot(3,3,9)
plot(t(1:steps), error_x, 'r', 'LineWidth', 1);
hold on;
plot(t(1:steps), error_y, 'g', 'LineWidth', 1);
plot(t(1:steps), error_theta, 'b', 'LineWidth', 1);
hold off;
title('Error e(t) vs t');
xlabel('Time [s]');
ylabel('e(t)');
legend('e_x(t)','e_y(t)','e_\theta(t)');
grid on;
%% COST FUNCTION UNTUK MPC
function cost = computeCost(x_rw, x_rw_dot, u_seq, xsp_rw, Q, R, A, B, K, N, dt)
    cost = 0;
    x_pred = x_rw; % Initial state
    x_dot_pred = x_rw_dot; % Initial velocity
    theta_pred = x_rw(3); % Initial orientation

    for i = 1:N
        u = u_seq(:, i); % Control input for step i

        % Transform control input to world frame
        T_robot_to_rw = [cos(theta_pred), -sin(theta_pred), 0;
                         sin(theta_pred),  cos(theta_pred), 0;
                         0,                0,               1];
        u_rw = T_robot_to_rw * u;

        % Predict next state
        x_dot_dot_pred = A * x_dot_pred + K * sign(x_dot_pred) + B * u_rw;
        x_dot_pred = x_dot_pred + dt * x_dot_dot_pred;
        x_pred = x_pred + dt * x_dot_pred;

        % Update orientation
        theta_pred = theta_pred + dt * x_dot_pred(3);
        x_pred(3) = theta_pred;

        % Compute tracking error
        e = x_pred - xsp_rw(:, min(i, size(xsp_rw, 3)));

        % Accumulate cost
        cost = cost + (e' * Q * e + u' * R * u) * dt;
    end
end