% 用作phase map相关算法实验
f = figure;
f.Position(3:4) = [900 400];

%
total_lenth = 99;
T = 20;
omega = 2*pi/T;
omega_2 = omega/1.5;
init_phase = pi;%2*pi;
phase_shift = 7*pi/4;
compute_y = @(query_x) mod((omega * (query_x-50) + init_phase), 2*pi)-pi;
compute_y_2 = @(query_x) mod((omega * (query_x-50) + init_phase - phase_shift), 2*pi)-pi;
compute_y_3 = @(query_x) mod((omega_2 * (query_x-50) + init_phase), 2*pi)-pi;
compute_y_4 = @(query_x) mod((omega_2 * (query_x-50) + init_phase - phase_shift/omega*omega_2), 2*pi)-pi;

x_line = [1:1:total_lenth];
y_line = compute_y(x_line);
y_line_2 = compute_y_2(x_line);
y_line_3 = compute_y_3(x_line);
y_line_4 = compute_y_4(x_line);

figure(f);
plot(x_line, y_line + 2*pi, 'b'); hold on; %plot(x_line, y_line_3 + 2*pi, 'c'); hold on;
plot(x_line, y_line_2 + 2*pi, 'r'); hold on; %plot(x_line, y_line_4 + 2*pi); hold on;
plot(x_line, y_line_2 - y_line, 'g'); hold on;


% plot(x_line, mod((y_line - y_line_3+pi), 2*pi)-pi, 'm'); hold on;
% plot(x_line, mod((y_line_2 - y_line_4+pi), 2*pi)-pi, 'y'); hold on;
plot(x_line, (mod((y_line_2 - y_line_4+pi), 2*pi)-pi) - (mod((y_line - y_line_3+pi), 2*pi)-pi), 'm'); hold on;
% plot(x_line, y_line - y_line_3, 'k'); hold on;