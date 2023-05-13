sample_rate = 200;
dura_time = 0.5;
time_lines = [0:round(sample_rate * dura_time)] ./ round(sample_rate * dura_time) .* dura_time;
motion_lines =0.8*sin(2*pi*12*time_lines) + 0.2*sin(2*pi*25*time_lines) + 0.3*sin(2*pi*75*time_lines);

% figure(f);plot(time_lines,motion_lines);
% title('Reference motion');
% xlabel('t /s');
% ylabel('s(t)');

figure(f);
plot(motion_lines);
hold on;
plot(0.85*signalout(4,:));