% 查看模拟的振动数据
f = figure;
f.Position(3:4) = [900 400];

%%
oe = 65;
% 生成模拟振动数据
sample_rate = 1000;
duration = 10;
% 振动方程
% compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x);

% % 一个等比位移数列来探究未加噪声的识别精度
% T  = 50; interval = 5; compute_y = @(query_x) (1/2).^(floor(abs(mod(query_x, 2*T) - floor(mod(query_x, 2*T)/T)*2*T) / interval));

% % 一个等比位移阻尼振荡数列来探究未加噪声的识别精度
% T  = 50; interval = 5; 
% compute_y = @(query_x) ((-1).^(floor(query_x./interval)) .* ((1/2).^(floor(mod(query_x, 2*T)/2/interval))));

% % 一个连续位移阻尼振荡数列来探究未加噪声的识别精度
% damp_ratio = 0.65;
% v_init = 40; x_init = 0;
% omega = 2*pi*1.2; 
% compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));

% % 包含多个频率的方程，但是每步位移都不超过识别上限
compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x)+1/12*sin(2*pi*1.5*query_x + pi/6)+1/16*sin(2*pi*10*query_x) + 1/14*sin(2*pi*2.8*query_x+pi/4);

% % 使用有一些位移超过上限的方程
% compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);

% % 一个等比增加位移数列来探究不同尺度单步位移上限
% T  = 30; interval = 5; compute_y = @(query_x) -10+1/2*(((floor(abs(mod(query_x, 2*T) - floor(mod(query_x, 2*T)/T)*2*T) / interval))).^2 + ((floor(abs(mod(query_x, 2*T) - floor(mod(query_x, 2*T)/T)*2*T) / interval))));
x = [1 : round(sample_rate * duration)];
% x = [1 : round(sample_rate * duration)] ./round(sample_rate * duration).*duration;
y = compute_y(x);

% 制作视频
video_rate = 30;
video_duration = duration;
video_lenth = round(video_rate * video_duration);
% lowrate_x = [1:video_lenth];
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;
% lowrate_y = compute_y(lowrate_x);
amp_factor = [1/12/0.45, 1/25/0.45, 1/50/0.45]; lowrate_y = amp_factor(3) .* lowrate_y;
motion_cable = [lowrate_x ; lowrate_y];

% noise_levels = [0.0001, 0.001, 0.005];
noise_levels = [0.0001, 0.0005, 0.001]; % 模拟算例
output_video = emulate_video(motion_cable,...
                            'Image_size',[32,32],...
                            'Angle_cable', oe, ...
                            'Width_cable', 1.0, ...
                            'Smooth_Sigma', 0.5, ...
                            'Discrete_rate', 1024, ...
                            'Noise_yes', 1, ...
                            'Noise_var', noise_levels(3), ...
                            'Background',1,...
                            'BackColor', 0.25);

ANIME(video_lenth) = struct('cdata',[],'colormap',[]);
    

for i = 1:video_lenth
    subplot(1,2,1);figure(f); surf(output_video(:,:,i)); view(66.4, 42.8);
    
    title(['frame:',num2str(i)]);
%     ANIME(i)= getframe(f);
end


%%
output_path = 'C:\Users\HIT\Desktop\相位图片\模拟数据生成_m-D3.avi';
outvideo = VideoWriter(output_path,'Motion JPEG AVI');
open(outvideo)
for i = 1:video_lenth
%     writeVideo(outvideo,ANIME(i).cdata);
    writeVideo(outvideo,output_video(:,:,i));
end
close(outvideo);
'输出完成'