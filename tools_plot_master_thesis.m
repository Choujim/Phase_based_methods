%% ͼ2-1
% description
% һ���������ϲ�
f = figure;
f.Position(3:4) = [800 400];

T = 320;
init_phi = 0 / pi;
sum_points = 800;
compute_y = @(query_x) (sin(2*pi/T*query_x + init_phi));
x = [0 : sum_points];
y = compute_y(x);

shift_phi = -1 / 3*pi;
x_shift = x + shift_phi/2/pi*T;
y_shift = compute_y(x_shift);

compute_y_phase_shift= @(query_x) (sin(2*pi/T*query_x + shift_phi));
y_phase_shift = compute_y_phase_shift(x);
x2show = (x-400) ./ T * 2* pi./2;
color_1 = [75,108,255]./255; %[194,94,136]./255;
color_2 = [22,196,28]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
plot(x2show,y,'LineWidth', 1.5, 'Color',color_1, 'LineStyle', '-'); hold on;
plot(x2show,y_shift ,'LineWidth', 1.5, 'Color', color_2,'LineStyle', '-.'); hold on;
plot(x2show,y_phase_shift, 'LineWidth', 1.5, 'Color', color_3,'LineStyle', '--'); hold on;

% legend('y = cos(2x)', 'y = cos(2(x-pi/6))', 'y = cos(2x-pi/3)');
legend(' cos(2x)', ' cos(2(x-pi/6))', ' cos(2x-pi/3)');
legend('boxoff')
% set(legend, 'Location','NorthInside' );
ylabel('Intensity');
% xlabel('Space (rad)');
xlabel('x(rad)');
ylim([-2,2]);
yticks(-2:1:2);
xticks(-4:2:4);

% ���
fig = figure(f);
jpgHeight = 7.48;
jpgWidth = 15;
fig.PaperPositionMode = 'manual';
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 jpgWidth jpgHeight];

set(fig,'Renderer','painters');
fileout = 'C:\Users\HIT\Desktop\thesis_materials\1.jpg';
print(fig, fileout, '-djpeg', '-r600');

%% ͼ2-2
% description
% 1D�ֲ���λ˵��
f = figure;
f.Position(3:4) = [800 400];

sum_points = 120;
windows_lenth = 10;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 

impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);
impluse_y = zeros(1, sum_points+1);
center_1 = 40;
center_2 = 100;
impluse_y(1, center_1) = 1;
impluse_y(1, center_2) = 1;

y_line_1 = conv(impluse_y, impluse_x, 'same');

% plot(y_line_1, 'Color', [86 118 165]./255); hold on; ylim([-0.2 1.2]);
% plot(impluse_y); hold on;
% plot(impluse_x); hold on;

% FFT
fft_y_1 = fftshift(fft(y_line_1)); %figure; plot(abs(fft_y_1));

center_1 = 38; impluse_y = zeros(1, sum_points+1); impluse_y(1, center_1) = 1;impluse_y(1, center_2) = 1;
y_line_2 = conv(impluse_y, impluse_x, 'same');
fft_y_2 = fftshift(fft(y_line_2));
plot(y_line_2, 'Color', [200,36,35]./255); hold on; ylim([-0.2 1.2]);
plot(y_line_1, 'Color', [40,120,181]./255); hold on; ylim([-0.2 1.2]);
ylabel('Intensity');
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);
% xticks(-4:2:4);
plot(center_1+2, y_line_1(center_1+2), 'ob'); hold on;
text(center_1+5, y_line_1(center_1+2), ['x=40'] );hold on;
plot(center_1, y_line_1(center_1+2), 'ob'); hold on;
text(center_1-3, y_line_1(center_1+2), ['x=38'], 'HorizontalAlignment','right');hold on;
legend('Shifted', 'Original');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');


lenth_fft_y_1 = length(fft_y_1(:));
max_ht = floor(log2(lenth_fft_y_1)) - 2; max_ht
f_center = floor(lenth_fft_y_1/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y_1/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);

% wavelenth: [4, 8, 16, 32]
x_shift = 2;
scales_phase = [x_shift/4*2*pi, x_shift/8*2*pi, x_shift/16*2*pi, 0];
scales = [];
total_scales = 4;
total_mask = zeros(1, sum_points+1);
total_phase_pre = zeros(total_scales, sum_points+1);
total_phase_cur = zeros(total_scales, sum_points+1);
total_phase_ref = zeros(total_scales, sum_points+1);
for s = 1:total_scales
    Xrcos = Xrcos_copy;
    if (s ==1)
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);himask(1:f_center-1) = 0;
        hi_res = 2 * himask .* fft_y_1 .* himask;
        total_mask = total_mask + himask.*himask;
%         figure; plot(himask);
    end
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    total_mask = total_mask + radial_mask .* radial_mask;
%     figure; plot(radial_mask);
% space_mask = ifft(ifftshift(radial_mask)); figure; plot(abs(space_mask)); hold on;
    temp_re = ifft(ifftshift(radial_mask .* fft_y_1));  %figure;plot(real(temp_re));figure;plot(imag(temp_re));
    temp_re_2 = ifft(ifftshift(radial_mask .* fft_y_2));
    temp_phase = angle(temp_re); total_phase_pre(s,:) = temp_phase;
    temp_phase_2 = angle(temp_re_2); total_phase_ref(s,:) = temp_phase_2;
    phase_shift = scales_phase(s);
% temp_phase = mod(pi + temp_phase + phase_shift, 2*pi)-pi; %figure;plot(temp_phase);
% ������λ����
    inv_temp_re = zeros(1, sum_points+1);
    inv_temp_re(1:20) = temp_re(1:20);
    inv_temp_re(21:60) = temp_re(21:60) .* exp(1i * phase_shift);
    inv_temp_re(61:end) = temp_re(61:end); %figure;plot(angle(inv_temp_re)); 
    total_phase_cur(s,:) = angle(inv_temp_re);
    inv_re = fftshift(fft(inv_temp_re));
    scales = [scales; 2*inv_re .* radial_mask];
    if (s == total_scales)
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);lomask(1:f_center-1) = 0;
        lo_res = 2 * lomask .* lomask .* fft_y_1; 
        lo_res(f_center) = lo_res(f_center)/2;
        total_mask = total_mask + lomask .* lomask;
%         figure; plot(lomask);
    end
end
% �ؽ�
% hi_res = himask .* fft_y_1 .* himask; hi_res(1:40) = 0;
% lo_res = lomask .* fft_y_1 .* lomask; lo_res(1:40) = 0;
f_res = lo_res + hi_res;
for s = 1:total_scales
    f_res = f_res + scales(s,:);
end
res = real(ifft(ifftshift(f_res)));
f_2 = figure; f_2.Position(3:4) = [800 400];
plot(res, 'Color', [200,36,35]./255, 'LineStyle', '-'); hold on; ylim([-0.2 1.2]);
plot(y_line_1, 'Color', [40,120,181]./255, 'LineStyle', '-'); hold on; ylim([-0.2 1.2]);

ylabel('Intensity');
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);
plot(center_1+2, y_line_1(center_1+2), 'ob'); hold on;
text(center_1+5, y_line_1(center_1+2), ['x=40'] );hold on;
plot(center_1, res(center_1), 'ob'); hold on;
text(center_1-3,res(center_1), ['x=38'], 'HorizontalAlignment','right');hold on;
legend('Phase shifted', 'Original');
legend('Location','northeast');
legend('boxoff');
% figure; plot(total_mask);
% figure; plot(abs(total_mask .* fft_y_1));
% figure; plot(real(ifft(ifftshift(total_mask .* fft_y_1))));

% ����ǰ��ֲ���λ
idx_level = 2;
f_3 = figure; f_3.Position(3:4) = [800 400];
plot(total_phase_pre(idx_level,:), 'Color', [40,120,181]./255, 'LineStyle', '-'); hold on;
plot(total_phase_cur(idx_level,:), 'Color', [200,36,35]./255, 'LineStyle', '-.');
ylabel('Local Phase(rad)');
xlabel('x(pixel)');
yticks(-4 : 2 : 4);
legend('Original Phase', 'Shifted Phase ');
legend('Location','northeast');
legend('boxoff');

f_4 = figure; f_4.Position(3:4) = [800 400];
plot(total_phase_ref(idx_level,:), 'Color', [40,120,181]./255, 'LineStyle', '-'); hold on;
plot(total_phase_cur(idx_level,:), 'Color', [200,36,35]./255, 'LineStyle', '-.');
ylabel('Local Phase(rad)');
xlabel('x(pixel)');
yticks(-4 : 2 : 4);
legend( 'True Phase', 'Shifted Phase ');
legend('Location','northeast');
legend('boxoff');

% plot(angle(fft_y_2)); hold on;
% plot(abs(fft_y_2)); hold on;

% % �鿴���ձ仯�Ƿ�Ծֲ�������Ӱ�� 
% y_line_3 = y_line_1 + 2;
% % y_line_3 = zeros(1,sum_points+1)+1;
% fft_y_3 = fftshift(fft(y_line_3));
% total_scales = 1;
% for s = 1:total_scales
%     Xrcos = Xrcos_copy;
%     if (s ==1)
%         himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);himask(1:f_center-1) = 0;
%         hi_res = 2 * himask .* fft_y_3 .* himask;
%         total_mask = total_mask + himask.*himask;
%     end
%     for i = 1:s
%         lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
%         Xrcos = Xrcos - 1;
%         himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
%     end
%     radial_mask = lomask .* himask; 
%     radial_mask(1:f_center-1) = 0;
%     total_mask = total_mask + radial_mask .* radial_mask;
% 
%     temp_re = ifft(ifftshift(radial_mask .* fft_y_3));  %figure;plot(abs(temp_re));
%     figure;plot(real(temp_re));
% %     figure;plot(imag(temp_re));
%     temp_re_2 = ifft(ifftshift(radial_mask .* fft_y_1)); %figure;plot(abs(temp_re_2));
%     figure;plot(real(temp_re_2));
%     figure;plot(imag(temp_re_2));

%     figure; plot(abs(temp_re_2)-abs(temp_re));
% end

%% ͼ2-3/2-4
% description
% 1D��Ե����˵��
f = figure;
f.Position(3:4) = [500 300];

sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (1/(sqrt(2*pi)*sigma)) * (exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 

impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);
impluse_y = zeros(1, sum_points+1);
center_1 = 50;
center_2 = 80;

% б��ģ��
edge_1 = ([center_1:1:center_2]-center_1)./(center_2-center_1);
impluse_y(center_1:center_2) = edge_1;
impluse_y(center_2+1:end) = 1;
slice_1_y = conv(impluse_y, impluse_x, 'same');
slice_1_y(center_2+windows_lenth:end) = 1;
plot(slice_1_y, 'Color', [200,36,35]./255); hold on;
ylim([-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);

% f_1 = figure; f_1.Position(3:4) = [500 300];
% noise_mean = 0;
% noise_var = 0.01;
% noise_y = imnoise(slice_1_y(:),'gaussian',noise_mean,noise_var);
% figure(f_1); plot(noise_y); hold on;
% ylim([-0.2 1.2]);
% ylabel('Intensity');
% xlim([0 121]);
% xlabel('x(pixel)');
% yticks(-0.2 : 0.2 : 1.2);

f_2 = figure; f_2.Position(3:4) = [500 100];
im_edge_height = 20;
im_edge_1 = zeros(im_edge_height, sum_points+1);
for i=1:im_edge_height
    im_edge_1(i,:) = 255 .* slice_1_y;
end
im_edge_1 = uint8(im_edge_1);
imshow(im_edge_1,'InitialMagnification','fit');
start_scale = 2;
end_scale = 6;

% % ��λһ����˵��
local_slice_1_y = slice_1_y;
total_fft = fftshift(fft(local_slice_1_y));
ctr = ceil((sum_points+1)/2);
half_fft = abs(total_fft(ctr: end));
[sorted_half_fft, sorted_idx] = sort(half_fft, 'descend');
f_2_1 = figure; f_2_1.Position(3:4) = [500 300];
f_2_3 = figure; f_2_3.Position(3:4) = [500 300];
scales_phase = zeros(2, sum_points+1);
% f_2_7 = figure; f_2_7.Position(3:4) = [500 300];
% ������
for i=start_scale:end_scale
    single_mask = zeros(1,sum_points+1);
    single_mask(sorted_idx(i)+ctr-1) = 1;
    single_mask(ctr-abs(sorted_idx(i))+1) = 1;
    single_fft = single_mask .* total_fft;
    figure(f_2_1);plot(ifft(ifftshift(single_fft)));hold on;
    ylim([-0.8 0.8]);
    ylabel('Intensity');
    xlim([0 121]);
    xlabel('x(pixel)');
    yticks(-0.8 : 0.2 : 0.8);
    
    inv_single = ifft(ifftshift(single_fft));
    zoom_factor = abs(sorted_half_fft(i)) / abs(sorted_half_fft(start_scale));
    complex_fft = single_fft;
    complex_fft(ctr-abs(sorted_idx(i))+1) = 0;
    inv_complex = ifft(ifftshift(complex_fft));
    global_phase = atan2(imag(inv_complex),...
                    real(inv_complex)) * zoom_factor;
    scales_phase(1,:) = scales_phase(1,:) + global_phase;
    figure(f_2_3);plot(global_phase);hold on;
    ylim([-pi-0.5 pi+0.5]);
    ylabel('Phase(rad)');
    xlim([0 121]);
    xlabel('x(pixel)');
    yticks(-pi : pi/2 : pi);
    yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});    
end
% figure(f_2_7);plot(mod(pi+scales_phase(1,:), 2*pi)-pi);hold on;
% ������
% f_2_1_2 = figure; f_2_1_2.Position(3:4) = [500 300];
% f_2_3_2 = figure; f_2_3_2.Position(3:4) = [500 300];
% local_slice_1_y = noise_y';
% total_fft_2 = fftshift(fft(local_slice_1_y));
% ctr = ceil((sum_points+1)/2);
% half_fft = abs(total_fft_2(ctr: end));
% [sorted_half_fft, sorted_idx] = sort(half_fft, 'descend');
% for i=start_scale:end_scale
%     single_mask = zeros(1,sum_points+1);
%     single_mask(sorted_idx(i)+ctr-1) = 1;
%     single_mask(ctr-abs(sorted_idx(i))+1) = 1;
%     single_fft = single_mask .* total_fft_2;
%     figure(f_2_1_2);plot(ifft(ifftshift(single_fft)));hold on;
%     ylim([-0.8 0.8]);
%     ylabel('Intensity');
%     xlim([0 121]);
%     xlabel('x(pixel)');
%     yticks(-0.8 : 0.2 : 0.8);
    
%     inv_single = ifft(ifftshift(single_fft));
%     zoom_factor = abs(sorted_half_fft(i)) / abs(sorted_half_fft(start_scale));
%     complex_fft = single_fft;
%     complex_fft(ctr-abs(sorted_idx(i))+1) = 0;
%     inv_complex = ifft(ifftshift(complex_fft));
%     global_phase = atan2(imag(inv_complex),...
%                     real(inv_complex)) * zoom_factor;
%     scales_phase(2,:) = scales_phase(2,:) + global_phase;
%     figure(f_2_3_2);plot(global_phase);hold on;
%     ylim([-pi-0.5 pi+0.5]);
%     ylabel('Phase(rad)');
%     xlim([0 121]);
%     xlabel('x(pixel)');
%     yticks(-pi : pi/2 : pi);
%     yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});    
% end
% figure(f_2_7);plot(mod(pi+scales_phase(2,:), 2*pi)-pi);hold on;
% ylim([-pi-0.5 pi+0.5]);
% ylabel('Local Phase(rad)');
% xlim([0 121]);
% xlabel('x(pixel)');
% yticks(-pi : pi/2 : pi);
% yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});

% % �ֲ���λʾ��
% ������
f_2_5 = figure; f_2_5.Position(3:4) = [500 300];
lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
total_scales = 4;
cur_scale = 4;
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    figure(f_2_5); plot(temp_phase); hold on;
end
ylim([-pi-0.5 pi+0.5]);
ylabel('Local Phase(rad)');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-pi : pi/2 : pi);
yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
% ������
lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft_2));
    temp_phase = angle(temp_re);
    figure(f_2_5); plot(temp_phase); hold on;
end
ylim([-pi-0.5 pi+0.5]);
ylabel('Local Phase(rad)');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-pi : pi/2 : pi);
yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});


% figure; plot(half_fft);

% ��ģ��
f_3 = figure; f_3.Position(3:4) = [500 300];
sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 
impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);

edge_2 = 60;
impluse_y = zeros(1, sum_points+1);
impluse_y(edge_2) = 1;
slice_2_y = conv(impluse_y, impluse_x, 'same');
slice_2_y(center_2+windows_lenth:end) = 0;
plot(slice_2_y, 'Color', [200,36,35]./255); hold on;
ylim([-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);

f_4 = figure; f_4.Position(3:4) = [500 100];
im_edge_2 = zeros(im_edge_height, sum_points+1);
for i=1:im_edge_height
    im_edge_2(i,:) = 255 .* slice_2_y;
end
im_edge_2 = uint8(im_edge_2);
imshow(im_edge_2,'InitialMagnification','fit');
start_scale = 2;
end_scale = 6;

% % ��λһ����˵��
local_slice_2_y = slice_2_y;
total_fft = fftshift(fft(local_slice_2_y));
ctr = ceil((sum_points+1)/2);
half_fft = abs(total_fft(ctr: end));
[sorted_half_fft, sorted_idx] = sort(half_fft, 'descend');
f_2_2 = figure; f_2_2.Position(3:4) = [500 300];
f_2_4 = figure; f_2_4.Position(3:4) = [500 300];
% f_2_6 = figure; f_2_6.Position(3:4) = [500 300];
scales_phase = zeros(1, sum_points+1);
for i=start_scale:end_scale
    single_mask = zeros(1,sum_points+1);
    single_mask(sorted_idx(i)+ctr-1) = 1;
    single_mask(ctr-abs(sorted_idx(i))+1) = 1;
    single_fft = single_mask .* total_fft;
    figure(f_2_2);plot(ifft(ifftshift(single_fft)));hold on;
    ylim([-0.8 0.8]);
    ylabel('Intensity');
    xlim([0 121]);
    xlabel('x(pixel)');
    yticks(-0.8 : 0.2 : 0.8);
    
    inv_single = ifft(ifftshift(single_fft));
    zoom_factor = abs(sorted_half_fft(i)) / abs(sorted_half_fft(1));
    complex_fft = single_fft;
    complex_fft(ctr-abs(sorted_idx(i))+1) = 0;
    inv_complex = ifft(ifftshift(complex_fft));
    global_phase = atan2(imag(inv_complex),...
                    real(inv_complex)) * zoom_factor;
    scales_phase(1,:) = scales_phase(1,:) + global_phase;
    figure(f_2_4);plot(global_phase);hold on;
    ylim([-pi-0.5 pi+0.5]);
    ylabel('Phase(rad)');
    xlim([0 121]);
    xlabel('x(pixel)');
    yticks(-pi : pi/2 : pi);
    yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
end
% figure(f_2_6);plot(mod(pi+scales_phase, 2*pi)-pi);hold on;

% % �ֲ���λʾ��
f_2_6 = figure; f_2_6.Position(3:4) = [500 300];
lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
total_scales = 4;
cur_scale = 3;
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    figure(f_2_6); plot(temp_phase); hold on;
end
ylim([-pi-0.5 pi+0.5]);
ylabel('Local Phase(rad)');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-pi : pi/2 : pi);
yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});

% figure; plot(half_fft);

%% ͼ2-5
% description
% ������1D��Ե����˵��

sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (1/(sqrt(2*pi)*sigma)) * (exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 

impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);
impluse_y = zeros(1, sum_points+1);
center_1 = 50;
center_2 = 80;

% б��ģ��
f_1 = figure; f_1.Position(3:4) = [500 300];
edge_1 = ([center_1:1:center_2]-center_1)./(center_2-center_1);
impluse_y(center_1:center_2) = edge_1;
impluse_y(center_2+1:end) = 1;
slice_1_y = conv(impluse_y, impluse_x, 'same');
slice_1_y(center_2+windows_lenth:end) = 1;

x_shift = 1;
noise_mean = 0;
noise_var = 0.005;
noise_tmp = zeros(1, sum_points+1); noise_tmp = noise_tmp +0.5;
noise_tmp = imnoise(noise_tmp(:),'gaussian',noise_mean,noise_var); noise_tmp = noise_tmp -0.5;
noise_y = slice_1_y + noise_tmp';
figure(f_1); plot(noise_y); hold on;
ylim([-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);

total_scales = 6;
cur_scale = 6;

% % ��λһ����˵��
scales_phase = zeros(2, sum_points+1);

% % �ֲ���λʾ��
f_2_1 = figure; f_2_1.Position(3:4) = [500 300];
ax_1 = axes('Parent', f_2_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_2_1, 'Position', [0.42 0.62 0.27 0.27], 'Box','on'); hold(ax_2, 'on');
xlim(ax_2, [59, 61]);
ylim(ax_2, [-0.6-0.05, -0.6+0.05]);
annotation(f_2_1, 'ellipse', [0.465 0.3 0.1 0.1], 'Color',[0 161 241]./255);
annotation(f_2_1, 'arrow', [0.47 0.5],[0.58 0.42], 'Color',[0 161 241]./255);

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% ������
local_slice_1_y = slice_1_y;
total_fft = fftshift(fft(local_slice_1_y));

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    figure(f_2_1); plot(temp_phase, 'Color',color_1, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_1, 'Parent',ax_2); hold on;
end

% ������
local_slice_1_y = noise_y;
total_fft_2 = fftshift(fft(local_slice_1_y));

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft_2));
    temp_phase = angle(temp_re);
    figure(f_2_1); plot(temp_phase, 'Color',color_2, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_2, 'Parent',ax_2); hold on;
end

% ��λ��
local_slice_1_y = zeros(1, sum_points+1);
local_slice_1_y(1: end-x_shift) = slice_1_y(1+x_shift: end);
local_slice_1_y(end-x_shift+1: end) = 1;
total_fft_3 = fftshift(fft(local_slice_1_y));
figure(f_1); plot(local_slice_1_y); hold on;
ylim([-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);
legend('with Noise', 'left-shift 1 pixel');
legend('Location','northwest');
legend('boxoff');

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft_3));
    temp_phase = angle(temp_re);
    figure(f_2_1); plot(temp_phase, 'Color',color_3, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_3, 'Parent',ax_2); hold on;
end
ylim(ax_1, [-pi/4-0.5 pi/4+0.5]);
ylabel(ax_1, 'Local Phase(rad)');
xlim(ax_1, [0 121]);
xlabel(ax_1, 'x(pixel)');
yticks(ax_1, -pi/4 : pi/4 : pi/4);
yticklabels(ax_1, {'-pi/4' '0' 'pi/4'});
legend(ax_1, 'Raw', 'with Noise', 'left-shift 1 pixel');
legend(ax_1,'Location','southeast');
legend(ax_1, 'boxoff');


% ��ģ��
f_3 = figure; f_3.Position(3:4) = [500 300];
sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 
impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);

edge_2 = 60;
impluse_y = zeros(1, sum_points+1);
impluse_y(edge_2) = 1;
slice_2_y = conv(impluse_y, impluse_x, 'same');
slice_2_y(center_2+windows_lenth:end) = 0;

x_shift = 1;
noise_mean = 0;
noise_var = 0.005;
noise_tmp = zeros(1, sum_points+1); noise_tmp = noise_tmp +0.5;
noise_tmp = imnoise(noise_tmp(:),'gaussian',noise_mean,noise_var); noise_tmp = noise_tmp -0.5;
noise_y = slice_2_y + noise_tmp';
figure(f_3); plot(noise_y); hold on;
ylim([-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);

total_scales = 6;
cur_scale = 6;

% % ��λһ����˵��
scales_phase = zeros(2, sum_points+1);

% % �ֲ���λʾ��
f_4_1 = figure; f_4_1.Position(3:4) = [500 300];
ax_1 = axes('Parent', f_4_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_4_1, 'Position', [0.61 0.23 0.27 0.23], 'Box','on'); hold(ax_2, 'on');
xlim(ax_2, [59, 61]);
ylim(ax_2, [0-0.05, 0+0.05]);
annotation(f_4_1, 'ellipse', [0.465 0.48 0.1 0.1], 'Color',[0 161 241]./255);
annotation(f_4_1, 'arrow', [0.74 0.58],[0.47 0.53], 'Color',[0 161 241]./255);

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% ������
local_slice_1_y = slice_2_y;
total_fft = fftshift(fft(local_slice_1_y));

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    figure(f_4_1); plot(temp_phase, 'Color',color_1, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_1, 'Parent',ax_2); hold on;
end

% ������
local_slice_1_y = noise_y;
total_fft_2 = fftshift(fft(local_slice_1_y));

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft_2));
    temp_phase = angle(temp_re);
    figure(f_4_1); plot(temp_phase, 'Color',color_2, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_2, 'Parent',ax_2); hold on;
end

% ��λ��
local_slice_1_y = zeros(1, sum_points+1);
local_slice_1_y(1: end-x_shift) = slice_2_y(1+x_shift: end);
local_slice_1_y(end-x_shift+1: end) = 0;
total_fft_3 = fftshift(fft(local_slice_1_y));
figure(f_3); plot(local_slice_1_y); hold on;
ylim(ax_1, [-0.2 1.2]);
ylabel('Intensity');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-0.2 : 0.2 : 1.2);
legend('with Noise', 'left-shift 1 pixel');
legend('Location','northwest');
legend('boxoff');

lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft_3));
    temp_phase = angle(temp_re);
    figure(f_4_1); plot(temp_phase, 'Color',color_3, 'Parent',ax_1); hold on;
    plot(temp_phase, 'Color',color_3, 'Parent',ax_2); hold on;
end
ylim(ax_1, [-pi/2-0.5 pi/2+0.5]);
ylabel(ax_1, 'Local Phase(rad)');
xlim(ax_1, [0 121]);
xlabel(ax_1, 'x(pixel)');
yticks(ax_1, -pi/2 : pi/2 : pi/2);
yticklabels(ax_1, {'-pi/2' '0' 'pi/2'});
legend(ax_1, 'Raw', 'with Noise', 'left-shift 1 pixel');
legend(ax_1, 'Location','northwest');
legend(ax_1, 'boxoff');

%% ͼ2-6/2-7
% description
% �ռ���λ�ݶ�˵��

% ͼ2-6
f_1 = figure; f_1.Position(3:4) = [500 300];
sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (1/(sqrt(2*pi)*sigma)) *(exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 
impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);

center_1 = 50;
center_2 = 80;
edge_1 = ([center_1:1:center_2]-center_1)./(center_2-center_1);
impluse_y(center_1:center_2) = edge_1;
impluse_y(center_2+1:end) = 1;
slice_y = conv(impluse_y, impluse_x, 'same');
slice_y(center_2+windows_lenth:end) = 1;

% edge_2 = 60;
% impluse_y = zeros(1, sum_points+1);
% impluse_y(edge_2) = 1;
% slice_y = conv(impluse_y, impluse_x, 'same');
% slice_y(edge_2+windows_lenth:end) = 0;

total_scales = 6;
cur_scale = 6;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;

local_slice_1_y = slice_y;
total_fft = fftshift(fft(local_slice_1_y));
lenth_fft_y = length(total_fft(:));
f_center = floor(lenth_fft_y/2)+1;
log_rad = [-sum_points/2:sum_points/2]./(lenth_fft_y/2);
log_rad = log2(sqrt(log_rad .^2)); log_rad(f_center) = log_rad(f_center+1);
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    figure(f_1); plot(temp_phase, 'Color',color_3); hold on;
    
    ylim([-pi-0.5 pi+0.5]);
    ylabel('Local Phase(rad)');
    xlim([0 121]);
    xlabel('x(pixel)');
    yticks(-pi : pi/2 : pi);
    yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});  
end

% ��ɢ���Ĳ��̼���ռ���λ�ݶ�
f_2 = figure; f_2.Position(3:4) = [500 300];
dif_phase = zeros(1, sum_points+1);
for i=2:sum_points-1
    dif_phase(i) = (temp_phase(i+1)-temp_phase(i-1))/2;
end
dif_phase(1) = dif_phase(2);
dif_phase(sum_points+1) = dif_phase(sum_points);
figure(f_2); plot(dif_phase, 'Color',color_2,  'LineStyle', '-', 'LineWidth', 1); hold on;


% ʹ�ø������Ĳ��̼���ռ���λ�ݶ�
dif_phase_2 = zeros(1, sum_points+1);
for i=2:sum_points-1
    delta_re = (real(temp_re(i+1))-real(temp_re(i-1)))/2;
    delta_im = (imag(temp_re(i+1))-imag(temp_re(i-1)))/2;
    dif_phase_2(i) = (real(temp_re(i)) * delta_im - imag(temp_re(i)) * delta_re)/(abs(temp_re(i)).^2);
end
dif_phase_2(1) = dif_phase_2(2);
dif_phase_2(sum_points+1) = dif_phase_2(sum_points);
figure(f_2); plot(dif_phase_2, 'Color',color_3, 'LineStyle', '-.', 'LineWidth', 1.5); hold on;
figure(f_2); plot(abs(temp_re)); hold on;
% figure(f_2); plot(real(temp_re)); hold on;
ylim([-pi-0.5 pi/2+0.5]);
ylabel('Phase Gradient(rad/pixel)');
xlim([0 121]);
yticks(-pi : pi/2 : pi/2);
yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
xlabel('x(pixel)');
legend('Wrapped', 'True');
legend('Location','northeast');
legend('boxoff');

% ͼ2-7
amp = zeros(2, sum_points+1);
f_3_1 = figure; f_3_1.Position(3:4) = [500 300];
total_scales = 2;
cur_scale = 2;
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    temp_re = temp_re ./abs(temp_re);
    amp(1,:) = abs(temp_re);
end
% ��ɢ���Ĳ��̼���ռ���λ�ݶ�
dif_phase = zeros(1, sum_points+1);
for i=2:sum_points-1
    dif_phase(i) = (temp_phase(i+1)-temp_phase(i-1))/2;
end
dif_phase(1) = dif_phase(2);
dif_phase(sum_points+1) = dif_phase(sum_points);
% ʹ�ø������Ĳ��̼���ռ���λ�ݶ�
dif_phase_2 = zeros(1, sum_points+1);
for i=2:sum_points-1
    delta_re = (real(temp_re(i+1))-real(temp_re(i-1)))/2;
    delta_im = (imag(temp_re(i+1))-imag(temp_re(i-1)))/2;
    dif_phase_2(i) = (real(temp_re(i)) * delta_im - imag(temp_re(i)) * delta_re)/(abs(temp_re(i)).^2);
end
dif_phase_2(1) = dif_phase_2(2);
dif_phase_2(sum_points+1) = dif_phase_2(sum_points);
for i=1:sum_points+1
    if (abs(dif_phase(i)-dif_phase_2(i)) > pi/2)
        dif_phase(i) = dif_phase(i)-sign(dif_phase(i)-dif_phase_2(i))*pi;
    end
end

figure(f_3_1); plot(dif_phase, 'Color',color_2,  'LineStyle', '-', 'LineWidth', 1); hold on;
figure(f_3_1); plot(dif_phase_2, 'Color',color_3, 'LineStyle', '-.', 'LineWidth', 1.5); hold on;
ylim([0-0.01 1.2+0.01]);
ylabel('Phase Gradient(rad/pixel)');
xlim([0 121]);
% yticks(-pi : pi/2 : pi/2);
% yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
xlabel('x(pixel)');
legend('differential', 'difference quotient');
legend('Location','northeast');
legend('boxoff');

f_3_2 = figure; f_3_2.Position(3:4) = [500 300];
total_scales = 6;
cur_scale = 6;
for s = total_scales:-1:cur_scale
    Xrcos = Xrcos_copy;
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - 1;
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);        
    end
    radial_mask = lomask .* himask; 
    radial_mask(1:f_center-1) = 0;
    temp_re = ifft(ifftshift(radial_mask .* total_fft));
    temp_phase = angle(temp_re);
    temp_re = temp_re ./abs(temp_re);
    amp(2,:) = abs(temp_re);
end
% ��ɢ���Ĳ��̼���ռ���λ�ݶ�
dif_phase = zeros(1, sum_points+1);
for i=2:sum_points-1
    dif_phase(i) = (temp_phase(i+1)-temp_phase(i-1))/2;
end
dif_phase(1) = dif_phase(2);
dif_phase(sum_points+1) = dif_phase(sum_points);
% ʹ�ø������Ĳ��̼���ռ���λ�ݶ�
dif_phase_2 = zeros(1, sum_points+1);
for i=2:sum_points-1
    delta_re = (real(temp_re(i+1))-real(temp_re(i-1)))/2;
    delta_im = (imag(temp_re(i+1))-imag(temp_re(i-1)))/2;
    dif_phase_2(i) = (real(temp_re(i)) * delta_im - imag(temp_re(i)) * delta_re)/(abs(temp_re(i)).^2);
end
dif_phase_2(1) = dif_phase_2(2);
dif_phase_2(sum_points+1) = dif_phase_2(sum_points);

figure(f_3_2); plot(dif_phase, 'Color',color_2,  'LineStyle', '-', 'LineWidth', 1); hold on;
figure(f_3_2); plot(dif_phase_2, 'Color',color_3, 'LineStyle', '-.', 'LineWidth', 1.5); hold on;
ylim([-0.1 0.05+0.01]);
ylabel('Phase Gradient(rad/pixel)');
xlim([0 121]);
% yticks(-pi : pi/2 : pi/2);
% yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
xlabel('x(pixel)');
legend('differential', 'difference quotient');
legend('Location','northeast');
legend('boxoff');

f_3_3 = figure; f_3_3.Position(3:4) = [500 300];
figure(f_3_3); plot(amp(1,:), 'Color',color_2,  'LineStyle', '-', 'LineWidth', 1.5); hold on;
figure(f_3_3); plot(amp(2,:), 'Color',color_3, 'LineStyle', '-.', 'LineWidth', 1.5); hold on;
ylim([-0-0.25 0.6+0.25]);
ylabel('Amplitude');
xlim([0 121]);
% yticks(-pi : pi/2 : pi/2);
% yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
xlabel('x(pixel)');
legend('small Amp', 'large Amp');
legend('Location','northwest');
legend('boxoff');

%% ͼ2-8
% description
% ��Ƶ��

f_1 = figure; f_1.Position(3:4) = [500 300];
sum_points = 120;
windows_lenth = 20;
mean_center = 0; sigma = 1/5*windows_lenth;
gaussian_filter = @(query_x) (1/(sqrt(2*pi)*sigma)) *(exp(-(query_x-mean_center).^2 / (2*sigma^2))); % (1/(sqrt(2*pi)*sigma)) * 
impluse_x = -windows_lenth:1:windows_lenth;
impluse_x = gaussian_filter(impluse_x);

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;

T = 20;
omega = 2*pi/T;
omega_2 = omega/1.3;
init_phase = pi;%2*pi;
ctr = floor((sum_points+1)/2);

compute_y = @(query_x) mod((omega * (query_x-ctr) + init_phase), 2*pi)-pi;
compute_y_3 = @(query_x) mod((omega_2 * (query_x-ctr) + init_phase), 2*pi)-pi;

x_line = [1:1:sum_points+1];
y_line = compute_y(x_line);
y_line_3 = compute_y_3(x_line);
figure(f_1); plot(y_line, 'Color',color_2, 'LineStyle', '-.','LineWidth', 1.5); hold on;
figure(f_1); plot(y_line_3, 'Color',color_1, 'LineStyle', '-.','LineWidth', 1.5); hold on;

new_phase = mod(pi+y_line-y_line_3,2*pi)-pi;
figure(f_1); plot(new_phase, 'Color',color_3, 'LineStyle', '-','LineWidth', 1.5); hold on;
ylim([-pi-1.5 pi+1.5]);
ylabel('Local Phase(rad)');
xlim([0 121]);
xlabel('x(pixel)');
yticks(-pi : pi/2 : pi);
yticklabels({'-pi' '-pi/2' '0' 'pi/2' 'pi'});
legend('High', 'Low', 'Compose');
legend('Location','northeast');
legend('boxoff');
legend('Orientation','horizontal');

%% ͼ2-9
% description
% CSPʾ��

sum_points = 120;
f_center = floor((sum_points+1)/2);
f_1 = figure; f_1.Position(3:4) = [400 400];
im_size = [600 600];
total_scales = 3;
total_angles = 4;

dims = im_size;
ctr = floor(im_size ./ 2)+1;
[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
                          ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle_map = atan2(yramp,xramp);                      
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);
log_rad  = log2(log_rad);

twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Xrcos_copy = Xrcos;
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);

lutsize = 1024;
Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;
alfa=	mod(pi+Xcosn,2*pi)-pi;
nbands = total_angles;
order = nbands-1;
const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
Ycosn = 2*sqrt(const) * (cos(Xcosn).^order) .* (abs(alfa)<pi/2);

total_mask = zeros(dims(1), dims(2), 3);
total_phase_pre = zeros(total_scales, sum_points+1);
total_phase_cur = zeros(total_scales, sum_points+1);
total_phase_ref = zeros(total_scales, sum_points+1);

% ��ɫ��
colors = zeros(total_scales, total_angles, 3);
colors(1,1,:) = [56,83,163]; colors(1,2,:) = [58,91,169]; colors(1,3,:) = [64,119,188]; colors(1,4,:) = [62,187,236];
colors(2,1,:) = [109,204,220]; colors(2,2,:) = [121,201,168]; colors(2,3,:) = [151,205,111]; colors(2,4,:) = [194,215,51];
colors(3,1,:) = [242,235,23]; colors(3,2,:) = [253,191,17]; colors(3,3,:) = [245,127,33]; colors(3,4,:) = [239,69,34];
color_hi_res = [127,19,21];
color_lo_res = [127,19,21];

for s = 1:total_scales
    Xrcos = Xrcos_copy;
    if (s ==1)
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
%         figure; surf(himask);
        for i = 1:dims(1)
            for j = 1:dims(2)
                if (himask(i,j)==1)
                    total_mask(i,j,:) = color_hi_res;
                end
            end
        end
    end
    for i = 1:s
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - log2(2);
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        radial_mask = lomask .* himask;
    end
    for angle = 1:total_angles
        anglemask = pointOp(angle_map, Ycosn, Xcosn(1)+pi*(angle-1)/nbands, Xcosn(2)-Xcosn(1));
        mask = radial_mask .* anglemask;
        if (s==1 && angle==2)
            test_mask = mask;
        end
        for i = 1:dims(1)
            for j = 1:dims(2)
                if (mask(i,j)>0 && (angle_map(i,j)>= pi*(angle-1.5)/nbands && angle_map(i,j)<= pi*(angle-0.5)/nbands))
                    total_mask(i,j,:) = colors(s,angle,:);
                end
            end
        end
    end
    

    if (s == total_scales)
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);lomask(1:f_center-1) = 0;
        
        for i = 1:dims(1)
            for j = 1:dims(2)
                if (lomask(i,j)>0)
                    total_mask(i,j,:) = color_lo_res;
                end
            end
        end
    end  
end
figure(f_1); imshow(uint8(total_mask),'InitialMagnification','fit');
% title( 'Frequency domain','FontName','Times New Roman');
% title( 'Frequency domain');
ylabel('\omega\it_y \rm(1/pixel)');
xlabel('\omega\it_x \rm(1/pixel)');

f_2 = figure; f_2.Position(3:4) = [400 400];
figure(f_2); imshow(test_mask,'InitialMagnification','fit');
% annotation(f_1, 'rectangle', [0.55 0.3 0.1 0.1], 'Color',[255 0 0]./255);

f_3 = figure; f_3.Position(3:4) = [460 400];
test_im = ifftshift(ifft2(test_mask));
test_width = 8;
re_test = real(test_im(ctr(1)-test_width:ctr(1)+test_width,ctr(2)-test_width:ctr(2)+test_width));
% figure(f_3); imshow(uint8((re_test+0.1)./0.3),'InitialMagnification','fit');
figure(f_3); 
surf(re_test, 'EdgeColor', 'none', 'FaceColor', 'interp'); view(0, -90);
colorbar('Position', [0.85 0.155 0.05 0.73]);
axis off;

f_4 = figure; f_4.Position(3:4) = [460 400];
im_test = imag(test_im(ctr(1)-test_width:ctr(1)+test_width,ctr(2)-test_width:ctr(2)+test_width));
figure(f_4); 
surf(im_test, 'EdgeColor', 'none', 'FaceColor', 'interp'); view(0, -90);
colorbar('Position', [0.85 0.155 0.05 0.73]);
axis off;

%% ͼ2-10/��2-1
% description
% �߶Ȳ���Ӱ��
% �ǵ������������ݵĴ���

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';

% 
% video_path = video_raw;
% vr_1 = VideoReader(video_path);
% im_1 = vr_1.read(1);
% f_1 = figure; f_1.Position(3:4) = [300 300];
% figure(f_1); imshow(im_1,'InitialMagnification','fit');

% 
video_path = video_back_origin;
vr_0 = VideoReader(video_path);
im_0 = vr_0.read(1);
f_0 = figure; f_0.Position(3:4) = [300 300];
figure(f_0); imshow(im_0,'InitialMagnification','fit');

% 
video_path = video_back_level_1;
vr_1 = VideoReader(video_path);
im_1 = vr_1.read(1);
f_1 = figure; f_1.Position(3:4) = [300 300];
figure(f_1); imshow(im_1,'InitialMagnification','fit');

% 
video_path = video_back_level_2;
vr_2 = VideoReader(video_path);
im_2 = vr_2.read(1);
f_2 = figure; f_2.Position(3:4) = [300 300];
figure(f_2); imshow(im_2,'InitialMagnification','fit');

% 
video_path = video_back_level_3;
vr_3 = VideoReader(video_path);
im_3 = vr_3.read(1);
f_3 = figure; f_3.Position(3:4) = [300 300];
figure(f_3); imshow(im_3,'InitialMagnification','fit');

% ��������
[psnr_2, snr_2] = psnr(im_3, im_0);
fprintf("psnr_2: %f\n", psnr_2);
fprintf("snr_2: %f\n", snr_2);

%% ͼ2-11
% description
% �߶Ȳ���Ӱ��
% ��ͬ�߶�ʶ����

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 4;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
colors = [color_1; color_2; color_3; color_4];

% cur_scale = 2;
% start_scale = cur_scale;
% end_scale = cur_scale;
nF = 400;
dispout = zeros(4, 800);
f = figure;
f.Position(3:4) = [900 400];
for s = 1:3
    line = animatedline('Color',colors(s,:));
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
    cur_scale = s;
    start_scale = cur_scale;
    end_scale = cur_scale;
    for i = 2:300
        phase_scales = zeros(2, scales, dim(1), dim(2));
        re_scales = zeros(2, scales, dim(1), dim(2));
        for scale = start_scale:end_scale
            vframein = vHandle.read(i);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
            im_dft = mask .* fftshift(fft2(image));
            temp_re = ifft2(ifftshift(im_dft));
            temp_phase = angle(temp_re);
            
            % ʹ�òο�֡��
            vframein = vHandle.read(1);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            im_dft = mask .* fftshift(fft2(image));
            temp_re_2 = ifft2(ifftshift(im_dft));
            temp_phase_2 = angle(temp_re_2);
            
            phase_scales(2,1,:,:) = temp_phase;
            phase_scales(1,1,:,:) = temp_phase_2;
            
            re_scales(2,1,:,:) = temp_re;
            re_scales(1,1,:,:) = temp_re_2;
            
        end
        % ʹ���ںϲ���
%         tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        % ʹ����λ�ݶ��Ƶ�ʽ
        tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
        addpoints(line,xline(i),double(disp(1)));
        figure(f);
        title(['frame:',num2str(i)]);
    end
    
end

%% ͼ2-11��plot��/��2-2
% description
% ��ͼ���֣��������ǰ�����

compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3); hold on;
figure(f_1); plot(dispout(1,1:300), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5); hold on;
figure(f_1); plot(dispout(2,1:300), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5); hold on;
figure(f_1); plot(dispout(3,1:300), 'Color',colors(4,:), 'LineStyle', '-.','LineWidth', 1.5); hold on;
ylim([-0.2-0.1 0.2+0.1]);
ylabel('Amplitude(pixels)');
xlabel('Time(s)');
xticks(0 : 60 : 300);
xticklabels({'0' '2' '4' '6' '8' '10'});
legend('True', 'scale 1', 'scale 2', 'scale 3');
legend('Location','northeast');
legend('boxoff');
legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
% figure(f_2); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3); hold on;
figure(f_2); plot(dispout(1,1:300)-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
figure(f_2); plot(dispout(2,1:300)-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
figure(f_2); plot(dispout(3,1:300)-lowrate_y(1,1:300), 'Color',colors(4,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
ylim([-0.1-0.05 0.1+0.05]);
ylabel('Error(pixels)');
xlabel('Time(s)');
xticks(0 : 60 : 300);
xticklabels({'0' '2' '4' '6' '8' '10'});
legend('scale 1', 'scale 2', 'scale 3');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

for i = 1:3
    cor_x = dispout(i,1:300);
    cor_y = lowrate_y(1,1:300);
    coeff = corr(cor_x', cor_y','type','pearson');
    fprintf("Scale: %d pearson_corr: %f\n", i, coeff);
    rmse = sqrt(mean((dispout(i,1:300)-lowrate_y(1,1:300)).^2));
    fprintf("Scale: %d rmse: %f\n", i,  rmse);
end

%% ͼ2-12
% description
% �������Ӱ��
% ��ͬ����ʶ����

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 1;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
colors = [color_1; color_2; color_3; color_4];

cur_scale = 3;
start_scale = cur_scale;
end_scale = cur_scale;
angle_scale = [1 2 4 8 16];
r_scales = zeros(5, 181);
rmse_scales = zeros(5, 181);
nF = 100;
dispout = zeros(5, 181, 800);
f = figure;
f.Position(3:4) = [900 400];
for n_angle = 3:3
     fprintf("/------\nN: %f\n------/\n", n_angle);
%     line = animatedline('Color',colors(n_angle,:));
%     line = animatedline;
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
%     cur_scale = s;
%     start_scale = cur_scale;
%     end_scale = cur_scale;
    for a = 0:180
        fprintf("angle: %f\n", a);
%         line = animatedline;
        for i = 2:nF
            phase_scales = zeros(2, scales, dim(1), dim(2));
            re_scales = zeros(2, scales, dim(1), dim(2));
            for scale = start_scale:end_scale
                vframein = vHandle.read(i);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                mask = mask_creator(size(im), 2, angle_scale(n_angle), [scale, a/180*pi], 1);
                im_dft = mask .* fftshift(fft2(image));
                temp_re = ifft2(ifftshift(im_dft));
                temp_phase = angle(temp_re);

                % ʹ�òο�֡��
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);

                phase_scales(2,1,:,:) = temp_phase;
                phase_scales(1,1,:,:) = temp_phase_2;

                re_scales(2,1,:,:) = temp_re;
                re_scales(1,1,:,:) = temp_re_2;

            end
            % ʹ���ںϲ���
    %         tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
            % ʹ����λ�ݶ��Ƶ�ʽ
            tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, a, iou_threshold);
%             dispout(n_angle,a+1,i) = (cos((oe-a)/180*pi)) * x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, a, iou_threshold);
            tdisp = tdisp * (cos((oe-a)/180*pi));
            disp(1) = tdisp; dispout(n_angle,a+1,i) = disp(1);               % ʹ�òο�֡��
%             addpoints(line,xline(i),double(disp(1)));
%             figure(f);
%             title(['frame:',num2str(i)]);
        end
        cor_x = squeeze(dispout(n_angle,a+1,1:nF));
        cor_x = cor_x';
        cor_y = lowrate_y(1,1:nF);
        coeff = corr(cor_x', cor_y','type','pearson'); r_scales(n_angle, a+1) = coeff;
        fprintf("Angle: %d pearson_corr: %f\t", a, coeff);
        rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(n_angle, a+1) = rmse;
        fprintf("Angle: %d rmse: %f\n", a,  rmse);
    end
    
end

%% ͼ2-12��plot��
% description
% ��ͼ���֣��������ǰ�����
r_mat_name = ['data_r_1.mat'; 'data_r_2.mat'; 'data_r_3.mat';];
rmse_mat_name = ['data_rmse_1.mat'; 'data_rmse_2.mat'; 'data_rmse_3.mat'];
        
color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
colors = [color_1; color_2; color_3; color_4];        
        
f_1 = figure; f_1.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_1, 'Position', [0.24 0.42 0.39 0.27], 'Box','on'); hold(ax_2, 'on');
load(r_mat_name(1,:));
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(1,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(1,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
load(r_mat_name(2,:));
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
load(r_mat_name(3,:));
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_1); plot(r_scales(3,1:181), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
ylim(ax_1, [-1-0.1 1+0.1]);
ylabel(ax_1, 'r');
xlim(ax_1, [0 181]);
xlabel(ax_1, '\theta (deg)');
yticks(ax_1, -1 : 0.5 : 1);
yticklabels(ax_1, {'-1' '-0.5' '0' '0.5' '1'});
legend(ax_1, 'Scale 1', 'Scale 2', 'Scale 3');
legend(ax_1, 'Location','southwest');
legend(ax_1, 'boxoff');
% legend('Orientation','horizontal');
ylim(ax_2, [0.998-0.001 0.999+0.002]);
xlim(ax_2, [0 150]);
yticks(ax_2, 0.997 : 0.001 : 0.999+0.002);
% yticklabels(ax_2, {'-1' '0' '1'});

f_2 = figure; f_2.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_2,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_2, 'Position', [0.245 0.57 0.39 0.27], 'Box','on'); hold(ax_2, 'on');
load(rmse_mat_name(1,:));
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(1,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(1,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
load(rmse_mat_name(2,:));
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
load(rmse_mat_name(3,:));
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(rmse_scales(3,1:181), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
ylim(ax_1, [0-0.005 0.15+0.01]);
ylabel(ax_1, 'RMSE (pixel)');
xlim(ax_1, [0 181]);
xlabel(ax_1, '\theta (deg)');
yticks(ax_1, -0.005 : 0.05 : 0.15);
% yticklabels(ax_1, {'-1' '-0.5' '0' '0.5' '1'});
legend(ax_1, 'Scale 1', 'Scale 2', 'Scale 3');
legend(ax_1, 'Location','southeast');
legend(ax_1, 'boxoff');
% legend('Orientation','horizontal');
ylim(ax_2, [0-0.005 0.01+0.005]);
xlim(ax_2, [0 150]);
yticks(ax_2, -0.005 : 0.005 : 0.01);

%% ͼ2-13
% description
% �������Ӱ��
% ��ͬ����ʶ����

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 1;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

% cur_scale = 2;
% start_scale = cur_scale;
% end_scale = cur_scale;
angle_scale = [1 2 4 8 16 32];
% r_scales = zeros(5, 1);
mae_scales = zeros(3, 5, 1);
rmse_scales = zeros(3, 5, 1);
nF = 100;
dispout = zeros(n_angle, 800);
f = figure;
f.Position(3:4) = [900 400];
for s=1:3
    start_scale = s;
    end_scale = s;
    for n_angle = 1:6
        fprintf("/------\nN: %f\n------/\n", n_angle);
        line = animatedline('Color',colors(n_angle,:));
        xline = [1:1:nF];
        oe = 65;
        iou_threshold = -1.33;
        scales = 1;
        disp = zeros(scales, 1);
        num_phase_bin = 4;
        line_width = 2;
        cos_ratio = 0.50;
        dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
        max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
        line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
        line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
    %     cur_scale = s;
    %     start_scale = cur_scale;
    %     end_scale = cur_scale;

        for i = 2:nF
            phase_scales = zeros(2, scales, dim(1), dim(2));
            re_scales = zeros(2, scales, dim(1), dim(2));
            for scale = start_scale:end_scale
                vframein = vHandle.read(i);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                mask = mask_creator(size(im), 2, angle_scale(n_angle), [scale, oe/180*pi], 1);
                im_dft = mask .* fftshift(fft2(image));
                temp_re = ifft2(ifftshift(im_dft));
                temp_phase = angle(temp_re);

                % ʹ�òο�֡��
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);

                phase_scales(2,1,:,:) = temp_phase;
                phase_scales(1,1,:,:) = temp_phase_2;

                re_scales(2,1,:,:) = temp_re;
                re_scales(1,1,:,:) = temp_re_2;

            end
            % ʹ���ںϲ���
    %         tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
            % ʹ����λ�ݶ��Ƶ�ʽ
            tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
    %             dispout(n_angle,a+1,i) = (cos((oe-a)/180*pi)) * x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, a, iou_threshold);
            disp(1) = tdisp; dispout(n_angle,i) = disp(1);               % ʹ�òο�֡��
            addpoints(line,xline(i),double(disp(1)));
            figure(f);
            title(['frame:',num2str(i)]);
        end
        cor_x = squeeze(dispout(n_angle,1:nF));
        cor_y = lowrate_y(1,1:nF);
        coeff = mean(abs(cor_x-cor_y)); mae_scales(s, n_angle, 1) = coeff;
    %     coeff = corr(cor_x', cor_y','type','pearson'); r_scales(n_angle, 1) = coeff;
        fprintf("angle_scale: %d mae: %f\t", n_angle, coeff);
        rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(s, n_angle, 1) = rmse;
        fprintf("angle_scale: %d rmse: %f\n", n_angle,  rmse);


    end
end

%% ͼ2-13��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 1;
compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
% color_3 = [255,34,0]./255; %[232,201,158]./255;
color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); 
b = bar(mae_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:5
    for i = 1:3
        b(j).CData(i,:) = colors(j,:);
    end
end
y_lim_1 = [[0.002 0.008+0.001]; [0.004 0.018]; [0.004 0.028]; [0.01 0.12]];
y_step_1 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('MAE (pixel)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('\sigma_a_f = \pi', '\sigma_a_f = \pi/2', '\sigma_a_f = \pi/4', '\sigma_a_f = \pi/8', '\sigma_a_f = \pi/16', '\sigma_a_f = \pi/32');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
figure(f_2); 
b = bar(rmse_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:5
    for i = 1:3
        b(j).CData(i,:) = colors(j,:);
    end
end
y_lim_2 = [[0.002 0.009+0.001]; [0.004 0.020]; [0.004 0.032]; [0.01 0.12]];
y_step_2 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_2(video_idx,:));
ylabel('RMSE (pixel)');
yticks(y_lim_2(video_idx,1) : y_step_2(video_idx) : y_lim_2(video_idx,2));
% yticklabels({'0' '2' '4' '6' '8' '10'});
% ytickformat('%.0f');
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('\sigma_a_f = \pi', '\sigma_a_f = \pi/2', '\sigma_a_f = \pi/4', '\sigma_a_f = \pi/8', '\sigma_a_f = \pi/16', '\sigma_a_f = \pi/32');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

%% ͼ2-14
% description
% ��ͬ�˲������Ӱ��
% ��ͬ�߶ȷֱ�Ƚ�

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 4;
video_path = video_paths(video_idx,:);
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

mae_scales = zeros(3, 2, 1);
rmse_scales = zeros(3, 2, 1);
nF = 100;
dispout = zeros(5, 800);
f = figure;
f.Position(3:4) = [900 400];
for s=1:3
    start_scale = s;
    end_scale = s;
    for n_type = 1:2
        line = animatedline('Color',colors(s,:));
        xline = [1:1:nF];
        oe = 65;
        iou_threshold = -1.33;
        scales = 1;
        disp = zeros(scales, 1);
        num_phase_bin = 4;
        line_width = 2;
        cos_ratio = 0.50;
        dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
        max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
        line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
        line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
        for i = 2:nF
            phase_scales = zeros(2, scales, dim(1), dim(2));
            re_scales = zeros(2, scales, dim(1), dim(2));
            for scale = start_scale:end_scale
                vframein = vHandle.read(i);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                if (n_type == 1)
                    mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
%                     figure; surf(mask, 'EdgeColor', 'none', 'FaceColor', 'interp'); view(0, -90);
                elseif (n_type == 2)
                    mask = x_mask_creator(size(im), 2, 4, [scale, oe/180*pi], 2);
%                     figure; surf(mask, 'EdgeColor', 'none', 'FaceColor', 'interp'); view(0, -90);
                end
                    
                im_dft = mask .* fftshift(fft2(image));
                temp_re = ifft2(ifftshift(im_dft));
                temp_phase = angle(temp_re);

                % ʹ�òο�֡��
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);

                phase_scales(2,1,:,:) = temp_phase;
                phase_scales(1,1,:,:) = temp_phase_2;

                re_scales(2,1,:,:) = temp_re;
                re_scales(1,1,:,:) = temp_re_2;

            end
            % ʹ���ںϲ���
    %         tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
            % ʹ����λ�ݶ��Ƶ�ʽ
            tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
    %             dispout(n_angle,a+1,i) = (cos((oe-a)/180*pi)) * x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, a, iou_threshold);
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
            addpoints(line,xline(i),double(disp(1)));
            figure(f);
            title(['frame:',num2str(i)]);
        end
        cor_x = squeeze(dispout(s,1:nF));
        cor_y = lowrate_y(1,1:nF);
        coeff = mean(abs(cor_x-cor_y)); mae_scales(s, n_type, 1) = coeff;
    %     coeff = corr(cor_x', cor_y','type','pearson'); r_scales(n_angle, 1) = coeff;
        fprintf("scale: %d\t", s);
        fprintf("filter_type: %d mae: %f\t", n_type, coeff);
        rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(s, n_type, 1) = rmse;
        fprintf("filter_type: %d rmse: %f\n", n_type,  rmse);


    end
end

%% ͼ2-14��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 4;
compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
% color_3 = [255,34,0]./255; %[232,201,158]./255;
color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); 
b = bar(mae_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:2
    for i = 1:3
        b(j).CData(i,:) = colors(j,:);
    end
end
y_lim_1 = [[0.000 0.01+0.001]; [0.000 0.016]; [0.000 0.024]; [0.00 0.08]];
y_step_1 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('MAE (pixel)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('CSP filter', 'Gabor filter');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
figure(f_2); 
b = bar(rmse_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:2
    for i = 1:3
        b(j).CData(i,:) = colors(j,:);
    end
end
y_lim_2 = [[0.000 0.01+0.001]; [0.000 0.016]; [0.000 0.024]; [0.00 0.08]];
y_step_2 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_2(video_idx,:));
ylabel('RMSE (pixel)');
yticks(y_lim_2(video_idx,1) : y_step_2(video_idx) : y_lim_2(video_idx,2));
% yticklabels({'0' '2' '4' '6' '8' '10'});
% ytickformat('%.0f');
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('CSP filter', 'Gabor filter');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');


%% ͼ3-3
% description
% ��ֱͬ�߼���㷨Ч���Ƚ�

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_true_1 = 'C:\Users\HIT\Desktop\220605_�ʺ���_����һ·��\VID_20220605_163141.mp4';
% video_true_2 = 'C:\Users\HIT\Desktop\stc2713-sup-0002-stabilizedvideo_v3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 4;
video_path = video_paths(video_idx,:);
vHandle = VideoReader(video_path);
% vHandle = VideoReader(video_true_1);
roi = [[0, 32];[0, 32]]; % ģ������
% roi = [[456, 480];[1080, 1104]]; %VID_20220605_163141

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

f_1 = figure; f_1.Position(3:4) = [900 400]; 
nF = 300;
xline = [1:1:nF];
oe = 65;

type_detector = 3;      % 1 -- canny��Ե���+����
                        % 2 -- FLD(��д��)
                        % 3 -- LSD
                        % 4 -- EDlines
                        % 5 -- ����������ǿ��Hough�����ֱ�߽Ƕ�

types = [1 3 4 5];
for m = 4:4
    line = animatedline('Color',colors(m,:));
    for i = 1:nF
        vframein = vHandle.read(i);
        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
        image = squeeze(mean(im,3));

        out_angle = tools_different_detect_angle(image, types(m));
        addpoints(line,xline(i),double(out_angle)); angle_out(video_idx, types(m), i) = out_angle;
        figure(f_1);
        title(['frame:',num2str(i)]);
    end
end

%% ͼ3-3��plot��
% description
% ��ͼ���֣��������ǰ�����

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_4 = [0.93 0.69 0.13];
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

nF = 300;
types = [1 3 4 5];
video_idx = 4;
f_1 = figure; f_1.Position(3:4) = [900 1000];

% y_min = 63; y_max = 67; y_step = 1;
y_min = 9; y_max = 13; y_step = 1;

i=1;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(a)');
legend('Canny + Hough');
legend('boxoff');
legend('Location','southeast');
grid on; 

i=2;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(b)');
legend('LSD');
legend('boxoff');
grid on; 

i=3;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(c)');
legend('EDLine');
legend('boxoff');
legend('Location','northeast');
grid on; 
ylabel('Orientation angle (deg)','position', [-20 68]);

i=4;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(d)');
legend('Proposed');
legend('boxoff');
legend('Location','northeast');
grid on; 
title('TIme (frame)', 'position', [151 61]);

% figure(f_1);subplot(4,1,4);
% ylabel('Orientation angle (deg)','position', [-20 -65 -1]);
% title('TIme (frame)');

%% ͼ3-4��plot��
% description
% ��ͼ���֣��������ǰ�����

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_4 = [0.93 0.69 0.13];
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

nF = 300;
types = [1 3 4 5];
video_idx = 4;
f_1 = figure; f_1.Position(3:4) = [900 1000];

% y_min = 63; y_max = 67; y_step = 1;
y_min = 9; y_max = 13; y_step = 1;

i=1;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(a)');
legend('Canny + Hough');
legend('boxoff');
legend('Location','northeast');
grid on; 

i=2;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(b)');
legend('LSD');
legend('boxoff');
grid on; 

i=3;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(c)');
legend('EDLine');
legend('boxoff');
legend('Location','northeast');
grid on; 
ylabel('Orientation angle (deg)','position', [-20 14]);

i=4;
tmp_angle(:) = angle_out(video_idx, types(i),:);
figure(f_1); subplot(4,1,i);
plot(tmp_angle, 'Color',colors(i,:), 'LineStyle', '-','LineWidth', 1.5); hold on;
xlim([1, nF]);
ylim([y_min, y_max]);
yticks(y_min: y_step: y_max);
xlabel('(d)');
legend('Proposed');
legend('boxoff');
legend('Location','northeast');
grid on; 
title('TIme (frame)', 'position', [151 7]);

% figure(f_1);subplot(4,1,4);
% ylabel('Orientation angle (deg)','position', [-20 -65 -1]);
% title('TIme (frame)');

%% ��3-1
% description
% ��ֱͬ�߼���㷨ƽ��ÿ֡��ʱ

video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_back_level_3_size_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3-64.avi';
video_back_level_3_size_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3-128.avi';
video_true_1 = 'C:\Users\HIT\Desktop\220605_�ʺ���_����һ·��\VID_20220605_163141.mp4';
% video_true_2 = 'C:\Users\HIT\Desktop\stc2713-sup-0002-stabilizedvideo_v3.avi';
% video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_path = video_back_level_3_size_3;
vHandle = VideoReader(video_path);
% vHandle = VideoReader(video_true_1);
% roi = [[0, 32];[0, 32]]; % ģ������
% roi = [[0, 64];[0, 64]];
roi = [[0, 128];[0, 128]];
% roi = [[456, 480];[1080, 1104]]; %VID_20220605_163141

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

% f_1 = figure; f_1.Position(3:4) = [900 400]; 
nF = 300;
xline = [1:1:nF];
oe = 65;

type_detector = 3;      % 1 -- canny��Ե���+����
                        % 2 -- FLD(��д��)
                        % 3 -- LSD
                        % 4 -- EDlines
                        % 5 -- ����������ǿ��Hough�����ֱ�߽Ƕ�

types = [1 3 4 5];
m = 4;
% line = animatedline('Color',colors(m,:));
vframein = vHandle.read(1);
im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
image = squeeze(mean(im,3));
time = zeros(1, nF);
t_1_start = tic;
for i = 1:nF
%     tic
    out_angle = tools_different_detect_angle(image, types(m));
%     time(i) = toc;
%     fprintf("cur frame: %f\n", i);
%     addpoints(line,xline(i),double(out_angle)); angle_out(video_idx, types(m), i) = out_angle;
%     figure(f_1);
%     title(['frame:',num2str(i)]);
end
total_time = toc(t_1_start);
% t_2 = sum(time(:)) / nF;
fprintf("per_frame_time: %f\n", total_time * 1000 / nF);
% fprintf("per_frame_time: %f\n", t_2*1000);

%% ͼ3-6
% description
% �����Ŷ���λ����Ч��

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 1;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
colors = [color_1; color_2; color_3; color_4];

compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

flag_roi = 0;
mae_scales = zeros(3, 2, 1);
rmse_scales = zeros(3, 2, 1);

% cur_scale = 2;
% start_scale = cur_scale;
% end_scale = cur_scale;
nF = 300;
dispout = zeros(4, 800);
f = figure;
f.Position(3:4) = [900 400];
for flag = 0:1
    flag_roi = flag;
    for s = 1:3
        line = animatedline('Color',colors(s,:));
        xline = [1:1:nF];
        oe = 65;
        iou_threshold = -1.33;
        scales = 1;
        disp = zeros(scales, 1);
        num_phase_bin = 4;
        line_width = 2;
        cos_ratio = 0.50;
        dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
        max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
        line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
        line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
        cur_scale = s;
        start_scale = cur_scale;
        end_scale = cur_scale;
        for i = 2:300
            phase_scales = zeros(2, scales, dim(1), dim(2));
            re_scales = zeros(2, scales, dim(1), dim(2));
            for scale = start_scale:end_scale
                vframein = vHandle.read(i);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                if (flag == 1)
                    [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                else
                    roi_mask = ones(32, 32);
                end
                
                mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
                im_dft = mask .* fftshift(fft2(image));
                temp_re = ifft2(ifftshift(im_dft));
                temp_phase = angle(temp_re);

                % ʹ�òο�֡��
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                if (flag == 1)
                    [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                else
                    roi_mask_2 = ones(32, 32);
                end
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);

                phase_scales(2,1,:,:) = temp_phase;
                phase_scales(1,1,:,:) = temp_phase_2;

                re_scales(2,1,:,:) = temp_re;
                re_scales(1,1,:,:) = temp_re_2;

            end
            % ʹ���ںϲ���
    %         tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
            % ʹ����λ�ݶ��Ƶ�ʽ
            tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
            addpoints(line,xline(i),double(disp(1)));
            figure(f);
            title(['frame:',num2str(i)]);
        end
        cor_x = squeeze(dispout(s,1:nF));
        cor_y = lowrate_y(1,1:nF);
        coeff = mean(abs(cor_x-cor_y)); mae_scales(s, flag_roi+1, 1) = coeff;
        fprintf("if ROI: %d mae: %f\t", flag_roi, coeff);
        rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(s, flag_roi+1, 1) = rmse;
        fprintf("if ROI: %d rmse: %f\n", flag_roi,  rmse);
    end
end

%% ͼ3-6��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 1;
compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
% color_3 = [255,34,0]./255; %[232,201,158]./255;
color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); 
b = bar(mae_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:2
    for i = 1:3
        b(j).CData(i,:) = colors(j+1,:);
    end
end
y_lim_1 = [[0.000 0.01+0.001]; [0.000 0.024]; [0.000 0.050]; [0.00 0.10]];
y_step_1 = [0.002; 0.006; 0.01; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('MAE (pixel)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('no HCPR', 'with HCPR');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
figure(f_2); 
b = bar(rmse_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:2
    for i = 1:3
        b(j).CData(i,:) = colors(j+1,:);
    end
end
y_lim_2 = [[0.000 0.01+0.001]; [0.000 0.024]; [0.000 0.050]; [0.00 0.10]];
y_step_2 = [0.002; 0.006; 0.01; 0.02];
ylim(y_lim_2(video_idx,:));
ylabel('RMSE (pixel)');
yticks(y_lim_2(video_idx,1) : y_step_2(video_idx) : y_lim_2(video_idx,2));
% yticklabels({'0' '2' '4' '6' '8' '10'});
% ytickformat('%.0f');
xlabel('Scale');
% xticks(0 : 60 : 300);
% xticklabels({'0' '2' '4' '6' '8' '10'});
legend('no HCPR', 'with HCPR');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

%% ��3-2
% description
% ��ͬ��С��ֱ�߿�ƽ��ÿ֡��ʱ

video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_back_level_3_size_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3-64.avi';
video_back_level_3_size_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3-128.avi';
video_true_1 = 'C:\Users\HIT\Desktop\220605_�ʺ���_����һ·��\VID_20220605_163141.mp4';
% video_true_2 = 'C:\Users\HIT\Desktop\stc2713-sup-0002-stabilizedvideo_v3.avi';
% video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_path = video_back_level_3_size_3;
vHandle = VideoReader(video_path);
% vHandle = VideoReader(video_true_1);
% roi = [[0, 32];[0, 32]]; % ģ������
% roi = [[0, 64];[0, 64]];
roi = [[0, 128];[0, 128]];
% roi = [[456, 480];[1080, 1104]]; %VID_20220605_163141

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

% f_1 = figure; f_1.Position(3:4) = [900 400]; 
nF = 300;
xline = [1:1:nF];
oe = 65;

type_detector = 3;      % 1 -- canny��Ե���+����
                        % 2 -- FLD(��д��)
                        % 3 -- LSD
                        % 4 -- EDlines
                        % 5 -- ����������ǿ��Hough�����ֱ�߽Ƕ�

types = [1 3 4 5];
m = 4;
% line = animatedline('Color',colors(m,:));
vframein = vHandle.read(1);
im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
image = squeeze(mean(im,3));

num_phase_bin = 4;
line_width = 2;
cos_ratio = 0.50;
dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];

% ����ʶ�����oe����ÿ��bin_masks�������Եõ��ض��Ƕȵ�phase map����
% ��ȿ���Ϊ1*scale
line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);

time = zeros(1, nF);
t_1_start = tic;
for i = 1:nF
%     tic
    % ���ָ��line_k, line_width��ֱ��bin��
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
%     time(i) = toc;
%     fprintf("cur frame: %f\n", i);
%     addpoints(line,xline(i),double(out_angle)); angle_out(video_idx, types(m), i) = out_angle;
%     figure(f_1);
%     title(['frame:',num2str(i)]);
end
total_time = toc(t_1_start);
% t_2 = sum(time(:)) / nF;
fprintf("per_frame_time: %f ms\n", total_time * 1000 / nF);
% fprintf("per_frame_time: %f\n", t_2*1000);

%% ͼ3-7
% description
% ��߶���λ�ں�Ч��

video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_9-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 1;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_3; color_4; color_5; color_6];

start_scale = 1;
end_scale = 3;

nF = 300;
% unfusion_dispout = zeros(4, 800);
fusion_dispout = zeros(1,800);
f = figure;
f.Position(3:4) = [900 400];
for s = 1:1
    line = animatedline('Color',colors(s,:));
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
    
%     cur_scale = s;
%     start_scale = cur_scale;
%     end_scale = cur_scale;
    
    for i = 2:300
        phase_scales = zeros(2, scales, dim(1), dim(2));
        re_scales = zeros(2, scales, dim(1), dim(2));
        for scale = start_scale:end_scale
            vframein = vHandle.read(i);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
            im_dft = mask .* fftshift(fft2(image));
            temp_re = ifft2(ifftshift(im_dft));
            temp_phase = angle(temp_re);
            
            % ʹ�òο�֡��
            vframein = vHandle.read(1);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            im_dft = mask .* fftshift(fft2(image));
            temp_re_2 = ifft2(ifftshift(im_dft));
            temp_phase_2 = angle(temp_re_2);
            
            phase_scales(2,scale,:,:) = temp_phase;
            phase_scales(1,scale,:,:) = temp_phase_2;
            
            re_scales(2,1,:,:) = temp_re;
            re_scales(1,1,:,:) = temp_re_2;
            
        end
        % ʹ���ںϲ���
        tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        disp(1) = tdisp; fusion_dispout(i) = disp(1);
        % ʹ����λ�ݶ��Ƶ�ʽ
%         tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
%         disp(1) = tdisp; unfusion_dispout(s,i) = disp(1);               % ʹ�òο�֡��
        addpoints(line,xline(i),double(disp(1)));
        figure(f);
        title(['frame:',num2str(i)]);
    end
    
end

%% ͼ3-7��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 1;
compute_y = @(query_x) 1/5*sin(2*pi*0.6*query_x);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
% color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_4; color_3; color_5; color_6];

mae_scales = zeros(2, 3, 1);
rmse_scales = zeros(2, 3, 1);
nF = 300;

cor_x = squeeze(fusion_dispout(1:nF));
cor_y = lowrate_y(1,1:nF);
coeff = mean(abs(cor_x-cor_y)); mae_scales(2, 2, 1) = coeff; mae_scales(2, 1, 1) = 0;mae_scales(2, 3, 1) = 0;
fprintf("fusion: %d mae: %f\t", 1, coeff);
rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(2, 2, 1) = rmse; rmse_scales(2, 1, 1) = 0;rmse_scales(2, 3, 1) = 0;
fprintf("fusion: %d rmse: %f\n", 1,  rmse);

for s = 1:3
    cor_x = squeeze(unfusion_dispout(s,1:nF));
    cor_y = lowrate_y(1,1:nF);
    coeff = mean(abs(cor_x-cor_y)); mae_scales(1, s, 1) = coeff;
    fprintf("fusion: %d mae: %f\t", 0, coeff);
    rmse = sqrt(mean((cor_x-lowrate_y(1,1:nF)).^2)); rmse_scales(1, s, 1) = rmse;
    fprintf("fusion: %d rmse: %f\n", 0,  rmse);
end

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); 
b = bar(mae_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:3
    b(j).CData(1,:) = colors(j+0,:);
    b(j).CData(2,:) = colors(2+2,:);
end
y_lim_1 = [[0.000 0.01+0.001]; [0.000 0.016]; [0.000 0.024]; [0.00 0.08]];
y_step_1 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('MAE (pixel)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
% xlabel('Scale');
xticks(1 : 1 : 2);
xticklabels({'no fusion' 'fusion'});
legend('scale 1', 'scale 2', 'scale 3');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
figure(f_2); 
b = bar(rmse_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
for j=1:3
    b(j).CData(1,:) = colors(j+0,:);
    b(j).CData(2,:) = colors(2+2,:);
end
y_lim_2 = [[0.000 0.01+0.001]; [0.000 0.016]; [0.000 0.024]; [0.00 0.08]];
y_step_2 = [0.002; 0.004; 0.008; 0.02];
ylim(y_lim_2(video_idx,:));
ylabel('RMSE (pixel)');
yticks(y_lim_2(video_idx,1) : y_step_2(video_idx) : y_lim_2(video_idx,2));
% yticklabels({'0' '2' '4' '6' '8' '10'});
% ytickformat('%.0f');
% xlabel('Scale');
xticks(1 : 1 : 2);
xticklabels({'no fusion' 'fusion'});
legend('scale 1', 'scale 2', 'scale 3');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

%% ͼ3-8
% description
% ��������ʾ��

damp_ratio = 0.65;
v_init = 40; x_init = 0;
omega = 2*pi*1.2; 
compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [800 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_1, 'Position', [0.60 0.58 0.27 0.31], 'Box','on'); hold(ax_2, 'on');
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_1); hold on;
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_2); hold on;
ylim(ax_1, [-4-0.1 5+0.1]);
ylabel(ax_1, 'Amplitude(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});

ylim(ax_2, [-0.08-0.01 0.1+0.02]);
xlim(ax_2, [180 300]);
xticks(ax_2, 180 : 60 : 300);
xticklabels(ax_2, {'6' '8' '10'});
legend(ax_2, 'True');
legend(ax_2, 'Location','northeast');
legend(ax_2, 'boxoff');
legend(ax_2, 'Orientation','horizontal');

%% ͼ3-9
% description
% �ο�֡��������֡��ʾ��
video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 1;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

damp_ratio = 0.65;
v_init = 40; x_init = 0;
omega = 2*pi*1.2; 
compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

nF = 400;
dispout = zeros(4, 800);
f = figure;
f.Position(3:4) = [900 400];
for s = 1:2
    line = animatedline('Color',colors(s,:));
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

    start_scale = 1;
    end_scale = 3;
    for i = 2:300
        phase_scales = zeros(2, scales, dim(1), dim(2));
        re_scales = zeros(2, scales, dim(1), dim(2));
        for scale = start_scale:end_scale
            vframein = vHandle.read(i);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
            im_dft = mask .* fftshift(fft2(image));
            temp_re = ifft2(ifftshift(im_dft));
            temp_phase = angle(temp_re);
            
            % ʹ������֡��
            if (s == 1)
                vframein = vHandle.read(i-1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            % ʹ�òο�֡��
            if (s == 2)
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
%                 [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                roi_mask_2 = roi_mask;
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            phase_scales(2,scale,:,:) = temp_phase;
            phase_scales(1,scale,:,:) = temp_phase_2;
            
            re_scales(2,1,:,:) = temp_re;
            re_scales(1,1,:,:) = temp_re_2;
            
        end
        % ʹ���ںϲ���
        tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        % ʹ����λ�ݶ��Ƶ�ʽ
%         tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        if (s == 1)
            disp(1) = disp(1)+tdisp; dispout(s,i) = disp(1);       % ʹ������֡��
        end
        if (s == 2)
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
        end
        addpoints(line,xline(i),double(disp(1)));
        figure(f);
        title(['frame:',num2str(i)]);
    end
    
end

%% ͼ3-9��plot��/��2-2
% description
% ��ͼ���֣��������ǰ�����

video_idx = 1;
damp_ratio = 0.65;
v_init = 40; x_init = 0;
omega = 2*pi*1.2; 
compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_1, 'Position', [0.60 0.58 0.27 0.31], 'Box','on'); hold(ax_2, 'on');
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_1); hold on;
figure(f_1); plot(dispout(1,1:300), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5,'Parent',ax_1); hold on;
figure(f_1); plot(dispout(2,1:300), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_2); hold on;
figure(f_1); plot(dispout(1,1:300), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5,'Parent',ax_2); hold on;
figure(f_1); plot(dispout(2,1:300), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;

y_lim_1 = [[-4 5]; [-4 5]; [-4 5]; [-4 5]];
sub_y_lim_1 = [[-0.02-0.018 0.02+0.01]; [-0.04-0.01 0.02+0.01]; [-0.05-0.01 0.03+0.01]; [-0.18-0.01 0.08+0.01]];
y_step_1 = [1; 1; 1; 1];
ylim(ax_1, y_lim_1(video_idx,:));
yticks(ax_1, y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
ylim(ax_1, [-4-0.1 5+0.1]);
ylabel(ax_1, 'Amplitude(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'True', 'adjacent frame', 'fixed frame');
legend(ax_1, 'Location','southeast');
legend(ax_1, 'boxoff');
% legend(ax_1, 'Orientation','horizontal');
ylim(ax_2, sub_y_lim_1(video_idx,:));
xlim(ax_2, [240 320]);
xticks(ax_2, 240 : 60 : 320);
xticklabels(ax_2, {'8' '10'});

f_2 = figure; f_2.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_2,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_2, 'Position', [0.60 0.60 0.27 0.29], 'Box','on'); hold(ax_2, 'on');
% figure(f_2); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3); hold on;
figure(f_2); plot(dispout(1,1:300)-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(dispout(2,1:300)-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(dispout(1,1:300)-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_2); hold on;
figure(f_2); plot(dispout(2,1:300)-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_2); hold on;

y_lim_2 = [[-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]];
sub_y_lim_2 = [[-0.02-0.018 0.02+0.01]; [-0.04-0.01 0.02+0.01]; [-0.05-0.01 0.03+0.01]; [-0.18-0.01 0.08+0.01]];
ylim(ax_1, y_lim_2(video_idx,:));
ylabel(ax_1, 'Error(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'adjacent frame', 'fixed frame');
legend(ax_1, 'Location','southeast');
legend(ax_1, 'boxoff');
% legend('Orientation','horizontal');

%% ͼ3-10
% description
% ǰ�˼Ӻ��ʾ��
video_raw = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 2;
video_path = video_paths(video_idx,:);
% video_path = video_raw;
vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

damp_ratio = 0.65;
v_init = 40; x_init = 0;
omega = 2*pi*1.2; 
compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

nF = 400;
dispout = zeros(4, 800);
f = figure;
f.Position(3:4) = [900 400];
for s = 1:3
    line = animatedline('Color',colors(s,:));
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

    start_scale = 1;
    end_scale = 3;
    for i = 2:300
        phase_scales = zeros(2, scales, dim(1), dim(2));
        re_scales = zeros(2, scales, dim(1), dim(2));
        for scale = start_scale:end_scale
            vframein = vHandle.read(i);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
            im_dft = mask .* fftshift(fft2(image));
            temp_re = ifft2(ifftshift(im_dft));
            temp_phase = angle(temp_re);
            
            % ʹ������֡��
            if (s == 1)
                vframein = vHandle.read(i-1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            % ʹ�òο�֡��
            if (s == 2)
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
%                 [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                roi_mask_2 = roi_mask;
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end
            
            
            % ʹ��ǰ��+���
            if (s == 3)
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            phase_scales(2,scale,:,:) = temp_phase;
            phase_scales(1,scale,:,:) = temp_phase_2;
            
            re_scales(2,1,:,:) = temp_re;
            re_scales(1,1,:,:) = temp_re_2;
            
        end
        % ʹ���ںϲ���
        tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        % ʹ����λ�ݶ��Ƶ�ʽ
%         tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        if (s == 1)
            disp(1) = disp(1)+tdisp; dispout(s,i) = disp(1);       % ʹ������֡��
        end
        if (s == 2)
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
        end
        if (s == 3)
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ��ǰ��+���
        end
        addpoints(line,xline(i),double(disp(1)));
        figure(f);
        title(['frame:',num2str(i)]);
    end
    
end

%% ͼ3-10��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 2;
damp_ratio = 0.65;
v_init = 40; x_init = 0;
omega = 2*pi*1.2; 
compute_y = @(query_x) (sqrt((omega .* x_init).^2 + (v_init + damp_ratio*x_init).^2) / omega) .* (exp(-damp_ratio .* query_x)) .* cos(omega .* query_x - atan((v_init + damp_ratio * x_init) / (omega * x_init)));
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_1, 'Position', [0.60 0.58 0.27 0.31], 'Box','on'); hold(ax_2, 'on');
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_1); hold on;
figure(f_1); plot(dispout(1,1:300), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5,'Parent',ax_1); hold on;
figure(f_1); plot(dispout(2,1:300), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_1); plot(dispout(3,1:300), 'Color',colors(4,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;

figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_2); hold on;
figure(f_1); plot(dispout(1,1:300), 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5,'Parent',ax_2); hold on;
figure(f_1); plot(dispout(2,1:300), 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;
figure(f_1); plot(dispout(3,1:300), 'Color',colors(4,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_2); hold on;

y_lim_1 = [[-4 5]; [-4 5]; [-4 5]; [-4 5]];
sub_y_lim_1 = [[-0.02-0.018 0.02+0.01]; [-0.04-0.01 0.02+0.01]; [-0.05-0.01 0.03+0.01]; [-0.18-0.01 0.08+0.01]];
y_step_1 = [1; 1; 1; 1];
% ylim(ax_1, y_lim_1(video_idx,:));
yticks(ax_1, y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
ylim(ax_1, [-4-0.1 5+0.1]);
ylabel(ax_1, 'Amplitude(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'True', 'adjacent frame', 'fixed frame', 'Proposed method');
legend(ax_1, 'Location','southeast');
legend(ax_1, 'boxoff');
% legend(ax_1, 'Orientation','horizontal');
ylim(ax_2, sub_y_lim_1(video_idx,:));
xlim(ax_2, [240 320]);
xticks(ax_2, 240 : 60 : 320);
xticklabels(ax_2, {'8' '10'});

f_2 = figure; f_2.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_2,'Box','on'); hold(ax_1, 'on');
ax_2 = axes('Parent', f_2, 'Position', [0.60 0.60 0.27 0.29], 'Box','on'); hold(ax_2, 'on');
% figure(f_2); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3); hold on;
figure(f_2); plot(dispout(1,1:300)-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(dispout(2,1:300)-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
figure(f_2); plot(dispout(3,1:300)-lowrate_y(1,1:300), 'Color',colors(4,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;

figure(f_2); plot(dispout(1,1:300)-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_2); hold on;
figure(f_2); plot(dispout(2,1:300)-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_2); hold on;
figure(f_2); plot(dispout(3,1:300)-lowrate_y(1,1:300), 'Color',colors(4,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_2); hold on;

y_lim_2 = [[-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]];
sub_y_lim_2 = [[-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]];
ylim(ax_1, y_lim_2(video_idx,:));
ylabel(ax_1, 'Error(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'adjacent frame', 'fixed frame', 'Proposed method');
legend(ax_1, 'Location','southeast');
legend(ax_1, 'boxoff');
% legend('Orientation','horizontal');
ylim(ax_2, sub_y_lim_2(video_idx,:));

%% ͼ3-11
% description
% ��������ʾ��

compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x)+1/12*sin(2*pi*1.5*query_x + pi/6)+1/16*sin(2*pi*10*query_x) + 1/14*sin(2*pi*2.8*query_x+pi/4);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [800 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
% ax_2 = axes('Parent', f_1, 'Position', [0.60 0.58 0.27 0.31], 'Box','on'); hold(ax_2, 'on');
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
% figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_2); hold on;
ylim(ax_1, [-0.4-0.1 0.4+0.1]);
ylabel(ax_1, 'Amplitude(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'True');
legend(ax_1, 'Location','northeast');
legend(ax_1, 'boxoff');
legend(ax_1, 'Orientation','horizontal');

% ylim(ax_2, [-0.08-0.01 0.1+0.02]);
% xlim(ax_2, [180 300]);
% xticks(ax_2, 180 : 60 : 300);
% xticklabels(ax_2, {'6' '8' '10'});
% legend(ax_2, 'True');
% legend(ax_2, 'Location','northeast');
% legend(ax_2, 'boxoff');
% legend(ax_2, 'Orientation','horizontal');

%% ͼ3-12
% description
% ģ������ʾ��

video_back_A_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-A1.avi';
video_back_A_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-A2.avi';
video_back_A_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-A3.avi';
video_back_B_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-B1.avi';
video_back_B_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-B2.avi';
video_back_B_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-B3.avi';
video_back_C_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-C1.avi';
video_back_C_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-C2.avi';
video_back_C_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-C3.avi';
video_back_D_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-D1.avi';
video_back_D_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-D2.avi';
video_back_D_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_m-D3.avi';
video_paths = [video_back_A_level_1; video_back_B_level_1; video_back_C_level_1; video_back_D_level_1;...
               video_back_A_level_2; video_back_B_level_2; video_back_C_level_2; video_back_D_level_2;...
               video_back_A_level_3; video_back_B_level_3; video_back_C_level_3; video_back_D_level_3;];

video_idx = 1;
% video_path = video_paths(video_idx,:);
% video_path = video_raw;
% vHandle = VideoReader(video_path);
roi = [[0, 32];[0, 32]]; % ģ������

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

nF = 400;
dispout = zeros(4,4, 800);
f = figure;
f.Position(3:4) = [900 400];
for d = 1:3
    for n = 1:3
        video_idx = n + 1 + (d-1)*4;
        video_path = video_paths(video_idx,:);
        vHandle = VideoReader(video_path);
        for s = 2:2
            line = animatedline('Color',colors(n,:));
            xline = [1:1:nF];
            oe = 65;
            iou_threshold = -1.33;
            scales = 1;
            disp = zeros(scales, 1);
            num_phase_bin = 4;
            line_width = 2;
            cos_ratio = 0.50;
            dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
            max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
            line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
            line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

            start_scale = 1;
            end_scale = 3;
            for i = 2:300
                phase_scales = zeros(2, scales, dim(1), dim(2));
                re_scales = zeros(2, scales, dim(1), dim(2));
                for scale = start_scale:end_scale
                    vframein = vHandle.read(i);
                    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                    image = im2single(squeeze(mean(im,3)));
                    [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                    mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
                    im_dft = mask .* fftshift(fft2(image));
                    temp_re = ifft2(ifftshift(im_dft));
                    temp_phase = angle(temp_re);

                    % ʹ������֡��
                    if (s == 1)
                        vframein = vHandle.read(i-1);
                        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                        image = im2single(squeeze(mean(im,3)));
                        [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                        im_dft = mask .* fftshift(fft2(image));
                        temp_re_2 = ifft2(ifftshift(im_dft));
                        temp_phase_2 = angle(temp_re_2);
                    end


                    % ʹ�òο�֡��
                    if (s == 2)
                        vframein = vHandle.read(1);
                        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                        image = im2single(squeeze(mean(im,3)));
        %                 [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                        roi_mask_2 = roi_mask;
                        im_dft = mask .* fftshift(fft2(image));
                        temp_re_2 = ifft2(ifftshift(im_dft));
                        temp_phase_2 = angle(temp_re_2);
                    end


                    % ʹ��ǰ��+���
                    if (s == 3)
                        vframein = vHandle.read(1);
                        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                        image = im2single(squeeze(mean(im,3)));
                        [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                        im_dft = mask .* fftshift(fft2(image));
                        temp_re_2 = ifft2(ifftshift(im_dft));
                        temp_phase_2 = angle(temp_re_2);
                    end


                    phase_scales(2,scale,:,:) = temp_phase;
                    phase_scales(1,scale,:,:) = temp_phase_2;

                    re_scales(2,1,:,:) = temp_re;
                    re_scales(1,1,:,:) = temp_re_2;

                end
                % ʹ���ںϲ���
                tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
                % ʹ����λ�ݶ��Ƶ�ʽ
        %         tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
                if (s == 1)
                    disp(1) = disp(1)+tdisp; dispout(n,d,i) = disp(1);       % ʹ������֡��
                end
                if (s == 2)
                    disp(1) = tdisp; dispout(n,d,i) = disp(1);               % ʹ�òο�֡��
                end
                if (s == 3)
                    disp(1) = tdisp; dispout(n,d,i) = disp(1);               % ʹ��ǰ��+���
                end
                addpoints(line,xline(i),double(disp(1)));
                figure(f);
                title(['frame:',num2str(i)]);
            end

        end
    end
end

%% ͼ3-12��plot��
% description
% ��ͼ���֣��������ǰ�����

d_video = 1;
compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x)+1/12*sin(2*pi*1.5*query_x + pi/6)+1/16*sin(2*pi*10*query_x) + 1/14*sin(2*pi*2.8*query_x+pi/4);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;
amp_factor = [1/12/0.45, 1/25/0.45, 1/50/0.45]; lowrate_y = amp_factor(mod(d_video-1,3)+1) .* lowrate_y;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

f_1 = figure; f_1.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_1,'Box','on'); hold(ax_1, 'on');
figure(f_1); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3, 'Parent',ax_1); hold on;
disp2plot(:,:) = dispout(1,d_video, 1:300);
figure(f_1); plot(disp2plot', 'Color',colors(2,:), 'LineStyle', '-.','LineWidth', 1.5,'Parent',ax_1); hold on;
disp2plot(:,:) = dispout(2,d_video, 1:300);
figure(f_1); plot(disp2plot', 'Color',colors(3,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
disp2plot(:,:) = dispout(3,d_video, 1:300);
figure(f_1); plot(disp2plot', 'Color',colors(4,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;
% figure(f_1); plot(dispout(4,1:300), 'Color',colors(5,:), 'LineStyle', '-.','LineWidth', 1.5, 'Parent',ax_1); hold on;

y_lim_1 = [[-0.12 0.12]; [-0.08 0.08]; [-0.12 0.12]; [-0.12 0.12]];
sub_y_lim_1 = [[-0.02-0.018 0.02+0.01]; [-0.04-0.01 0.02+0.01]; [-0.05-0.01 0.03+0.01]; [-0.18-0.01 0.08+0.01]];
y_step_1 = [0.04; 0.04; 0.04; 0.04];
ylim(ax_1, y_lim_1(d_video,:));
yticks(ax_1, y_lim_1(d_video,1) : y_step_1(d_video) : y_lim_1(d_video,2));

ylabel(ax_1, 'Amplitude(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1, 'True', 'noise 1', 'noise 2', 'noise 3');
legend(ax_1, 'Location','northeast');
legend(ax_1, 'boxoff');
legend(ax_1, 'Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
ax_1 = axes('Parent', f_2,'Box','on'); hold(ax_1, 'on');
% figure(f_2); plot(lowrate_y(1,1:300), 'Color',colors(1,:), 'LineStyle', '-','LineWidth', 3); hold on;
disp2plot(:,:) = dispout(1,d_video, 1:300);
figure(f_2); plot(disp2plot'-lowrate_y(1,1:300), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
disp2plot(:,:) = dispout(2,d_video, 1:300);
figure(f_2); plot(disp2plot'-lowrate_y(1,1:300), 'Color',colors(3,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
disp2plot(:,:) = dispout(3,d_video, 1:300);
figure(f_2); plot(disp2plot'-lowrate_y(1,1:300), 'Color',colors(4,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;
% figure(f_2); plot(dispout(4,1:300)-lowrate_y(1,1:300), 'Color',colors(5,:), 'LineStyle', '-','LineWidth', 1.5, 'Parent',ax_1); hold on;

y_lim_2 = [[-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]; [-0.1-0.05 0.1+0.05]];
sub_y_lim_2 = [[-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]; [-0.5-0.01 0.5+0.01]];
ylim(ax_1, y_lim_2(d_video,:));
ylabel(ax_1, 'Error(pixels)');
xlabel(ax_1, 'Time(s)');
xticks(ax_1, 0 : 60 : 300);
xticklabels(ax_1, {'0' '2' '4' '6' '8' '10'});
legend(ax_1,'noise 1', 'noise 2', 'noise 3');
legend(ax_1, 'Location','northeast');
legend(ax_1, 'boxoff');
% legend('Orientation','horizontal');

%% ͼ3-13��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 2;
compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x)+1/12*sin(2*pi*1.5*query_x + pi/6)+1/16*sin(2*pi*10*query_x) + 1/14*sin(2*pi*2.8*query_x+pi/4);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;
amp_factor = [1/12/0.45, 1/25/0.45, 1/50/0.45]; %lowrate_y = amp_factor(mod(video_idx-1,3)+1) .* lowrate_y;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
% color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_4; color_3; color_5; color_6];

mae_scales = zeros(3, 3, 1);
rmse_scales = zeros(3, 3, 1);
nF = 300;
for d = 1:3
    for s = 1:3
        cor_x = squeeze(dispout(s,d, 1:nF)) ./ 0.45 ./ amp_factor(mod(d-1,3)+1) .* 2;
        cor_y = lowrate_y(1,1:nF) ./ 0.45 .* 2;
%         cor_x = squeeze(dispout(s,d, 1:nF));
%         cor_y = amp_factor(mod(d-1,3)+1) .* lowrate_y(1,1:nF);
        coeff = mean(abs(cor_x'-cor_y)); mae_scales(d, s, 1) = coeff;
        fprintf("dist: %d noise: %d mae: %f\t", d, s, coeff);
        rmse = sqrt(mean((cor_x'-cor_y).^2)); rmse_scales(d, s, 1) = rmse;
        fprintf("dist: %d noise: %d rmse: %f\n", d, s, rmse);
    end
end

f_1 = figure; f_1.Position(3:4) = [500 400];
figure(f_1); 
b = bar(mae_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
% for j=1:3
% %     b(j).CData(1,:) = colors(j+0,:);
% %     b(j).CData(2,:) = colors(j+1,:);
% %     b(j).CData(3,:) = colors(j+2,:);
% end
y_lim_1 = [[0.000 0.04+0.01]; [0.000 2]; [0.000 0.024]; [0.00 0.08]];
y_step_1 = [0.005; 0.2; 0.008; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('MAE (mm)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Distance (m)');
xticks(1 : 1 : 3);
xticklabels({'25' '50' '100'});
legend('noise 1', 'noise 2', 'noise 3');
legend('Location','northwest');
legend('boxoff');
% legend('Orientation','horizontal');

f_2 = figure; f_2.Position(3:4) = [500 400];
figure(f_2); 
b = bar(rmse_scales(:,:,1), 1, 'FaceColor','flat'); hold on;
% for j=1:1
% %     b(j).CData(1,:) = colors(j+0,:);
% %     b(j).CData(2,:) = colors(j+1,:);
% %     b(j).CData(3,:) = colors(j+2,:);
% end
y_lim_2 = [[0.000 0.04+0.01]; [0.000 2]; [0.000 0.024]; [0.00 0.08]];
y_step_2 = [0.005; 0.2; 0.008; 0.02];
ylim(y_lim_2(video_idx,:));
ylabel('RMSE (mm)');
yticks(y_lim_2(video_idx,1) : y_step_2(video_idx) : y_lim_2(video_idx,2));
% yticklabels({'0' '2' '4' '6' '8' '10'});
% ytickformat('%.0f');
xlabel('Distance (m)');
xticks(1 : 1 : 3);
xticklabels({'25' '50' '100'});
legend('noise 1', 'noise 2', 'noise 3');
legend('Location','northwest');
legend('boxoff');
% legend('Orientation','horizontal');

%% ͼ3-14��plot��
% description
% ��ͼ���֣��������ǰ�����

video_idx = 2;
compute_y = @(query_x) 1/4*sin(2*pi*0.6*query_x)+1/12*sin(2*pi*1.5*query_x + pi/6)+1/16*sin(2*pi*10*query_x) + 1/14*sin(2*pi*2.8*query_x+pi/4);
video_rate = 30;
video_duration = 10;
video_lenth = round(video_rate * video_duration);
lowrate_x = [1:video_lenth]/video_lenth.*video_duration;
lowrate_y = zeros(1,video_lenth); lowrate_y(2:end) = compute_y(lowrate_x(1:end-1)); lowrate_y(1)=0;
amp_factor = [1/12/0.45, 1/25/0.45, 1/50/0.45]; %lowrate_y = amp_factor(mod(video_idx-1,3)+1) .* lowrate_y;

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
% color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_4; color_3; color_5; color_6];

f_1 = figure; f_1.Position(3:4) = [900 400];
nF = 300;
samplerate = 30;
method = 2;
for d = 3:3
    for s = 1:3
        cor_x = squeeze(dispout(s,d, 1:nF)) ./ 0.45 ./ amp_factor(mod(d-1,3)+1) .* 2;
        cor_y = lowrate_y(1,1:nF) ./ 0.45 .* 2;
        % Welch����
        if (method == 1)
            window_lenth = 0.5 * nF;
            overlapping = 0.5 * window_lenth;
            nDFT_points = nF;
            fs = samplerate;
            [pxx, freq] = pwelch(cor_x, window_lenth, overlapping, nDFT_points, fs);
            figure(f_1); plot(freq, 10*log10(pxx), 'LineStyle', '-','LineWidth', 1.5); hold on;
        end

        % ����ͼ��/ֱ�ӷ�������
        if (method == 2)
            sigfft = fft(cor_x);
            L = max(size(cor_x));
            sf2 = abs(sigfft/L);
            sf1 = sf2(1 : floor(L/2)+1);
            sf1(2 : end-1) = 2*sf1(2 : end-1);
            L2time = (1:L)/samplerate;
            L2freq = samplerate*(0:floor(L/2))/L;
            psd = (1/(samplerate*L)) * abs(sigfft(1 : floor(L/2)+1)).^2;
            spsd = psd;
            spsd(2 : end-1) = 2*spsd(2 : end-1);
            figure(f_1);plot(L2freq,10*log10(spsd), 'LineStyle', '-','LineWidth', 1.5); hold on;
        end
        
        % FFTƵ��
        if (method == 3)
            sigfft = fft(cor_x);
            L = max(size(cor_x));
            sf2 = abs(sigfft/L);
            sf1 = sf2(1 : floor(L/2)+1);
            sf1(2 : end-1) = 2*sf1(2 : end-1);
            L2time = (1:L)/samplerate;
            L2freq = samplerate*(0:floor(L/2))/L;
            psd = (1/(samplerate*L)) * abs(sigfft(1 : floor(L/2)+1)).^2;
            spsd = psd;
            spsd(2 : end-1) = 2*spsd(2 : end-1);
            figure(f_1);plot(L2freq,sf1, 'LineStyle', '-','LineWidth', 1.5); hold on;
%             figure(f_1);plot(L2freq,10*log10(sf1),'LineStyle', '-','LineWidth', 1.5); hold on;
        end
%         cor_x = squeeze(dispout(s,d, 1:nF));
%         cor_y = amp_factor(mod(d-1,3)+1) .* lowrate_y(1,1:nF);
%         coeff = mean(abs(cor_x'-cor_y)); mae_scales(d, s, 1) = coeff;
%         fprintf("dist: %d noise: %d mae: %f\t", d, s, coeff);
%         rmse = sqrt(mean((cor_x'-cor_y).^2)); rmse_scales(d, s, 1) = rmse;
%         fprintf("dist: %d noise: %d rmse: %f\n", d, s, rmse);
    end
end

y_lim_1 = [[-50 30]; [-50 30]; [-50 30]; [0.00 0.08]];
y_step_1 = [10; 10; 10; 0.02];
ylim(y_lim_1(video_idx,:));
ylabel('PSD (dB/Hz)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Frequency (Hz)');
xticks(0 : 3 : 15);
% xticklabels({'25' '50' '100'});
legend('noise 1', 'noise 2', 'noise 3');
legend('Location','northeast');
legend('boxoff');
% legend('Orientation','horizontal');

%% ͼ3-16
% description
% ROIʾ��
video_raw = 'C:\Users\HIT\Desktop\stc2713-sup-0002-stabilizedvideo_v3.avi';
video_back_origin = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-0.avi';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 2;
% video_path = video_paths(video_idx,:);
video_path = video_raw;
vHandle = VideoReader(video_path);

% [ymin ymax xmin xmax]
roi_1 = [[259, 291];[244, 276]]; % 1���� 32*32
% roi_1 = [[255, 279];[271, 295]]; % 1���� 32*32
roi_2 = [[282, 298];[282, 298]]; % 2���� 24*24
roi_3 = [[299, 323];[274, 298]]; % 3���� 24*24
roi_4 = [[308, 324];[312, 328]]; % 4���� 16*16
% roi_5 = [[318, 334];[333, 349]]; % 5���� 16*16
roi_5 = [[342, 358];[298, 314]]; % 5���� 16*16
% roi = [[350, 366];[286, 302]]; % 5����
% ����Ϊ��ͼ���ϻ����Σ������ѭ[xmin xmax ymin ymax]
roi_sets = [roi_1(2,:); roi_1(1,:); roi_2(2,:); roi_2(1,:); ...
            roi_3(2,:); roi_3(1,:); roi_4(2,:); roi_4(1,:); roi_5(2,:); roi_5(1,:)];
[[243, 267];[304, 328]];
f_1 = figure; f_1.Position(3:4) = [900 400];
vframein = vHandle.read(70);
figure(f_1); hold on
imshow(vframein,'InitialMagnification','fit');
for i = 1:5
    roi(1,:) = roi_sets(2*i-1, :);
    roi(2,:) = roi_sets(2*i, :);
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    rectangle('Position',[roi(1,1)+1 roi(2,1)+1 dim(1) dim(2)],'LineWidth',1,'EdgeColor','r');
end

%% ͼ3-17
% description
% ʵ������ʾ��
video_raw = 'C:\Users\HIT\Desktop\stc2713-sup-0002-stabilizedvideo_v3.avi';
% video_raw = 'C:\Users\HIT\Desktop\220115_rainbowbridge\dub_bridge_Trim.mp4';
video_back_level_1 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-1.avi';
video_back_level_2 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-2.avi';
video_back_level_3 = 'C:\Users\HIT\Desktop\��λͼƬ\ģ����������_8-3.avi';
video_paths = [video_back_origin; video_back_level_1; video_back_level_2; video_back_level_3];

video_idx = 2;
% video_path = video_paths(video_idx,:);
video_path = video_raw;
vHandle = VideoReader(video_path);

% ����Ϊ�ھ�����ʹ��ROI�������ѭ[ymin ymax xmin xmax]
roi_1 = [[259, 291];[244, 276]]; % 1���� 32*32
% roi_1 = [[255, 279];[271, 295]]; % 1���� 32*32
roi_2 = [[282, 298];[282, 298]]; % 2���� 24*24
roi_3 = [[299, 323];[274, 298]]; % 3���� 24*24
roi_4 = [[308, 324];[293, 309]]; % 4���� 16*16
% roi_5 = [[318, 334];[333, 349]]; % 5���� 16*16
roi_5 = [[342, 358];[298, 314]]; % 5���� 16*16
% roi = [[350, 366];[286, 302]]; % 5����
roi_sets = [roi_1; roi_2; roi_3; roi_4; roi_5];
scales_sets = [3; 2; 2; 2; 2];


color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
color_4 = [194,94,136]./255;
color_5 = [0.93 0.69 0.13];
colors = [color_1; color_2; color_3; color_5; color_4];

nF = 411;
dispout = zeros(5, 800);
f_1 = figure;
f = figure; 
f.Position(3:4) = [900 400];
method = 3;
for s = 1:1
    line = animatedline('Color',colors(s,:));
    xline = [1:1:nF];
    oe = 65;
    iou_threshold = -1.33;
    scales = 1;
    disp = zeros(scales, 1);
    num_phase_bin = 4;
    line_width = 2;
    cos_ratio = 0.50;
    roi(1,:) = roi_sets(2*s-1, :);
    roi(2,:) = roi_sets(2*s, :);
    dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
    max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
    line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

    start_scale = 1;
    end_scale = scales_sets(s);
    for i = 2:411
        phase_scales = zeros(2, scales, dim(1), dim(2));
        re_scales = zeros(2, scales, dim(1), dim(2));
        
        vframein = vHandle.read(i);
        figure(f_1); hold on
%         im = vframein(roi(2,1)-9:roi(1,2)+10,roi(1,1)-9:roi(2,2)+10);
%         imshow(im,'InitialMagnification','fit');        
%         rectangle('Position',[1+9 1+9 dim(1) dim(2)],'LineWidth',4,'EdgeColor','r');
        imshow(vframein, 'InitialMagnification','fit');
        rectangle('Position',[roi(2,1)+1 roi(1,1)+1 dim(1) dim(2)],'LineWidth',1,'EdgeColor','r');
        
        for scale = start_scale:end_scale
            vframein = vHandle.read(i);
            im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
            image = im2single(squeeze(mean(im,3)));
            [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
            mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
            im_dft = mask .* fftshift(fft2(image));
            temp_re = ifft2(ifftshift(im_dft));
            temp_phase = angle(temp_re);
            
            % ʹ������֡��
            if (method == 1)
                vframein = vHandle.read(i-1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            % ʹ�òο�֡��
            if (method == 2)
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
%                 [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                roi_mask_2 = roi_mask;
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end
            
            
            % ʹ��ǰ��+���
            if (method == 3)
                vframein = vHandle.read(1);
                im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
                image = im2single(squeeze(mean(im,3)));
                [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
                im_dft = mask .* fftshift(fft2(image));
                temp_re_2 = ifft2(ifftshift(im_dft));
                temp_phase_2 = angle(temp_re_2);
            end

            
            phase_scales(2,scale,:,:) = temp_phase;
            phase_scales(1,scale,:,:) = temp_phase_2;
            
            re_scales(2,1,:,:) = temp_re;
            re_scales(1,1,:,:) = temp_re_2;
            
        end
        % ʹ���ںϲ���
        tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        % ʹ����λ�ݶ��Ƶ�ʽ
%         tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
        if (method == 1)
            disp(1) = disp(1)+tdisp; dispout(s,i) = disp(1);       % ʹ������֡��
        end
        if (method == 2)
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ�òο�֡��
        end
        if (method == 3)
            disp(1) = tdisp; dispout(s,i) = disp(1);               % ʹ��ǰ��+���
        end
        figure(f);
        addpoints(line,xline(i),double(disp(1))); drawnow;
        figure(f);
        title(['frame:',num2str(i)]);
    end
    
end

%% ͼ3-17��plot��
% description
% ��ͼ���֣��������ǰ�����

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
% color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_4; color_3; color_5; color_6];

f_0 = figure; f_0.Position(3:4) = [500 400];
f_1 = figure; f_1.Position(3:4) = [500 400];
nF = 512;
samplerate = 30;
method = 2;
for s = 1:1
    video_idx = s;
    cor_x = zeros(1, nF);
    cor_x(1:411) =  dispout(s, 1:411);
%     cor_x = dispout(s, 1:nF);
    L = 411;
    L2time = (1:L)/samplerate; 
    figure(f_0); plot(L2time, cor_x(1:411), 'LineStyle', '-','LineWidth', 1.5); hold on;
    y_lim_1 = [[-50 30]; [-50 30]; [-50 30]; [-50 30]; [-50 30]];
    y_step_1 = [1; 1; 1; 1; 1];
%     ylim(y_lim_1(video_idx,:));
    ylabel('Amplitude (pixels)');
    yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
    xlabel('Time (s)');
%     xticks(0 : 30 : 411);
    % Welch����
    if (method == 1)
        window_lenth = 0.5 * nF;
        overlapping = 0.5 * window_lenth;
        nDFT_points = nF;
        fs = samplerate;
        [pxx, freq] = pwelch(cor_x, window_lenth, overlapping, nDFT_points, fs);
        figure(f_1); plot(freq, 10*log10(pxx), 'LineStyle', '-','LineWidth', 1.5); hold on;
    end

    % ����ͼ��/ֱ�ӷ�������
    if (method == 2)
        sigfft = fft(cor_x);
        L = max(size(cor_x));
        sf2 = abs(sigfft/L);
        sf1 = sf2(1 : floor(L/2)+1);
        sf1(2 : end-1) = 2*sf1(2 : end-1);
        L2time = (1:L)/samplerate;
        L2freq = samplerate*(0:floor(L/2))/L;
        psd = (1/(samplerate*L)) * abs(sigfft(1 : floor(L/2)+1)).^2;
        spsd = psd;
        spsd(2 : end-1) = 2*spsd(2 : end-1);
        figure(f_1);plot(L2freq,10*log10(spsd), 'LineStyle', '-','LineWidth', 1.5); hold on;
    end

    % FFTƵ��
    if (method == 3)
        sigfft = fft(cor_x);
        L = max(size(cor_x));
        sf2 = abs(sigfft/L);
        sf1 = sf2(1 : floor(L/2)+1);
        sf1(2 : end-1) = 2*sf1(2 : end-1);
        L2time = (1:L)/samplerate;
        L2freq = samplerate*(0:floor(L/2))/L;
        psd = (1/(samplerate*L)) * abs(sigfft(1 : floor(L/2)+1)).^2;
        spsd = psd;
        spsd(2 : end-1) = 2*spsd(2 : end-1);
        figure(f_1);plot(L2freq,sf1, 'LineStyle', '-','LineWidth', 1.5); hold on;
%             figure(f_1);plot(L2freq,10*log10(sf1),'LineStyle', '-','LineWidth', 1.5); hold on;
    end
%         cor_x = squeeze(dispout(s,d, 1:nF));
%         cor_y = amp_factor(mod(d-1,3)+1) .* lowrate_y(1,1:nF);
%         coeff = mean(abs(cor_x'-cor_y)); mae_scales(d, s, 1) = coeff;
%         fprintf("dist: %d noise: %d mae: %f\t", d, s, coeff);
%         rmse = sqrt(mean((cor_x'-cor_y).^2)); rmse_scales(d, s, 1) = rmse;
%         fprintf("dist: %d noise: %d rmse: %f\n", d, s, rmse);
end


y_lim_1 = [[-50 30]; [-50 30]; [-50 30];  [-50 30]; [-50 30]];
y_step_1 = [10; 10; 10; 10; 10];
ylim(y_lim_1(video_idx,:));
ylabel('PSD (dB/Hz)');
yticks(y_lim_1(video_idx,1) : y_step_1(video_idx) : y_lim_1(video_idx,2));
xlabel('Frequency (Hz)');
xticks(0 : 3 : 15);
% xticklabels({'25' '50' '100'});
% legend('noise 1', 'noise 2', 'noise 3');
% legend('Location','northeast');
% legend('boxoff');
% legend('Orientation','horizontal');


%% ͼ4-4
% description
% ������������쳣ʾ��

color_1 = [124 187 0]./255; %[194,94,136]./255;
color_2 = [0 114.75 188.7]./255; %[130,172,109]./255;
color_3 = [255,34,0]./255; %[232,201,158]./255;
% color_3 = [0.85,0.33,0.10];
color_4 = [0.93 0.69 0.13];
% color_4 = [194,94,136]./255;
color_5 = [0.49,0.18,0.56];
color_6 = [48,151,164]./255;
colors = [color_1; color_2; color_4; color_3; color_5; color_6];

f_0 = figure; f_0.Position(3:4) = [900 240];
% [SJX11 SJX08 SJX13 SJS13]
anomaly_idx = [11, 8, 13, 6];
idx_cable = 3;
figure(f_0); plot(temp_data(:,anomaly_idx(idx_cable)), 'Color',colors(2,:), 'LineStyle', '-','LineWidth', 1); hold on;
% ylim([-0.4-0.1 0.4+0.1]);
ylabel('Cable Force(kN)');
xlabel('Time(day)');
xticks(0 : 172800 : 1728000);
xticklabels({'1' '2' '3' '4' '5' '6' '7' '8' '9' '10'});
% legend('True');
% legend('Location','northeast');
% legend('boxoff');
% legend('Orientation','horizontal');

%% ͼ4-5
% description
% ��������

true_1 = zeros(10, 10*108);
for i = 1:10
    true_1(i, (i-1)*108+1:i*108) = 1;
end
lenth = 108;
predict = zeros(10, 10*108);
% cable 0
x_shift = 0;
predict(1, x_shift+1:x_shift+round(lenth*0.98)) = 1;
predict(4, x_shift+round(lenth*0.98)+1:x_shift+round(lenth*(0.98+0.01))) = 1;
predict(9, x_shift+round(lenth*(0.98+0.01))+1:x_shift+round(lenth*(0.98+0.01+0.01))) = 1;
% cable 1
x_shift = 1*108;
predict(2, x_shift+1:x_shift+round(lenth*0.86)) = 1;
predict(8, x_shift+round(lenth*0.86)+1:x_shift+round(lenth*(0.86+0.14))) = 1;
% cable 2
x_shift = 2*108;
predict(3, x_shift+1:x_shift+round(lenth*0.81)) = 1;
predict(2, x_shift+round(lenth*0.81)+1:x_shift+round(lenth*(0.81+0.10))) = 1;
predict(8, x_shift+round(lenth*(0.81+0.10))+1:x_shift+round(lenth*(0.81+0.10+0.09))) = 1;
% cable 3
x_shift = 3*108;
predict(4, x_shift+1:x_shift+round(lenth*0.93)) = 1;
predict(9, x_shift+round(lenth*0.93)+1:x_shift+round(lenth*(0.93+0.07))) = 1;
% cable 4
x_shift = 4*108;
predict(5, x_shift+1:x_shift+round(lenth*0.49)) = 1;
predict(1, x_shift+round(lenth*0.49)+1:x_shift+round(lenth*(0.49+0.23))) = 1;
predict(6, x_shift+round(lenth*(0.49+0.23))+1:x_shift+round(lenth*(0.49+0.23+0.13))) = 1;
predict(10, x_shift+round(lenth*(0.49+0.23+0.13))+1:x_shift+round(lenth*(0.49+0.23+0.13+0.15))) = 1;
% cable 5
x_shift = 5*108;
predict(6, x_shift+1:x_shift+round(lenth*0.77)) = 1;
predict(9, x_shift+round(lenth*0.77)+1:x_shift+round(lenth*(0.77+0.23))) = 1;
% cable 6
x_shift = 6*108;
predict(7, x_shift+1:x_shift+round(lenth*0.90)) = 1;
predict(6, x_shift+round(lenth*0.90)+1:x_shift+round(lenth*(0.90+0.1))) = 1;
% cable 7
x_shift = 7*108;
predict(8, x_shift+1:x_shift+round(lenth*0.52)) = 1;
predict(2, x_shift+round(lenth*0.52)+1:x_shift+round(lenth*(0.52+0.25))) = 1;
predict(3, x_shift+round(lenth*(0.52+0.25))+1:x_shift+round(lenth*(0.52+0.25+0.23))) = 1;
% cable 8
x_shift = 8*108;
predict(9, x_shift+1:x_shift+round(lenth*0.82)) = 1;
predict(4, x_shift+round(lenth*0.82)+1:x_shift+round(lenth*(0.82+0.18))) = 1;
% cable 9
x_shift = 9*108;
predict(10, x_shift+1:x_shift+round(lenth*0.23)) = 1;
predict(5, x_shift+round(lenth*0.23)+1:x_shift+round(lenth*(0.23+0.29))) = 1;
predict(6, x_shift+round(lenth*(0.23+0.29))+1:x_shift+round(lenth*(0.23+0.29+0.35))) = 1;
predict(9, x_shift+round(lenth*(0.23+0.29+0.35))+1:x_shift+round(lenth*(0.23+0.29+0.35+0.13))) = 1;


f_0 = figure; f_0.Position(3:4) = [800 800];
figure(f_0); plotconfusion(true_1, predict);


