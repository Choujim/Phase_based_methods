f = figure;
f.Position(3:4) = [900 400];

%%
line = animatedline('Color','r');
xline = [1:1:nF];

disp = 0;
oe = 64;
ANIME(411) = struct('cdata',[],'colormap',[]);

num_phase_bin = 4;
line_width = 2;
cos_ratio = 0.50;
dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];

% 根据识别出的oe遍历每个bin_masks，即可以得到特定角度的phase map区域
% 宽度可设为1*scale
line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
% 获得指定line_k, line_width的直线bin库
line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);


% 查看line_bins覆盖范围
% dim_bins = size(line_bins);
% total_bins = single(zeros(dim(1), dim(2)));
% for d = 1:dim_bins(3)
%     total_bins = total_bins + line_bins(:,:,d);
% end
% figure; surf(single(logical(total_bins)));

for i = 2:300
vframein = vHandle.read(i);
im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
image = im2single(squeeze(mean(im,3)));


[ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);

% figure(f); surf(ratio); view(0.5, 0.4);
% figure(f); plot(mean(ratio));

% t_ratio = mean(mean(ratio));
% addpoints(line,xline(i),t_ratio);

% 比较ROI
% subplot(1,2,1);figure(f);%imshow(im,'InitialMagnification','fit');
% mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
% im_dft = mask .* fftshift(fft2(image));
% temp_re = ifft2(ifftshift(im_dft));
% temp_phase = angle(temp_re);
% surf(temp_phase);view(0, -90);
% 
% subplot(1,2,2);figure(f); surf(roi_mask); view(0, -90);
% 查看ROI截取效果
% temp_phase(~logical(roi_mask)) = 0;
% subplot(1,2,2);figure(f); surf(temp_phase); view(0, -90);

% 输出相位差
mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
im_dft = mask .* fftshift(fft2(image));
temp_re = ifft2(ifftshift(im_dft));
temp_phase = angle(temp_re);

vframein = vHandle.read(i-1);
im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
image = im2single(squeeze(mean(im,3)));
[ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
im_dft = mask .* fftshift(fft2(image));
temp_re_2 = ifft2(ifftshift(im_dft));
temp_phase_2 = angle(temp_re_2);
% figure(f); surf(single((roi_mask + roi_mask_2)>=1));view(0, -90);

delta_phase = (temp_phase - temp_phase_2);
t_phase = mod(pi+delta_phase(:), 2*pi)-pi;

t_mask = logical(roi_mask)&logical(roi_mask_2);                     % 使用ROI
if (sum(sum(single(t_mask))) ~= 0)
    tdisp = mean(t_phase(t_mask));% 防止mask全空报错
else
    tdisp = 0;
end
% tdisp = mean(mean(t_phase));                                       % 不加ROI
% temp_amp = abs(temp_re); temp_amp = temp_amp(:);                     % 原幅值加权
% t_phase = temp_amp .* t_phase;
% sum_amp = sum(temp_amp(:));
% tdisp = sum(t_phase(:)) / sum_amp;

disp = disp+tdisp;signalout(4,i) = disp;
addpoints(line,xline(i),double(disp));
figure(f);




% 输出复值响应的幅值图
% mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
% im_dft = mask .* fftshift(fft2(image));
% temp_re = ifft2(ifftshift(im_dft));
% subplot(1,2,1);figure(f); surf(im);view(0, -90);
% temp_amp = abs(temp_re);
% subplot(1,2,2);figure(f);surf(temp_amp);view(0, -90);

% temp_phase = angle(temp_re);
% subplot(1,2,2);figure(f);surf(temp_phase);view(0, -90);

% 输出相位差map
% dim = size(im);
% subplot(1,2,1);figure(f); surf(im);view(0, -90);
% subplot(1,2,2);figure(f);surf(delta_phase);view(0, -90);
% subplot(1,2,1);figure(f);surf(temp_amp);view(0, -90);
% subplot(1,2,2);figure(f);surf(reshape(t_phase,dim));view(0, -90);


% % 使用若干bin来分割phase map
% % [-pi, pi]分成4*scale个bin
% num_bin = 4;
% angle_bin = [-num_bin : num_bin] / num_bin * pi;
% phase_bin_masks = single(zeros(dim(1),dim(2), num_bin));
% dim = size(im);
% 
% mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
% im_dft = mask .* fftshift(fft2(image));
% temp_re = ifft2(ifftshift(im_dft));
% temp_phase = angle(temp_re);
% for num = 1:num_bin 
%     left_mask = temp_phase>=angle_bin(2*num-1);
%     right_mask = temp_phase<=angle_bin(2*num+1);
%     phase_bin_masks(:,:, num) = single(left_mask==right_mask);
% %     figure; surf(single(phase_bin_masks(:,:, num)));
% end
% 
% % 根据识别出的oe遍历每个bin_masks，即可以得到特定角度的phase map区域
% % 宽度可设为2*scale
% line_width = 2;
% % line_bins = zeros(dim(1),dim(2),dim(1)/line_width);
% line_k = (oe-90)/abs(oe-90) * tan(abs(oe-90)/180*pi);
% total_temp = zeros(dim(1),dim(2));
% % 获得指定line_k, line_width的直线bin库
% line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
% 
% size_p = size(phase_bin_masks);
% size_l = size(line_bins);
% refine_roi = single(zeros(dim(1), dim(2), size_p(3)));
% cos_ratio = 0.46;
% for it_p = 1:size_p(3)
%     temp_phase_bin = phase_bin_masks(:,:,it_p);
%     
%     for it_l = 1:size_l(3)
%         temp = line_bins(:,:,it_l);
%         query = temp_phase_bin .* temp;
%         temp_ref = sum(temp(:));
%         temp_query = sum(query(:));
% %         temp_query / temp_ref
%         if (temp_query > 8) && ((temp_query / temp_ref) >= cos_ratio)
%             refine_roi(:,:,it_p) = refine_roi(:,:,it_p) + query + temp;
%         end
%     end
%     refine_roi(:,:,it_p) = single(logical(refine_roi(:,:,it_p)));
% end
% 
% refine_mask = single(zeros(dim(1), dim(2)));
% for it_p = 1:size_p(3)
%     refine_mask = refine_mask + refine_roi(:,:,it_p);
% end

% % 考虑形状信息
% shape_image = de_background(image, 2);


% mean_shape = mean(abs(shape_image(:)));
% mean_shape_mask = abs(shape_image) > mean_shape;

% mean_shape = sqrt(mean(shape_image(:).^2));
% mean_shape_mask = abs(shape_image) > mean_shape;
% figure(f);surf(double(mean_shape_mask));view(0, -90)


% 尺度1
% mask = mask_creator(size(im), 2, 4, [1, 65/180*pi], 1);
% im_dft = mask .* fftshift(fft2(image));
% temp_re = ifft2(ifftshift(im_dft));
% temp_phase = angle(temp_re);
% roi_index = abs(temp_phase)<=0.785; 

% 考虑phase map信息
% roi_index = roi_index & mean_shape_mask;

% 简单ROI
%     mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re = ifft2(ifftshift(im_dft));
%     temp_phase = angle(temp_re);
%     
%     vframein = vHandle.read(i-1);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     image = im2single(squeeze(mean(im,3)));
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re_2 = ifft2(ifftshift(im_dft));
%     temp_phase_2 = angle(temp_re_2);
%     shape_image = de_background(image, 2);
%     mean_shape = mean(abs(shape_image));
%     mean_shape_mask_2 = shape_image> mean_shape;
    
% 融合ROI    
%     mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re = ifft2(ifftshift(im_dft));
%     temp_phase = angle(temp_re);
%     roi_index = abs(temp_phase)<=pi/4;
%     roi_index = roi_index & mean_shape_mask;
%     
%     vframein = vHandle.read(i-1);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     image = im2single(squeeze(mean(im,3)));
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re_2 = ifft2(ifftshift(im_dft));
%     temp_phase_2 = angle(temp_re_2);
%     shape_image = de_background(image, 2);
%     mean_shape = mean(abs(shape_image));
%     mean_shape_mask_2 = shape_image> mean_shape;
%     roi_index_2 = abs(temp_phase)<=pi/4;
%     roi_index_2 = roi_index_2 & mean_shape_mask;

% 输出    
%     delta_phase = (temp_phase - temp_phase_2);
%     t_phase = mod(pi+delta_phase(:), 2*pi)-pi;
%     tdisp = mean(t_phase(roi_index|roi_index_2));
% %     tdisp = mean(t_phase(:));
%     disp = disp+tdisp;signalout(4,i) = disp;
%     addpoints(line,xline(i),double(disp));
%     figure(f);
%     title(['frame:',num2str(i)]);


% 尺度2
% mask = mask_creator(size(im), 2, 4, [2, 65/180*pi], 1);
% dims = size(im);
% ctr = ceil((dims+0.5)/2);
% lodims = ceil((dims-0.5)/2);
% loctr = ceil((lodims+0.5)/2);
% lostart = ctr-loctr+1;
% loend = lostart+lodims-1;
% 
% im_dft = fftshift(fft2(image));
% im_dft = im_dft(lostart(1):loend(1),lostart(2):loend(2));
% image = imresize(image, 0.5, 'bilinear');
% temp_re = ifft2(ifftshift(mask .* im_dft));
% temp_phase = angle(temp_re);
% roi_index = abs(temp_phase)<=0.785; %(2pi/4)/2 



% figure(f);
% subplot(1,2,1);surf(image);view(0,-90);
% % view(59.3, 67.6);
% % surf(temp_phase);
% subplot(1,2,2);surf(double(roi_index));view(0,-90);
% % view(63.6,70.8);
title(['frame:',num2str(i)]);
% ANIME(i)= getframe(f);

end