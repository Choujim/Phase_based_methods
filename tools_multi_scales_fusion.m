f = figure;
f.Position(3:4) = [900 400];

%%
% profile on;
line = animatedline('Color','b');
line_2 = animatedline('Color','g');
line_3 = animatedline('Color','r');
xline = [1:1:nF];

% tic;fprintf('Start now ...\n');

oe = 65;
iou_threshold = -1.33;
scales = 3;
disp = zeros(scales, 1);
% signalout = zeros(4,800);

num_phase_bin = 4;
line_width = 2;
cos_ratio = 0.50;
dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
max_ht = floor(log2(min(dim(:)))) - 2;             % �������������Ƿ񳬹�����
if (scales > max_ht)
    scales = max_ht;
end
% ����ʶ�����oe����ÿ��bin_masks�������Եõ��ض��Ƕȵ�phase map����
% ��ȿ���Ϊ1*scale
line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
% ���ָ��line_k, line_width��ֱ��bin��
line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

for i = 2:300
% �����λ��  
    delta_phase = zeros(dim(1), dim(2), scales);
    phase_scales = zeros(2, scales, dim(1), dim(2));
    re_scales = zeros(2, scales, dim(1), dim(2));
    for scale = 1:scales
        vframein = vHandle.read(i);
        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
        image = im2single(squeeze(mean(im,3)));
        [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
        mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
        im_dft = mask .* fftshift(fft2(image));
        temp_re = ifft2(ifftshift(im_dft));
        temp_phase = angle(temp_re);
        
        % ʹ������֡��
%         vframein = vHandle.read(i-1);
%         im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%         image = im2single(squeeze(mean(im,3)));
%         [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
%         im_dft = mask .* fftshift(fft2(image));
%         temp_re_2 = ifft2(ifftshift(im_dft));
%         temp_phase_2 = angle(temp_re_2);
        
        % ʹ�òο�֡��
        vframein = vHandle.read(1);
        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
        image = im2single(squeeze(mean(im,3)));
        [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
        im_dft = mask .* fftshift(fft2(image));
        temp_re_2 = ifft2(ifftshift(im_dft));
        temp_phase_2 = angle(temp_re_2);
        
%         delta_phase(:,:,scale) = (temp_phase - temp_phase_2);
        phase_scales(2,scale,:,:) = temp_phase;
        phase_scales(1,scale,:,:) = temp_phase_2;
        
        re_scales(2,scale,:,:) = temp_re;
        re_scales(1,scale,:,:) = temp_re_2;
        
    end
    
%     % ʹ���ںϲ���
%     tdisp = multi_scales_fusion(phase_scales, roi_mask_2, roi_mask, oe, iou_threshold);
    % ʹ����λ�ݶ��Ƶ�ʽ
    tdisp = x_multi_scales_fusion(re_scales, roi_mask_2, roi_mask, oe, iou_threshold);
%     disp(1) = disp(1)+tdisp;signalout(2,i) = disp(1);       % ʹ������֡��
    disp(1) = tdisp;signalout(2,i) = disp(1);               % ʹ�òο�֡��
    addpoints(line,xline(i),double(disp(1)));
    
    % δʹ���ںϲ���
%     t_phase_1 = mod(pi+delta_phase(:,:,1), 2*pi)-pi;
%     t_phase_2 = mod(pi+delta_phase(:,:,2), 2*pi)-pi;
%     t_phase_3 = mod(pi+delta_phase(:,:,3), 2*pi)-pi;
%     t_mask = logical(roi_mask)&logical(roi_mask_2);                     % ʹ��ROI
%     tdisp = zeros(scales, 1);
%     if (sum(sum(single(t_mask))) ~= 0)
%         tdisp_1 = mean(t_phase_1(t_mask));% ��ֹmaskȫ�ձ���
%         tdisp_2 = mean(t_phase_2(t_mask));
%         tdisp_3 = mean(t_phase_3(t_mask));
%     else
%         tdisp_1 = 0; tdisp_2 = 0; tdisp_3 = 0;
%     end
%     disp(1) = disp(1)+tdisp_1;signalout(2,i) = disp(1);
%     disp(2) = disp(2)+tdisp_2;signalout(3,i) = disp(2);
%     disp(3) = disp(3)+tdisp_3;signalout(4,i) = disp(3);
    
    % �ֱ�鿴�����߶ȵ�����
%     addpoints(line,xline(i),double(disp(1)));
%     addpoints(line_2,xline(i),double(disp(2)));
%     addpoints(line_3,xline(i),double(disp(3)));
    
    % �鿴�����߶ȵı�ֵ
%     addpoints(line, xline(i),double(sign(tdisp_2) * tdisp_1 / (abs(tdisp_2)+1e-17)));
%     addpoints(line_2, xline(i),double((disp(1)) / (disp(2)+1e-17)));

    % �鿴��ֵmap
%     subplot(1, 3, 1); figure(f); surf(t_phase_1 ./ t_phase_2); view(0, -90);
%     subplot(1, 3, 2); figure(f); surf(t_phase_1 ./ t_phase_3); view(0, -90);
%     subplot(1, 3, 3); figure(f); surf(t_phase_2 ./ t_phase_3); view(0, -90);

    % �鿴��ͬ�߶���λ��map
%     subplot(1, 3, 1); figure(f); surf(delta_phase(:,:,1)); view(0, -90);
%     subplot(1, 3, 2); figure(f); surf(delta_phase(:,:,2)); view(0, -90);
%     subplot(1, 3, 3); figure(f); surf(delta_phase(:,:,3)); view(0, -90);
    
%     subplot(1, 3, 1); figure(f); surf(t_phase_1); view(0, -90);
%     subplot(1, 3, 2); figure(f); surf(t_phase_2); view(0, -90);
%     subplot(1, 3, 3); figure(f); surf(t_phase_3); view(0, -90);
    
%     subplot(1, 3, 1); figure(f); surf(t_phase_1 .* double(t_mask)); view(0, -90);
%     subplot(1, 3, 2); figure(f); surf(t_phase_2 .* double(t_mask)); view(0, -90);
%     subplot(1, 3, 3); figure(f); surf(t_phase_3 .* double(t_mask)); view(0, -90);



    figure(f);
    title(['frame:',num2str(i)]);

end

% profile viewer;
% toc;
% fprintf('Total used time: %f s\n', toc);