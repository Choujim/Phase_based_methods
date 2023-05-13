% 查看各种phase unwrap算法的效果
f = figure;
f.Position(3:4) = [900 400];

%%

oe = 65;
scales = 3;
num_phase_bin = 4;
line_width = 2;
cos_ratio = 0.50;
dim = [(roi(1,2)-roi(1,1)), (roi(2,2)-roi(2,1))];
% 根据识别出的oe遍历每个bin_masks，即可以得到特定角度的phase map区域
% 宽度可设为1*scale
line_k = (oe-90)/abs(oe-90+1e-17) * tan(abs(oe-90)/180*pi);
% 获得指定line_k, line_width的直线bin库
line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);

for i = 2:2
% 输出相位差  
    delta_phase = zeros(dim(1), dim(2), scales);
    phase_scales = zeros(2, scales, dim(1), dim(2));
    for scale = 1:scales
        vframein = vHandle.read(i);
        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
        image = im2single(squeeze(mean(im,3)));
        [ratio,roi_mask] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
        mask = mask_creator(size(im), 2, 4, [scale, oe/180*pi], 1);
        im_dft = mask .* fftshift(fft2(image));
        temp_re = ifft2(ifftshift(im_dft));
        temp_phase = angle(temp_re);
        %
        [temp_phase_unwrap,N]=Unwrap_TIE_DCT_Iter(temp_phase);

        vframein = vHandle.read(i-1);
        im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
        image = im2single(squeeze(mean(im,3)));
        [ratio_2, roi_mask_2] = high_confidence_roi(image, line_bins, oe, num_phase_bin, cos_ratio);
        im_dft = mask .* fftshift(fft2(image));
        temp_re_2 = ifft2(ifftshift(im_dft));
        temp_phase_2 = angle(temp_re_2);
        %
        [temp_phase_2_unwrap,N_2]=Unwrap_TIE_DCT_Iter(temp_phase_2);
        
%         delta_phase(:,:,scale) = (temp_phase - temp_phase_2);
        phase_scales(2,scale,:,:) = temp_phase;
        phase_scales(1,scale,:,:) = temp_phase_2;
        
    end
end