% 高置信度相位区域的选取
% 融合了索段的形状信息和相位结构信息
% phase_bin：phase map 上的 bin 大小，可参考[-pi, pi]分成 4*scale 个bin
% cos_ratio：余弦相似度阈值，可作为超参调试

function [ratio,roi_mask] = high_confidence_roi(image, line_bins, orient_angle, num_phase_bin, cos_ratio)
    dim = size(image);
    % 划分相位
    % [-pi, pi]可分成 4*scale 个bin
%     angle_bin = [-num_phase_bin : 1 : num_phase_bin] / num_phase_bin * pi;
%     phase_bin_masks = single(zeros(dim(1), dim(2), num_phase_bin));
%     
%     mask = mask_creator(dim, 2, 4, [1, orient_angle/180*pi], 1);
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re = ifft2(ifftshift(im_dft));
%     temp_phase = angle(temp_re);
    
    shape_image = de_background(image, 2);
    mean_shape = mean(abs(shape_image(:)));
    mean_shape_mask = abs(shape_image) > mean_shape;
    
    size_l = size(line_bins);
    refine_roi = single(zeros(dim(1), dim(2)));
    
    for it_l = 1: size_l(3)
        temp = line_bins(:,:,it_l);
        query = single(mean_shape_mask) .* temp;
        temp_ref = sum(temp(:));
        temp_query = sum(query(:));
        if (temp_query > 8)  && ((temp_query / temp_ref) >= cos_ratio)
            refine_roi(:,:) = refine_roi(:,:) + temp;
        end
    end
    refine_roi = single(logical(refine_roi)); 
    
    %%
    % 以0相位为中心
%     left_mask = temp_phase>=angle_bin(2*num_phase_bin);
%     right_mask = temp_phase<=angle_bin(2*1);
%     phase_bin_masks(:,:, num_phase_bin) = single((left_mask | right_mask) & mean_shape_mask);
%     for num = 1:num_phase_bin-1 
%         left_mask = temp_phase>=angle_bin(2*num);
%         right_mask = temp_phase<=angle_bin(2*(num+1));
%         phase_bin_masks(:,:, num) = single((left_mask==right_mask)&mean_shape_mask);
% %         figure; surf(single(phase_bin_masks(:,:, num)));view(0, -90);
%     end
%     
%     size_p = size(phase_bin_masks);
%     size_l = size(line_bins);
%     refine_roi = single(zeros(dim(1), dim(2), size_p(3)));
%     
%     ratio = zeros(size_p(3), size_l(3));
%     for it_p = 1:size_p(3)
%         temp_phase_bin = phase_bin_masks(:,:,it_p);
%         
%         for it_l = 1:size_l(3)
%             temp = line_bins(:,:,it_l);
%             query = temp_phase_bin .* temp;
%             temp_ref = sum(temp(:));
%             temp_query = sum(query(:));
%     %         temp_query / temp_ref
%             if (temp_query > 8)  && ((temp_query / temp_ref) >= cos_ratio)
%                 refine_roi(:,:,it_p) = refine_roi(:,:,it_p) + temp;
% %                 refine_roi(:,:,it_p) = refine_roi(:,:,it_p) + query;
%             end
%             % 查看平均cos_ratio曲线
%             ratio(it_p, it_l) = (temp_query / temp_ref);
%             % 查看temp_query曲线
% %             ratio(it_p, it_l) = temp_query;
%             
%         end
%         refine_roi(:,:,it_p) = single(logical(refine_roi(:,:,it_p))); 
%     end
% 
%     roi_mask = single(zeros(dim(1), dim(2)));
%     for it_p = 1:size_p(3)
%         roi_mask = roi_mask + refine_roi(:,:,it_p);
%     end
    %%
%     roi_mask = single(logical(roi_mask));
%     roi_mask = single(logical(roi_mask) & mean_shape_mask);
%     roi_mask = single(mean_shape_mask);ratio=0;
    roi_mask = single(logical(refine_roi)); ratio=0;
    
    
end