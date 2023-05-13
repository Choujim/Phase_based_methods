% 高置信度相位区域的选取
% 融合了索段的形状信息和相位结构信息
% phase_bin：phase map 上的 bin 大小，可参考[-pi, pi]分成 4*scale 个bin
% line_width：直线库的直线线宽，一般设置为与等相位条带2倍宽，参考宽度可设为 2*scale
% cos_ratio：余弦相似度阈值，可作为超参调试
function [ratio,roi_mask] = high_confidence_roi(image, orient_angle, num_phase_bin, line_width, cos_ratio)
    dim = size(image);
    % 划分相位
    % [-pi, pi]可分成 4*scale 个bin
    angle_bin = [-num_phase_bin : 2 : num_phase_bin] / num_phase_bin * pi;
    phase_bin_masks = single(zeros(dim(1), dim(2), num_phase_bin));
    
    mask = mask_creator(dim, 2, 4, [1, orient_angle/180*pi], 1);
    im_dft = mask .* fftshift(fft2(image));
    temp_re = ifft2(ifftshift(im_dft));
    temp_phase = angle(temp_re);
    for num = 1:num_phase_bin 
        left_mask = temp_phase>=angle_bin(num);
        right_mask = temp_phase<=angle_bin(num+1);
        phase_bin_masks(:,:, num) = single(left_mask==right_mask);
%         figure; surf(single(phase_bin_masks(:,:, num)));
    end
    
    % 根据识别出的oe遍历每个bin_masks，即可以得到特定角度的phase map区域
    % 宽度可设为1*scale
    line_k = (orient_angle-90)/abs(orient_angle-90) * tan(abs(orient_angle-90)/180*pi);
    % 获得指定line_k, line_width的直线bin库
    line_bins = line_bin_mask(dim(1), dim(2), line_k, line_width);
    
    size_p = size(phase_bin_masks);
    size_l = size(line_bins);
    refine_roi = single(zeros(dim(1), dim(2), size_p(3)));
    
    ratio = zeros(size_p(3), size_l(3));
    for it_p = 1:size_p(3)
        temp_phase_bin = phase_bin_masks(:,:,it_p);
        
        for it_l = 1:size_l(3)
            temp = line_bins(:,:,it_l);
            query = temp_phase_bin .* temp;
            temp_ref = sum(temp(:));
            temp_query = sum(query(:));
    %         temp_query / temp_ref
            if (temp_query > 8)  && ((temp_query / temp_ref) >= cos_ratio)
                refine_roi(:,:,it_p) = refine_roi(:,:,it_p) + query + temp;
            end
            % 查看平均cos_ratio曲线
            ratio(it_p, it_l) = (temp_query / temp_ref);
            % 查看temp_query曲线
%             ratio(it_p, it_l) = temp_query;
            
        end
        refine_roi(:,:,it_p) = single(logical(refine_roi(:,:,it_p)));
        
    end

    roi_mask = single(zeros(dim(1), dim(2)));
    for it_p = 1:size_p(3)
        roi_mask = roi_mask + refine_roi(:,:,it_p);
    end
    roi_mask = single(logical(roi_mask));


end