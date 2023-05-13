% 多尺度相位融合
% 旨在融合多尺度的相位信息以帮助解决大尺度运动发生的相位包裹现象

% 使用相位梯度推导式代替相位梯度

function result = x_multi_scales_fusion(result_scales, pre_roi, cur_roi, orient_angle, iou_threshold)
%     pre_re_scales = squeeze(result_scales(1,:,:,:));
%     cur_re_scales = squeeze(result_scales(2,:,:,:));
    pre_re_scales(1,:,:) = result_scales(1,1,:,:);
    cur_re_scales(1,:,:) = result_scales(2,1,:,:);
    % 归一化
    pre_re_scales = pre_re_scales ./ abs(pre_re_scales);
    cur_re_scales = cur_re_scales ./ abs(cur_re_scales);
    s_size = size(result_scales);
    scales = s_size(2);
    iterations = 10;
    result_temp = zeros(scales-1, 1);
    
    phase_to_pixel = 1; % 是否把相位差转成像素位移（需要进行空间梯度计算）
    
    % 通过ROI检查两帧之间发生的位移是大位移还是小位移
    cross_roi = logical(pre_roi) & logical(cur_roi);%fprintf('sum_cross: %f',sum(sum(single(cross_roi))));
    union_roi = logical(pre_roi) | logical(cur_roi);%fprintf('sum_union: %f\n',sum(sum(single(union_roi))));
    iou = sum(sum(single(cross_roi))) / sum(sum(single(union_roi)));
    if (iou >= iou_threshold)
        %%
        % 此时判定为小位移，可以直接使用相位差
        delta_phase = zeros(scales, s_size(3), s_size(4));
        sum_points = sum(sum(single(cross_roi)));
        roi_index = find(cross_roi);
        
        displacement_cost = zeros(scales, 3);
        for i = 1:scales
            displacement_cost(i,:) = ...
                x_msf_large_motion(squeeze(pre_re_scales(i,:,:)), ...
                                 squeeze(cur_re_scales(i,:,:)), ...
                                 pre_roi, ...
                                 cur_roi, ...
                                 orient_angle,...
                                 (scales-i)*5+10);
%             fprintf('scale: %d large motion: %f\n', i, displacement_cost(i, 1));
%             fprintf('scale: %d disp: %f \t mean: %f \t var: %f\n', i, displacement_cost(i, 1), displacement_cost(i, 2), displacement_cost(i, 3));
        end
        
        % kalman 数据融合
        % 观测矩阵
        H = zeros(scales, 1); H = H + 1;
        % 观测协方差
        R = zeros(scales, scales);
        for s = 1:scales
            R(s,s) = displacement_cost(s, 3) + 1e-17;
        end
%         R = R ./ 6000
        % 系统噪声协方差，假定为1e-4
        Q = 1e-4;
        % 误差协方差, 假定为一较小值，假定为0.1
        P = 0.001;
        % 预测值
%         result_weight = (sum(displacement_cost(:,2))-displacement_cost(:,2)) ./ (sum(displacement_cost(:,2)+1e-17));
%         x_init = sum(displacement_cost(:,1) .* result_weight) / (scales-1);
        x_init = mean(displacement_cost(:,1));
%         x_init = 0;
        P_cur = 1 * P * 1 + Q;
        Z_pre = H * x_init;
        % kalman 增益
        K = (P_cur * H') / (H * P_cur * H' + R);
        % 更新
        x_optim = x_init + K * (displacement_cost(:,1) - Z_pre);
        % 按矩阵加权线性最小方差
%         A_total = R \ ones(scales,1) / (ones(scales,1)' / R * ones(scales,1));
%         x_optim = A_total' * displacement_cost(:,1);
        P_optim = (eye(1,1) - (K * H)) * P_cur;
        result = x_optim;
    end
end