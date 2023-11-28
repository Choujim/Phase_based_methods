% ��߶���λ�ں�
% ���ڶԱȻҶ���Ϣ����λ��Ϣ��Ч��

function result = xx_multi_scales_fusion(phase_scales, gray_scales, pre_roi, cur_roi, orient_angle, iou_threshold)
    pre_phase_scales = squeeze(phase_scales(1,:,:,:));
    cur_phase_scales = squeeze(phase_scales(2,:,:,:));
    pre_gray_scales = squeeze(gray_scales(1,:,:,:));
    cur_gray_scales = squeeze(gray_scales(2,:,:,:));
%     pre_phase_scales(1,:,:) = squeeze(phase_scales(1,1,:,:));
%     cur_phase_scales(1,:,:) = squeeze(phase_scales(2,1,:,:));
    s_size = size(phase_scales);
    scales = s_size(2);
    iterations = 10;
    result_temp = zeros(scales-1, 1);
    
    phase_to_pixel = 1; % �Ƿ����λ��ת������λ�ƣ���Ҫ���пռ��ݶȼ��㣩
    
    % ͨ��ROI�����֮֡�䷢����λ���Ǵ�λ�ƻ���Сλ��
    cross_roi = logical(pre_roi) & logical(cur_roi);%fprintf('sum_cross: %f',sum(sum(single(cross_roi))));
    union_roi = logical(pre_roi) | logical(cur_roi);%fprintf('sum_union: %f\n',sum(sum(single(union_roi))));
    iou = sum(sum(single(cross_roi))) / sum(sum(single(union_roi)));
    if (iou >= iou_threshold)
        %%
        % ��ʱ�ж�ΪСλ�ƣ�����ֱ��ʹ����λ��
        delta_phase = zeros(scales, s_size(3), s_size(4));
        sum_points = sum(sum(single(cross_roi)));
        roi_index = find(cross_roi);
        
        % ʹ�ò�Ƶ��У׼��λ��map
        ref_delta_phase = zeros(3, s_size(3), s_size(4));
        if (scales > 1)
            ref_delta_phase(1,:,:) = (mod(pre_phase_scales(1,:,:) - pre_phase_scales(2,:,:) + pi, 2*pi)-pi);
            ref_delta_phase(2,:,:) = (mod(pre_phase_scales(1,:,:) - pre_phase_scales(2,:,:) + pi, 2*pi)-pi);
            ref_delta_phase(3,:,:) = mod(pi+(mod(cur_phase_scales(1,:,:) - cur_phase_scales(2,:,:) + pi, 2*pi)-pi)...
                                      - (mod(pre_phase_scales(1,:,:) - pre_phase_scales(2,:,:) + pi, 2*pi)-pi), 2*pi)-pi;
        end
        for i = 1:scales
            delta_phase(i,:,:) = cur_phase_scales(i,:,:) - pre_phase_scales(i,:,:);
        end
        
%         points_static = zeros(2, scales);
        for p = 1:sum_points
            % ����ref_delta��У׼delta_phase
            [row, col] = ind2sub([s_size(3), s_size(4)], roi_index(p));
            for s = 1: scales
               if ((sign(delta_phase(s, row, col)) ~= sign(ref_delta_phase(3, row, col))))
%                    delta_phase(s, row, col) = sign(ref_delta_phase(3, row, col))*2*pi + delta_phase(s, row, col);
               end
               % ��ԭphase mapҲ����չ��
               if (sign(pre_phase_scales(s, row, col)) ~= sign(ref_delta_phase(1, row, col))...
                       && abs(pre_phase_scales(s, row, col) - ref_delta_phase(1, row, col)) > pi) 
                   pre_phase_scales(s, row, col) = sign(ref_delta_phase(1, row, col))*2*pi + pre_phase_scales(s, row, col);
               end
               if (sign(cur_phase_scales(s, row, col)) ~= sign(ref_delta_phase(2, row, col))...
                       && abs(cur_phase_scales(s, row, col) - ref_delta_phase(2, row, col)) > pi) 
                   cur_phase_scales(s, row, col) = sign(ref_delta_phase(2, row, col))*2*pi + cur_phase_scales(s, row, col);
               end
               % �ٴ�У׼delta_phase
               if((sign(delta_phase(s, row, col)) ~= sign(cur_phase_scales(s,row,col) - pre_phase_scales(s,row,col))))
%                    delta_phase(s, row, col) = cur_phase_scales(s,row,col) - pre_phase_scales(s,row,col);
               end
               % mean
%                points_static(1, s) = points_static(1, s) + delta_phase(s, row, col);
               
            end
            
        end
        % ͳ�ƾ�ֵ�ͷ���
%         points_static(1,:) = points_static(1,:) ./ sum_points;
%         for p = 1:sum_points
%             [row, col] = ind2sub([s_size(3), s_size(4)], roi_index(p));
%             for s = 1:scales
%                 % var
%                 points_static(2, s) = points_static(2, s) + (delta_phase(s, row, col) - points_static(1,s)).^2;
%             end
%         end
%         points_static(2,:) = points_static(2,:) ./sum_points;

        
        % Gauss-Newton
%         for s = 1:scales
%             for it = 1:iterations
%                 H = 0;
%                 b = 0;
%                 cost_temp = 0;
%                 num_good = 0;
%                 for p = 1:sum_points
%                     % Outliers detect
%                     if (sign(delta_phase(1, row, col)) ~= sign(ref_delta_phase(row, col))...
%                             && abs(delta_phase(s, row, col) - points_static(1,s)) > 2*pi)
%                         continue;
%                     end
%                     num_good = num_good + 1;
%                     error = delta_phase(1, row, col) - alpha_temp * delta_phase_cur(row, col);
%                     J = -1.0 * (delta_phase_cur(row, col));
%                     hessian = hessian + J'*J;
%                     bias = bias - J * error;
%                     cost_temp = cost_temp + error * error; 
%                 end
%             end
%         end
        
        displacement_cost = zeros(scales, 3);
        for i = 1:scales
            displacement_cost(i,:) = ...
                xx_msf_large_motion(squeeze(pre_phase_scales(i,:,:)), ...
                                    squeeze(cur_phase_scales(i,:,:)), ...
                                    squeeze(pre_gray_scales(i,:,:)), ...
                                    squeeze(cur_gray_scales(i,:,:)), ...
                                    pre_roi, ...
                                    cur_roi, ...
                                    orient_angle,...
                                    (scales-i)*5+10);
%             fprintf('scale: %d large motion: %f\n', i, displacement_cost(i, 1));
            fprintf('scale: %d disp: %f \t mean: %f \t var: %f\n', i, displacement_cost(i, 1), displacement_cost(i, 2), displacement_cost(i, 3));
        end
        
        % kalman �����ں�
        % �۲����
        H = zeros(scales, 1); H = H + 1;
        % �۲�Э����
        R = zeros(scales, scales);
        for s = 1:scales
            R(s,s) = displacement_cost(s, 3) + 1e-17;
        end
%         R = R ./ 6000
        % ϵͳ����Э����ٶ�Ϊ1e-4
        Q = 1e-4;
        % ���Э����, �ٶ�Ϊһ��Сֵ���ٶ�Ϊ0.1
        P = 0.1;
        % Ԥ��ֵ
%         result_weight = (sum(displacement_cost(:,2))-displacement_cost(:,2)) ./ (sum(displacement_cost(:,2)+1e-17));
%         x_init = sum(displacement_cost(:,1) .* result_weight) / (scales-1);
        x_init = mean(displacement_cost(:,1));
%         x_init = 0;
        P_cur = 1 * P * 1 + Q;
        Z_pre = H * x_init;
        % kalman ����
        K = (P_cur * H') / (H * P_cur * H' + R);
        % ����
        x_optim = x_init + K * (displacement_cost(:,1) - Z_pre);
        % �������Ȩ������С����
%         A_total = R \ ones(scales,1) / (ones(scales,1)' / R * ones(scales,1));
%         x_optim = A_total' * displacement_cost(:,1);
        P_optim = (eye(1,1) - (K * H)) * P_cur;
        result = x_optim;
%         fprintf('disp: %f \n', result);

    else
        %%
        % ��ʱ�ж�Ϊ��λ�ƣ�����ֱ��ʹ����λ��
%         fprintf('current motion is large motion\n');
%         result = 0;
        displacement_cost = zeros(scales, 3);
        for i = 1:scales
            displacement_cost(i,:) = ...
                msf_large_motion(squeeze(pre_phase_scales(i,:,:)), ...
                                 squeeze(cur_phase_scales(i,:,:)), ...
                                 pre_roi, ...
                                 cur_roi, ...
                                 orient_angle,...
                                 (scales-i)*5+10);
%              fprintf('scale: %d large motion: %f\n', i, displacement_cost(i, 1));
%             fprintf('scale: %d disp: %f \t mean: %f \t var: %f\n', ...
%                 i, displacement_cost(i, 1), displacement_cost(i, 2), displacement_cost(i, 3));
        end
%         result = mean(displacement(3));
        result_weight = (sum(displacement_cost(:,2))-displacement_cost(:,2)) ./ (sum(displacement_cost(:,2)+1e-17));
        result = sum(displacement_cost(:,1) .* result_weight) / (scales-1);
        
        
        
    end
    
    
end