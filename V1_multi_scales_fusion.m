% ��߶���λ�ں�
% ּ���ں϶�߶ȵ���λ��Ϣ�԰��������߶��˶���������λ��������

function result = multi_scales_fusion(phase_scales, pre_roi, cur_roi, orient_angle, iou_threshold)
    pre_phase_scales = squeeze(phase_scales(1,:,:,:));
    cur_phase_scales = squeeze(phase_scales(2,:,:,:));
    s_size = size(phase_scales);
    scales = s_size(2);
    iterations = 10;
    result_temp = zeros(scales-1, 1);
    
    phase_to_pixel = 1; % �Ƿ����λ��ת������λ�ƣ���Ҫ���пռ��ݶȼ��㣩
    
    % ͨ��ROI�����֮֡�䷢����λ���Ǵ�λ�ƻ���Сλ��
    cross_roi = logical(pre_roi) & logical(cur_roi);
    union_roi = logical(pre_roi) | logical(cur_roi);
    iou = sum(sum(single(cross_roi))) / sum(sum(single(union_roi)));
    if (iou >= iou_threshold)
        %%
        % ��ʱ�ж�ΪСλ�ƣ�����ֱ��ʹ����λ��
        alpha = 2.0 .* (1:scales-1);
        delta_phase = zeros(scales, s_size(3), s_size(4));
        sum_points = sum(sum(single(cross_roi)));
        roi_index = find(cross_roi);
        
        unwrap_phase_pre = zeros(scales, s_size(3), s_size(4));
        unwrap_phase_cur = zeros(scales, s_size(3), s_size(4));
        unwrap_delta_phase = zeros(scales, s_size(3), s_size(4));
        for i = 1:scales
            delta_phase(i,:,:) = cur_phase_scales(i,:,:) - pre_phase_scales(i,:,:);
%             % ��TIE-based algorithm unwrap
% %             [unwrap_delta_phase_pre, N] = Unwrap_TIE_DCT_Iter(squeeze(cur_phase_scales(i,:,:)));
% %             [unwrap_delta_phase_cur, N] = Unwrap_TIE_DCT_Iter(squeeze(pre_phase_scales(i,:,:)));
% %             unwrap_delta_phase(i,:,:) = unwrap_delta_phase_cur - unwrap_delta_phase_pre;
%             [unwrap_phase_pre(i,:,:), N] = Unwrap_TIE_DCT_Iter(squeeze(pre_phase_scales(i,:,:)));
%             [unwrap_phase_cur(i,:,:), N] = Unwrap_TIE_DCT_Iter(squeeze(cur_phase_scales(i,:,:)));
%              % unwrap�������λͬ��
%             unwrap_pre_temp = squeeze(unwrap_phase_pre(i,:,:));  sum_pre = mean(unwrap_pre_temp(logical(pre_roi)));
%             unwrap_cur_temp = squeeze(unwrap_phase_cur(i,:,:));  sum_cur = mean(unwrap_cur_temp(logical(cur_roi)));
%             pre_temp = squeeze(pre_phase_scales(i,:,:));  ref_sum_pre = mean(pre_temp(logical(pre_roi)));
%             cur_temp = squeeze(cur_phase_scales(i,:,:));  ref_sum_cur = mean(cur_temp(logical(cur_roi)));
%             unwrap_phase_pre(i,:,:) = unwrap_phase_pre(i,:,:) + (ref_sum_pre - sum_pre);
%             unwrap_phase_cur(i,:,:) = unwrap_phase_cur(i,:,:) + (ref_sum_cur - sum_cur);
%             unwrap_delta_phase(i,:,:) = unwrap_phase_cur(i,:,:) - unwrap_phase_pre(i,:,:);
%             delta_phase(i,:,:) = unwrap_delta_phase(i,:,:);
        end
        
        for i = scales:-1:2
            delta_phase_cur = squeeze(delta_phase(i,:,:));
            cost = 0; last_cost = 0;
%             for p = 1:sum_points
%                 [row, col] = ind2sub([s_size(3), s_size(4)], roi_index(p));
%                 for s = scales-1:-1:1
%                     % coarse to fine����λunwrap����
%                     if ((sign(delta_phase(s, row, col)) ~= sign(delta_phase(scales, row, col)))...
%                             && abs(delta_phase(s, row, col) - delta_phase(scales, row, col)) >= pi) 
%                         delta_phase(s, row, col) = sign(delta_phase(scales, row, col))*2*pi + delta_phase(s, row, col);
%                     end
%                     % ��ԭphase map��unwrap
%                     if (sign(pre_phase_scales(s, row, col)) ~= sign(pre_phase_scales(scales, row, col)) ...
%                             && abs(pre_phase_scales(s, row, col) - pre_phase_scales(scales, row, col)) >= pi) 
%                         pre_phase_scales(s, row, col) = sign(pre_phase_scales(scales, row, col))*2*pi + pre_phase_scales(s, row, col);
%                     end
%                     if (sign(cur_phase_scales(s, row, col)) ~= sign(cur_phase_scales(scales, row, col)) ...
%                             && abs(cur_phase_scales(s, row, col) - cur_phase_scales(scales, row, col)) >= pi) 
%                         cur_phase_scales(s, row, col) = sign(cur_phase_scales(scales, row, col))*2*pi + cur_phase_scales(s, row, col);
%                     end
%                 end
%             end
            
            for it = 1:iterations
                alpha_temp = alpha(i-1);
                hessian = 0;
                bias = 0;
                cost_temp = 0;
                num_good = 0;
                for p = 1:sum_points
                    % ������һ��У׼wrapped phase
                    [row, col] = ind2sub([s_size(3), s_size(4)], roi_index(p));
                    for s = i-1:-1:1
                        % coarse to fine����λunwrap����
                        if ((sign(delta_phase(s, row, col)) ~= sign(delta_phase_cur(row, col)))...
                                && abs(delta_phase(s, row, col) - delta_phase_cur(row, col)) >= pi) 
                            delta_phase(s, row, col) = sign(delta_phase_cur(row, col))*2*pi + delta_phase(s, row, col);
                        end
                        % ��ԭphase map��unwrap
                        if (sign(pre_phase_scales(s, row, col)) ~= sign(pre_phase_scales(i, row, col)) ...
                                && abs(pre_phase_scales(s, row, col) - pre_phase_scales(i, row, col)) >= pi) 
                            pre_phase_scales(s, row, col) = sign(pre_phase_scales(i, row, col))*2*pi + pre_phase_scales(s, row, col);
                        end
                        if (sign(cur_phase_scales(s, row, col)) ~= sign(cur_phase_scales(i, row, col)) ...
                                && abs(cur_phase_scales(s, row, col) - cur_phase_scales(i, row, col)) >= pi) 
                            cur_phase_scales(s, row, col) = sign(cur_phase_scales(i, row, col))*2*pi + cur_phase_scales(s, row, col);
                        end
                    end

                    % Gauss-Newton
                    if (sign(delta_phase(1, row, col)) ~= sign(delta_phase_cur(row, col)))
                        continue;
                    end
%                     if (abs(delta_phase(1, row, col)) > 1.5*pi...
%                         || abs(delta_phase_cur(row, col)) > 1.5*pi)
%                         continue;
%                     end
                    num_good = num_good + 1;
                    error = delta_phase(1, row, col) - alpha_temp * delta_phase_cur(row, col);
                    J = -1.0 * (delta_phase_cur(row, col));
                    hessian = hessian + J'*J;
                    bias = bias - J * error;
                    cost_temp = cost_temp + error * error; 
                end
                
                cost = cost_temp / num_good;
                % cost�������ֹ
                if (it > 1 && cost > last_cost)
                    break;
                end
                % ��Ⲣupdate alpha
                if (num_good > 0)
                    update = (hessian+1e-17) \ bias;
                    alpha_temp = alpha_temp + update;
                else
                    continue;
                end
                alpha(i-1) = alpha_temp;
%                 fprintf('iteration: %d alpha now = %f, cost = %f\n',it,alpha_temp, cost);
                % udpate̫С����ֹ
                if (abs(update) < 1e-5)
                    break;
                end
                last_cost = cost;     
            end
            
            refine_phase = squeeze(delta_phase(1,:,:)); refine_phase = refine_phase(cross_roi);
            refine_phase = refine_phase + alpha(i-1) .* delta_phase_cur(cross_roi);
            result_temp(i-1) = 1/2 * mean(refine_phase);
%             fprintf('result_delta_phase in scale %d is: %f\n',i-1, result_temp(i-1));
        end
%         result = mean(result_temp(:));
%         result = mean(mean(squeeze(mod(pi+delta_phase(2,:,:), 2*pi)-pi)));
        
        result_total = zeros(scales, 1);
        for i = 1:scales
            re_temp = squeeze(delta_phase(i,:,:));
            result_total(i) = mean(mod(pi + re_temp(cross_roi), 2*pi)-pi);
        end
        
        if (phase_to_pixel == 1)
            % ��pca���ƽ����Ƹ����߶ȵ�б��
            omega = zeros(scales, 1);
%             % ���ɲ鿴��֤�㼯
%             test_points = zeros(s_size(3), s_size(4));
%             for p = 1:sum_points
%                 [row, col] = ind2sub([s_size(3), s_size(4)], roi_index(p));
%                 test_points(row,col) = 1-row-col;
%             end
%             
            for i=1:scales
%                 [omega(i), normal_vector]= plane_fitting(squeeze(unwrap_phase_pre(i,:,:)), pre_roi, orient_angle, 0);
%                 result_total(i) = result_total(i) / omega(i);
                [omega(i), normal_vector]= plane_fitting(squeeze(pre_phase_scales(i,:,:)), pre_roi, orient_angle, 0);
%                 fprintf('scales: %d normal vector: [%f, %f, %f]\n', i, normal_vector(1),normal_vector(2),normal_vector(3));
%                 fprintf('scale: %d, omega: %f\n', i, omega(i));
                % �鿴��֤�㼯�ķ�����
%                 [omega(i), normal_vector]= plane_fitting(test_points, cross_roi, orient_angle, 0);
%                 fprintf('normal vector: [%f, %f, %f]\n', normal_vector(1),normal_vector(2),normal_vector(3));
            end
            refine_omega = omega(1);
            for i = 1:scales-1
                refine_omega = refine_omega + omega(i+1) * alpha(i);
            end
            refine_omega = refine_omega / scales;
%             refine_omega = mean([omega(1), omega(2)*alpha(1), omega(3)*alpha(2)]);
%             result = result / refine_omega;
%             refine_omega = [omega(2), omega(3)];
%             result = 0.5*(result_temp' * refine_omega');
            
            result = mean(result_total);
            
        end 
    else
        %%
        % ��ʱ�ж�Ϊ��λ�ƣ�����ֱ��ʹ����λ��
%         fprintf('current motion is large motion\n');
%         result = 0;
        displacement_cost = zeros(scales, 2);
        for i = 1:scales
            displacement_cost(i,:) = ...
                msf_large_motion(squeeze(pre_phase_scales(i,:,:)), ...
                                 squeeze(cur_phase_scales(i,:,:)), ...
                                 pre_roi, ...
                                 cur_roi, ...
                                 orient_angle,...
                                 (scales-i)*5+10);
%              fprintf('scale: %d large motion: %f\n', i, displacement_cost(i, 1));
        end
%         result = mean(displacement(3));
        result_weight = (sum(displacement_cost(:,2))-displacement_cost(:,2)) ./ (sum(displacement_cost(:,2)+1e-17));
        result = sum(displacement_cost(:,1) .* result_weight) / (scales-1);
        
        
        
    end
    
    
end