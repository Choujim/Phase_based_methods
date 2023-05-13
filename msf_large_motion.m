% 用作大位移（>5 pixel)时的位移估计

function displacement_cost = msf_large_motion(pre_phase_scales, cur_phase_scales, pre_roi, cur_roi, orient_angle, iterations, method)
%     iterations = 10;
    plane_constraint_ratio = 0.3;
    s_size = size(pre_roi);
    roi_index = find(logical(pre_roi));
    % 先计算前后帧ROI之间的距离作为初步估计值
    t_init = 0;
    orient_angle = orient_angle / 180 * pi;
    t_unit = [sin(orient_angle), cos(orient_angle)];
    t_pre_roi = zeros(2,1);
    t_cur_roi = zeros(2,1);
    sum_pre_roi = sum(sum(pre_roi)); index_pre = find(logical(pre_roi));
    sum_cur_roi = sum(sum(cur_roi)); index_cur = find(logical(cur_roi));
    % 估计每个ROI到原点的距离在索段法向方向的投影值
    for i=1:sum_pre_roi
        [temp_row,temp_col] = ind2sub([s_size(1), s_size(2)], index_pre(i));
        t_pre_roi(1) = t_pre_roi(1) + temp_row;
        t_pre_roi(2) = t_pre_roi(2) + temp_col;
    end
%     fprintf("t_pre_roi: %f\n", t_pre_roi);
    t_pre_roi = t_pre_roi ./ sum_pre_roi;
    for j=1:sum_cur_roi
        [temp_row,temp_col] = ind2sub([s_size(1), s_size(2)], index_cur(j));
        t_cur_roi(1) = t_cur_roi(1) + temp_row;
        t_cur_roi(2) = t_cur_roi(2) + temp_col;
    end
%     fprintf("t_cur_roi: %f\n", t_cur_roi);
    t_cur_roi = t_cur_roi ./ sum_cur_roi; 
    t_init = t_init + dot((t_cur_roi - t_pre_roi), t_unit); fprintf("t_init: %f\n", t_init);
%     fprintf('current motion is %d\n', t_init);

    % 最小化相位误差
    t_optim = t_init;
%     t_optim = sign(t_init) * 6.0;
    cost = 0; last_cost = 0; t_last = 0;
    half_patch_size = 1;
    
    right_point = zeros(s_size(1),s_size(2));
    right_point_2 = zeros(s_size(1),s_size(2));
    
    for it = 1:iterations
        H = 0;
        b = 0;
        cost_temp = 0; cost_copy = zeros(sum_pre_roi); %cost_data = [];
        num_good = 0;
        for p = 1:sum_pre_roi
            hessian = 0;
            bias = 0;
            [row, col] = ind2sub([s_size(1), s_size(2)], roi_index(p));
            cur_row = row + t_optim * sin(orient_angle);
            cur_col = col + t_optim * cos(orient_angle);
            % 以最近整数点作为patch的中心点确定局部平面
            cur_row_int = round(cur_row);
            cur_col_int = round(cur_col);
            if (row <= half_patch_size || row >= s_size(1)-half_patch_size ||...
                col <= half_patch_size || col >= s_size(2)-half_patch_size )
                continue;
            end
            if (cur_row_int <= half_patch_size || cur_row_int >= s_size(1)-half_patch_size ||...
                cur_col_int <= half_patch_size || cur_col_int >= s_size(2)-half_patch_size )
                continue;
            end
            if (abs(pre_phase_scales(row, col) - cur_phase_scales(cur_row_int, cur_col_int)) > pi)
               continue; 
            end
            % 舍弃局部不满足平面约束的点
            %  前后帧都需要满足
            y_query = cur_row_int + [-half_patch_size : half_patch_size];
            x_query = cur_col_int + [-half_patch_size : half_patch_size];
            y_line = cur_phase_scales(y_query, cur_col_int);
            x_line = cur_phase_scales(cur_row_int, x_query);
            y_patch = zeros(half_patch_size + 1, 1); y_patch(1) = y_line(half_patch_size+1);
            x_patch = zeros(half_patch_size + 1, 1); x_patch(1) = x_line(half_patch_size+1);
            for patch = 1:half_patch_size
                y_patch(patch+1) = 0.5 * (y_line(patch) + y_line(patch+half_patch_size+1));
                x_patch(patch+1) = 0.5 * (x_line(patch) + x_line(patch+half_patch_size+1));
            end
            if (std(y_patch) > abs(mean(y_patch)) * plane_constraint_ratio)
                continue;
            end
            if (std(x_patch) > abs(mean(x_patch)) * plane_constraint_ratio)
                continue;
            end
            %  前后帧都需要满足
            y_query = row + [-half_patch_size : half_patch_size];
            x_query = col + [-half_patch_size : half_patch_size];
            y_line = pre_phase_scales(y_query, col);
            x_line = pre_phase_scales(row, x_query);
            y_patch = zeros(half_patch_size + 1, 1); y_patch(1) = y_line(half_patch_size+1);
            x_patch = zeros(half_patch_size + 1, 1); x_patch(1) = x_line(half_patch_size+1);
            for patch = 1:half_patch_size
                y_patch(patch+1) = 0.5 * (y_line(patch) + y_line(patch+half_patch_size+1));
                x_patch(patch+1) = 0.5 * (x_line(patch) + x_line(patch+half_patch_size+1));
            end
            if (std(y_patch) > abs(mean(y_patch)) * plane_constraint_ratio)
                continue;
            end
            if (std(x_patch) > abs(mean(x_patch)) * plane_constraint_ratio)
                continue;
            end
            num_good = num_good + 1;
            right_point(roi_index(p)) = 1;
            right_point_2(cur_row_int, cur_col_int) = 1;
            % 中心点周围的局部平面作为jacobi
            for y = -half_patch_size : half_patch_size
               for x = -half_patch_size : half_patch_size
                   % 靠近相位突变的附近点不进行计算
                   if (abs(pre_phase_scales(row+y, col+x) - cur_phase_scales(cur_row_int+y, cur_col_int+x)) > pi)
                       continue; 
                   end
                   J_phase_pixel = zeros(2,1);
                   if (cur_row_int+y+1 > s_size(1) || cur_row_int+y-1 < 1)
                       J_phase_pixel(1) = cur_phase_scales(min(cur_row_int+y+1, s_size(1)), cur_col_int+x)...
                                          - cur_phase_scales(max(cur_row_int+y-1, 1), cur_col_int+x);
                   else
                       J_phase_pixel(1) = 0.5 * (cur_phase_scales(cur_row_int+y+1, cur_col_int+x) - ...
                                                 cur_phase_scales(cur_row_int+y-1, cur_col_int+x));
                   end
                   if (cur_col_int+x+1 > s_size(2) || cur_col_int+x-1 < 1)
                       J_phase_pixel(2) = cur_phase_scales(cur_row_int+y, min(cur_col_int+x+1, s_size(2)))...
                                          - cur_phase_scales(cur_row_int+y, max(cur_col_int+x-1, 1));
                   else
                       J_phase_pixel(2) = 0.5 * (cur_phase_scales(cur_row_int+y, cur_col_int+x+1) - ...
                                                 cur_phase_scales(cur_row_int+y, cur_col_int+x-1));
                   end
                   J_pixel_delta = [sin(orient_angle); cos(orient_angle)];
                   error = msf_get_pixel_value(pre_phase_scales, row, col) - ...
                           msf_get_pixel_value(cur_phase_scales, cur_row, cur_col);
                           
                   J = -1.0 * J_phase_pixel' * J_pixel_delta;
                   hessian = hessian + J * J';
                   bias = bias + (- error * J);
                   cost_temp = cost_temp + error * error; cost_copy(p) = cost_copy(p) + (error / J) * (error / J);
%                    cost_data = [cost_data, error / J];
               end
            end
            if (num_good > 0)
                H = H + hessian;
                b = b + bias;
            end
        end
%         figure;surf(right_point);figure;surf(right_point_2);
        cost = cost_temp / num_good;
        % 求解并update t_init
        update = (H+1e-17) \ b;
        t_optim = t_optim + update;
%         fprintf('iteration: %d t_optim now = %f, cost = %f\n',it,t_optim, cost);
        % cost变大则终止
        if (it > 1 && cost > last_cost)
            break;
        end
        if (abs(update) < 1e-5)
            break;
        end
        last_cost = cost;
        t_last = t_optim;
        
    end
    % 注意
    displacement_cost(1) = t_last;
    displacement_cost(2) = last_cost; % mean e^2
    displacement_cost(3) = sum(cost_copy(:)) / num_good;% var 
%     fprintf('cost_copy_sum: %f , num_good: %d\n', sum(cost_copy(:)), num_good);
    
end