% 制作line_bin_mask，根据指定斜率获得遍历图片的直线库
% 按截距范围遍历图片可以减少bin库的维度
% unit_lenth：pixel-perfect 的自增轴
% query_lenth：pixel-perfect 的查询轴

function ret = line_bin_mask(dim_h, dim_w, line_k, line_width)
    % 边界点
%     min_h = 1;  max_h = dim_h;
%     min_w = 1;  max_w = dim_w;
    
    % 对k值分类
    %% k < 0
    if (line_k < 0) && (line_k > -1)
        line_k_copy = line_k;
        theta = atan(line_k);
%         max_dis = (dim_h - (dim_w * line_k)) * cos(theta);
        max_dis = dim_h * cos(theta) - dim_w * sin(theta);
        unit_lenth = dim_w;
        query_lenth = dim_h;
        start_b = 0;
    elseif (line_k < 0) && (line_k <= -1)
        line_k_copy = line_k;
        line_k = 1 / line_k;
        theta = atan(line_k);
        max_dis = dim_h * cos(theta) - dim_w * sin(theta);
        unit_lenth = dim_h;
        query_lenth = dim_w;
        start_b = 0;
    elseif (line_k >= 0) && (line_k < 1)
        line_k_copy = line_k;
        theta = atan(line_k);
        max_dis = dim_h * cos(theta) + dim_w * sin(theta);
        unit_lenth = dim_w;
        query_lenth = dim_h;
        start_b = (-query_lenth) * line_k;
    elseif (line_k >= 0) && (line_k >= 1)
        line_k_copy = line_k;
        line_k = 1 / line_k;
        theta = atan(line_k);
        max_dis = dim_h * cos(theta) + dim_w * sin(theta);
        unit_lenth = dim_h;
        query_lenth = dim_w;
        start_b = (-query_lenth) * line_k;   
    end
    
    % 遍历[min_dis, max_dis] (Pixel Perfect)
    dense_ratio = 0.707;          % 弥补截距步距太稀疏
    lenth = round(max_dis/dense_ratio);
    line_width = round(line_width/dense_ratio);
    ret = single(zeros(dim_h, dim_w, lenth));
    for i = 0 : lenth
        % 生成每个bin的mask
        line_b = start_b + i*dense_ratio/cos(theta);
        for unit = 1 : unit_lenth
            for width = 1 : line_width
                y_query = unit * line_k + line_b + (width-1)*dense_ratio/cos(theta);
                if (y_query >= 1) && (y_query <= query_lenth)
                    if abs(line_k_copy) < 1
                        ret(round(y_query), unit, i+1) = 1;
                    elseif abs(line_k_copy) >= 1
                        ret(unit, round(y_query), i+1) = 1;
                    end
                    
                end
            end
        end
    end

end