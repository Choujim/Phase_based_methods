% ����line_bin_mask������ָ��б�ʻ�ñ���ͼƬ��ֱ�߿�
% ���ؾ෶Χ����ͼƬ���Լ���bin���ά��
% unit_lenth��pixel-perfect ��������
% query_lenth��pixel-perfect �Ĳ�ѯ��

function ret = line_bin_mask(dim_h, dim_w, line_k, line_width)
    % �߽��
%     min_h = 1;  max_h = dim_h;
%     min_w = 1;  max_w = dim_w;
    
    % ��kֵ����
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
    
    % ����[min_dis, max_dis] (Pixel Perfect)
    dense_ratio = 0.707;          % �ֲ��ؾಽ��̫ϡ��
    lenth = round(max_dis/dense_ratio);
    line_width = round(line_width/dense_ratio);
    ret = single(zeros(dim_h, dim_w, lenth));
    for i = 0 : lenth
        % ����ÿ��bin��mask
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