% 制作模拟数据视频

function ret = emulate_video(motion_cable, varargin)
    p = inputParser();
    
    addOptional(p, 'Image_size',[32,32], @isnumeric);
    addOptional(p, 'Video_rate',30,      @isnumeric);
    addOptional(p, 'Video_duraiton', 10, @isnumeric);
    addOptional(p, 'Width_cable',     4, @isnumeric);
    addOptional(p, 'Angle_cable',    65, @isnumeric);
    addOptional(p, 'View_distance',  50, @isnumeric);
    addOptional(p, 'Smooth_Sigma',  0.5, @isnumeric);
    addOptional(p, 'Discrete_rate', 512, @isnumeric);
    addOptional(p, 'Noise_yes',       0, @isnumeric);
    addOptional(p, 'Noise_mean',    0.0, @isnumeric);
    addOptional(p, 'Noise_var',    0.01, @isnumeric);
    addOptional(p, 'Background',      1, @isnumeric);
    addOptional(p, 'BackColor',     0.5, @isnumeric);
    
    parse(p, varargin{:});
    % 视频帧率
    video_rate = p.Results.Video_rate;
    % 视频时长 /s
    video_duration = p.Results.Video_duraiton;
    % 总帧数
    video_lenth = round(video_rate * video_duration);
    % 直线段角度
    oe = p.Results.Angle_cable;
    % 高斯核sigma
    sigma = p.Results.Smooth_Sigma;
    
    % 设置添加噪声的均值和方差
    noise_mean = p.Results.Noise_mean;
    noise_var =  p.Results.Noise_var;
    line_k = (oe-90)/abs(oe-90+eps) * tan(abs(oe-90)/180*pi);
    line_width = p.Results.Width_cable;
    dim_h = p.Results.Image_size(1);
    dim_w = p.Results.Image_size(2);
    
    ret = single(zeros(dim_h, dim_w, video_lenth));
    
    ctr = ceil((p.Results.Image_size+0.5)/2);
    if (line_k > -1) && (line_k < 1)
        line_k_copy = line_k;
        unit_lenth = dim_w;
        query_lenth = dim_h;
        start_b = ctr(2) - line_k * ctr(1);
        theta = atan(line_k);
    elseif (line_k <= -1) || (line_k >= 1)
        line_k_copy = line_k;
        line_k = 1 / line_k;
        unit_lenth = dim_h;
        query_lenth = dim_w;
        start_b = ctr(1) - (line_k) * ctr(2);
        theta = atan(line_k);
    end
        

    % 亚像素坐标需要指定离散精度，delta = 1 pixel/discrete_rate
    discrete_rate = p.Results.Discrete_rate;
    conv_lenth = discrete_rate * round(line_width / cos(theta));
    discrete_section = [0 : conv_lenth]./discrete_rate;
    % 高斯核
    filter_gaussian = @(query_x) (exp(-(query_x).^2 / (2*sigma*sigma)));
    radius_filter = round(3 * sigma) + 1;
    sum_gaussian = 2 * sum(filter_gaussian([0 : discrete_rate*radius_filter]./discrete_rate)) - filter_gaussian(0);
    % 根据 motion_cable 制作图片流
    for i = 1:video_lenth
        next_b = start_b + motion_cable(2, i) / cos(theta);
        for unit = 1 : unit_lenth
            y_base = unit * line_k + next_b;
%             query_section_lenth = ceil(y_base + line_width/cos(theta)) - floor(y_base);
%             for width = 0 : query_section_lenth
            for width = 1 : query_lenth
                y_query = width;
%                 y_query = floor(y_base) + width;
                if (y_query >= 1) && (y_query <= query_lenth)
                    y_section  = discrete_section + y_base - y_query;
                    discrete_val = filter_gaussian(y_section);
                    discrete_val = sum(discrete_val) / sum_gaussian;
                    if abs(line_k_copy) < 1
                        ret(y_query, unit, i) = discrete_val;
                    elseif abs(line_k_copy) >= 1
                        ret(unit, y_query, i) = discrete_val;
                    end
                end
            end
        end
        % 加上高斯滤波平滑边缘
        ret(:,:,i) = imgaussfilt(ret(:,:,i), sigma);
        % 加上灰度背景
        if (p.Results.Background)
            back_color = p.Results.BackColor;
            ret(:,:,i) = (ret(:,:,i) + back_color)./(1+back_color);
        end
        
        if (p.Results.Noise_yes)
            % 加上高斯随机噪声
            ret(:,:,i) = imnoise(ret(:,:,i),'gaussian',noise_mean,noise_var);
        end
    end
    
    
end