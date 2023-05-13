% 制作line_bin_mask，根据指定斜率获得遍历图片的直线库
function ret = line_bin_mask(dim_h, dim_w, line_k, line_width)  
    delta_x = 1;
    if line_k < 0
        start_y = 1;                          start_x = 1;
        end_y = dim_h - round(dim_w*line_k);  end_x = 1;
        lenth = size(start_y : end_y);
        ret = single(zeros(dim_h, dim_w, lenth(2)));
        
%         ret(start_y, start_x, 1) = 1;
        for cur_y = start_y : end_y
            cur_y_copy = cur_y;
            for width = 1 : line_width
                next_y = cur_y_copy + (width-1);
                next_x = start_x;
   
                while (round(next_y) >= 1) && (next_x <= dim_w)
                    if (round(next_y) <= dim_h)
                        ret(round(next_y), next_x, cur_y) = 1;
                    end
                    next_x = next_x + delta_x;
                    next_y = next_y + line_k * delta_x;  
                end
            end
        end
    elseif line_k > 0
        start_y = 1 - round(dim_w*line_k);     start_x = 1;
        end_y = dim_h;                         end_x = 1;
        lenth = size(start_y : line_width : end_y);
        ret = single(zeros(dim_h, dim_w, lenth(2)));
        
%         ret(end_y, end_x, lenth(2)-1) = 1;
        for cur_y = start_y : end_y
            cur_y_copy = cur_y;
            for width = 1 : line_width
                next_y = cur_y_copy + (width-1);
                next_x = start_x;
   
                while (round(next_y) <= dim_h) && (next_x <= dim_w)
                    if (round(next_y) >= 1)
                        ret(round(next_y), next_x, (cur_y-start_y)) = 1;
                    end
                    next_x = next_x + delta_x;
                    next_y = next_y + line_k * delta_x;  
                end
            end
        end
            
    end

%     next_y = y0 + line_k * delta_x;
%     next_x = x0;
%     while (round(next_y) >= 1) && (next_y < dim_h) && (next_x < dim_w)
%         next_x = next_x + delta_x;
%         ret(round(next_y), next_x) = 1;
%         next_y = next_y + line_k * delta_x;        
%     end
    
end