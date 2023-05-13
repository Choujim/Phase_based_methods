% 用于按照双线性插值计算小数坐标
function value = msf_get_pixel_value(image, row, col)
    img_size = size(image);
    % min max 功能较多，速度较慢
%     row = min(max(1, row), img_size(1));
%     col = min(max(1, col), img_size(2));
    if (row > 1)
        if (row > img_size(1))
            row = img_size(1);
        end
    else
        row = 1;
    end
    if (col > 1)
        if (col > img_size(2))
            col = img_size(2);
        end
    else
        col = 1;
    end
        
    
    row_int = floor(row);
    col_int = floor(col);
    yy = row - row_int;
    xx = col - col_int;
    value = (1 - xx) * (1 - yy) * image(row_int, col_int) + ...
             xx * (1 - yy) * image(row_int, col_int + 1) + ...
            (1 - xx) * yy * image(row_int + 1, col_int) + ...
            xx * yy * image(row_int + 1, col_int + 1);
end