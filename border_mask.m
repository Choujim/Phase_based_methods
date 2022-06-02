% 为图像创建衰减边缘mask

function mask = border_mask(image_size, border_width, method)

%%
if (border_width < 1 || border_width > max(image_size(:)))
    error(sprintf('Cannot build border with %d pixels.',border_width));
end

mask = zeros(image_size);
mask((1+border_width) : (image_size(1)-border_width), (1+border_width) : (image_size(2)-border_width)) = 1;

%% 椭圆边界--余弦下降
if (method == 1)
    % 生成极坐标直径图
    dims = image_size;
    ctr = ceil((dims+0.5)/2);
    [xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
        ([1:dims(1)]-ctr(1))./(dims(1)/2) );
    rad = sqrt(xramp.^2 + yramp.^2);
    rad_in_x = min(((1+border_width-ctr(2))./(dims(2)/2)).^2 , ((dims(2)-border_width-ctr(2))./(dims(2)/2)).^2);
    rad_in_x = sqrt(rad_in_x);
    rad_in_y = min(((1+border_width-ctr(1))./(dims(1)/2)).^2 , ((dims(1)-border_width-ctr(1))./(dims(1)/2)).^2);
    rad_in_y = sqrt(rad_in_y);
    rad_in = min(rad_in_x, rad_in_y);
    
    
    % 余弦函数下降
    twidth = 1 - rad_in;
    [Xrcos,Yrcos] = rcosFn(twidth,(twidth/2),[1,0]);
    Yrcos = sqrt(Yrcos);
    rad = rad - rad_in;
    rad_mask = pointOp(rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0); 
    mask = rad_mask;


%% 方形边界--余弦下降
elseif (method == 2)
    dims = image_size;
    [Xrcos,Yrcos] = rcosFn(border_width-1,((border_width-1)/2),[0,1]);
    Xrcos = Xrcos + 1;
    Yrcos = sqrt(Yrcos);
    temp = (1:1:border_width);
    border = interp1(Xrcos, Yrcos, temp(:), 'linear', 'extrap');
    for i = 1:border_width
        mask(i, i:dims(2)+1-i) = border(i);
        mask(i:dims(1)+1-i, i) = border(i);
        mask(dims(1)+1-i, i:dims(2)+1-i) = border(i);
        mask(i:dims(1)+1-i, dims(2)+1-i) = border(i);
%         mask(i, (1+border_width) : (dims(2)-border_width)) = border(i);
%         mask((1+border_width) : (dims(1)-border_width), i) = border(i);
%         mask(dims(1)+1-i, (1+border_width) : (dims(2)-border_width)) = border(i);
%         mask((1+border_width) : (dims(1)-border_width), dims(2)+1-i) = border(i);
    end
        
end
end
% mask = rad_mask;
