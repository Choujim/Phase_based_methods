function result = de_background(image, downsample_depth)
    temp = image;
    %% bilinear模糊
    for i = 1:downsample_depth
        temp = imresize(temp, 0.5, 'bilinear');
        temp = imresize(temp, 2, 'bilinear');
    end   
    %% bilinear重建
%     for i = 1:downsample_depth
%         temp = imresize(temp, 2, 'bilinear');
%     end
%     
    %% 返回差分图
    if downsample_depth > 0
        result = image - temp;
    else
        result = image;
    end
%     result = result + rot90(result, -2);
    

end