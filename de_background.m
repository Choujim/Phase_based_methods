function result = de_background(image, downsample_depth)
    temp = image;
    %% bilinearģ��
    for i = 1:downsample_depth
        temp = imresize(temp, 0.5, 'bilinear');
        temp = imresize(temp, 2, 'bilinear');
    end   
    %% bilinear�ؽ�
%     for i = 1:downsample_depth
%         temp = imresize(temp, 2, 'bilinear');
%     end
%     
    %% ���ز��ͼ
    if downsample_depth > 0
        result = image - temp;
    else
        result = image;
    end
%     result = result + rot90(result, -2);
    

end