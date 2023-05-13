% ��÷���Ƶ��ͼ���ܺͳ��� ָ��threshold �ĸ������������

function [index, threshold_index] = sort_and_threshold(fft_image, threshold)
    temp_sort = zeros(numel(fft_image),1);
    [val, index] = sort(fft_image(:));
    temp_sort(1) = fft_image(index(end));
    
    for i = 2:numel(fft_image)
        temp_sort(i) = temp_sort(i-1)+fft_image(index(end-i+1));
    end
    
    j = 1;
    while (temp_sort(j) <= threshold * temp_sort(end))
        j=j+1;
    end
%     disp(j);  % ���ѡȡ�ĵ���
    threshold_index = j-1;



end
