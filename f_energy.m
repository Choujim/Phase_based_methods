%% 频谱法计算方向能量

function energy = f_energy(image, scale_total, orientation_total, band_index, method)
    temp = fftshift(fft2(image));
    mask = mask_creator(size(image), scale_total, orientation_total, band_index, method);
    
    %% 缩放事项
    for i = 2: band_index(1)
        dims = size(temp);
        ctr = ceil((dims+0.5)/2);
        lodims = ceil((dims-0.5)/2);
        loctr = ceil((lodims+0.5)/2);
        lostart = ctr-loctr+1;
        loend = lostart+lodims-1;
        temp = temp(lostart(1):loend(1),lostart(2):loend(2));
    end
    
    temp = mask .* temp;
    energy = sum(abs(temp(:)) .^2 );
end

