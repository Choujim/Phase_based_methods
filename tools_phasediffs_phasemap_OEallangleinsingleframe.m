f = figure;
f.Position(3:4) = [1200 400];

%%
f = figure;

%%
line = animatedline('Color','r');
xline = [1:1:nF];
disp = 0;
oe = 65;
for i = 2:20
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = im2single(squeeze(mean(im,3)));
    
    % 尺度1相位差输出
    mask = mask_creator(size(im), 2, 4, [1, oe/180*pi], 1);
    im_dft = mask .* fftshift(fft2(image));
    temp_re = ifft2(ifftshift(im_dft));
    temp_phase = angle(temp_re);
    
    vframein = vHandle.read(i-1);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = im2single(squeeze(mean(im,3)));
    im_dft = mask .* fftshift(fft2(image));
    temp_re_2 = ifft2(ifftshift(im_dft));
    temp_phase_2 = angle(temp_re_2);
    
    delta_phase = temp_phase - temp_phase_2;
    
    surf(delta_phase);
    
%     t_phase = mod(pi+delta_phase(:), 2*pi)-pi;
%     tdisp = mean(t_phase);
%     disp = disp+tdisp;signalout(4,i) = disp;
%     addpoints(line,xline(i),double(disp));
    figure(f);
%     view(-115.5,18.8);
    
    % 尺度2相位差输出
%     mask = mask_creator(size(im), 2, 4, [2, oe/180*pi], 1);
%     dims = size(im);
%     ctr = ceil((dims+0.5)/2);
%     lodims = ceil((dims-0.5)/2);
%     loctr = ceil((lodims+0.5)/2);
%     lostart = ctr-loctr+1;
%     loend = lostart+lodims-1;
%     
%     im_dft = fftshift(fft2(image));
%     im_dft = im_dft(lostart(1):loend(1),lostart(2):loend(2));
%     im_dft = mask .* im_dft;
%     temp_re = ifft2(ifftshift(im_dft));
%     temp_phase = angle(temp_re);
%     
%           % 下一帧
%     vframein = vHandle.read(i-1);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     image = im2single(squeeze(mean(im,3)));
%     im_dft = fftshift(fft2(image));
%     im_dft = im_dft(lostart(1):loend(1),lostart(2):loend(2));
%     im_dft = mask .* im_dft;
%     temp_re_2 = ifft2(ifftshift(im_dft));
%     temp_phase_2 = angle(temp_re_2);
%     
%     delta_phase = temp_phase - temp_phase_2;
%     t_phase = mod(pi+delta_phase(:), 2*pi)-pi;
%     tdisp = 4/3*mean(t_phase);
%     disp = disp+tdisp;
%     addpoints(line,xline(i),double(disp));
%     figure(f);
%     
%     
    title(['frame:',num2str(i)]);
    
    
    
    
    
    % 尺度1
%     mask = mask_creator(size(im), 2, 4, [1, 65/180*pi], 1);
%     im_dft = mask .* fftshift(fft2(image));
%     temp_re = ifft2(ifftshift(im_dft));
%     temp_phase = angle(temp_re);
% %     
% %     % 尺度2
% %     mask = mask_creator(size(im), 2, 4, [2, 65/180*pi], 1);
% %     dims = size(im);
% %     ctr = ceil((dims+0.5)/2);
% %     lodims = ceil((dims-0.5)/2);
% %     loctr = ceil((lodims+0.5)/2);
% %     lostart = ctr-loctr+1;
% %     loend = lostart+lodims-1;
% %     
% %     im_dft = fftshift(fft2(image));
% % %     im_dft = im_dft(lostart(1):loend(1),lostart(2):loend(2));
% % %     image = imresize(image, 0.5, 'bilinear');
% %     temp_re = ifft2(ifftshift(mask .* im_dft));
% %     temp_phase = angle(temp_re);
% %     
% %     %
%     figure(f); 
%     subplot(1,2,1); surf(image);
%     view(59.3, 67.6);
%     
%     subplot(1,2,2);surf(temp_phase);
% %     view(-118.8,75.6);
%     view(63.6,70.8);
%     title(['frame:',num2str(i)]);

    
    








end
%%    











%     f3 = figure;
%     for i = 2:411
%     vframein = vHandle.read(i);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% %     figure; imshow(vframein,'InitialMagnification','fit')
% %     im = de_background(im, 2);
% %     figure; surf(im);
% 
% %     image = im2single(squeeze(mean(im,3)));
% %     edge_im = edge(image, 'canny');
% %     figure(f3); imshow(edge_im,'InitialMagnification','fit');
% %     title(['frame:',num2str(i)]);
%     
%     end
%     im_dft = fftshift(fft2(im));
%     dims = size(im);
%     ctr = ceil((dims+0.5)/2);
%     im_dft(ctr(1),ctr(2)) = 0;
%     figure; surf(abs(im_dft));
    
%     figure; surf(abs(fftshift(fft2(im))));
    
%     mask = mask_creator(size(im), 2, 4, [2,90/180*pi], 1);
%     figure; surf(mask);
    
%     i = 10;
%     vframein = vHandle.read(i);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     im = de_background(im, 2);
%     image = im2single(squeeze(mean(im,3)));
%     energy_angle = [];
%     scale_index = 1;
%     for j = 1:180
%         energy_angle(j) = f_energy(image, 2, 4, [scale_index, j/180*pi], 1);
%         
%     end
%     figure; plot(energy_angle);
    

%     %%
%     [optimal_phase, optimal_index, optimal_mask, optimal_fft] = ...
%         calculate_phase(im, nscalesin-1, norientationsin, 12/180*pi);
%     figure; surf(real(optimal_fft));
%     figure; surf(imag(optimal_fft));
%     figure; surf(angle(optimal_fft));
%     figure; surf(abs(optimal_fft));
%     
% %     figure; surf(imag(log(optimal_fft)));
%     
%     % 相位差
%     vframein = vHandle.read(i+1);
%     im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     [next_optimal_phase, next_optimal_index, next_optimal_mask, next_optimal_fft] = ...
%         calculate_phase(im, nscalesin-1, norientationsin, 12/180*pi);   
%     figure; surf(angle(next_optimal_fft));
%     figure; surf(angle(next_optimal_fft)-angle(optimal_fft));
%     delta_phase_map = mod(pi+(angle(next_optimal_fft)-angle(optimal_fft)),2*pi)-pi;
%     figure; surf(mod(pi+(angle(next_optimal_fft)-angle(optimal_fft)),2*pi)-pi);
% %     delta_phase_map = atan2(sin(imag(log(optimal_fft))-imag(log(next_optimal_fft))), cos(imag(log(optimal_fft))-imag(log(next_optimal_fft))));
% %     delta_phase_map = atan2(sin(imag(log(next_optimal_fft))-imag(log(optimal_fft))), cos(imag(log(next_optimal_fft))-imag(log(optimal_fft))));
%     figure; surf(delta_phase_map);
    
    
    