%%
f1 = figure;
f2 = figure;
%%
line = animatedline('Color','r');
xline = [1:1:nF];
disp = 0;
ANIME(411) = struct('cdata',[],'colormap',[]);
% mask = mask_creator([26,26],2,4,[1,1],0);
for i = 2:411
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     subplot(1,2,1);
    figure(f1), hold on
    imshow(im,'InitialMagnification','fit');
                                                             % 最优角度相位
%     t_phase = mod(pi+anime3(:,i)-anime3(:,i-1), 2*pi)-pi;    % 相邻帧法
%     t_mask = anime3_mask(:,i)|anime3_mask(:,i-1);
%     tdisp = mean(t_phase(:)); 
%     disp = disp+tdisp; signalout(4,i) = disp;
%     disp = mean(mod(pi+anime3(:,i)-anime3(:,1), 2*pi)-pi);   % 参考帧法
%     addpoints(line,xline(i),double(disp)); 

%     addpoints(line,xline(i),atan(temp_energy2./temp_energy)/pi*180);
    
%     subplot(1,2,2);
%     energy_angle = detect_angle(im, 2, 4, 0); angle_out(1,i-1) = energy_angle;% 方向能量法
%     energy_angle = detect_angle(im, 2, 4, 1); energy_angle = energy_angle/pi*180;  angle_out(2,i-1) = energy_angle;% 频谱加权法
    energy_angle = detect_angle(im, 2, 4, 2); angle_out(3,i-1) = energy_angle;% 方向能量增强的Hough法检测直线角度
%     energy_angle = detect_angle(im, 2, 4, 3); angle_out(4,i-1) = energy_angle;% 频谱非极大值抑制
%     energy_angle = detect_angle(im, 2, 4, 4); angle_out(5,i-1) = energy_angle;% canny边缘检测+霍夫法
    addpoints(line,xline(i),energy_angle);

%     angle_orient(i,1) = atan(double(sum(sum(abs(anime1(:,:,1,1,i)).^2)) ./ sum(sum(abs(anime1(:,:,1,3,i)).^2))));  
%     angle_orient(i,2) = atan(double(sum(sum(abs(anime1(:,:,1,2,i)).^2)) ./ sum(sum(abs(anime1(:,:,1,4,i)).^2))));
%     addpoints(line,xline(i),angle_orient(i,1));
    
%     re_mask = ((-1i)^3) .* mask .* (fftshift(fft2(im2single(squeeze(mean(im,3))))));
%     re = ifft2(ifftshift(re_mask)); 
%     addpoints(line,xline(i),2.*double(sum(sum(abs(re).^2))));
    
%     addpoints(line,xline(i),double(sum(sum(abs(anime1(:,:,1,1,i)).^2)))...
%                            +double(sum(sum(abs(anime1(:,:,1,2,i)).^2)))...
%                            +double(sum(sum(abs(anime1(:,:,1,3,i)).^2)))...
%                            +double(sum(sum(abs(anime1(:,:,1,4,i)).^2))));
    
%     addpoints(line,xline(i),double(sum(sum(abs(im2single(squeeze(mean(im,3)))).^2)))); %总能量
%     addpoints(line,xline(i),double(sum(sum(p_anime1(:,:,1,2,i))))/26/19); % 原始相位差策略
%     addpoints(line,xline(i),2*double(sum(sum(angle(anime1(:,:,1,3,i)))))/26/19);
    
%     orient = 1;    % 多尺度平均相邻帧相位差法
%     tdisp = (sum(sum(mod(pi+angle(anime1(:,:,1,orient,i))-angle(anime1(:,:,1,orient,i-1)), 2*pi)-pi))/26/26 ...
%            +sum(sum(mod(pi+angle(anime2(:,:,2,orient,i))-angle(anime2(:,:,2,orient,i-1)), 2*pi)-pi))/13/13)/2;
%     tdisp = sum(sum(mod(pi+angle(anime1(:,:,1,1,i))-angle(anime1(:,:,1,1,i-1)), 2*pi)-pi))/26/26;
%     tdisp = sum(sum(angle(anime1(:,:,1,2,i))-angle(anime1(:,:,1,2,i-1))))/26/19;
%     tdisp = sum(sum(angle(anime1(:,:,1,2,i-1))))/26/19;
%     disp = disp+tdisp; signalout(1,i) = disp;

    figure(f2)%surf(abs(fftshift(fft2(im))));
%      surf(im);view(15.7,59.6);

%     subplot(1,2,2);imagesc(reshape(anime3(1 : prod(optimal_index(1,:)),i),optimal_index(1,1),optimal_index(1,2))); view(0,90);
%     subplot(1,2,2);surf(reshape(anime3(1 : prod(optimal_index(1,:)),i),optimal_index(1,1),optimal_index(1,2))); view(-165,56);%view(-122,68);
%     reshape_image = reshape(anime3(1 : prod(optimal_index(1,:)),i),optimal_index(1,1),optimal_index(1,2))...
%          - reshape(anime3(1 : prod(optimal_index(1,:)),i-1),optimal_index(1,1),optimal_index(1,2));
%     subplot(1,2,2);surf(mod(reshape_image+pi, 2*pi)-pi);zlim([-1,1]);view(30,20);%view(-122,68);
%      
%     surf(reshape(anime3(1 : prod(optimal_index(1,:)),i),optimal_index(1,1),optimal_index(1,2))...
%          - reshape(anime3(1 : prod(optimal_index(1,:)),i-1),optimal_index(1,1),optimal_index(1,2)));zlim([-9,9]);view(36.4,17.6);
%     surf(reshape(anime3(prod(optimal_index(1,:))+1 : end ,i),optimal_index(2,1),optimal_index(2,2)));view(32.5,63.6);
%     imhist(im); ylim([0, 80]);
%     addpoints(line,xline(i),double(disp));
%     drawnow
    title(['frame:',num2str(i)]);
%     colorbar;
%     view(0,-90);
%     ANIME(i)= getframe(f1);
end