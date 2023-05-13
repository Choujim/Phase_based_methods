% LSDֱ�߼��ʵ��


% addpath('C:\Users\HIT\Desktop\Vibration From Video\LSD_angel_detect\LSD-OpenCV-MATLAB-master\matlab\x64');
addpath('C:\Users\HIT\Desktop\Vibration_From_Video\LSD-Matlab_fromOpencvToMEX');
addpath('C:\Users\HIT\Desktop\Vibration_From_Video\EDlines');
%% 
f = figure;
f.Position(3:4) = [900 400];
%%
% ����LSD���ֱ��edge����Ƕ�
line = animatedline('Color','b');
nF = 300;
xline = [1:1:nF];
angle = 0;
for i = 1:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % ����ת����single,mex����double
    subplot(1,2,1);
    figure(f);
    imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlabΪ����չ������Ҫת�ú���lsd
    lines = mex_lsd(image');
    
    temp_angle = 0;
    for k = 1:size(lines, 2)
%         figure(f);
%         plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]);

        % imshow֮���plot����ϵ�Զ����룬����ֱ��ʹ��
        figure(f);
        plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]); 
        
        % ��ֱ��ƽ���Ƕ�
        % ���LSD���ص�������˳��
        if (lines(3, k) >= lines(4, k))
            max_y = lines(3, k);
            max_x = lines(1, k);
            min_y = lines(4, k);
            min_x = lines(2, k);
        else
            max_y = lines(4, k);
            max_x = lines(2, k);
            min_y = lines(3, k);
            min_x = lines(1, k);
        end
        temp_angle = temp_angle+atan2(max_y-min_y, max_x-min_x);
%         temp_angle = temp_angle+atan2((lines(1, k)-lines(2, k)), (lines(3, k)-lines(4, k)));
        xlim([0,26]);
        ylim([0,26]);
        hold on;
    end
%     hold off;

    % �����Ƕ�ͼ
    subplot(1,2,2);
    angle = temp_angle / size(lines, 2);
    angle = angle/pi*180 - 90;
    angle = round(angle);
    addpoints(line,xline(i),double(angle)); 
    figure(f);
    title(['frame:',num2str(i)]);
end
% imwrite(vframein, 'C:\Users\HIT\Desktop\Vibration_From_Video\LSD_angel_detect\LSD-OpenCV-MATLAB-master\matlab\x64\images\test3.jpg');
%%
% ����LSD��ȡֱ������map����Ƕ�
line = animatedline('Color','r');
nF = 300;
xline = [1:1:nF];
angle = 0;
for i = 2:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % ����ת����single,mex����double
%     subplot(1,2,1);
%     imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlabΪ����չ������Ҫת�ú���lsd
    position = mex_lsd_2(image');
    lsd_map = zeros(size(im, 2), size(im, 1));
    lsd_map(position) = 255;
    lsd_map = lsd_map';
    subplot(1,2,1);
    imshow(lsd_map, 'InitialMagnification','fit');
    [lsd_map_y, lsd_map_x] = find(lsd_map~=0);
    % ��ֱ���������ֱ��
    p = polyfit(lsd_map_x, lsd_map_y, 1);
    temp_angle = atan2(-p(1), -1);
    

    % �����Ƕ�ͼ
    subplot(1,2,2);
    angle = temp_angle;
    angle = angle/pi*180 - 90;
    addpoints(line,xline(i),double(angle));    
    title(['frame:',num2str(i)]);
end

%%
% ����EDline��ȡֱ������map����Ƕ�
line = animatedline('Color','b');
nF = 300;
xline = [1:1:nF];
angle = 0;

for i = 1:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % ����ת����single,mex����double
    subplot(1,2,1);
    figure(f);
    imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlabΪ����չ������Ҫת�ú���edlines
    lines = mex_edlines(image');
    
    temp_angle = 0;
    for k = 1:size(lines, 2)
%         figure(f);
%         plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]);

        % imshow֮���plot����ϵ�Զ����룬����ֱ��ʹ��
        figure(f);subplot(1,2,1);
        plot(lines(1:2, k), lines(3:4, k), 'Color', [1, 0, 0]); hold on;
        
        % ��ֱ��ƽ���Ƕ�
        % ���LSD���ص�������˳��
        if (lines(3, k) >= lines(4, k))
            max_y = lines(3, k);
            max_x = lines(1, k);
            min_y = lines(4, k);
            min_x = lines(2, k);
        else
            max_y = lines(4, k);
            max_x = lines(2, k);
            min_y = lines(3, k);
            min_x = lines(1, k);
        end
        temp_angle = temp_angle+atan2(max_y-min_y, max_x-min_x);
%         temp_angle = temp_angle+atan2((lines(1, k)-lines(2, k)), (lines(3, k)-lines(4, k)));
        xlim([0,26]);
        ylim([0,26]);
        hold on;
    end
%     hold off;

    % �����Ƕ�ͼ
    subplot(1,2,2);
    angle = temp_angle / size(lines, 2);
    angle = angle/pi*180 - 90;
    angle = round(angle);
    addpoints(line,xline(i),double(angle)); 
    figure(f);
    title(['frame:',num2str(i)]);
end