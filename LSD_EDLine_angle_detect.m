% LSD直线检测实现


% addpath('C:\Users\HIT\Desktop\Vibration From Video\LSD_angel_detect\LSD-OpenCV-MATLAB-master\matlab\x64');
addpath('C:\Users\HIT\Desktop\Vibration_From_Video\LSD-Matlab_fromOpencvToMEX');
addpath('C:\Users\HIT\Desktop\Vibration_From_Video\EDlines');
%% 
f = figure;
f.Position(3:4) = [900 400];
%%
% 调用LSD检测直线edge后求角度
line = animatedline('Color','b');
nF = 300;
xline = [1:1:nF];
angle = 0;
for i = 1:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % 无需转换成single,mex中是double
    subplot(1,2,1);
    figure(f);
    imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlab为按列展开，需要转置后传入lsd
    lines = mex_lsd(image');
    
    temp_angle = 0;
    for k = 1:size(lines, 2)
%         figure(f);
%         plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]);

        % imshow之后的plot坐标系自动对齐，所以直接使用
        figure(f);
        plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]); 
        
        % 求直线平均角度
        % 针对LSD返回的坐标无顺序
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

    % 画出角度图
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
% 调用LSD求取直线区域map后求角度
line = animatedline('Color','r');
nF = 300;
xline = [1:1:nF];
angle = 0;
for i = 2:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % 无需转换成single,mex中是double
%     subplot(1,2,1);
%     imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlab为按列展开，需要转置后传入lsd
    position = mex_lsd_2(image');
    lsd_map = zeros(size(im, 2), size(im, 1));
    lsd_map(position) = 255;
    lsd_map = lsd_map';
    subplot(1,2,1);
    imshow(lsd_map, 'InitialMagnification','fit');
    [lsd_map_y, lsd_map_x] = find(lsd_map~=0);
    % 对直线区域拟合直线
    p = polyfit(lsd_map_x, lsd_map_y, 1);
    temp_angle = atan2(-p(1), -1);
    

    % 画出角度图
    subplot(1,2,2);
    angle = temp_angle;
    angle = angle/pi*180 - 90;
    addpoints(line,xline(i),double(angle));    
    title(['frame:',num2str(i)]);
end

%%
% 调用EDline求取直线区域map后求角度
line = animatedline('Color','b');
nF = 300;
xline = [1:1:nF];
angle = 0;

for i = 1:nF
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = squeeze(mean(im,3)); % 无需转换成single,mex中是double
    subplot(1,2,1);
    figure(f);
    imshow(im, 'InitialMagnification','fit');hold on;
    
    % matlab为按列展开，需要转置后传入edlines
    lines = mex_edlines(image');
    
    temp_angle = 0;
    for k = 1:size(lines, 2)
%         figure(f);
%         plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]);

        % imshow之后的plot坐标系自动对齐，所以直接使用
        figure(f);subplot(1,2,1);
        plot(lines(1:2, k), lines(3:4, k), 'Color', [1, 0, 0]); hold on;
        
        % 求直线平均角度
        % 针对LSD返回的坐标无顺序
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

    % 画出角度图
    subplot(1,2,2);
    angle = temp_angle / size(lines, 2);
    angle = angle/pi*180 - 90;
    angle = round(angle);
    addpoints(line,xline(i),double(angle)); 
    figure(f);
    title(['frame:',num2str(i)]);
end