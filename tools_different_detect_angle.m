function angle = tools_different_detect_angle(image, method)
    % 1 -- canny��Ե���+����
    if (method == 1)
        angle = detect_angle(image, 2, 4, 4);
        
    % 2 -- FLD
    elseif (method == 2)
    
    % 3 -- LSD
    elseif (method == 3)
        addpath('C:\Users\HIT\Desktop\Vibration_From_Video\LSD-Matlab_fromOpencvToMEX');
        lines = mex_lsd(image');
        temp_angle = 0;
        for k = 1:size(lines, 2)
            % imshow֮���plot����ϵ�Զ����룬����ֱ��ʹ��
%             figure(f);
%             plot(lines(1:2, k), lines(3:4, k), 'LineWidth', lines(5, k) / 2, 'Color', [1, 0, 0]); 

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
%             xlim([0,26]);
%             ylim([0,26]);
%             hold on;
        end
        angle = temp_angle / size(lines, 2);
        angle = angle/pi*180 - 90;
        angle = round(angle);
        
    % 4 -- EDlines
    elseif (method == 4)
        addpath('C:\Users\HIT\Desktop\Vibration_From_Video\EDlines');
        lines = mex_edlines(image');
        temp_angle = 0;
%         size(lines, 2)
        for k = 1:size(lines, 2)
            % imshow֮���plot����ϵ�Զ����룬����ֱ��ʹ��
%             figure(f);
%             plot(lines(1:2, k), lines(3:4, k), 'Color', [1, 0, 0]); 

            % ��ֱ��ƽ���Ƕ�
            % ��Է��ص�������˳��
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
%             xlim([0,26]);
%             ylim([0,26]);
%             hold on;
        end
        angle = temp_angle / size(lines, 2);
        angle = angle/pi*180 - 90;
        angle = round(angle);
        
    % 5 -- ����������ǿ��Hough�����ֱ�߽Ƕ�
    elseif (method == 5)
        angle = detect_angle(image, 2, 4, 2);
        
%         addpath('C:\Users\HIT\Desktop\Vibration_From_Video\detect_angle_MEX');
%         tmp_angle = mex_detect_angle(image');
%         angle = tmp_angle;
    end

end