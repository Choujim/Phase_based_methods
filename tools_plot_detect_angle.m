% ���Ʒ����������ͼ
f = figure;
f.Position(3:4) = [1200 800];
for i = 1:4
    subplot(4,1,i)
    y_data_index = [3,1,4,5];
    x_label = {'(a)','(b)','(c)','(d)'};
    y_data = angle_out(y_data_index(i),:);
    plot(y_data,'b');

%     title('');
    grid on;                         % ��ʾ������
%     legend('Method', 'Location','northwest');
%     legend('boxoff');
    
%     legend()                       % ͼ��ע��
    xlabel(x_label(i));                 % x�������ǩ
%     ylabel('Normal angle (Deg)');    % y�������ǩ
    xlim([1,420]);
    ylim([60,70]);
%     ylim([10,12])
    
%     xticks(1:100:400);
    
end
% xlabel({'''Time (frame)', 'position', [210, 57]);
ylabel('Normal angle (deg)','position', [-16,91]);
title({'Time (frame)'},'position', [211.4,49.5]);

% ylabel('Normal angle (deg)','position', [-15,16.5]);
% title({'Time (frame)'},'position', [211.4,7.948]);