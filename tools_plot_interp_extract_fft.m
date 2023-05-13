

lenth = 10;
fs = lenth/pi;
n = [-lenth:1:lenth];
n = n.*pi./lenth;
% figure; plot(n);
y = sin(2*n)+0.5*sin(8*n);

% 0插值
% y_interp = zeros(1,2*numel(y));
% itp_index = [1:2:2*numel(y)];
% y_interp(itp_index) = y;

% % 线性插值
itp_index_2 = (y_interp == 0);
if (itp_index_2(1) == 1)
    y_interp(1) = y_interp(2)/2;
end
if (itp_index_2(2*numel(y)) == 1)
    y_interp(2*numel(y)) = y_interp(2*numel(y)-1)/2;
end

for i = 2 : 2*numel(y)-1
    if (itp_index_2(i) == 1)
       y_interp(i) = (y_interp(i-1)+y_interp(i+1))/2; 
    end
end
%     
        

% figure; plot(y);
% figure; plot(abs(fftshift(fft(y))));

% 未插值归一化频率作图
% ret = fft(y)/(2*lenth);
% % ret = ret(1:floor(lenth)+1);
% % ret(2 : end-1) = 2*ret(2 : end-1);
% 
% % % 真实频率
% freq = (0:fs/2/lenth:fs)-fs/2;
% % % 归一化频率
% % freq = (0:pi/lenth:2*pi)-pi;
% 
% figure; plot(freq, fftshift(abs(ret)));

% 插值后归一化频率作图
lenth = numel(y_interp)/2;
fs = lenth/pi;
% % 真实频率
freq = (0:fs/2/lenth:fs-fs/2/lenth)-fs/2;
% 归一化频率
% freq = (0:pi/lenth:2*pi-pi/lenth)-pi;

ret = fft(y_interp)/(2*lenth);
% freq = [0-fs/2/lenth:fs/2/lenth:2*fs]-fs;
% figure;plot(abs(ret));
ret = fftshift(abs(ret));
% figure; plot(freq(12:32), ret(12:32));
figure; plot(freq, ret);
