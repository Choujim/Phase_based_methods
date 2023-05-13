%获得基频幅值
function result = base_freq_amp(signal, samplerate)

sigfft = fft(signal);
L = max(size(signal));
sf2 = abs(sigfft/L);
sf1 = L*sf2(1 : floor(L/2)+1);
sf1(2 : end-1) = 2*sf1(2 : end-1);
L2time = (1:L)/samplerate;
L2freq = samplerate*(0:floor(L/2))/L;
psd = (1/(samplerate*L)) * abs(sigfft(1 : floor(L/2)+1)).^2;
spsd = psd;
spsd(2 : end-1) = 2*spsd(2 : end-1);
% figure;plot(L2freq,spsd);
% title('Single PSD');
% xlabel('f (Hz)');
% ylabel('Power/Hz (W/Hz)');

% 指定输出基频频率处的幅值
result = spsd(10);
%% 
% figure;plot(L2time,signal);
% title('Raw Signal');
% xlabel('t (seconds)');
% ylabel('X(t)');
% figure;plot(L2freq,sf1);
% title('signal spectrum')
% xlabel('f (Hz)');
% ylabel('|P1(f)|');

end
