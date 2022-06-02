function result = fftspec(signal, samplerate)

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
figure;plot(L2freq,spsd);
title('Single PSD');
xlabel('f (Hz)');
ylabel('Power/Hz (W/Hz)');

% result = spsd(10);
%% 
figure;plot(L2time,signal);
title('Raw Signal');
xlabel('t (seconds)');
ylabel('X(t)');
% figure;plot(L2freq,sf1);
% title('signal spectrum')
% xlabel('f (Hz)');
% ylabel('|P1(f)|');

end
