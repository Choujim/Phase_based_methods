%%Visual Microphone: this is the function that extracts sound from video
%%Based on "The Visual Microphone: Passive Recovery of Sound from Video"
%%by Abe Davis, Michael Rubinstein, Neal Wadhwa, Gautham J. Mysore,
%%Fredo Durand, and William T. Freeman
%%from SIGGRAPH 2014

%%This code written by Abe Davis

%%(c) Myers Abraham Davis (Abe Davis), MIT 
%%Note from Abe: It's probably worth mentioning that MIT has a patent
%%pending on this work.

%%
function [S] = vmSoundFromVideo(vHandle, nscalesin, norientationsin, varargin)
%   Extracts audio from tiny vibrations in video.
%   Optional argument DownsampleFactor lets you specify some factor to
%   downsample by to make processing faster. For example, 0.5 will
%   downsample to half size, and run the algorithm.

tic;
startTime = toc;
% Parameters

% roi = [[1239,1290];[732,799]];   % rainbowbridge area
% roi2 = [[14, 35];[149, 372]]; % rainbowbridge stable area
% roi = [[268, 287];[245, 263]];
% roi = [[262, 288];[247, 266]];   % cable area
% roi = [[306, 336];[635, 677]];   % stable area  
% roi = [[1017, 1061];[636,688]];
addpath('E:\MATLAB\R2017b\bin\VMSlim')

defaultnframes = 0;
defaultDownsampleFactor = 1;
defaultsamplingrate = -1;
defaultROI = -1;
defaultROI2 = -1;
p = inputParser();
addOptional(p, 'DownsampleFactor', defaultDownsampleFactor, @isnumeric);   
addOptional(p, 'NFrames', defaultnframes, @isnumeric);   
addOptional(p, 'SamplingRate', defaultsamplingrate, @isnumeric);
addOptional(p, 'ROI', defaultROI, @isnumeric);
addOptional(p, 'ROI2', defaultROI2, @isnumeric);
parse(p, varargin{:});
nScales = nscalesin;
nOrients = norientationsin;
dSampleFactor = p.Results.DownsampleFactor;
numFramesIn = p.Results.NFrames;
samplingrate = p.Results.SamplingRate;

roi = p.Results.ROI;
% roi2 = p.Results.ROI2;

if(samplingrate<0)
    samplingrate = vHandle.FrameRate;
end


% 'Reading first frame of video'
colorframe = vHandle.read(1);
% 'Successfully read first frame of video'

if(dSampleFactor~=1)
    colorframe = imresize(colorframe,dSampleFactor);% downsample 1st frames
end
%fullFrame = im2single(squeeze(colorframe));
%fullFrame = rgb2gray(colorframe);
fullFrame = im2single(squeeze(mean(colorframe,3)));% 'mean()'convert the frame to Grayscale image
                                                   % squeeze Remove singleton dimensions
                                                   % im2single Convert image to single precision
% refFrame = fullFrame;
refFrame = fullFrame(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% refFrame = de_background(refFrame, 2);               % 使用bilinear模糊重建（非锐化掩蔽）
% rad_mask = border_mask(size(refFrame), 5, 1);        % 新增边界衰减抑制边界阶跃效应(没用）
% refFrame = refFrame .* rad_mask;
% stable_refFrame = fullFrame(roi2(1,1)+1:roi2(1,2),roi2(2,1)+1:roi2(2,2));
[h,w] = size(refFrame);%height and width of video in pixels

% nF = numFramesIn;
nF = 0;
if(nF==0)
    %depending on matlab and type of video you are using, may need to read
    %the last frame
    %lastFrame = read(vHandle, inf); 
    nF = vHandle.NumberOfFrames;%number of frames
end


%params.nScales = nScales;
%params.nOrientations = nOrients;
%params.dSampleFactor = dSampleFactor;
%params.nFrames = nF;

% ok
%%

[pyrRef, pind] = buildSCFpyr(refFrame, nScales, nOrients-1); % refFrame= first frame
                                                             % nScales: n个尺度

% [stable_pyrRef, stable_pind] = buildSCFpyr(stable_refFrame, nScales, nOrients-1);                                                             
for j = 1:nScales
    for k = 1:nOrients
        bandIdx = 1+nOrients*(j-1)+k;    
    end
end

%

totalsigs = nScales*nOrients;
signalffs = zeros(nScales,nOrients,nF);
% stable_signalffs = zeros(nScales,nOrients,nF);
ampsigs = zeros(nScales,nOrients,nF);

%


% Process
% nF

for q = 1:nF                  % progress bar
    if(mod(q,floor(nF/5))==1)
        progress = q/nF;
        currentTime = toc;
        ['Progress:' num2str(progress*100) '% done after ' num2str(currentTime-startTime) ' seconds.']
    end
    
    vframein = vHandle.read(q);
    if(dSampleFactor == 1)
        fullFrame = im2single(squeeze(mean(vframein,3)));
    else
        fullFrame = im2single(squeeze(mean(imresize(vframein,dSampleFactor),3)));
    end
    
    im = fullFrame(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     im = de_background(im, 2); 
%     im2 = fullFrame(roi2(1,1)+1:roi2(1,2),roi2(2,1)+1:roi2(2,2));
    
    pyr = buildSCFpyr(im, nScales, nOrients-1);
%     stable_pyr = buildSCFpyr(im2, nScales, nOrients-1);
    pyrAmp = abs(pyr);
    pyrDeltaPhase = mod(pi+angle(pyr)-angle(pyrRef), 2*pi) - pi; 
%     stable_pyrDeltaPhase = mod(pi+angle(stable_pyr)-angle(stable_pyrRef), 2*pi) - pi; 
    
    % 最优角度子带相位响应
    [optimal_phase, optimal_index, optimal_mask] = calculate_phase(im, nscalesin-1, norientationsin, 65/180*pi);
    anime3(:,q) = optimal_phase;
    anime3_mask(:,q) = optimal_mask;
    
    for j = 1:nScales
        bandIdx = 1 + (j-1)*nOrients + 1;
        curH = pind(bandIdx,1);
        curW = pind(bandIdx,2);        
        for k = 1:nOrients
            bandIdx = 1 + (j-1)*nOrients + k;
            amp = pyrBand(pyrAmp, pind, bandIdx); %figure; surf(amp); title('Amplitude')
            phase = pyrBand(pyrDeltaPhase, pind, bandIdx); %figure; surf(phase); title('Delta Phase')
            phasemap = pyrBand(pyr,pind,bandIdx); %figure; surf(angle(phasemap)); title('Phase')
%             phase2 = pyrBand(stable_pyrDeltaPhase, stable_pind, bandIdx);
            if (j == 1)
                anime1(:,:,j,k,q)= phasemap;
                p_anime1(:,:,j,k,q)= phase;
            elseif (j == 2)
                anime2(:,:,j,k,q)= phasemap;  
                p_anime2(:,:,j,k,q)= phase; 
            end
            % 创建动画
%             if (j == 2)
%                 if (k == 2)
%                     surf(angle(phasemap)); colorbar;
%                     view(76,36);
%                     anime(j,k,q)= getframe;
%                 end
%             end
            %weighted signals with amplitude square weights. 
            phasew = phase.*(abs(amp).^2);
            
            %sumamp = sum(abs(amp(:)));
            sumamp = sum(abs(amp(:)).^2);
            ampsigs(j,k,q)= sumamp;
            
            % only using mean values of delta phase here
            signalffs(j,k,q)=mean(phase(:));%/sumamp;
%             stable_signalffs(j,k,q)=mean(phase2(:));
        end
    end    
end

%avx is average x
S.samplingRate = samplingrate;

signalout = zeros(4,411);
%%
f1 = figure;
f2 = figure;
%%
line = animatedline('Color','b');
xline = [1:1:nF];
disp = 0;
ANIME(411) = struct('cdata',[],'colormap',[]);
% mask = mask_creator([26,26],2,4,[1,1],0);
for i = 2:411
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
%     subplot(1,2,1);
%     figure(f1), hold on
%     imshow(im,'InitialMagnification','fit');
                                                             % 最优角度相位
    t_phase = mod(pi+anime3(:,i)-anime3(:,i-1), 2*pi)-pi;    % 相邻帧法
%     t_mask = anime3_mask(:,i)|anime3_mask(:,i-1);
    tdisp = mean(t_phase(:)); 
    disp = disp+tdisp; signalout(4,i) = disp;
%     disp = mean(mod(pi+anime3(:,i)-anime3(:,1), 2*pi)-pi);   % 参考帧法
    addpoints(line,xline(i),double(disp)); 

%     subplot(1,2,2);
%     surf(angle(anime1(:,:,1,2,i)) - angle(anime1(:,:,1,2,i-1)));view(0,-90); colorbar;
%     subplot(1,3,3);
%     surf(mod(pi + angle(anime1(:,:,1,2,i)) - angle(anime1(:,:,1,2,i-1)),2*pi)-pi);
%     surf(angle(anime1(:,:,1,1,i)));
%     surf(abs(anime1(:,:,1,4,i)).^2);
%     surf(real(anime1(:,:,1,1,i)).^2 + imag(anime1(:,:,1,1,i)).^2);
%     addpoints(line,xline(i),double(sum(sum(abs(anime2(:,:,2,4,i)).^2))));
    
%     im = de_background(im, 2);  % 频域法求方向能量
% %     temp_energy = f_energy(im, 2, 4, [2, 1]);   
% %     temp_energy2 = f_energy(im, 2, 4, [2, 3]);
%     temp_energy = f_energy(im, 2, 2, [2, 1]);   
%     temp_energy2 = f_energy(im, 2, 2, [2, 2]);    
%     
%     addpoints(line,xline(i),atan(temp_energy2./temp_energy)/pi*180);
    
%     subplot(1,2,2);
%     energy_angle = detect_angle(im, 2, 4, 0); angle_out(1,i-1) = energy_angle;% 方向能量法
%     energy_angle = detect_angle(im, 2, 4, 1); energy_angle = energy_angle/pi*180;  angle_out(2,i-1) = energy_angle;% 频谱加权法
%     energy_angle = detect_angle(im, 2, 4, 2); angle_out(3,i-1) = energy_angle;% 方向能量增强的Hough法检测直线角度
%     energy_angle = detect_angle(im, 2, 4, 3); angle_out(4,i-1) = energy_angle;% 频谱非极大值抑制
%     energy_angle = detect_angle(im, 2, 4, 4); angle_out(5,i-1) = energy_angle;% canny边缘检测+霍夫法
%     addpoints(line,xline(i),energy_angle);

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
%% 保存anime中的动画
    outfilename = 'C:\Users\HIT\Desktop\相位图片\高置信度ROI.avi';
    outvideo = VideoWriter(outfilename,'Motion JPEG AVI');
    open(outvideo)
    for i = 2:411
        writeVideo(outvideo,ANIME(i).cdata);
    end
    close(outvideo);
    '输出完成'

%%
    i = 1;    
    f1 = figure;
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    figure(f1), hold on
    imshow(im,'InitialMagnification','fit');
    im = im2single(squeeze(mean(im,3)));
    [optimal_phase, optimal_amp, optimal_index] = get_amp_phase(im, nscalesin-1, norientationsin, 50/180*pi);
%     figure(f2), hold on
%     surf(reshape(optimal_amp, optimal_index(1,1), optimal_index(1,2)));
%     figure(f3), hold on
%     surf(optimal_phase);
%%
% movie(anime(:),5,15);

% sigOut = zeros(nF, nScales);
% stable_sigOut = zeros(nF, nScales);
% for q=1:nScales
%     for p=1:nOrients
% %         [sigaligned, shiftam] = vmAlignAToB(squeeze(signalffs(q,p,:)), squeeze(signalffs(1,1,:)));
% %         [stable_sigaligned, stable_shiftam] = vmAlignAToB(squeeze(stable_signalffs(q,p,:)), squeeze(stable_signalffs(1,1,:)));
% %         sigOut(:,q) = sigOut(:,q)+sigaligned;
% %         stable_sigOut(:,q) = stable_sigOut(:,q)+stable_sigaligned;
%         sigOut(:,q) = sigOut(:,q)+squeeze(signalffs(q,p,:));
%         stable_sigOut(:,q) = stable_sigOut(:,q)+squeeze(stable_signalffs(q,p,:));
% %         shiftam
%     end
% end

% S.aligned = sigOut;

% fftspec(S.aligned, samplingrate);
fftspec(signalout(4,:), samplingrate);
% S.freq_amp = fftspec(signalout(4,:), samplingrate); return

%sometimes the alignment aligns on noise and boosts it, in which case just
%use averaging with no alignment, or highpass before alignment
S.averageNoAlignment = mean(reshape(double(signalffs),nScales*nOrients,nF)).';

[b,a] = butter(3,[0.3 3]*2/samplingrate);
S.x = filter(b,a,S.aligned);
fftspec(S.x, samplingrate); 

highpassfc = 0.05;
[b,a] = butter(3,highpassfc,'high');
S.x = filter(b,a,S.aligned);

%sometimes butter doesn't fix the first few entries
S.x(1:10)=mean(S.x);

maxsx = max(S.x);
minsx = min(S.x);
if(maxsx~=1.0 || minsx ~= -1.0)
    range = maxsx-minsx;
    S.x = 2*S.x/range;
    newmx = max(S.x);
    offset = newmx-1.0;
    S.x = S.x-offset;
end

%

end
