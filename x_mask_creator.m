% 返回指定子带的模板
% method：指定通过子带方向编号 or 指定角度 生成mask
% method = 0: 通过子带方向编号
% method = 1: 指定角度， 此时band_idx(2)为具体角度值 [0 ~ pi]
% method = 2: 指定角度， 构建Gabor

function mask = x_mask_creator(image_size, scale_total, orientation_total, band_index, method)

%% 通用参数组
dims = image_size;
ctr = ceil((dims+0.5)/2);
[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
                          ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle_map = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);    % set center point with a nonzero value
log_rad  = log2(log_rad);

%% 径向分解参数组
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);

%% 角向分解参数组
lutsize = 1024;
Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;
alfa=	mod(pi+Xcosn,2*pi)-pi;
nbands = orientation_total;
order = nbands-1;
const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
Ycosn = 2*sqrt(const) * (cos(Xcosn).^order) .* (abs(alfa)<pi/2);

%% 多尺度多方向滤波模板  
% 通过子带方向编号
if (method == 0)
    for i = 1:band_index(1)
        
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - log2(2);
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        radial_mask = lomask .* himask;
        anglemask = pointOp(angle_map, Ycosn, Xcosn(1)+pi*(band_index(2)-1)/nbands, Xcosn(2)-Xcosn(1));
        %       下采样
        dims = size(log_rad);
        ctr = ceil((dims+0.5)/2);
        lodims = ceil((dims-0.5)/2);
        loctr = ceil((lodims+0.5)/2);
        lostart = ctr-loctr+1;
        loend = lostart+lodims-1;
        log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
        angle_map = angle_map(lostart(1):loend(1),lostart(2):loend(2));
        
    end
    mask = radial_mask .* anglemask;
    
% 指定角度 
%   输入角度应为弧度
elseif (method == 1)
    for i = 1:band_index(1)
        
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - log2(2);
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        radial_mask = lomask .* himask;
        anglemask = pointOp(angle_map, Ycosn, Xcosn(1)+band_index(2), Xcosn(2)-Xcosn(1)); % 修改了角度部分
        %       下采样
        if false
            dims = size(log_rad);
            ctr = ceil((dims+0.5)/2);
            lodims = ceil((dims-0.5)/2);
            loctr = ceil((lodims+0.5)/2);
            lostart = ctr-loctr+1;
            loend = lostart+lodims-1;
            log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
            angle_map = angle_map(lostart(1):loend(1),lostart(2):loend(2));
        end
        
    end
    mask = radial_mask .* anglemask;

% Gabor构建
% 指定角度 
elseif (method == 2)
    lambda = 2 .^ (band_index(1)+1);
    theta = -band_index(2)/pi*180;
    bandwidth = 1.5;
    sigma_af = 1.5;
    G = gabor(lambda, theta, 'SpatialFrequencyBandwidth', bandwidth, 'SpatialAspectRatio', sigma_af);
    tmp_im = zeros(dims(1), dims(2));
    tmp_im(ctr(1), ctr(2)) = 1;
%     tmp_mask = fftshift(fft2(tmp_im));
    [amp_mask, phase_mask] = imgaborfilt(tmp_im, G);
    tmp_mask = amp_mask .* exp(-1i .* phase_mask);
%     figure; surf(real(tmp_mask), 'EdgeColor', 'none', 'FaceColor', 'interp');
%     figure; surf(imag(tmp_mask), 'EdgeColor', 'none', 'FaceColor', 'interp');
    tmp_mask = real(fftshift(ifft2(ifftshift(tmp_mask))));
%     figure; surf(tmp_mask);
    mask = tmp_mask;
end

% mask = ((-1i)^(nbands-1)) .* radial_mask .* anglemask;
% mask = radial_mask .* anglemask;
% mask = lomask;
% mask = himask;
end
    
    

    





