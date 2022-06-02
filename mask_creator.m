% ����ָ���Ӵ���ģ��
% method��ָ��ͨ���Ӵ������� or ָ���Ƕ� ����mask
% method = 0: ͨ���Ӵ�������
% method = 1: ָ���Ƕȣ� ��ʱband_idx(2)Ϊ����Ƕ�ֵ [0 ~ pi]

function mask = mask_creator(image_size, scale_total, orientation_total, band_index, method)

%% ͨ�ò�����
dims = image_size;
ctr = ceil((dims+0.5)/2);
[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
                          ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle_map = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);    % set center point with a nonzero value
log_rad  = log2(log_rad);

%% ����ֽ������
twidth = 1;
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);

%% ����ֽ������
lutsize = 1024;
Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;
alfa=	mod(pi+Xcosn,2*pi)-pi;
nbands = orientation_total;
order = nbands-1;
const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
Ycosn = 2*sqrt(const) * (cos(Xcosn).^order) .* (abs(alfa)<pi/2);

%% ��߶ȶ෽���˲�ģ��  
% ͨ���Ӵ�������
if (method == 0)
    for i = 1:band_index(1)
        
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - log2(2);
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        radial_mask = lomask .* himask;
        anglemask = pointOp(angle_map, Ycosn, Xcosn(1)+pi*(band_index(2)-1)/nbands, Xcosn(2)-Xcosn(1));
        %       �²���
        dims = size(log_rad);
        ctr = ceil((dims+0.5)/2);
        lodims = ceil((dims-0.5)/2);
        loctr = ceil((lodims+0.5)/2);
        lostart = ctr-loctr+1;
        loend = lostart+lodims-1;
        log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
        angle_map = angle_map(lostart(1):loend(1),lostart(2):loend(2));
        
    end
    
% ָ���Ƕ�
elseif (method == 1)
    for i = 1:band_index(1)
        
        lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        Xrcos = Xrcos - log2(2);
        himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
        radial_mask = lomask .* himask;
        anglemask = pointOp(angle_map, Ycosn, Xcosn(1)+band_index(2), Xcosn(2)-Xcosn(1)); % �޸��˽ǶȲ���
        %       �²���
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

% mask = ((-1i)^(nbands-1)) .* radial_mask .* anglemask;
mask = radial_mask .* anglemask;
% mask = lomask;
% mask = himask;
end
    
    

    





