% 处理phasemap获取delta-phase时，对可能发生phase-wrap的位置的数据进行处理
% phasemap1--前一帧数据
% phasemap2--后一帧数据
function deltaphase = phaseUnwrap(phasemap1, phasemap2, rule)
%% 
if (rule == 0)
    signmap = phasemap1 .* phasemap2;
    sign = signmap >= 0;
    maps_ize = size(phasemap1);
    positive_map = zeros(maps_ize);
    negative_map = zeros(maps_ize);
    positive_map(sign) = 1;
    negative_map(~sign) = 1;
    delta1 = 
end