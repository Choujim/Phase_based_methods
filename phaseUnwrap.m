% ����phasemap��ȡdelta-phaseʱ���Կ��ܷ���phase-wrap��λ�õ����ݽ��д���
% phasemap1--ǰһ֡����
% phasemap2--��һ֡����
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