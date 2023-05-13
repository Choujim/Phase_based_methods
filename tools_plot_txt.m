% txtfile = 'D:\Download\ic-shm2021-data_proj3\data_proj3\ground_truth\Damage_multi1\GroundTruthData.txt';
txtfile = 'D:\Download\ic-shm2021-data_proj3\data_proj3\ground_truth\Damage_small1\GroundTruthData.txt';
% txtfile = 'D:\Download\ic-shm2021-data_proj3\data_proj3\ground_truth\Damage1\GroundTruthData.txt';

fs = fopen(txtfile);
data_in = textscan(fs, '%f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64, %f64');
fclose(fs);
whos data_in;
data2plot = zeros(1199, 16);
for i=1:16
    rowinfo = data_in(1,i);
    data2plot(:,i) = cell2mat(rowinfo);
end

% data2plot = reshape(data2plot, 16);

txt2plot = data2plot(:,12);
f = figure;
f.Position(3:4) = [900 400];
plot(-txt2plot);
