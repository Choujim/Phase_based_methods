line = animatedline('Color','b');
xline = [1:1:nF];

for i = 2:411
    vframein = vHandle.read(i);
    im = vframein(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
    image = de_background(im, 2);
    
    figure(f1), hold on
    imshow(im,'InitialMagnification','fit');
    energy_angle = [];
    scale_index = 2;
    for j = 0:179
        energy_angle(j+1) = f_energy(image, 2, 4, [scale_index, j/180*pi], 1);   
    end
    [max_val, max_angle] = max(energy_angle);
    
    max_angle = max_angle-1
    addpoints(line,xline(i),max_angle);
    figure(f2)
    title(['frame:',num2str(i)]);
end