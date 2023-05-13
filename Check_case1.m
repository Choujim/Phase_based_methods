
% Case 1
%%
% µÚ7Ö¡Í¼Æ¬
i = 7;
im_7 = vHandle.read(i);
im_7 = im_7(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
figure; hold on
imshow(im_7,'InitialMagnification','fit');
energy_angle = detect_angle(im_7, 2, 4, 2);

pre_im_7 = vHandle.read(i-1);
pre_im_7 = pre_im_7(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
pre_energy_angle  = detect_angle(pre_im_7, 2, 4, 2);

post_im_7 = vHandle.read(i+1);
post_im_7 = post_im_7(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
post_energy_angle  = detect_angle(post_im_7, 2, 4, 2);

figure;plot([6,7,8],[pre_energy_angle,energy_angle,post_energy_angle]);


%%
% µÚ164Ö¡Í¼Æ¬
i = 378;
im_164 = vHandle.read(i);
im_164 = im_164(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% figure; hold on
% imshow(im_164,'InitialMagnification','fit');
energy_angle = detect_angle(im_164, 2, 4, 2);

% pre_im_164 = vHandle.read(i-1);
% pre_im_164 = pre_im_164(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% pre_energy_angle  = detect_angle(pre_im_164, 2, 4, 2);
% 
% post_im_164 = vHandle.read(i+1);
% post_im_164 = post_im_164(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% post_energy_angle  = detect_angle(post_im_164, 2, 4, 2);
% 
% figure;plot([163,164,165],[pre_energy_angle,energy_angle,post_energy_angle]);

% figure; surf(im_164);
% figure; surf(abs(fftshift(fft2(im_164))));

%%
% µÚ165Ö¡Í¼Æ¬
i = 165;
im_165 = vHandle.read(i);
im_165 = im_165(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% figure; hold on
% imshow(im_165,'InitialMagnification','fit');
energy_angle = detect_angle(im_165, 2, 4, 2);

% pre_im_165 = vHandle.read(i-1);
% pre_im_165 = pre_im_165(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% pre_energy_angle  = detect_angle(pre_im_165, 2, 4, 2);
% 
% post_im_165 = vHandle.read(i+1);
% post_im_165 = post_im_165(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
% post_energy_angle  = detect_angle(post_im_165, 2, 4, 2);
% 
% figure;plot([164,165,166],[pre_energy_angle,energy_angle,post_energy_angle]);

figure; surf(im_165);

%%
% µÚ191Ö¡Í¼Æ¬
i = 191;
im_191 = vHandle.read(i);
im_191 = im_191(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
figure; hold on
imshow(im_191,'InitialMagnification','fit');
energy_angle = detect_angle(im_191, 2, 4, 2);

pre_im_191 = vHandle.read(i-1);
pre_im_191 = pre_im_191(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
pre_energy_angle  = detect_angle(pre_im_191, 2, 4, 2);

post_im_191 = vHandle.read(i+1);
post_im_191 = post_im_191(roi(1,1)+1:roi(1,2),roi(2,1)+1:roi(2,2));
post_energy_angle  = detect_angle(post_im_191, 2, 4, 2);

figure;plot([190,191,192],[pre_energy_angle,energy_angle,post_energy_angle]);

