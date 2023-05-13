% 验证SVD用来做多平行平面估计

f = figure;
f.Position(3:4) = [900 400];
%%
dim = [26, 26];
% (Y, X, Z)
normal_vector = [0.5601, 0.3179, -sqrt(1-0.5601*0.5601-0.3179*0.3179)];
dist = 2.0;
% 生成单一平面
plane = zeros(dim(1), dim(2));
for i=1:dim(1)
    for j=1:dim(2)
        plane(i,j) = (dist - dot([i, j, 0], normal_vector)) ./  (-sqrt(1-0.5601*0.5601-0.3179*0.3179));
    end
end
% figure; surf(plane);

% 生成wrap平面
wrap_plane = mod((plane + pi), 2*pi) - pi;
% figure; surf(wrap_plane);
% figure; surf(wrap_plane .* roi_mask_2);

% SVD分析
test_plane = wrap_plane;
sum_points = double(sum(sum(roi_mask)));
roi_index = find(logical(roi_mask));
center_point = zeros(3, 1);
for p = 1:sum_points
    [row, col] = ind2sub([dim(1), dim(2)], roi_index(p));
    center_point(1) = center_point(1) + row;
    center_point(2) = center_point(2) + col;
    center_point(3) = center_point(3) + test_plane(row, col);
end
center_point = center_point ./ sum_points;

decenter_mat = zeros(sum_points, 3);
for p = 1:sum_points
    [row, col] = ind2sub([dim(1), dim(2)], roi_index(p));
    decenter_mat(p, 1) = row - center_point(1);
    decenter_mat(p, 2) = col - center_point(2);
    decenter_mat(p, 3) = test_plane(row, col) - center_point(3);
end
cov_mat = decenter_mat' * decenter_mat ./sum_points;
[eigen_vector, eigen_value] = eig(cov_mat);
abs_eigen = abs([eigen_value(1,1);eigen_value(2,2);eigen_value(3,3)]);
[min_value, min_index] = min(abs_eigen(:));
esti_normal = eigen_vector(:,min_index);

[U, S, V] = svd(decenter_mat, 'econ');
u_mat = decenter_mat * decenter_mat' ./ sum_points;
[u_eigen_vector, u_eigen_value] = eig(u_mat);
u_abs_eigen = abs(u_eigen_value);
[min_value, min_index] = min(abs_eigen(:));

% 查看向法向量投影的结果
theta_y = 0; theta_x = 0; theta_z = 0;
Ry = [1, 0, 0;
      0, cos(theta_y), -sin(theta_y);
      0, sin(theta_y), cos(theta_y)];
Rx = [cos(theta_x), 0, sin(theta_x);
      0, 1, 0;
      -sin(theta_x), 0, cos(theta_x)];
Rz = [cos(theta_z), -sin(theta_z), 0;
      sin(theta_z), cos(theta_z), 0;
      0, 0, 1;];
rotate_normal_vector = Ry * Rx * Rz * normal_vector';
result = decenter_mat * rotate_normal_vector;
figure; plot(result, 'r');

% 验证法向量
% proj_1 = decenter_mat * normal_vector';
% proj_2 = decenter_mat * esti_normal;
% figure; hold on;
% plot(proj_1, 'r');
% plot(proj_2, 'b');
% 可视化
figure; hold on;
[row, col] = ind2sub([dim(1), dim(2)], roi_index);
surf(test_plane .* roi_mask);
% surf(wrap_plane);
% scatter3(row, col, test_plane(roi_index), 'filled');
normal_point = [center_point(1), center_point(2), center_point(3)];
normal_line = [normal_point; normal_point + 10*normal_vector];

normal_line_2 = [normal_point; normal_point + 10*esti_normal'];
normal_line_3 = [normal_point; normal_point + 15*eigen_vector(:,1)'];
normal_line_4 = [normal_point; normal_point + 15*eigen_vector(:,2)'];
normal_line_5 = [normal_point; normal_point + 15*eigen_vector(:,3)'];
plot3(normal_line(:,2), normal_line(:,1), normal_line(:,3), 'b');
plot3(normal_line_2(:,2), normal_line_2(:,1), normal_line_2(:,3), 'r');

plot3(normal_line_3(:,2), normal_line_3(:,1), normal_line_3(:,3), 'r', 'LineWidth', 3);
plot3(normal_line_4(:,2), normal_line_4(:,1), normal_line_4(:,3), 'r', 'LineWidth', 3);
plot3(normal_line_5(:,2), normal_line_5(:,1), normal_line_5(:,3), 'r', 'LineWidth', 3);

