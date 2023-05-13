% ���ƽ�棬�ṩ��λmap�ϵ��ݶȹ���

function [omega, normal_vector] = plane_fitting(phase_scales, plane_roi, orient_angle, method)

    if (method == 0)
        %% ��pca���ƽ��
        sum_points = sum(sum(single(plane_roi)));
        roi_index = find(logical(plane_roi));
        map_size = size(plane_roi);

        % ����������ĵ�
        center_point = zeros(3, 1); %[Y, X, Z]
        for p = 1:sum_points
            [row, col] = ind2sub([map_size(1), map_size(2)], roi_index(p));
            center_point(1) = center_point(1) + row;
            center_point(2) = center_point(2) + col;
            center_point(3) = center_point(3) + phase_scales(row, col);
        end
        center_point = center_point ./ sum_points;

        % ����������Э�������
        % ȥ���Ļ��㼯д��3 * N�ľ���
        decenter_mat = zeros(sum_points, 3);
        for p = 1:sum_points
            [row, col] = ind2sub([map_size(1), map_size(2)], roi_index(p));
            decenter_mat(p, 1) = row - center_point(1);
            decenter_mat(p, 2) = col - center_point(2);
            decenter_mat(p, 3) = phase_scales(row, col) - center_point(3);
        end
        cov_mat = decenter_mat' * decenter_mat ./sum_points;
        % ����С����ֵ��Ӧ����������
        [eigen_vector, eigen_value] = eig(cov_mat);
        abs_eigen = abs([eigen_value(1,1);eigen_value(2,2);eigen_value(3,3)]);
        [min_value, min_index] = min(abs_eigen(:));
        normal_vector = eigen_vector(:,min_index);
        orient_angle = orient_angle / 180 * pi;
        unit_vector = [sin(orient_angle), cos(orient_angle), 0];
        sign_normal = sign(dot(sign(normal_vector(3)) .* normal_vector', unit_vector));
        omega = abs(dot(normal_vector', unit_vector)) / (norm(normal_vector)*norm(unit_vector));
        omega = sign_normal .* 1/tan(acos(omega));
    else if (method == 1)
        %% �þ������pca���
        
    end
end