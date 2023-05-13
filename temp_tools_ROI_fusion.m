size_p = size(phase_bin_masks);
size_l = size(line_bins);
refine_roi = single(zeros(dim(1), dim(2), size_p(3)));
cos_ratio = 0.46;
for it_p = 1:size_p(3)
    temp_phase_bin = phase_bin_masks(:,:,it_p);
    
    for it_l = 1:size_l(3)
        temp = line_bins(:,:,it_l);
        query = temp_phase_bin .* temp;
        temp_ref = sum(temp(:));
        temp_query = sum(query(:));
%         temp_query / temp_ref
        if (temp_query > 8) && ((temp_query / temp_ref) >= cos_ratio)
            refine_roi(:,:,it_p) = refine_roi(:,:,it_p) + query + temp;
        end
    end
    refine_roi(:,:,it_p) = single(logical(refine_roi(:,:,it_p)));
end

refine_mask = single(zeros(dim(1), dim(2)));
for it_p = 1:size_p(3)
    refine_mask = refine_mask + refine_roi(:,:,it_p);
end
% refine_mask = single(logical(refine_mask));
figure; surf(refine_mask);