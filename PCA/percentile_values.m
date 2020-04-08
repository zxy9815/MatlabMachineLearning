function [percentile_values] = percentile_values(v, percentiles)
    % assumes v as a column vector
    % percentiles is an array of percentiles 
    % percentile_values is an array of percentile-values corresponding
    % to percentiles 
    [n, ~] = size(v);
    [sorted_v,~] = sort(v, 'ascend');
    percentile_indices = ceil(n*percentiles/100);
    percentile_values = sorted_v(percentile_indices,:);
end
