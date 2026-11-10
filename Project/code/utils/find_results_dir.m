function results_dir = find_results_dir()
    % FIND_RESULTS_DIR Returns the path to the results directory containing
    % valid pre-computed feature matrices. Searches multiple candidate paths.

    candidates = {
        fullfile('..', 'results'), ...          % Project/results  (Python output)
        fullfile('..', '..', 'results'), ...    % OE-2/results     (MATLAB fallback)
    };

    for ci = 1:length(candidates)
        f = fullfile(candidates{ci}, 'feature_matrix_train.mat');
        if exist(f, 'file')
            try
                tmp = load(f, 'X_train');
                if isfield(tmp, 'X_train') && size(tmp.X_train, 1) > 10
                    results_dir = candidates{ci};
                    return;
                end
            catch
            end
        end
    end

    % Default: two levels up (existing MATLAB convention)
    results_dir = fullfile('..', '..', 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
end
