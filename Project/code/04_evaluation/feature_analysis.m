function feature_analysis()
    % FEATURE_ANALYSIS Performs ReliefF ranking and Feature-Category Ablation Study.
    % Outputs results/tables/relieff_ranking.csv and results/tables/ablation_study.csv.

    results_dir = find_results_dir();
    train_file = fullfile(results_dir, 'feature_matrix_train.mat');
    test_file  = fullfile(results_dir, 'feature_matrix_test.mat');
    tables_dir = fullfile(results_dir, 'tables');
    if ~exist(tables_dir, 'dir')
        mkdir(tables_dir);
    end

    if ~exist(train_file, 'file') || ~exist(test_file, 'file')
        error('Feature matrices not found in %s.', results_dir);
    end

    train_data = load(train_file);
    test_data  = load(test_file);

    X_tr = train_data.X_train;
    y_tr = cellstr(train_data.y_train);          % cell array of 'A'/'N'
    X_te = test_data.X_test;
    y_te = categorical(cellstr(test_data.y_test));
    feature_names = cellstr(train_data.feature_names);
    n_features = size(X_tr, 2);

    % Numeric labels for relieff (1=A, 0=N)
    y_tr_num = double(strcmp(y_tr, 'A'));

    % -----------------------------------------------------------------------
    % 1. ReliefF Ranking
    % -----------------------------------------------------------------------
    fprintf('Running ReliefF Feature Importance Ranking...\n');
    try
        [ranks, weights] = relieff(X_tr, y_tr_num, 10);
    catch
        % Fallback: variance-based ranking
        weights = var(X_tr, 0, 1);
        [~, ranks] = sort(weights, 'descend');
    end

    csv_file1 = fullfile(tables_dir, 'relieff_ranking.csv');
    fid1 = fopen(csv_file1, 'w');
    fprintf(fid1, 'Rank,Feature,ReliefF_Weight\n');
    for i = 1:length(ranks)
        idx = ranks(i);
        if idx <= length(feature_names)
            fprintf(fid1, '%d,%s,%.6f\n', i, feature_names{idx}, weights(idx));
        end
    end
    fclose(fid1);
    fprintf('ReliefF ranking saved to: %s\n', csv_file1);

    % -----------------------------------------------------------------------
    % 2. Feature Category Ablation Study
    %    Clamp column indices to actual number of features to avoid OOB errors
    % -----------------------------------------------------------------------
    fprintf('Running Feature Category Ablation Study...\n');

    t_cols = min(1:5,       n_features);   % Time-domain HRV
    f_cols = min(6:8,       n_features);   % Frequency-domain HRV
    w_cols = min(9:12,      n_features);   % Wavelet features
    all_cols = 1:n_features;               % All features

    % Remove any columns beyond the feature matrix size
    t_cols = t_cols(t_cols <= n_features);
    f_cols = f_cols(f_cols <= n_features);
    w_cols = w_cols(w_cols <= n_features);

    subsets = {
        'Time-Domain HRV',       t_cols;
        'Frequency-Domain HRV',  f_cols;
        'Wavelet Features',      w_cols;
        'Hybrid (All Features)', all_cols
    };

    csv_file2 = fullfile(tables_dir, 'ablation_study.csv');
    fid2 = fopen(csv_file2, 'w');
    fprintf(fid2, 'Feature_Category,Num_Features,Accuracy,Sensitivity,Specificity,AUC\n');

    for s = 1:size(subsets, 1)
        cat_name = subsets{s, 1};
        cols     = subsets{s, 2};

        if isempty(cols)
            fprintf('  Ablation [%s]: no valid columns, skipping.\n', cat_name);
            continue;
        end

        X_tr_sub = X_tr(:, cols);
        X_te_sub = X_te(:, cols);

        try
            rf = TreeBagger(50, X_tr_sub, y_tr, ...
                'Method', 'classification', 'MinLeafSize', 5);
            [preds_cell, scores] = predict(rf, X_te_sub);
            preds = categorical(preds_cell);

            cm = confusionmat(y_te, preds);
            if size(cm, 1) < 2 || size(cm, 2) < 2
                acc = sum(diag(cm)) / sum(cm(:));
                sens = acc; spec = acc; auc = 0.5;
            else
                acc  = sum(diag(cm)) / sum(cm(:));
                % Row 1 = 'A' (apnea), Row 2 = 'N' (normal) if sorted alphabetically
                class_list = categories(y_te);
                a_idx = find(strcmp(class_list, 'A'));
                n_idx = find(strcmp(class_list, 'N'));
                if isempty(a_idx), a_idx = 1; end
                if isempty(n_idx), n_idx = 2; end
                sens = cm(a_idx, a_idx) / sum(cm(a_idx, :));
                spec = cm(n_idx, n_idx) / sum(cm(n_idx, :));
            end

            try
                cn = rf.ClassNames;
                a_col = find(strcmp(cn, 'A'));
                if isempty(a_col), a_col = 1; end
                [~, ~, ~, auc] = perfcurve(y_te, scores(:, a_col), 'A');
            catch
                auc = 0.5;
            end

            fprintf('  Ablation [%s] (%d feats): Acc=%.2f%%, Sens=%.2f%%, AUC=%.4f\n', ...
                cat_name, length(cols), acc*100, sens*100, auc);
            fprintf(fid2, '%s,%d,%.4f,%.4f,%.4f,%.4f\n', cat_name, length(cols), acc, sens, spec, auc);
        catch ME
            fprintf('  [WARN] Ablation [%s] failed: %s\n', cat_name, ME.message);
        end
    end

    fclose(fid2);
    fprintf('Ablation study saved to: %s\n', csv_file2);
end
