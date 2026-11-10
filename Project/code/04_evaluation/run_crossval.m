function run_crossval()
    % RUN_CROSSVAL Evaluates SVM, Random Forest, and Neural Network models on the benchmark test split.
    % Computes Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC, and Cohen's Kappa.
    % Saves performance metrics to results/tables/model_performance.csv.

    results_dir = find_results_dir();
    test_file = fullfile(results_dir, 'feature_matrix_test.mat');
    models_dir = fullfile(results_dir, 'models');
    tables_dir = fullfile(results_dir, 'tables');
    if ~exist(tables_dir, 'dir')
        mkdir(tables_dir);
    end

    if ~exist(test_file, 'file')
        error('Test feature matrix not found: %s. Run build_feature_matrix first.', test_file);
    end

    data = load(test_file);
    X_test = data.X_test;
    y_test = cellstr(data.y_test);   % keep as cell array; convert per-model below

    csv_file = fullfile(tables_dir, 'model_performance.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Model,Accuracy,Sensitivity,Specificity,Precision,F1_Score,AUC,Kappa\n');

    models = {'svm_model', 'rf_model', 'nn_model'};
    model_names = {'SVM (RBF)', 'Random Forest', 'Neural Network'};

    for i = 1:length(models)
        m_file = fullfile(models_dir, [models{i}, '.mat']);
        if exist(m_file, 'file')
            m_struct = load(m_file);
            fn = fieldnames(m_struct);
            mdl = m_struct.(fn{1});

            try
                if strcmp(models{i}, 'rf_model')
                    [y_pred_cell, scores] = predict(mdl, X_test);
                    y_pred = y_pred_cell;   % already cell array
                    cn = mdl.ClassNames;
                    a_col = find(strcmp(cn, 'A'));
                    if isempty(a_col), a_col = 1; end
                    score_apnea = scores(:, a_col);
                else
                    [y_pred_raw, scores] = predict(mdl, X_test);
                    % Convert to cellstr regardless of whether output is categorical/char/cell
                    if iscategorical(y_pred_raw)
                        y_pred = cellstr(char(y_pred_raw));
                    elseif ischar(y_pred_raw)
                        y_pred = cellstr(y_pred_raw);
                    else
                        y_pred = y_pred_raw;
                    end
                    cn = mdl.ClassNames;
                    if iscategorical(cn)
                        cn = cellstr(char(cn));
                    end
                    a_col = find(strcmp(cn, 'A'));
                    if isempty(a_col), a_col = 1; end
                    score_apnea = scores(:, a_col);
                end

                metrics = compute_classification_metrics(y_test, y_pred, score_apnea);
                fprintf('%s Performance: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%, F1=%.2f%%, AUC=%.4f, Kappa=%.4f\n', ...
                    model_names{i}, metrics.acc*100, metrics.sens*100, metrics.spec*100, metrics.f1*100, metrics.auc, metrics.kappa);

                fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
                    model_names{i}, metrics.acc, metrics.sens, metrics.spec, metrics.prec, metrics.f1, metrics.auc, metrics.kappa);
            catch ME
                fprintf('[ERROR] Failed to evaluate %s: %s\n', model_names{i}, ME.message);
            end
        end
    end

    fclose(fid);
    fprintf('Evaluation complete. Results saved to: %s\n', csv_file);
end

function metrics = compute_classification_metrics(y_true, y_pred, scores)
    % Normalize both to cellstr so confusionmat always gets matching types
    if iscategorical(y_true),  y_true = cellstr(char(y_true)); end
    if ischar(y_true),         y_true = cellstr(y_true); end
    if iscategorical(y_pred),  y_pred = cellstr(char(y_pred)); end
    if ischar(y_pred),         y_pred = cellstr(y_pred); end

    cm = confusionmat(y_true, y_pred);
    classes = unique(y_true);   % sorted: {'A', 'N'}
    pos_idx = find(strcmp(classes, 'A'));
    neg_idx = find(strcmp(classes, 'N'));

    if isempty(pos_idx), pos_idx = 1; end
    if isempty(neg_idx), neg_idx = 2; end

    TP = cm(pos_idx, pos_idx);
    FN = cm(pos_idx, neg_idx);
    FP = cm(neg_idx, pos_idx);
    TN = cm(neg_idx, neg_idx);

    N = TP + TN + FP + FN;
    acc = (TP + TN) / N;
    sens = TP / (TP + FN);
    spec = TN / (TN + FP);
    prec = TP / (TP + FP);
    f1 = 2 * (sens * prec) / (sens + prec);

    p0 = acc;
    pe = (((TP + FN)*(TP + FP)) + ((TN + FP)*(TN + FN))) / (N^2);
    kappa = (p0 - pe) / (1 - pe);

    try
        [~, ~, ~, auc] = perfcurve(y_true, scores, 'A');
    catch
        auc = 0.5;
    end

    metrics.acc = acc;
    metrics.sens = sens;
    metrics.spec = spec;
    metrics.prec = prec;
    metrics.f1 = f1;
    metrics.auc = auc;
    metrics.kappa = kappa;
end
