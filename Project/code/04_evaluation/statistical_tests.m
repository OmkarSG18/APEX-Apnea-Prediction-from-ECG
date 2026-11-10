function statistical_tests()
    % STATISTICAL_TESTS Computes McNemar's statistical significance test between classifier pairs.
    % Generates results/tables/mcnemar_tests.csv.

    results_dir = find_results_dir();
    test_file = fullfile(results_dir, 'feature_matrix_test.mat');
    models_dir = fullfile(results_dir, 'models');
    tables_dir = fullfile(results_dir, 'tables');
    if ~exist(tables_dir, 'dir')
        mkdir(tables_dir);
    end

    if ~exist(test_file, 'file')
        error('Test file not found: %s', test_file);
    end

    data = load(test_file);
    X_test = data.X_test;
    y_test = categorical(cellstr(data.y_test));

    svm_struct = load(fullfile(models_dir, 'svm_model.mat'));
    rf_struct  = load(fullfile(models_dir, 'rf_model.mat'));
    nn_struct  = load(fullfile(models_dir, 'nn_model.mat'));

    % Normalize all predictions to categorical for == comparison
    y_svm_raw = predict(svm_struct.svm_model, X_test);
    if iscategorical(y_svm_raw), y_svm = y_svm_raw;
    else, y_svm = categorical(cellstr(y_svm_raw)); end

    y_rf_raw = predict(rf_struct.rf_model, X_test);
    if iscategorical(y_rf_raw), y_rf = y_rf_raw;
    else, y_rf = categorical(cellstr(y_rf_raw)); end

    y_nn_raw = predict(nn_struct.nn_model, X_test);
    if iscategorical(y_nn_raw), y_nn = y_nn_raw;
    else, y_nn = categorical(cellstr(y_nn_raw)); end

    csv_file = fullfile(tables_dir, 'mcnemar_tests.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Comparison_Pair,ChiSquare_Stat,P_Value,Significant_p05\n');

    pairs = {
        'SVM vs Random Forest', y_svm, y_rf;
        'SVM vs Neural Network', y_svm, y_nn;
        'Random Forest vs Neural Network', y_rf, y_nn
    };

    fprintf('Computing McNemar Statistical Tests...\n');

    for p = 1:size(pairs, 1)
        name = pairs{p, 1};
        p1 = pairs{p, 2};
        p2 = pairs{p, 3};

        c1 = (p1 == y_test);
        c2 = (p2 == y_test);

        b = sum(c1 & ~c2);
        c = sum(~c1 & c2);

        chi2 = (abs(b - c) - 1)^2 / (b + c + eps);
        p_val = 1 - chi2cdf(chi2, 1);
        sig = p_val < 0.05;

        fprintf('  %s: Chi2=%.4f, p-val=%.6f (Sig: %d)\n', name, chi2, p_val, sig);
        fprintf(fid, '%s,%.4f,%.6f,%d\n', name, chi2, p_val, sig);
    end

    fclose(fid);
    fprintf('Statistical tests complete. Saved to: %s\n', csv_file);
end
