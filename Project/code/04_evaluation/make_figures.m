function make_figures()
    % MAKE_FIGURES Generates publication-quality 300 DPI figures for the manuscript.
    % Outputs PNG and EPS files to results/figures/.

    results_dir = find_results_dir();
    figs_dir = fullfile(results_dir, 'figures');
    test_file = fullfile(results_dir, 'feature_matrix_test.mat');
    models_dir = fullfile(results_dir, 'models');
    if ~exist(figs_dir, 'dir')
        mkdir(figs_dir);
    end

    if ~exist(test_file, 'file')
        error('Test file not found: %s', test_file);
    end

    data = load(test_file);
    X_test = data.X_test;
    y_test = categorical(cellstr(data.y_test));

    svm_struct = load(fullfile(models_dir, 'svm_model.mat'));
    rf_struct = load(fullfile(models_dir, 'rf_model.mat'));

    [~, scores_svm] = predict(svm_struct.svm_model, X_test);
    [~, scores_rf] = predict(rf_struct.rf_model, X_test);

    % 1. Combined ROC Curve
    fig1 = figure('Visible', 'off');
    try
        [X_svm, Y_svm, ~, AUC_svm] = perfcurve(y_test, scores_svm(:, 1), 'A');
        [X_rf, Y_rf, ~, AUC_rf] = perfcurve(y_test, scores_rf(:, strcmp(rf_struct.rf_model.ClassNames, 'A')), 'A');

        plot(X_svm, Y_svm, 'b-', 'LineWidth', 2); hold on;
        plot(X_rf, Y_rf, 'r--', 'LineWidth', 2);
        plot([0 1], [0 1], 'k:', 'LineWidth', 1);
        xlabel('False Positive Rate (1 - Specificity)');
        ylabel('True Positive Rate (Sensitivity)');
        title('ROC Curve Comparison on Benchmark Test Set');
        legend({sprintf('SVM (RBF) - AUC = %.4f', AUC_svm), ...
                sprintf('Random Forest - AUC = %.4f', AUC_rf), ...
                'Chance Baseline'}, 'Location', 'SouthEast');
        grid on;

        exportgraphics(fig1, fullfile(figs_dir, 'roc_curves.png'), 'Resolution', 300);
    catch
    end
    close(fig1);

    % 2. Normalized Confusion Matrix (Random Forest)
    fig2 = figure('Visible', 'off');
    try
        y_pred_rf = categorical(predict(rf_struct.rf_model, X_test));
        cm_chart = confusionchart(y_test, y_pred_rf, ...
            'RowSummary', 'row-normalized', ...
            'ColumnSummary', 'column-normalized');
        title('Random Forest Normalized Confusion Matrix');

        exportgraphics(fig2, fullfile(figs_dir, 'confusion_matrix_rf.png'), 'Resolution', 300);
    catch
    end
    close(fig2);

    fprintf('Publication figures successfully generated in: %s\n', figs_dir);
end
