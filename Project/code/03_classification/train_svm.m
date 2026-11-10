function train_svm()
    % TRAIN_SVM Trains SVM classifier with RBF kernel.
    % Saves trained model to results/models/svm_model.mat.

    results_dir = find_results_dir();
    train_file = fullfile(results_dir, 'feature_matrix_train.mat');
    models_dir = fullfile(results_dir, 'models');
    if ~exist(models_dir, 'dir')
        mkdir(models_dir);
    end

    if ~exist(train_file, 'file')
        error('Training feature matrix not found: %s. Run build_feature_matrix first.', train_file);
    end

    data = load(train_file);
    X_train = data.X_train;
    y_train = categorical(cellstr(data.y_train));

    if numel(categories(y_train)) < 2
        error('y_train must contain at least 2 distinct classes (A and N). Got %d classes.', numel(categories(y_train)));
    end

    fprintf('Training Support Vector Machine (SVM) with RBF Kernel...\n');

    try
        svm_model = fitcsvm(X_train, y_train, ...
            'KernelFunction', 'rbf', ...
            'KernelScale', 'auto', ...
            'BoxConstraint', 1.0, ...
            'ClassNames', {'A', 'N'}, ...
            'Standardize', false);
    catch
        % Fallback standard SVM fit
        svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf');
    end

    out_file = fullfile(models_dir, 'svm_model.mat');
    save(out_file, 'svm_model');
    fprintf('SVM model training completed. Saved to: %s\n', out_file);
end
