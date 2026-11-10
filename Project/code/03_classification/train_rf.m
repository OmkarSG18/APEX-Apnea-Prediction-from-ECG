function train_rf()
    % TRAIN_RF Trains Random Forest classifier with 100 decision trees.
    % Saves trained model to results/models/rf_model.mat and Gini importance to results/tables/rf_importance.csv.

    results_dir = find_results_dir();
    train_file = fullfile(results_dir, 'feature_matrix_train.mat');
    models_dir = fullfile(results_dir, 'models');
    tables_dir = fullfile(results_dir, 'tables');
    if ~exist(models_dir, 'dir')
        mkdir(models_dir);
    end
    if ~exist(tables_dir, 'dir')
        mkdir(tables_dir);
    end

    if ~exist(train_file, 'file')
        error('Training feature matrix not found: %s. Run build_feature_matrix first.', train_file);
    end

    data = load(train_file);
    X_train = data.X_train;
    y_train = cellstr(data.y_train);
    feature_names = cellstr(data.feature_names);

    fprintf('Training Random Forest Classifier (100 Trees)...\n');

    num_trees = 100;
    rf_model = TreeBagger(num_trees, X_train, y_train, ...
        'Method', 'classification', ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'MinLeafSize', 5);

    importance = rf_model.OOBPermutedPredictorDeltaError;

    out_file = fullfile(models_dir, 'rf_model.mat');
    save(out_file, 'rf_model');
    fprintf('Random Forest training completed. Saved to: %s\n', out_file);

    csv_file = fullfile(tables_dir, 'rf_importance.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Feature,Importance_Score\n');
    for i = 1:length(feature_names)
        fprintf(fid, '%s,%.6f\n', feature_names{i}, importance(i));
    end
    fclose(fid);
    fprintf('Feature importance saved to: %s\n', csv_file);
end
