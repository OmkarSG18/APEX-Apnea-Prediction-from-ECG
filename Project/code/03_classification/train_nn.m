function train_nn()
    % TRAIN_NN Trains Feedforward Neural Network classifier [64, 32, 16].
    % Saves trained model to results/models/nn_model.mat.

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

    fprintf('Training Feedforward Neural Network [64 32 16]...\n');

    cv = cvpartition(y_train, 'HoldOut', 0.15);
    idx_tr = training(cv);
    idx_val = test(cv);

    X_tr = X_train(idx_tr, :);
    y_tr = y_train(idx_tr);
    X_val = X_train(idx_val, :);
    y_val = y_train(idx_val);

    try
        if exist('fitcnet', 'file') || exist('fitcnet', 'builtin')
            nn_model = fitcnet(X_train, y_train, ...
                'LayerSizes', [64 32 16], ...
                'Activations', 'relu', ...
                'Standardize', false);
        else
            nn_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf');
        end

        out_file = fullfile(models_dir, 'nn_model.mat');
        save(out_file, 'nn_model');
        fprintf('Neural Network model training completed. Saved to: %s\n', out_file);
    catch ME
        fprintf('[WARNING] Neural network fit failed: %s\n', ME.message);
    end
end
