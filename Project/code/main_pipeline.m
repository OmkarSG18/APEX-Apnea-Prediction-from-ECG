function main_pipeline()
    % MAIN_PIPELINE Master execution pipeline for Sleep Apnea Detection from Single-Lead ECG.
    %
    % Steps:
    %   1. Environment setup & verification
    %   2. Preprocessing & R-peak validation
    %   3. Feature extraction & normalization
    %   4. Model training (SVM, Random Forest, Neural Network)
    %   5. Comparative evaluation, ablation study, statistical tests & figures

    fprintf('\n======================================================================\n');
    fprintf('  Sleep Apnea ECG Comparative ML Pipeline (SDG 3 Alignment)\n');
    fprintf('======================================================================\n\n');

    % Add subdirectories to MATLAB Path
    addpath(genpath('.'));

    % Step 1: Environment Check
    fprintf('--- STEP 1: Environment Setup Check ---\n');
    check_environment();

    % Step 2: Preprocessing & R-peak Validation
    fprintf('\n--- STEP 2: Preprocessing & R-Peak Validation ---\n');
    validate_rpeaks();

    % Step 3: Feature Matrix Assembly
    fprintf('\n--- STEP 3: Feature Matrix Assembly ---\n');
    build_feature_matrix();

    % Step 4: Classifier Model Training
    fprintf('\n--- STEP 4: Classifier Model Training ---\n');
    train_svm();
    train_rf();
    train_nn();

    % Step 5: Comparative Evaluation & Figures
    fprintf('\n--- STEP 5: Evaluation & Figures Generation ---\n');
    run_crossval();
    feature_analysis();
    statistical_tests();
    make_figures();
    external_validation();

    fprintf('\n======================================================================\n');
    fprintf('  Pipeline execution finished successfully.\n');
    fprintf('  Results exported to results/ directory.\n');
    fprintf('======================================================================\n');
end
