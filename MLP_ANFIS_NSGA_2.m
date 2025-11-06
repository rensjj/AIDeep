%% ------------------------------------------------------------------------
% 项目：基于 MLP+ANFIS 混合模型和 NSGA-II 的多目标优化
% 平台：MATLAB 2023b
% -------------------------------------------------------------------------

clear;
clc;
close all;

% 设置随机种子以保证结果可复现
rng(42); 


    data = readtable('LE-P-h-E-Date.xlsx');
    disp('已加载真实数据文件 "LE-P-h-E-Date.xlsx"。');
    data = data{:,:}; % 转换为矩阵


disp('--- 数据加载与预处理开始 ---');


%% ------------------------------------------------------------------------
% 第 1 节：数据加载、预处理与划分

disp('--- 数据加载与预处理开始 ---');
% 1.1 加载数据
if istable(data)
    data = data{:,:}; 
end

X = data(:, 1:3); % 输入: LE, P, h
Y = data(:, 4);   % 输出: Eav

% 1.2 预处理：检查异常值 
validIdx = (Y > 10) & (X(:,2) > 0); 
X = X(validIdx, :);
Y = Y(validIdx, :);

% 1.3 划分数据 
N_total = size(X, 1);
cv_train = cvpartition(N_total, 'HoldOut', 0.3); 
idxTrain = training(cv_train);
idxTestValid = test(cv_train);
cv_test = cvpartition(sum(idxTestValid), 'HoldOut', 0.5); 
idxValid_rel = training(cv_test);
idxTest_rel = test(cv_test);
idxValid = find(idxTestValid);
idxValid = idxValid(idxValid_rel);
idxTest = find(idxTestValid);
idxTest = idxTest(idxTest_rel);

X_train_orig = X(idxTrain, :);
Y_train_orig = Y(idxTrain, :);
X_val_orig = X(idxValid, :);
Y_val_orig = Y(idxValid, :);
X_test_orig = X(idxTest, :);
Y_test_orig = Y(idxTest, :);

% 噪声系数 
noise_factor = 0.001; 

% 计算每个特征的范围 (Max - Min)
range_LE = max(X(:,1)) - min(X(:,1)); 
range_P  = max(X(:,2)) - min(X(:,2));
range_h  = max(X(:,3)) - min(X(:,3));

% 计算每个特征的噪声标准差 (σ)
noise_std_LE = range_LE * noise_factor;
noise_std_P  = range_P  * noise_factor;
noise_std_h  = range_h  * noise_factor;

% 生成并添加高斯噪声
noise_matrix = [ ...
    noise_std_LE * randn(size(X_train_orig, 1), 1), ...
    noise_std_P * randn(size(X_train_orig, 1), 1), ...
    noise_std_h * randn(size(X_train_orig, 1), 1)  ...
];

X_train_noise = X_train_orig + noise_matrix;

% 确保噪声后的数据仍在原始的物理边界内
X_train_noise(:, 1) = max(min(X_train_noise(:, 1), max(X(:,1))), min(X(:,1)));
X_train_noise(:, 2) = max(min(X_train_noise(:, 2), max(X(:,2))), min(X(:,2)));
X_train_noise(:, 3) = max(min(X_train_noise(:, 3), max(X(:,3))), min(X(:,3)));

% 整合回原始数据集，准备进行归一化
X(idxTrain, :) = X_train_noise; 
Y(idxTrain, :) = Y_train_orig; 
% ------------------------------------------------------------------

% 1.4 归一化: 对整个数据集（包含带噪声的训练集）进行 [0, 1] 归一化
[X_norm, scalerX] = mapminmax(X'); 
scalerX.ymin = 0; scalerX.ymax = 1;
X_norm = mapminmax('apply', X', scalerX)';

[Y_norm, scalerY] = mapminmax(Y');
scalerY.ymin = 0; scalerY.ymax = 1;
Y_norm = mapminmax('apply', Y', scalerY)';

% 1.5 重新划分归一化后的数据
X_train = X_norm(idxTrain, :);
Y_train = Y_norm(idxTrain, :);
X_val = X_norm(idxValid, :);
Y_val = Y_norm(idxValid, :);
X_test = X_norm(idxTest, :);
Y_test = Y_norm(idxTest, :);


fprintf('数据划分: 训练集=%d, 验证集=%d, 测试集=%d\n', ...
    length(Y_train), length(Y_val), length(Y_test));
fprintf('噪声标准差 (LE/P/h) 约为: [%.4f, %.4f, %.4f]\n', ...
    noise_std_LE, noise_std_P, noise_std_h);

disp('--- 数据预处理完成 ---');


%% ------------------------------------------------------------------------
% 第 2 节：MLP + ANFIS 混合预测模型构建 
% -------------------------------------------------------------------------
% ---------------------------------
% 2.1 阶段一：MLP 预测模型
% ---------------------------------
disp('--- 阶段一：MLP 模型训练开始 ---');
hiddenLayer1Size = 10;
hiddenLayer2Size = 6;
net = feedforwardnet([hiddenLayer1Size, hiddenLayer2Size]);
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';

% 配置训练数据
X_train_val = [X_train; X_val]';
Y_train_val = [Y_train; Y_val]';
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:length(Y_train);
net.divideParam.valInd = (length(Y_train)+1):length(Y_train_val);
net.divideParam.testInd = [];

% 添加 L2 正则化
net.performFcn = 'mse'; 
net.performParam.regularization = 0.2; % 设置 L2 正则化系数
% ------------------------------------

net.trainParam.showWindow = false; 
[mlp_model, tr] = train(net, X_train_val, Y_train_val);
disp('--- MLP 模型训练完成 ---');

% ---------------------------------
% 2.2 阶段二：ANFIS 残差修正模型
% ---------------------------------
disp('--- 阶段二：ANFIS 残差修正模型训练开始 ---');
Y_train_orig = mapminmax('reverse', Y_train', scalerY)';
Y_mlp_norm_train = mlp_model(X_train');
Y_mlp_orig_train = mapminmax('reverse', Y_mlp_norm_train, scalerY)';
Residuals_train = Y_train_orig - Y_mlp_orig_train;
X_train_orig = mapminmax('reverse', X_train', scalerX)';
anfis_train_data = [X_train_orig, Residuals_train];

ruleNum = 3; 
opt = genfisOptions('FCMClustering', 'NumClusters', ruleNum);
fis = genfis(X_train_orig, Residuals_train, opt);

trainOptions = [50, 0, 0.01, 0.9, 1.1];
anfis_model = anfis(anfis_train_data, fis, trainOptions);
disp('--- ANFIS 模型训练完成 ---');

% ---------------------------------
% 2.3 模型性能评估 (使用测试集)
% ---------------------------------
disp('--- 模型性能评估 (测试集) ---');
X_test_orig = mapminmax('reverse', X_test', scalerX)';
Y_test_orig = mapminmax('reverse', Y_test', scalerY)';

Y_mlp_norm_test = mlp_model(X_test');
Y_mlp_pred = mapminmax('reverse', Y_mlp_norm_test, scalerY)';

Delta_E_pred = evalfis(anfis_model, X_test_orig);
Y_hybrid_pred = Y_mlp_pred + Delta_E_pred;

metrics = @(true, pred) ...
    struct('RMSE', sqrt(mean((true - pred).^2)), ...
           'MAE', mean(abs(true - pred)), ...
           'R2', 1 - sum((true - pred).^2) / sum((true - mean(true)).^2));

metrics_mlp = metrics(Y_test_orig, Y_mlp_pred);
metrics_hybrid = metrics(Y_test_orig, Y_hybrid_pred);

fprintf('MLP 单独模型:\n\tRMSE: %.4f\n\tMAE: %.4f\n\tR2: %.4f\n', ...
    metrics_mlp.RMSE, metrics_mlp.MAE, metrics_mlp.R2);
fprintf('MLP+ANFIS 混合模型:\n\tRMSE: %.4f\n\tMAE: %.4f\n\tR2: %.4f\n', ...
    metrics_hybrid.RMSE, metrics_hybrid.MAE, metrics_hybrid.R2);

% (F) 绘制对比图 (R2 散点图)
figure('Name', '模型预测性能对比');
subplot(1, 2, 1);
scatter(Y_test_orig, Y_mlp_pred, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(Y_test_orig), max(Y_test_orig)], [min(Y_test_orig), max(Y_test_orig)], 'r--', 'LineWidth', 1.5);
grid on; xlabel('真实 Eav (lx)'); ylabel('预测 Eav (lx)');
title(sprintf('MLP (R^2 = %.4f)', metrics_mlp.R2)); axis square;

subplot(1, 2, 2);
scatter(Y_test_orig, Y_hybrid_pred, 30, 'g', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(Y_test_orig), max(Y_test_orig)], [min(Y_test_orig), max(Y_test_orig)], 'r--', 'LineWidth', 1.5);
grid on; xlabel('真实 Eav (lx)'); ylabel('预测 Eav (lx)');
title(sprintf('MLP+ANFIS (R^2 = %.4f)', metrics_hybrid.R2)); axis square;
sgtitle('模型预测性能对比 (测试集)');
saveas(gcf, 'model_performance_comparison.png');


% 预测值 vs 真实值曲线 (图一：MLP 独立模型)
figure('Name', 'MLP 独立模型预测曲线');
% 真实值：蓝色实线
plot(Y_test_orig, 'b-', 'LineWidth', 1.5, 'DisplayName', '真实值');
hold on;
% MLP 预测值：红色虚线 (与混合模型区分线型)
plot(Y_mlp_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'MLP 预测值');
grid on;
xlabel('测试集样本索引');
ylabel('Eav (lx)');
title('MLP 独立模型预测值 vs 真实值');
legend('show', 'Location', 'best');
saveas(gcf, 'prediction_curve_MLP_only.png');


% 预测值 vs 真实值曲线 (图二：MLP+ANFIS 混合模型)
figure('Name', 'MLP+ANFIS 混合模型预测曲线');
% 真实值：蓝色实线
plot(Y_test_orig, 'b-', 'LineWidth', 1.5, 'DisplayName', '真实值');
hold on;
% 混合模型预测值：红色实线
plot(Y_hybrid_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'MLP+ANFIS 预测值');
grid on;
xlabel('测试集样本索引');
ylabel('Eav (lx)');
title('MLP+ANFIS 混合模型预测值 vs 真实值');
legend('show', 'Location', 'best');
saveas(gcf, 'prediction_curve_Hybrid_only.png');


%% ------------------------------------------------------------------------
% 第 3 节：多目标优化 (NSGA-II)
% -------------------------------------------------------------------------
disp('--- NSGA-II 多目标优化开始 ---');

% 3.1 定义决策变量边界
nvars = 3;
LB = [100, 20, 2.8]; 
UB = [160, 60, 3.2]; 

% 3.2 定义混合模型预测函数
predict_Eav_hybrid = @(x, mlp_model, anfis_model, scalerX, scalerY) ...
    predictHybrid(x, mlp_model, anfis_model, scalerX, scalerY);

% 3.3 定义目标函数
objective_function = @(x) objFun(x, predict_Eav_hybrid, mlp_model, anfis_model, scalerX, scalerY);

% 3.4 定义非线性约束
constraint_function = @(x) conFun(x, predict_Eav_hybrid, mlp_model, anfis_model, scalerX, scalerY);

% 3.5 NSGA-II 算法设置
options = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 200, ...
    'CrossoverFraction', 0.8, ...
    'MutationFcn', {@mutationgaussian, 1, 1}, ...
    'Display', 'iter', ...
    'PlotFcn', @gaplotpareto); 

% 3.6 运行 NSGA-II
[x_pareto, f_pareto, exitflag, output] = gamultiobj(...
    objective_function, nvars, ...
    [], [], [], [], LB, UB, ...
    constraint_function, options);

disp('--- NSGA-II 优化完成 ---');


%% ------------------------------------------------------------------------
% 第 4 节：结果分析、保存与可视化
% -------------------------------------------------------------------------

% 4.1 保存模型、缩放器和日志
disp('正在保存模型和结果...');
results_log.random_seed = 42;
results_log.mlp_metrics = metrics_mlp;
results_log.hybrid_metrics = metrics_hybrid;
results_log.nsga_output = output;

save('Hybrid_Optimization_Workspace.mat', ...
    'mlp_model', 'anfis_model', 'scalerX', 'scalerY', ...
    'x_pareto', 'f_pareto', 'results_log');

% 4.2 整理并保存 Pareto 解集
Eav_pareto = zeros(size(x_pareto, 1), 1);
for i = 1:size(x_pareto, 1)
    Eav_pareto(i) = predict_Eav_hybrid(x_pareto(i, :), mlp_model, anfis_model, scalerX, scalerY);
end
Pareto_solutions = [x_pareto, f_pareto, Eav_pareto];
Pareto_T = array2table(Pareto_solutions, 'VariableNames', ...
    {'LE', 'P', 'h', 'Obj_Power', 'Obj_Eav_Deviation', 'Obj_Cost', 'Predicted_Eav'});
writetable(Pareto_T, 'Pareto_Solutions_NSGA-II.xlsx');


% 4.3 选取折中解 (TOPSIS)
if ~isempty(f_pareto)
    f_norm = mapminmax(f_pareto')';
    distances = sqrt(sum(f_norm.^2, 2));
    [min_dist, idx_topsis] = min(distances);
    best_solution_topsis = Pareto_T(idx_topsis, :);
    disp('--- 最佳折中解 (Topsis) ---');
    disp(best_solution_topsis);
else
    disp('未找到 Pareto 解，无法选取最佳解。');
    idx_topsis = [];
end


% 4.4 绘制 3D 帕累托前沿散点图 (带颜色和最佳解高亮)
figure('Name', 'NSGA-II 帕累托前沿 (高亮最佳解)');

if ~isempty(f_pareto)
    % 使用 决策变量 P (x_pareto(:, 2)) 作为颜色数据
    color_data = x_pareto(:, 2); 
    
    % 绘制所有 Pareto 解
    scatter3(f_pareto(:,1), f_pareto(:,2), f_pareto(:,3), ...
        50, color_data, 'filled', 'MarkerFaceAlpha', 0.7);
    
    hold on;
    
    % 如果找到了最佳解，则高亮显示
    if ~isempty(idx_topsis)
        best_f = f_pareto(idx_topsis, :);
        plot3(best_f(1), best_f(2), best_f(3), ...
            'r*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', '最佳折中解 (Topsis)');
    end
    
    xlabel('f1: 功率 (W)');
    ylabel('f2: 照度上限偏差 max(0, Eav-550) (lx)');
    zlabel('f3: 初始成本 (RMB)');
    title('NSGA-II 帕累托前沿解集 (颜色: 功率 P)');
    grid on;
    
    % 添加颜色条
    cb = colorbar;
    ylabel(cb, '功率 P (W)');
    
    % 添加图例
    legend('show', 'Location', 'best');
    
    hold off;
else
    title('NSGA-II 未找到帕累托解');
end
saveas(gcf, 'Pareto_Front_3D_Color_Highlight.png');


% 4.5 帕累托质量指标 (超体积 - Hypervolume)
if ~isempty(f_pareto)
    [~, num_objectives] = size(f_pareto);
    if num_objectives < 3
        fprintf(2, '警告：帕累托前沿目标维度不足 (%d)。无法计算 3 目标参考点。\n', num_objectives);
    else
        ref_point = max(f_pareto) * 1.1;
        fprintf('帕累托超体积 (HV) 计算参考点: [%.2f, %.2f, %.2f]\n', ...
            ref_point(1), ref_point(2), ref_point(3));
        disp('注意: MATLAB 核心库中没有 hypervolume 函数，计算需额外脚本。');
    end
else
    disp('未找到 Pareto 解，无法计算 HV。');
end

disp('--- 所有任务已完成 ---');


%% ------------------------------------------------------------------------
% 辅助函数定义
% -------------------------------------------------------------------------

% 辅助函数 1：混合模型预测
function E_pred = predictHybrid(x, mlp_model, anfis_model, scalerX, scalerY)
    x_norm = mapminmax('apply', x', scalerX)';
    E_mlp_norm = mlp_model(x_norm');
    E_mlp_orig = mapminmax('reverse', E_mlp_norm, scalerY)';
    Delta_E = evalfis(anfis_model, x);
    E_pred = E_mlp_orig + Delta_E;
end

% 辅助函数 2：NSGA-II 目标函数 
function F = objFun(x, predict_Eav_hybrid, mlp_model, anfis_model, scalerX, scalerY)
    LE = x(1);
    P = x(2);
    Eav_pred = predict_Eav_hybrid(x, mlp_model, anfis_model, scalerX, scalerY);
    
    % (A) 目标 1：min(P)
    f1 = P;
    
    % (B) 目标 2：min(|Eav - 500|)
    target_Eav_center = 500;
    f2 = abs(Eav_pred - target_Eav_center);
    
    % (C) 目标 3：min(初始成本 - Initial Cost)
    a = 200;   % LE 的权重 (高光效高成本)
    b = 5;     % P 的权重 
    c = 1000;  % 固定成本
    
    f3 = a * LE + b * P + c;
    
    F = [f1, f2, f3];
end

% 辅助函数 3：NSGA-II 约束函数
function [c_ineq, c_eq] = conFun(x, predict_Eav_hybrid, mlp_model, anfis_model, scalerX, scalerY)
    Eav_pred = predict_Eav_hybrid(x, mlp_model, anfis_model, scalerX, scalerY);
    
    % 不等式约束 (硬性下限): Eav >= 500  =>  500 - Eav <= 0
    c_ineq = 500 - Eav_pred;
    
    % 等式约束 (无)
    c_eq = [];
end