% Q2. Gaussian Naïve Bayes

% Initialisation
clear;
load('spamData.mat'); % Load variables from file into workspace
clc; % clear command window

% Log-transform of Xtrain & Xtest
[xtrain_log, xtest_log] = log_trans(Xtrain, Xtest);
[Xtrain_rows, Xtrain_cols] = size(Xtrain);
[Xtest_rows, Xtest_cols] = size(Xtest);

% MLE of ytrain
% Nc_y1: count of y=1 within XTrain_rows
% Nc_y0: count of y=0 within XTrian_rows
Nc_y1 = 0;

for i = 1:Xtrain_rows

    if ytrain(i, 1) == 1
        Nc_y1 = Nc_y1 + 1;
    end

end

Nc_y0 = Xtrain_rows - Nc_y1;
PAIc_y1 = Nc_y1 / Xtrain_rows;
PAIc_y0 = Nc_y0 / Xtrain_rows;

% MLE of miu & sigma
for j = 1:Xtrain_cols
    sum_y1 = 0; num_x_y1 = 0;
    sum_y0 = 0; num_x_y0 = 0;

    for i = 1:Xtrain_cols

        if ytrain(i, 1) == 1
            sum_y1 = sum_y1 + xtrain_log(i, j);
            num_x_y1 = num_x_y1 + 1;
        else
            sum_y0 = sum_y0 + xtrain_log(i, j);
            num_x_y0 = num_x_y0 + 1;
        end

    end

    miu_y1(1, j) = sum_y1 / num_x_y1;
    miu_y0(1, j) = sum_y0 / num_x_y0;
    sum_y1 = 0; num_x_y1 = 0;
    sum_y0 = 0; num_x_y0 = 0;

    for i = 1:Xtrain_rows

        if ytrain(i, 1) == 1
            sum_y1 = sum_y1 + (xtrain_log(i, j) - miu_y1(1, j)) * (xtrain_log(i, j) - miu_y1(1, j));
            num_x_y1 = num_x_y1 + 1;
        else
            sum_y0 = sum_y0 + (xtrain_log(i, j) - miu_y0(1, j)) * (xtrain_log(i, j) - miu_y0(1, j));
            num_x_y0 = num_x_y0 + 1;
        end

    end

    sigma_y1(1, j) = (sum_y1 / num_x_y1)^0.5;
    sigma_y0(1, j) = (sum_y0 / num_x_y0)^0.5;
end

%calculate the training and test error
logPr_y1 = log(PAIc_y1) * ones(Xtrain_rows, 1);
logPr_y0 = log(PAIc_y0) * ones(Xtrain_rows, 1);
train_error = 0;
test_error = 0;

for i = 1:Xtrain_rows

    for j = 1:Xtrain_cols
        logPr_y1(i, 1) = logPr_y1(i, 1) + log(normpdf(xtrain_log(i, j), miu_y1(1, j), sigma_y1(1, j)));
        logPr_y0(i, 1) = logPr_y0(i, 1) + log(normpdf(xtrain_log(i, j), miu_y0(1, j), sigma_y0(1, j)));
    end

    if ytrain(i, 1) == 1

        if logPr_y1(i, 1) < logPr_y0(i, 1)
            train_error = train_error + 1;
        end

    elseif ytrain(i, 1) == 0

        if logPr_y1(i, 1) > logPr_y0(i, 1)
            train_error = train_error + 1;
        end

    end

end

logPr_y1 = log(PAIc_y1) * ones(Xtest_rows, 1);
logPr_y0 = log(PAIc_y0) * ones(Xtest_rows, 1);

for i = 1:Xtest_rows

    for j = 1:Xtest_cols
        logPr_y1(i, 1) = logPr_y1(i, 1) + log(normpdf(xtest_log(i, j), miu_y1(1, j), sigma_y1(1, j)));
        logPr_y0(i, 1) = logPr_y0(i, 1) + log(normpdf(xtest_log(i, j), miu_y0(1, j), sigma_y0(1, j)));
    end

    if ytest(i, 1) == 1

        if logPr_y1(i, 1) < logPr_y0(i, 1)
            test_error = test_error + 1;
        end

    elseif ytest(i, 1) == 0

        if logPr_y1(i, 1) > logPr_y0(i, 1)
            test_error = test_error + 1;
        end

    end

end

train_error_rate = train_error / Xtrain_rows;
train_error_rate_percentage = train_error_rate * 100;
test_error_rate = test_error / Xtest_rows;
test_error_rate_percentage = test_error_rate * 100;
%output
% fprintf('Traing Error Rate = %.6f\n', train_error_rate);
% fprintf('Test Error Rate   = %.6f\n', test_error_rate);
fprintf('Traing Error Rate = %.4f %%\n', train_error_rate_percentage);
fprintf('Test Error Rate   = %.4f %%\n', test_error_rate_percentage);
%=============== end of program =======================

%======================================================
% Function of Log Transform
% transform each feature using log(xij + 0.1)
%======================================================
function [xtrain_log, xtest_log] = log_trans(Xtrain, Xtest)
    % Initialization
    Xtrain_cols = size(Xtrain, 2);
    Xtrain_rows = size(Xtrain, 1);
    xtrain_log = zeros(Xtrain_rows, Xtrain_cols);
    Xtest_cols = size(Xtest, 2);
    Xtest_rows = size(Xtest, 1);
    xtest_log = zeros(Xtest_rows, Xtest_cols);
    % log-transform of Xtrain
    for j = 1:Xtrain_cols

        for i = 1:Xtrain_rows
            xtrain_log(i, j) = log(Xtrain(i, j) + 0.1);
        end

    end

    % log-transform of Xtest
    for j = 1:Xtest_cols

        for i = 1:Xtest_rows
            xtest_log(i, j) = log(Xtest(i, j) + 0.1);
        end

    end

end
