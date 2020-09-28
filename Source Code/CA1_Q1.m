% Q1. Beta-binomial Naïve Bayes

% Initialisation
clear;
load('spamData.mat'); % Load variables from file into workspace
clc; % clear command window

% Binarization of Xtrain & Xtest
[xTrain_rows, xTrain_cols] = size(Xtrain);
[Xtest_rows, Xtest_cols] = size(Xtest);
[xtrain_binary, xtest_binary] = binarization(Xtrain, Xtest);

% MLE of ytrain
% Nc_y1: count of y=1 within XTrain_rows
% Nc_y0: count of y=0 within XTrian_rows
Nc_y1 = 0;

for i = 1:xTrain_rows

    if ytrain(i, 1) == 1
        Nc_y1 = Nc_y1 + 1;
    end

end

Nc_y0 = xTrain_rows - Nc_y1;
PAIc_y1 = Nc_y1 / xTrain_rows;
PAIc_y0 = Nc_y0 / xTrain_rows;
%======================================================
% Calculate the error rate VS different parameter alpha
% Assume prior Beta with hyperparameter
alpha = 0:0.5:100; % alpha = {0,0.5,1,1.5,2,...,100}

for a_counter = 1:201% a_couter: counter for alpha
    a = alpha(a_counter);
    % Initialise zero matrix for thera and Njc
    % j: dimension， c: class
    Njc_y1 = zeros(1, xTrain_cols);
    Njc_y0 = zeros(1, xTrain_cols);
    thetajc_y1 = zeros(xTrain_rows, 1);
    thetajc_y0 = zeros(xTrain_rows, 1);
    logPr_y1 = log(PAIc_y1) * ones(xTrain_rows, 1);
    logPr_y0 = log(PAIc_y0) * ones(xTrain_rows, 1);
    logPr_y1_test = log(PAIc_y1) * ones(Xtest_rows, 1);
    logPr_y0_test = log(PAIc_y0) * ones(Xtest_rows, 1);
    train_error = 0; % init train_error value as 0
    test_error = 0; % init test_error value as 0

    % calculate the theta and Njc
    for j = 1:xTrain_cols

        for i = 1:xTrain_rows

            if ytrain(i, 1) == 1

                if xtrain_binary(i, j) == 1
                    % calculate Njc_y1.ie P(x=1|y=1,T)
                    Njc_y1(1, j) = Njc_y1(1, j) + 1;
                end

            elseif ytrain(i, 1) == 0

                if xtrain_binary(i, j) == 1
                    % calculate Njc_y1.ie P(x=1|y=0,T)
                    Njc_y0(1, j) = Njc_y0(1, j) + 1;
                end

            end

        end

        % calculate thetajc_y
        thetajc_y1(1, j) = (Njc_y1(1, j) + a) / (Nc_y1 + a + a);
        thetajc_y0(1, j) = (Njc_y0(1, j) + a) / (Nc_y0 + a + a);
    end

    % calculate training error rate
    for i = 1:xTrain_rows

        for j = 1:xTrain_cols

            if xtrain_binary(i, j) == 1
                logPr_y1(i, 1) = logPr_y1(i, 1) + log(thetajc_y1(1, j));
                logPr_y0(i, 1) = logPr_y0(i, 1) + log(thetajc_y0(1, j));
            elseif xtrain_binary(i, j) == 0
                logPr_y1(i, 1) = logPr_y1(i, 1) + log(1 - thetajc_y1(1, j));
                logPr_y0(i, 1) = logPr_y0(i, 1) + log(1 - thetajc_y0(1, j));
            end

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

    % calculate test error rate
    for i = 1:Xtest_rows

        for j = 1:Xtest_cols

            if xtest_binary(i, j) == 1
                logPr_y1_test(i, 1) = logPr_y1_test(i, 1) + log(thetajc_y1(1, j));
                logPr_y0_test(i, 1) = logPr_y0_test(i, 1) + log(thetajc_y0(1, j));
            elseif xtest_binary(i, j) == 0
                logPr_y1_test(i, 1) = logPr_y1_test(i, 1) + log(1 - thetajc_y1(1, j));
                logPr_y0_test(i, 1) = logPr_y0_test(i, 1) + log(1 - thetajc_y0(1, j));
            end

        end

        if ytest(i, 1) == 1

            if logPr_y1_test(i, 1) < logPr_y0_test(i, 1)
                test_error = test_error + 1;
            end

        elseif ytest(i, 1) == 0

            if logPr_y1_test(i, 1) > logPr_y0_test(i, 1)
                test_error = test_error + 1;
            end

        end

    end

    train_error_rate(a_counter) = train_error / xTrain_rows;
    train_error_rate_percentage(a_counter) = train_error_rate(a_counter) * 100;
    test_error_rate(a_counter) = test_error / Xtest_rows;
    test_error_rate_percentage(a_counter) = test_error_rate(a_counter) * 100;
end

% plot
plot(alpha, train_error_rate, 'black', alpha, test_error_rate, 'red')
grid on;
title('Error Changing Rate of Training and Test Error vs Alpha');
xlabel('Alpha');
ylabel('Error Rate');
legend('Training Error Rate', 'Test Error Rate', 'Location', 'NorthWest');
% output : alpha(i x 2 +1) where, i=1,10,100
% fprintf('a= %d,   Traing Error Rate=%.6f, Test Error Rate=%.6f\n', alpha(3), train_error_rate(3), test_error_rate(3))
% fprintf('a= %d,  Traing Error Rate=%.6f, Test Error Rate=%.6f\n', alpha(21), train_error_rate(21), test_error_rate(21))
% fprintf('a= %d, Traing Error Rate=%.6f, Test Error Rate=%.6f\n', alpha(201), train_error_rate(201), test_error_rate(201))
fprintf('a= %d,   Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', alpha(3), train_error_rate_percentage(3), test_error_rate_percentage(3))
fprintf('a= %d,  Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', alpha(21), train_error_rate_percentage(21), test_error_rate_percentage(21))
fprintf('a= %d, Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', alpha(201), train_error_rate_percentage(201), test_error_rate_percentage(201))
%=============== end of program =======================

%======================================================
% Function of binarization
% Input: Xtrain (i,j), Xtest (i,j) I ( xij > 0 )
%        If feature >  0, set to 1
%        If feature <= 0, set to 0
% Return: Binarized Xtrian (i,j), Xtest(i,j)
%======================================================
function [xtrain_binary, xtest_binary] = binarization(Xtrain, Xtest)
    % xTest         % Rows: 3065      Columns: 57
    % xTrain        % Rows: 1536      Columns: 57
    [xTrain_rows, xTrain_cols] = size(Xtrain);
    [Xtest_rows, Xtest_cols] = size(Xtest);
    xtrain_binary = zeros(xTrain_rows, xTrain_cols);
    xtest_binary = zeros(Xtest_rows, Xtest_cols);
    % Binarization of xTrain
    for i = 1:xTrain_rows

        for j = 1:xTrain_cols

            if Xtrain(i, j) > 0
                xtrain_binary(i, j) = 1;
            else
                xtrain_binary(i, j) = 0;
            end

        end

    end

    % Binarization of xTest
    for i = 1:Xtest_rows

        for j = 1:Xtest_cols

            if Xtest(i, j) > 0
                xtest_binary(i, j) = 1;
            else
                xtest_binary(i, j) = 0;
            end

        end

    end

end
