% Q3. Logistic Regression

% Initialisation
clear;
load('spamData.mat'); % Load variables from file into workspace
clc; % clear command window

% Log-transform of Xtrain & Xtest
[xtrain_log, xtest_log] = log_trans(Xtrain, Xtest);
[Xtrain_rows, Xtrain_cols] = size(Xtrain);
[Xtest_rows, Xtest_cols] = size(Xtest);

% Initialize para_w, x_train,x_test
w = zeros(Xtrain_cols + 1, 1);
X_train = ones(Xtrain_rows, 1);
X_test = ones(Xtest_rows, 1);
Xtrain_log_c = [X_train, xtrain_log];
Xtest_log_c = [X_test, xtest_log];

%calculate the training and test error rates VS parameter lamda
para_lamda = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10:5:100]; % given lamda jump in interval from 10 to 15 and beyound

for ai = 1:28 % lamda parameter contains 28 items
    K = K_parameter(ai);
    lamda = para_lamda(ai); 
    u = zeros(Xtrain_rows, 1);
    u_inv = zeros(Xtrain_rows, 1);
    X_comput = Xtrain_log_c';
    judgment = 1; % to enter the first cycle

    % cycle for changing the w with iteration
    while (judgment > 0.0001)

        for i = 1:Xtrain_rows
            u(i, 1) = sigm(w' * X_comput(:, i));
            u_inv(i, 1) = 1 - u(i, 1);
        end

        y = ytrain;
        %w_tmp to make sure w0 not applied lamda
        w_tmp = w;
        w_tmp(1) = 0;
        g = X_comput * (u - y) + lamda * w_tmp;
        H = X_comput * diag(u) * diag(u_inv) * X_comput' + lamda * eye(Xtrain_cols + 1);
        dk = -inv(H) * g;
        wk = w;
        w = w + dk;
        judgment = dk' * dk;
    end

    % calculate the training and test error rate
    train_error = 0; test_error = 0;

    for i = 1:Xtrain_rows
        judge_cls = Xtrain_log_c(i, :) * w;

        if ytrain(i, 1) == 1

            if judge_cls < 0
                train_error = train_error + 1;
            end

        else

            if judge_cls > 0
                train_error = train_error + 1;
            end

        end

    end

    for i = 1:Xtest_rows
        judge_cls = Xtest_log_c(i, :) * w;

        if ytest(i, 1) == 1

            if judge_cls < 0
                test_error = test_error + 1;
            end

        else

            if judge_cls > 0
                test_error = test_error + 1;
            end

        end

    end

    train_error_rate(ai) = train_error / Xtrain_rows;
    test_error_rate(ai) = test_error / Xtest_rows;
    train_error_rate_percentage(ai) = train_error_rate(ai) * 100;
    test_error_rate_percentage(ai) = test_error_rate(ai) * 100;
end

% plot
plot(para_lamda, train_error_rate, 'black', para_lamda, test_error_rate, 'red')
grid on;
title('Error Changing Rate of Training and Test Error vs Lambda');
xlabel('Lambda');
ylabel('Error Rate');
legend('Training Error Rate', 'Test Error Rate', 'Location', 'NorthWest');
%output
% fprintf('lamda= %d,    Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', para_lamda(1), train_error_rate(1), test_error_rate(1))
% fprintf('lamda= %d,   Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', para_lamda(10), train_error_rate(10), test_error_rate(10))
% fprintf('lamda= %d,  Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', para_lamda(28), train_error_rate(28), test_error_rate(28))
fprintf('lamda= %d,   Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', para_lamda(1), train_error_rate_percentage(1), test_error_rate_percentage(1))
fprintf('lamda= %d,  Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', para_lamda(10), train_error_rate_percentage(10), test_error_rate_percentage(10))
fprintf('lamda= %d, Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', para_lamda(28), train_error_rate_percentage(28), test_error_rate_percentage(28))
%=============== end of program =======================

%======================================================
% Function: sigm function
%======================================================
function [s] = sigm(x)
    s = 1 / (1 + exp(-x));
end

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
