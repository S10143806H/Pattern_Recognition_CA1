% Q4. K-Nearest Neighbours

% Initialisation
clear;
load('spamData.mat'); % Load variables from file into workspace
clc; % clear command window

% Log-transform of Xtrain & Xtest
[xtrain_log, xtest_log] = log_trans(Xtrain, Xtest);
[Xtrain_rows, Xtrain_cols] = size(Xtrain);
[Xtest_rows, Xtest_cols] = size(Xtest);

% compute the training and test error rates vs K
K_parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10:5:100];

for ai = 1:28% k_parameter contains 28 items
    K = K_parameter(ai);
    %calculate training and test error rate
    dis_train = dis_matrix(xtrain_log, xtrain_log);
    dis_test = dis_matrix(xtest_log, xtrain_log);
    k_nearst = zeros(K, 2); % (distance, sequence)
    k_nearst = zeros(K, 2);
    train_error = 0; test_error = 0;

    for j = 1:Xtrain_rows
        [B, IX] = sort(dis_train(:, j));

        for ik = 1:K

            k_nearst(ik, 1) = B(ik);
            k_nearst(ik, 2) = IX(ik);

        end

        y0_knear = 0;
        y1_knear = 0;

        for ik = 1:K

            if ytrain(k_nearst(ik, 2), 1) == 1
                y1_knear = y1_knear + 1;
            else
                y0_knear = y0_knear + 1;
            end

        end

        if ytrain(j, 1) == 1

            if y1_knear < y0_knear
                train_error = train_error + 1;
            end

        else

            if y1_knear > y0_knear
                train_error = train_error + 1;
            end

        end

    end

    for j = 1:Xtest_rows
        [B, IX] = sort(dis_test(:, j));

        for ik = 1:K

            k_nearst(ik, 1) = B(ik);
            k_nearst(ik, 2) = IX(ik);

        end

        y0_knear = 0;
        y1_knear = 0;

        for ik = 1:K

            if ytrain(k_nearst(ik, 2), 1) == 1
                y1_knear = y1_knear + 1;
            else
                y0_knear = y0_knear + 1;
            end

        end

        if ytest(j, 1) == 1

            if y1_knear < y0_knear
                test_error = test_error + 1;
            end

        else

            if y1_knear > y0_knear
                test_error = test_error + 1;
            end

        end

    end

    train_error_rate(ai) = train_error / Xtrain_rows;
    test_error_rate(ai) = test_error / Xtest_rows;
    train_error_rate_percentage(ai) = test_error_rate(ai) * 100;
    test_error_rate_percentage(ai) = test_error_rate(ai) * 100;
end

% plot
plot(K_parameter, train_error_rate, 'black', K_parameter, test_error_rate, 'red')
grid on;
title('(log)Error Chaing Rate of Training and Test Error vs K');
xlabel('K');
ylabel('Error Rate');
legend('Training Error Rate', 'Test Error Rate', 'Location', 'NorthWest');
% output
% fprintf('K= %d,    Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', K_parameter(1), train_error_rate(1), test_error_rate(1))
% fprintf('K= %d,   Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', K_parameter(10), train_error_rate(10), test_error_rate(10))
% fprintf('K= %d,  Traing Error Rate=%.6f,  Test Error Rate=%.6f\n', K_parameter(28), train_error_rate(28), test_error_rate(28))
fprintf('K= %d,   Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', K_parameter(1), train_error_rate_percentage(1), test_error_rate_percentage(1))
fprintf('K= %d,  Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', K_parameter(10), train_error_rate_percentage(10), test_error_rate_percentage(10))
fprintf('K= %d, Traing Error Rate=%.4f %%, Test Error Rate=%.4f %%\n', K_parameter(28), train_error_rate_percentage(28), test_error_rate_percentage(28))
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

%======================================================
% Function of computing distance
% input： matrix of test & training data
% return: Euclidean distance between inputs
%======================================================
function [dis] = dis_matrix(X_test, X_train)

    dis = zeros(size(X_train, 1), size(X_test, 1));

    for i = 1:size(X_test, 1)
        X = X_test(i, :);

        for i_compare = 1:size(X_train, 1)
            dis(i_compare, i) = distance(X, X_train(i_compare, :));
        end

    end

end

%======================================================
% Function of computing distance
% input： matrix of Xand X_start
% return: Euclidean distance between inputs
%======================================================
function [d] = distance(X, X_start)
    sum = 0;

    for i = 1:size(X_start, 2)
        sum = sum + (X(1, i) - X_start(1, i))^2;
    end

    d = sum^0.5;

end
