function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_find = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_find = [0.01 0.03 0.1 0.3 1 3 10 30];

error = 0;
error = zeros(length(C_find), length(sigma_find));

for i = 1:length(C_find)
  for j = 1:length(sigma_find)
    model = svmTrain(X, y, C_find(i), @(x1, x2) gaussianKernel(x1, x2, sigma_find(j)));
    predictions = svmPredict(model, Xval);
    error(i,j) = mean(double(predictions ~= yval));
  endfor
endfor

[min_value, row] = min(min(error, [], 2));
[min_value, col] = min(min(error, [], 1));

C = C_find(row);
sigma = sigma_find(col);




% =========================================================================

end
