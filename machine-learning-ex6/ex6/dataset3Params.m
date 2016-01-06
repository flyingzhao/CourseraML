function [Cvec, sigma_vec] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cvec = 1;
sigma_vec = 0.3;

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

C=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma=[0.01 0.03 0.1 0.3 1 3 10 30];
mat=zeros(size(C,2),size(sigma,2));
for j=1:size(C,2)
    for k=1:size(sigma,2)    
    model= svmTrain(X, y, C(j), @(x1, x2) gaussianKernel(x1, x2, sigma(k)));
    pred = svmPredict(model, Xval);
    mat(j,k)=sum(yval~=pred);
    end
end

min_mat=min(min(mat));
[x,y]=find(mat==min_mat);
Cvec=C(x);
sigma_vec=sigma(y);





% =========================================================================

end