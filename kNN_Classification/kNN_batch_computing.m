
% Problem 4.3 Nearest Neighbor Classifier
% d)

clc, clear

fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
imshow(reshape(X_train(200,:), 28,28)')

% determine size of dataset
[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test);

% precompute components
%prevd = 0;
%d_best = ones(1,2);
prediction = zeros(Ntest,1);
% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
batch_size = 500;  % fit 4 GB of memory
num_batches = Ntest / batch_size;


% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
for bn = 1:num_batches
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  
  % calculate cross term
  y = X_train(:,:)';
  x = X_test(batch_start:batch_stop,:); %matrix of one batch of test images
  XX = sum(x.*x,2); %(Row_sum)Vector of each value of (X dot X)
  YY = sum(y.*y,1); %(Column_sum)Transposed Vector of each (y dot y)
  % compute euclidean distance
  D = sqrt(bsxfun(@plus,XX,YY)-2*x*y);
  
  
  fprintf("==== Doing 1-NN classification for batch %d\n", bn);
  % find minimum distance for k = 1
  [minValues, minIndices] = min(D,[],2);
  for i = batch_start:batch_stop
      prediction(i) = Y_train(minIndices(i+1-batch_start));
  end
end

% compute confusion matrix
conf_mat = confusionmat(Y_test(:), prediction(:));
% compute CCR from confusion matrix
ccr = trace(conf_mat)/Ntest;
