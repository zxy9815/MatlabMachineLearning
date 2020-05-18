
clc, clear

fprintf("==== Loading data\n");
load("prostateStnd.mat")

Ntrain = size(Xtrain,1);
Ntest = size(Xtest,1);

%% a.Normalization
xmean = mean(Xtrain,1);
xstd = std(Xtrain,0,1);
ymean = mean(ytrain);
ystd = std(ytrain);
Xtrain_N = (Xtrain-xmean)./xstd;
ytrain_N = (ytrain-ymean).*(1/ystd);

Xtest_N = (Xtest-xmean)./xstd;
ytest_N = (ytest-ymean).*(1/ystd);
%% b,c.Ridge Regression
lambda = exp(-5:1:10);
w_ridge = zeros(length(lambda),8);
b_ridge = zeros(length(lambda),1);
for i = 1:length(lambda)
    [w_ridge(i,:),b_ridge(i,1)] = r_regression(Xtrain_N,ytrain_N,lambda(i));
end

e = -5:1:10;
figure
for i = 1:8
    plot(e,w_ridge(:,i));
    hold on
end
hold off
xlabel('ln \lambda');
ylabel('Ridge Regression Coefficient');
title('Ridge Regression Coefficient vs. ln \lambda');
legend(names(1:8));

%% d.MSE(w,b)
MSE_train = zeros(length(lambda),1);
MSE_test = zeros(length(lambda),1);
for i = 1:length(lambda)
    for j = 1:Ntrain
        MSE_train(i) = MSE_train(i) + (ytrain_N(j)- w_ridge(i,:)*Xtrain_N(j,:)')^2;
    end
    MSE_train(i) = MSE_train(i)/Ntrain;
    
    for j = 1:Ntest
        MSE_test(i) = MSE_test(i) + (ytest_N(j)- w_ridge(i,:)*Xtest_N(j,:)')^2;
    end
    MSE_test(i) = MSE_test(i)/Ntest;
    
end
figure
plot(e,MSE_train);
hold on
plot(e,MSE_test);
hold off
xlabel('ln \lambda');
ylabel('MSE(w,b)');
title('MSE(w,b) vs. ln \lambda');
legend('MSE train','MSE test');
        

%%
function [w_ridge,b_ridge] = r_regression(X,Y,lambda)
d = length(X(1,:));
n = length(Y);
muy = mean(Y);
mux = mean(X);
sx = zeros(d,d);
sxy = zeros(d,1);
for i = 1:n
    sx = sx + ((X(i,:)-mux)'*(X(i,:)-mux));
    sxy = sxy + ((X(i,:)-mux)'.*(Y(i,:)-muy));
end
sx = sx./n;
sxy = sxy./n;
w_ridge = inv(sx+((lambda/n)*eye(d)))*sxy;
b_ridge = muy - (w_ridge' * mux');
end

