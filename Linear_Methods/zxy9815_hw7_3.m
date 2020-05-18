%Xinyuan Zhao EC414 HW7

load('iris.mat')
n_train = length(X_data_train(:,1));
n_test = length(X_data_test(:,1));
d = length(X_data_train(1,:));
m = 3;
%% 7.3a. Data Analysis
label_set = [Y_label_train;Y_label_test];
figure
histogram(label_set,'BinMethod','integers');
ylim([0,100]);
ylabel('Counts');
xlabel('Classes');
title('Histogram of Class Labels');
%% 7.3a.Correlation Matrix
X = [X_data_train;X_data_test];
corr = zeros(4);
N_X = n_train+n_test;

for i = 1:4
    for j = 1:4
        f1 = X(:,i);
        f2 = X(:,j);
        cov = sum((f1-mean(f1)).*(f2-mean(f2)))/N_X;
        var1 = sum((f1-mean(f1)).^2)/N_X;
        var2 = sum((f2-mean(f2)).^2)/N_X;
        corr(i,j) = cov/sqrt(var1*var2);
    end
end
fprintf('7.3a Correlation matrix');
disp(corr);
%% 7.3a.Scatter Plots
count = 1;
figure
hold on
for i = 1:3
    for j = (i+1):4
        subplot(3,2,count)
        gscatter(X(:,i),X(:,j),label_set(:));
        xlabel(['Feature ',num2str(i)]);
        ylabel(['Feature ',num2str(j)]);
        title(['Scatter of feature ',num2str(i),' , ',num2str(j)]);
        count = count + 1;
    end
end
hold off
%% SGD Algorithm

theta = zeros(d+1,m);
x_test_ext = [X_data_test, ones(n_test,1)];
ccr_test = zeros(300,1);
logloss = zeros(300,1);
tmax = 6000;
lambda = 0.1;
x_ext = [X_data_train, ones(n_train,1)];
iter = 0;
ccr = zeros(300,1);
g0 = zeros(300,1);
for t = 1:tmax
    j = randi([1,n_train]);
    iter = iter+1;
    p_k = zeros(m,1);
    gk = zeros(d+1,m);
   
    for k = 1:m
        kyj = 0;
        if k == Y_label_train(j)
            kyj = 1;
        end
        summ = 0;
        for ss = 1:m
            summ = summ + exp(theta(:,ss)' * x_ext(j,:)');
        end
        p_k(k) = exp(theta(:,k)' * x_ext(j,:)')/summ;
        if p_k(k) < 10^(-10)
            p_k(k) = 10^(-10);
        end
        gk(:,k) = 2*lambda*theta(:,k) + n_train*(p_k(k)-kyj)*x_ext(j,:)';
    end
    
    for k2 = 1:m
        theta(:,k2) = theta(:,k2) - (0.01/t)*gk(:,k2);
    end
    
    if iter == 20
        
        ypred = zeros(n_train,1);
        label_train = zeros(n_train,1);
        f0 = 0;
        for f = 1:m
            f0 = f0 + sum(theta(:,f).^2);
        end
        f0 = lambda*f0;
        
        fj = 0;
        for n = 1:n_train
            term1 = 0;
            term2 = 0;
            for ss = 1:m
                term1 = term1 + exp(theta(:,ss)' * x_ext(n,:)');
                if ss == Y_label_train(n)
                    term2 = term2 + (theta(:,ss)' * x_ext(n,:)');
                end
            end
            term1 = log(term1);
            fj = fj + (term1 - term2);
            
            [yj,ii] = max(theta'*x_ext(n,:)');
            label_train(n) = ii;
            if ii == Y_label_train(n)
                ypred(n) = 1;
            end
        end
        g0(t/iter) = f0 + fj;
        ccr(t/iter) = sum(ypred)/n_train;
        %test set
        ypred_test = zeros(n_test,1);
        label_test = zeros(n_test,1);
        NLL = 0;
        for n = 1:n_test
            summ2 = 0;
            pyj = 0;
            for ss = 1:m
                summ2 = summ2 + exp(theta(:,ss)' * x_test_ext(n,:)');
            end
            pyj = exp(theta(:,Y_label_test(n))' * x_test_ext(n,:)')/summ2;
            if pyj < 10^(-10)
                pyj = 10^(-10);
            end
            NLL = NLL+log(pyj);
            
            
            [yj,ii] = max(theta'*x_test_ext(n,:)');
            label_test(n) = ii;
            if ii == Y_label_test(n)
                ypred_test(n) = 1;
            end
        end
        ccr_test(t/iter) = sum(ypred_test)/n_test;
        logloss(t/iter) = -(1/n_test)*NLL;
        iter = 0;
    end
end

%% 7.3b.Regularized LL vs. Iter
figure
plot([1:20:6000],(g0/n_train));
xlabel('Iteration Number');
ylabel('Regularized Logistic Loss of Training Set');
title('Training Set Regularized Logistic Loss vs. #Iteration');

%% 7.3c. CCR training vs. Iter
figure
plot([1:20:6000],ccr);
xlabel('Iteration Number');
ylabel('CCR of Training Set');
title('Training Set CCR vs. #Iteration');


%% 7.3d. CCR test vs. Iter
figure
plot([1:20:6000],ccr_test);
xlabel('Iteration Number');
ylabel('CCR of Test Set');
title('Test Set CCR vs. #Iteration');

%% 7.3e. NLL of test set vs. Iter
figure
plot([1:20:6000],logloss);
xlabel('Iteration Number');
ylabel('Log-Loss of Test Set');
title('Test Set Log-Loss vs. #Iteration');


%% 7.3f. Final Values
fprintf('7.3f: Theta of Training Set:\n');
disp(theta);
fprintf('7.3f: Training CCR:\n');
disp(ccr(300));
fprintf('7.3f: Test CCR:\n');
disp(ccr_test(300));
fprintf('7.3f: Training Confusion Matrix:\n');
disp(confusionmat(Y_label_train,label_train));
fprintf('7.3f: Test Confusion Matrix:\n');
disp(confusionmat(Y_label_test,label_test));

%% 7.3g. Decision Boundaries
count = 1;
figure
hold on
for i = 1:3
    for j = (i+1):4                                   
        fx1 = @(x) theta(i,1)*x+theta(5,1);
        fx2 = @(x) theta(i,2)*x+theta(5,2);
        fx3 = @(x) theta(i,3)*x+theta(5,3);
        subplot(3,2,count)
        gscatter(X(:,i),X(:,j),label_set(:));
        hold on
        fplot(fx1,[min(X(:,i)),max(X(:,i))]);
        fplot(fx2,[min(X(:,i)),max(X(:,i))]);
        fplot(fx3,[min(X(:,i)),max(X(:,i))]);
        xlabel(['Feature ',num2str(i)]);
        ylabel(['Feature ',num2str(j)]);
        title(['Boundaries of feature ',num2str(i),' , ',num2str(j)]);
        count = count + 1;
    end
end
hold off






