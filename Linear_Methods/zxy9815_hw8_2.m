%Xinyuan Zhao EC414 HW8

load('iris.mat')
n_train = length(X_data_train(:,1));
n_test = length(X_data_test(:,1));
%Only Consider features X2 and X4
d = 2;
X_test = [X_data_test(:,2),X_data_test(:,4)];
X_train = [X_data_train(:,2),X_data_train(:,4)];
m = 3;

ytrain_binary = [ones(35,1);-1*ones(35,1)];
ytest_binary = [ones(15,1);-1*ones(15,1)];
Xtrain12 = X_train(1:70,:);
Xtrain13 = [X_train(1:35,:);X_train(71:105,:)];
Xtrain23 = X_train(36:105,:);
ntrain_binary = 70;

Xtest12 = X_test(1:30,:);
Xtest13 = [X_test(1:15,:);X_test(31:45,:)];
Xtest23 = X_test(16:45,:);
ntest_binary = 30;
    
%% SSGD Algorithm

theta = zeros(d+1,3);
tmax = 2*10^5;
C = 1.2;
x12_ext = [Xtrain12, ones(ntrain_binary,1)];
x13_ext = [Xtrain13, ones(ntrain_binary,1)];
x23_ext = [Xtrain23, ones(ntrain_binary,1)];
xtrain_ext = [x12_ext;x13_ext;x23_ext];
test12_ext = [Xtest12, ones(ntest_binary,1)];
test13_ext = [Xtest13, ones(ntest_binary,1)];
test23_ext = [Xtest23, ones(ntest_binary,1)];
xtest_ext = [test12_ext;test13_ext;test23_ext];


norm_cost = zeros(3,200);
ccr = zeros(3,200);
ccr_test = zeros(3,200);
ypred_train = zeros(ntrain_binary,3);
ypred_test = zeros(ntest_binary,3);

for i = 1:3
    ind = (i-1)*70+1;
    x_ext = xtrain_ext(ind:(i*70),:);
    ind2 = (i-1)*30+1;
    x_ext2 = xtest_ext(ind2:(i*30),:);
    iter = 0;
    for t = 1:tmax
        j = randi([1,ntrain_binary]);
        iter = iter+1;

        v = [theta(1:d,i);0];
        term1 = ytrain_binary(j)*theta(:,i)'*x_ext(j,:)';
        if term1 < 1
            v = v - ntrain_binary*C*ytrain_binary(j)*x_ext(j,:)';
        end
        theta(:,i) = theta(:,i) - (0.5/t)*v;
        
        if iter == 1000
            f0 = 1/2*sqrt(theta(1,i)^2+theta(2,i)^2);
            fj = 0;
            ccrk = 0;
            ypred = zeros(ntrain_binary,1);
            for k = 1:ntrain_binary
                termk = ytrain_binary(k)*theta(:,i)'*x_ext(k,:)';
                fj = fj + C*max(0,1-termk);
                %ccr computation
                ypred(k) = sign(theta(:,i)'*x_ext(k,:)');
                if ypred(k) == ytrain_binary(k)
                    ccrk = ccrk+1;
                end
            end
            
            ccrtk = 0;
            for k = 1:ntest_binary
                ypred_test(k,i) = sign(theta(:,i)'*x_ext2(k,:)');
                if ypred_test(k,i) == ytest_binary(k)
                    ccrtk = ccrtk+1;
                end
            end
            ccr_test(i,t/iter) = (1/ntest_binary)*ccrtk;
            ccr(i,t/iter) = (1/ntrain_binary)*ccrk;
            norm_cost(i,t/iter) = (1/ntrain_binary)*(f0+fj);
            iter = 0;
        end
            
    end
    ypred_train(:,i) = ypred;
end

%% 8.2a. normalized cost vs. iter
class = {'1 & 2','1 & 3','2 & 3'};
for i = 1:3
    figure
    plot([1:1000:2*10^5],norm_cost(i,:));
    xlabel('Iteration Number');
    ylabel('Normalized Cost');
    title(['Normalized Cost vs. #Iteration for class ', class{i}]);
end
%% 8.2b. training ccr vs. iter
for i = 1:3
    figure
    plot([1:1000:2*10^5],ccr(i,:));
    xlabel('Iteration Number');
    ylabel('Training CCR');
    title(['Training CCR vs. #Iteration for class ', class{i}]);
end

%% 8.2c. test ccr vs. iter
for i = 1:3
    figure
    plot([1:1000:2*10^5],ccr_test(i,:));
    xlabel('Iteration Number');
    ylabel('Test CCR');
    title(['Test CCR vs. #Iteration for class ', class{i}]);
end

%% 8.2d. Results
for i = 1:3
    fprintf('Theta for classifying classes %s\n',class{i});
    disp(theta(:,i));
    fprintf('Training CCR for classes %s\n',class{i});
    disp(ccr(i,end));
    fprintf('Test CCR for classes %s\n',class{i});
    disp(ccr_test(i,end));
    fprintf('Training Confusion Matrix for classes %s\n',class{i});
    disp(confusionmat(ytrain_binary,ypred_train(:,i)));
    fprintf('Test Confusion Matrix for classes %s\n',class{i});
    disp(confusionmat(ytest_binary,ypred_test(:,i)));
end

%% 8.2e. All Pairs Method
label_trainAP = zeros(n_train,1);
ccr_AP = 0;
label_testAP = zeros(n_test,1);
ccr_testAP = 0;
%training 
for j = 1:n_train
    wins = zeros(1,m);
    ypredAP = sign(theta'*[X_train(j,:),1]');
    if ypredAP(1) > 0
        wins(1) = wins(1)+1;
    elseif ypredAP(1) < 0
        wins(2) = wins(2)+1;
    end
    if ypredAP(2) > 0
        wins(1) = wins(1)+1;
    elseif ypredAP(2) < 0
        wins(3) = wins(3)+1;
    end
    if ypredAP(3) > 0
        wins(2) = wins(2)+1;
    elseif ypredAP(3) < 0
        wins(3) = wins(3)+1;
    end
    [maxx,ii] = max(wins);
    label_trainAP(j) = ii;
    if ii == Y_label_train(j)
        ccr_AP = ccr_AP + 1;
    end
end
ccr_AP = ccr_AP/n_train;

%test
for j = 1:n_test
    wins = zeros(1,m);
    ypredAP = sign(theta'*[X_test(j,:),1]');
    if ypredAP(1) > 0
        wins(1) = wins(1)+1;
    elseif ypredAP(1) < 0
        wins(2) = wins(2)+1;
    end
    if ypredAP(2) > 0
        wins(1) = wins(1)+1;
    elseif ypredAP(2) < 0
        wins(3) = wins(3)+1;
    end
    if ypredAP(3) > 0
        wins(2) = wins(2)+1;
    elseif ypredAP(3) < 0
        wins(3) = wins(3)+1;
    end
    [maxx,ii] = max(wins);
    label_testAP(j) = ii;
    if ii == Y_label_test(j)
        ccr_testAP = ccr_testAP + 1;
    end
end
ccr_testAP = ccr_testAP/n_test;

fprintf('Results for All Pairs Method\n');
fprintf('Training CCR\n');
disp(ccr_AP);
fprintf('Test CCR\n');
disp(ccr_testAP);
fprintf('Training Confusion Matrix\n');
disp(confusionmat(Y_label_train,label_trainAP));
fprintf('Test Confusion Matrix\n');
disp(confusionmat(Y_label_test,label_testAP));

            
            
            





