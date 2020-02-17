% EC 414 Introduction to Machine Learning
% Spring semester, 2020
% Homework 3
% by (Xinyuan Zhao)
%
% Problem 4.3 Nearest Neighbor Classifier
% a), b), c), and d)

clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,'rgb');
% label axis and include title
xlabel('Feature 1');
ylabel('Feature 2');
title('Training Data');


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute probabilities of being in class 2 for each point on grid
[a,b] = size(Xgrid);
%Initialize a matrix containing all counts
ky = zeros(a,b);
for r = 1:a
    for c = 1:b
        
        count = 0;
        x1 = (-3.5+c*0.1)* ones(200,1);
        y1 = (-3.0+r*0.1)* ones(200,1);
        d = ((Xtrain(:,1)-x1).^2 + (Xtrain(:,2)-y1).^2).^(1/2);
        [d1,I] = sort(d,'ascend');
        for j = 1:K
            if ytrain(I(j)) == 2
                count = count+1;
            end
        end
        ky(r,c) = count;
    end
end            
              
probabilities = ky/K;

% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Feature 1');
ylabel('Feature 2');
title('Probability of Classification as 2');


% repeat steps above for class 3 below
ky3 = zeros(a,b);
for r1 = 1:a
    for c1 = 1:b
        
        count1 = 0;
        x2 = (-3.5+c1*0.1)* ones(200,1);
        y2 = (-3.0+r1*0.1)* ones(200,1);
        d2 = ((Xtrain(:,1)-x2).^2 + (Xtrain(:,2)-y2).^2).^(1/2);
        [d3,II] = sort(d2,'ascend');
        for j = 1:K
            if ytrain(II(j)) == 3
                count1 = count1+1;
            end
        end
        ky3(r1,c1) = count1;
    end
end            
              
probabilities3 = ky3/K;

% Figure for class 3
figure
class3ProbonGrid = reshape(probabilities3,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Feature 1');
ylabel('Feature 2');
title('Probability of Classification as 3');

%% c) Class label predictions
K = 1 ; % K = 1 case

% compute predictions 
%initialize a matrix of predictions
ypred = zeros(a,b);
for r = 1:a
    for c = 1:b
       
        x1 = (-3.5+c*0.1)* ones(200,1);
        y1 = (-3.0+r*0.1)* ones(200,1);
        d = ((Xtrain(:,1)-x1).^2 + (Xtrain(:,2)-y1).^2).^(1/2);
        [d1,I] = sort(d,'ascend');
        
        ypred(r,c) = ytrain(I(1,1));
    end
end           

figure
gscatter(Xgrid(:),Ygrid(:),ypred(:),'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
legend('1','2','3','location','eastoutside');
% remember to include title and labels!
xlabel('Feature 1');
ylabel('Feature 2');
title('kNN Classification for k=1');

% repeat steps above for the K=5 case. Include code for this below.
K = 5;
ypred5 = zeros(a,b);
m = zeros(K,1);
for r = 1:a
    for c = 1:b
        
        x1 = (-3.5+c*0.1)* ones(200,1);
        y1 = (-3.0+r*0.1)* ones(200,1);
        d = ((Xtrain(:,1)-x1).^2 + (Xtrain(:,2)-y1).^2).^(1/2);
        [d1,I] = sort(d,'ascend');
        for i = 1:K
            m(i,1) = ytrain(I(i,1));
        end
        ypred5(r,c) = mode(m);
    end
end           

figure
gscatter(Xgrid(:),Ygrid(:),ypred5(:),'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
legend('1','2','3','location','eastoutside');
% remember to include title and labels!
xlabel('Feature 1');
ylabel('Feature 2');
title('kNN Classification for k=5');

%% d) LOOCV CCR computations

for k = 1:2:11
    % determine leave-one-out predictions for k
    ypred = zeros(200,1);
    m = zeros(k,1);
    for j = 1:length(ytrain)
        x1 = Xtrain(j,1)* ones(200,1);
        y1 = Xtrain(j,2)* ones(200,1);
        d = ((Xtrain(:,1)-x1).^2 + (Xtrain(:,2)-y1).^2).^(1/2);
        [d1,I] = sort(d,'ascend');
        %start from 2nd since first is comparing with itself
        for i = 2:(k+1) 
            m(i-1,1) = ytrain(I(i,1));
        end
        ypred(j,1) = mode(m);

    end       


    % compute confusion matrix
    conf_mat = confusionmat(ytrain(:), ypred(:));
    % from confusion matrix, compute CCR
    CCR = trace(conf_mat)/length(ytrain);
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title 
figure
plot([1:2:11],CCR_values);
ylim([0.5,1]);
xlabel('Value of k');
ylabel('LOOCV CCR');
title('LOOCV CCR for Different k');
