%Xinyuan Zhao EC414 HW9_2

load('kernel-svm-2rings.mat')
n = length(y);
d = length(x(:,1));

K = zeros(n,n);
for i = 1:n
    for j = 1:n
        dif = x(:,i) - x(:,j);
        nom = sum(dif.^2);
        K(i,j) = exp(-(1/(2*0.5^2))*nom);
    end
end

K_ext = [K;ones(1,n)];

    
%% SSGD Algorithm

phi = zeros(n+1,1);
tmax = 1000;
nC = 256;
iter = 0;

normCost = zeros(100,1);
ccr = zeros(100,1);

for t = 1:tmax
    j = randi([1,n]);
    iter = iter+1;

    v = [K zeros(n,1);zeros(1,n) 0]*phi;
    
    term1 = y(j) * phi' * K_ext(:,j);
    if term1 < 1
        v = v - nC * y(j) *  K_ext(:,j);
    end
    phi = phi - (0.256/t)*v;
    
    if iter == 10
        f0 = (1/2)*phi'*[K zeros(n,1);zeros(1,n) 0]*phi;
        fj = 0;
        ccrk = 0;
        ypred = zeros(n,1);
        for k = 1:n
            termHinge = y(k) * phi' * K_ext(:,k);
            fj = fj + (nC/n)*max(0,(1-termHinge));
            %ccr computation
            ypred(k) = sign(phi'*K_ext(:,k));
            if ypred(k) == y(k)
                ccrk = ccrk+1;
            end
        end
        normCost(t/iter) = (1/n)*(f0 + fj);
        ccr(t/iter) = (1/n)*ccrk;
        iter = 0;
    end
end
   

%% 9.2a. normalized cost vs. iter

figure
plot([1:10:1000],normCost);
xlabel('Iteration Number');
ylabel('Normalized Cost');
title('Normalized Cost vs. #Iteration');
ylim([0 100]);

%% 9.2b. training ccr vs. iter

figure
plot([1:10:1000],ccr);
xlabel('Iteration Number');
ylabel('Training CCR');
title('Training CCR vs. #Iteration');

%% 9.2c. Results

fprintf('Training Confusion Matrix: \n');
disp(confusionmat(y,ypred));

%% 9.2d. Creating a Dense Grid and Evaluating decisions on each point
xGrid = linspace(-2,2,100);
[gridX,gridY] = meshgrid(xGrid,xGrid);
decision = zeros(100,100);
for i = 1:100
    for j = 1:100
        kij = ones(n+1,1);
        for k = 1:n
            dif = x(:,k) - [gridX(i,j);gridY(i,j)];
            nom = sum(dif.^2);
            kij(k) = exp(-(1/(2*0.5^2))*nom);
        end
        decision(i,j) = sign(phi'*kij);
    end
end
%% 9.2d. Locating points where decision changes
xBound = [];
yBound = [];
for i = 2:99
    for j = 2:99
        if decision(i+1,j+1) ~= decision(i-1,j-1)
            xBound = [xBound, gridX(i,j)];
            yBound = [yBound, gridY(i,j)];
        end
    end
end
            
%% Plotting Training data and Decision Boundary
figure
gscatter(x(1,:),x(2,:),y(:));
hold on
scatter(xBound,yBound,'g*');
hold off
xlabel('Feature 1');
ylabel('Feature 2');
title('Training data & Decision Boundary');
legend('Class 1','Class 2','Boundary');





