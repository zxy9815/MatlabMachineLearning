%Xinyuan Zhao EC414 HW9_3

load('kernel-kmeans-2rings.mat')
d = 2;
k = 2;
n = length(data(:,1));

K = zeros(n,n);
for i = 1:n
    for j = 1:n
        dif = data(i,:) - data(j,:);
        nom = sum(dif.^2);
        K(i,j) = exp(-(1/(2*0.16))*nom);
    end
end

%% Initialize Mean Coefficients

ranD1 = rand(n,1);
ranD2 = rand(n,1);

u1 = ranD1./(norm(ranD1));
u2 = ranD2./(norm(ranD2));

%% Kmeans Algorithm
tmax = 60;
labels = zeros(n,1);
u = [u1,u2];
iter = 0;
for t = 1:tmax
    iter  = iter + 1;
    fprintf('iteration: %d\n',iter);
    %Update Labels
    for j = 1:n
        class = zeros(k,1);
        ej = zeros(n,1);
        ej(j) = 1;
        for l = 1:k
            term = (ej-u(:,l))' * K * (ej-u(:,l));
            class(l) = term;
        end
        [mini,ii] = min(class);
        labels(j) = ii;
    end
    %Update Cluster Mean Coes
    for l = 1:k
        nl = 0;
        eVector = zeros(n,1);
        for j = 1:n
            if labels(j) == l
                nl = nl+1;
                eVector(j) = 1;
            end
        end
        if nl == 0
            u(:,l) = eVector;
        else
            u(:,l) = eVector./nl;
        end
    end
end

%% Plotting Results
figure
gscatter(data(:,1),data(:,2),labels(:));
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter of Kernel K-means Clustering Result');
legend('Class 1','Class 2');





            