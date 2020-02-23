% EC 414 - HW 4 - Spring 2020
% K-Means starter code
%Xinyuan Zhao

clear, clc, close all,

%% Generate Gaussian data (4.2a):
% Add code below:
R1 = mvnrnd([2;2],0.02*eye(2),50);
R2 = mvnrnd([-2;2],0.05*eye(2),50);
R3 = mvnrnd([0;-3.25],0.07*eye(2),50);
scatter(R1(:,1),R1(:,2),'r');
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter of 3 Clusters of Dataset');

hold on
scatter(R2(:,1),R2(:,2),'g');
scatter(R3(:,1),R3(:,2),'b');
legend('1','2','3','location','EastOutside');
hold off


%% Generate NBA data: (4.2e)
% Add code below:
[num,txt,raw] = xlsread('NBA_stats_2018_2019');
MPG = num(:,2);
PPG = num(:,4);
figure
scatter(MPG,PPG);
xlabel('MPG');
ylabel('PPG');
title('Scatter PPG Against MPG for NBA Players');
% HINT: readmatrix might be useful here

%% Problem 4.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 4 folder on Blackboard.
[data_cc ,label_cc] = sample_circle( 3, [500;500;500] );
figure
gscatter(data_cc(:,1),data_cc(:,2),label_cc(:),'rgb');
xlabel('Feature X');
ylabel('Feature Y');
title('Scatter of Concentric Data');
%% K-Means implementation (4.2a)
% Add code below
DATA = [R1;R2;R3];
K = 3;
MU_init = [3,3;-4,-1;2,-4];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
%Run k-means
while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    d =[];
    for i = 1:K
        u = [MU_init(i,1),MU_init(i,2)].*ones(length(DATA),2);
        dtoi = (DATA(:,1)-u(:,1)).^2 + (DATA(:,2)-u(:,2)).^2;
        d = [d,dtoi];
    end
    
    [dd,ii] = min(d,[],2);
    labels = ii';
        
    % CODE - Mean Updating - Update the cluster means
    % Write code below here:
    MU_previous = MU_current;
    clusters = zeros(K,2);
    for z = 1:K
        c = zeros(K);
        for j = 1:length(labels)
            if labels(j) == z
                clusters(z,1) = clusters(z,1)+DATA(j,1);
                clusters(z,2) = clusters(z,2)+DATA(j,2);
                c(z) = c(z)+1;
               
            end
        end
        clusters(z,1) = clusters(z,1)/c(z);
        clusters(z,2) = clusters(z,2)/c(z);
    end
    MU_current = clusters;

            
            
    %CODE 4 - Check for convergence 
    % Write code below here:
    conv = mean(mean(MU_current-MU_previous));
    if (conv < convergence_threshold)
        converged=1;
    end
    
    % CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        figure
        gscatter(DATA(:,1),DATA(:,2),labels','rgb');
        xlabel('Feature 1');
        ylabel('Feature 2');
        title('Scatter Dataset Tested by K-means');
        legend('1','2','3','location','EastOutside');
        
        
        %If converged, get WCSS metric
        % Add code below
        D_wcss=zeros(1,K);
        for i2 = 1:K
            for j2 = 1:length(labels)
                if labels(j2) == i2
                    D_wcss(1,i2) = D_wcss(1,i2)+((DATA(j2,1)-MU_current(i2,1)).^2 ... 
                        + (DATA(j2,2)-MU_current(i2,2)).^2);

                end
            end
        end
        
        WCSS = sum(D_wcss);
    end
end

fprintf('4.2a WCSS: %f\n',WCSS);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check different Initialization (4.2b)
% Add code below
MU_init = [-0.14,2.61;3.15,-0.84;-3.28,-1.58];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
%Run k-means
while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    d =[];
    for i = 1:K
        u = [MU_init(i,1),MU_init(i,2)].*ones(length(DATA),2);
        dtoi = (DATA(:,1)-u(:,1)).^2 + (DATA(:,2)-u(:,2)).^2;
        d = [d,dtoi];
    end
    
    [dd,ii] = min(d,[],2);
    labels = ii';
        
    % CODE - Mean Updating - Update the cluster means
    % Write code below here:
    MU_previous = MU_current;
    clusters = zeros(K,2);
    for z = 1:K
        c = zeros(K);
        for j = 1:length(labels)
            if labels(j) == z
                clusters(z,1) = clusters(z,1)+DATA(j,1);
                clusters(z,2) = clusters(z,2)+DATA(j,2);
                c(z) = c(z)+1;
               
            end
        end
        clusters(z,1) = clusters(z,1)/c(z);
        clusters(z,2) = clusters(z,2)/c(z);
    end
    MU_current = clusters;

            
            
    %CODE 4 - Check for convergence 
    % Write code below here:
    conv = mean(mean(MU_current-MU_previous));
    if (conv < convergence_threshold)
        converged=1;
    end
    
    % CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        figure
        gscatter(DATA(:,1),DATA(:,2),labels','rgb');
        xlabel('Feature 1');
        ylabel('Feature 2');
        title('Test of Different Initialization K-means');
        legend('1','2','3','location','EastOutside');
        
        
        %If converged, get WCSS metric
        % Add code below
        D_wcss=zeros(1,K);
        for i2 = 1:K
            for j2 = 1:length(labels)
                if labels(j2) == i2
                    D_wcss(1,i2) = D_wcss(1,i2)+((DATA(j2,1)-MU_current(i2,1)).^2 ... 
                        + (DATA(j2,2)-MU_current(i2,2)).^2);

                end
            end
        end
        
        WCSS2 = sum(D_wcss);
    end
end

fprintf('4.2b WCSS: %f\n',WCSS2);
%Observation: For this initialization, clusters 1 and 2 converged to the
%same cluster, which is not desired. Compared to WCSS of part a which is
%11.8, WCSS for this initialization is 405. As the initial u2 is too far
%away from actual dataset, the k-means algorithm fails to converge
%correctly. This algorithm is very sensitive to initial centers.

%% Random Initializations (4.2c)
DATA = [R1;R2;R3];
K = 3;
WCSS_random = zeros(1,10);
labels = ones(length(DATA),10);
%Run k-means for 10 times at random
for r = 1:10
    MU_init = ones(3,2);
    MU_init(1,:) = [3,3].*rand(1,2);
    MU_init(2,:) = [-3,3].*rand(1,2);
    MU_init(3,:) = [1,-3].*rand(1,2);
    

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        d =[];
        for i = 1:K
            u = [MU_init(i,1),MU_init(i,2)].*ones(length(DATA),2);
            dtoi = (DATA(:,1)-u(:,1)).^2 + (DATA(:,2)-u(:,2)).^2;
            d = [d,dtoi];
        end

        [dd,ii] = min(d,[],2);
        labels(:,r) = ii';

        % CODE - Mean Updating - Update the cluster means
        % Write code below here:
        MU_previous = MU_current;
        clusters = zeros(K,2);
        for z = 1:K
            c = zeros(K);
            for j = 1:length(labels(:,1))
                if labels(j,r) == z
                    clusters(z,1) = clusters(z,1)+DATA(j,1);
                    clusters(z,2) = clusters(z,2)+DATA(j,2);
                    c(z) = c(z)+1;

                end
            end
            clusters(z,1) = clusters(z,1)/c(z);
            clusters(z,2) = clusters(z,2)/c(z);
        end
        MU_current = clusters;



        %CODE 4 - Check for convergence 
        % Write code below here:
        conv = mean(mean(MU_current-MU_previous));
        if (conv < convergence_threshold)
            converged=1;
        end

        % CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            %If converged, get WCSS metric
            % Add code below
            D_wcss=zeros(1,K);
            for i2 = 1:K
                for j2 = 1:length(labels(:,1))
                    if labels(j2,r) == i2
                        D_wcss(1,i2) = D_wcss(1,i2)+((DATA(j2,1)-MU_current(i2,1)).^2 ... 
                            + (DATA(j2,2)-MU_current(i2,2)).^2);

                    end
                end
            end
        end
    end
    WCSS_random(r) = sum(D_wcss);
end
[mini,ind] = min(WCSS_random);

fprintf('4.2c Minimum WCSS: %f\n',mini);
figure
gscatter(DATA(:,1),DATA(:,2),labels(:,ind)','rgb');
xlabel('Feature 1');
ylabel('Feature 2');
title(['Best of Random Init: WCSS = ',num2str(mini),' ']);
legend('1','2','3','location','EastOutside');


%% Choose K (4.2d)
DATA = [R1;R2;R3];
K = [2,3,4,5,6,7,8,9,10];
WCSS_k = zeros(1,9);
%Run Random k-means for 9 times
for ik = 1:length(K)
    K_val = K(ik);
    WCSS_random = zeros(1,10);
    labels = ones(length(DATA),10);
    for r = 1:10
        MU_init = [6,7].*rand(K_val,2);
        MU_init(:,1) = MU_init(:,1)+((-3)*ones(K_val,1));
        MU_init(:,2) = MU_init(:,2)+((-4)*ones(K_val,1));
        


        MU_previous = MU_init;
        MU_current = MU_init;

        % initializations

        converged = 0;
        iteration = 0;
        convergence_threshold = 0.025;

        while (converged==0)
            iteration = iteration + 1;
            fprintf('Iteration: %d\n',iteration)

            % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
            % Write code below here:
            d =[];
            for i = 1:K_val
                u = [MU_init(i,1),MU_init(i,2)].*ones(length(DATA),2);
                dtoi = (DATA(:,1)-u(:,1)).^2 + (DATA(:,2)-u(:,2)).^2;
                d = [d,dtoi];
            end

            [dd,ii] = min(d,[],2);
            labels(:,r) = ii';

            % CODE - Mean Updating - Update the cluster means
            % Write code below here:
            MU_previous = MU_current;
            clusters = zeros(K_val,2);
            for z = 1:K_val
                c = zeros(K_val);
                for j = 1:length(labels(:,1))
                    if labels(j,r) == z
                        clusters(z,1) = clusters(z,1)+DATA(j,1);
                        clusters(z,2) = clusters(z,2)+DATA(j,2);
                        c(z) = c(z)+1;

                    end
                end
                clusters(z,1) = clusters(z,1)/c(z);
                clusters(z,2) = clusters(z,2)/c(z);
            end
            MU_current = clusters;



            %CODE 4 - Check for convergence 
            % Write code below here:
            conv = max(max(MU_current-MU_previous));
            if (conv < convergence_threshold)
                converged=1;
            end

            % CODE 5 - Plot clustering results if converged:
            % Write code below here:
            if (converged == 1)
                fprintf('\nConverged.\n')
                %If converged, get WCSS metric
                % Add code below
                D_wcss=zeros(1,K_val);
                for i2 = 1:K_val
                    for j2 = 1:length(labels(:,1))
                        if labels(j2,r) == i2
                            D_wcss(1,i2) = D_wcss(1,i2)+((DATA(j2,1)-MU_current(i2,1)).^2 ... 
                                + (DATA(j2,2)-MU_current(i2,2)).^2);

                        end
                    end
                end
            end
        end
        WCSS_random(r) = sum(D_wcss);
    end
    WCSS_k(ik) = min(WCSS_random);
end
figure
plot(K,WCSS_k);
xlabel('Value of K');
ylabel('WCSS(K)');
title('Plot of WCSS against K from 2-10');

%Observation: From this plot, we can see that the elbow occurs at k=3,
%which means k=3 is the best choice for minimizing WCSS. This choice of k
%matches the number of clusters in the training dataset.

%% NBA Stats (4.2e)
NBA_data = [MPG,PPG];
K = 10;
WCSS_random = zeros(1,10);
labels = ones(length(NBA_data(:,1)),10);
%Run k-means for 10 times at random
for r = 1:10
    MU_init = 40*rand(10,2);
    

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.000001;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        d =[];
        for i = 1:K
            u = [MU_init(i,1),MU_init(i,2)].*ones(length(NBA_data(:,1)),2);
            dtoi = (NBA_data(:,1)-u(:,1)).^2 + (NBA_data(:,2)-u(:,2)).^2;
            d = [d,dtoi];
        end

        [dd,ii] = min(d,[],2);
        labels(:,r) = ii';

        % CODE - Mean Updating - Update the cluster means
        % Write code below here:
        MU_previous = MU_current;
        clusters = zeros(K,2);
        for z = 1:K
            c = zeros(K);
            for j = 1:length(labels(:,1))
                if labels(j,r) == z
                    clusters(z,1) = clusters(z,1)+NBA_data(j,1);
                    clusters(z,2) = clusters(z,2)+NBA_data(j,2);
                    c(z) = c(z)+1;

                end
            end
            clusters(z,1) = clusters(z,1)/c(z);
            clusters(z,2) = clusters(z,2)/c(z);
        end
        MU_current = clusters;



        %CODE 4 - Check for convergence 
        % Write code below here:
        conv = max(max(MU_current-MU_previous));
        if (conv < convergence_threshold)
            converged=1;
        end

        % CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            %If converged, get WCSS metric
            % Add code below
            D_wcss=zeros(1,K);
            for i2 = 1:K
                for j2 = 1:length(labels(:,1))
                    if labels(j2,r) == i2
                        D_wcss(1,i2) = D_wcss(1,i2)+((NBA_data(j2,1)-MU_current(i2,1)).^2 ... 
                            + (NBA_data(j2,2)-MU_current(i2,2)).^2);

                    end
                end
            end
        end
    end
    WCSS_random(r) = sum(D_wcss);
end
[mini,ind] = min(WCSS_random);
fprintf('4.2e Minimum WCSS: %f\n',mini);
figure
gscatter(NBA_data(:,1),NBA_data(:,2),labels(:,ind)');
xlabel('MPG');
ylabel('PPG');
title(['Best of Random Init: WCSS = ',num2str(mini),' ']);

%% Concentric Data (4.2f)
K = 3;
WCSS_random = zeros(1,10);
labels = ones(length(data_cc(:,1)),10);
%Run k-means for 10 times at random
for r = 1:10
    MU_init = zeros(3,2);
    MU_init(1,:) = 4*rand(1,2);
    MU_init(2,:) = rand(1,2);
    MU_init(3,:) = -4*rand(1,2);
    

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        % CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        d =[];
        for i = 1:K
            u = [MU_init(i,1),MU_init(i,2)].*ones(length(data_cc(:,1)),2);
            dtoi = (data_cc(:,1)-u(:,1)).^2 + (data_cc(:,2)-u(:,2)).^2;
            d = [d,dtoi];
        end

        [dd,ii] = min(d,[],2);
        labels(:,r) = ii';

        % CODE - Mean Updating - Update the cluster means
        % Write code below here:
        MU_previous = MU_current;
        clusters = zeros(K,2);
        for z = 1:K
            c = zeros(K);
            for j = 1:length(labels(:,1))
                if labels(j,r) == z
                    clusters(z,1) = clusters(z,1)+data_cc(j,1);
                    clusters(z,2) = clusters(z,2)+data_cc(j,2);
                    c(z) = c(z)+1;

                end
            end
            clusters(z,1) = clusters(z,1)/c(z);
            clusters(z,2) = clusters(z,2)/c(z);
        end
        MU_current = clusters;



        %CODE 4 - Check for convergence 
        % Write code below here:
        conv = max(max(MU_current-MU_previous));
        if (conv < convergence_threshold)
            converged=1;
        end

        % CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            %If converged, get WCSS metric
            % Add code below
            D_wcss=zeros(1,K);
            for i2 = 1:K
                for j2 = 1:length(labels(:,1))
                    if labels(j2,r) == i2
                        D_wcss(1,i2) = D_wcss(1,i2)+((data_cc(j2,1)-MU_current(i2,1)).^2 ... 
                            + (data_cc(j2,2)-MU_current(i2,2)).^2);

                    end
                end
            end
        end
    end
    WCSS_random(r) = sum(D_wcss);
end
[mini,ind] = min(WCSS_random);
figure
gscatter(data_cc(:,1),data_cc(:,2),labels(:,ind)');
xlabel('Feature X');
ylabel('Feature Y');
title(['Best of Random Init: WCSS = ',num2str(mini),' ']);

%Observation: As we can see from 4.2e and 4.2f, K-means clustering fails to
%find meaningful clusters if data is linear or concentric. The WCSSs of
%these kinds of data are very high. So we can only use this algorithm for
%spherical datas.

