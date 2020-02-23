% EC 414 - HW 4 - Spring 2020
% DP-Means starter code

clear, clc, close all,

%% Role of lamda parameter: 4.4a

%lamda is a panelty parameter that is checked each iteration.
%For each iteration, we check if the squared distance btw xj and its cluster
%center is larger than lamda. If yes, then we add a new cluster with center
%of xj.

%% Generate Gaussian data:
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

%% Generate NBA data:
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

%% DP Means method:

% Parameter Initializations
DATA = [R1;R2;R3];
LAMBDA = [0.15,0.4,3,20];
convergence_threshold = 0.1;
num_points = length(DATA);
total_indices = [1:num_points];
for ll = 1:4
    la = LAMBDA(ll);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% DP Means - Initializations for algorithm %%%
    % cluster count
    K = 1;
    K_prev = K;
    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points);

    % means
    MU = [];
    MU = [MU; mean(DATA,1)];
    MU_prev = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;
    t = 0;

    while (converged == 0)
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)
        K_prev = K;
        MU_prev = MU;
        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            d = [];
            for j = 1:K
                dtoi = (DATA(i,1)-MU(j,1)).^2 + (DATA(i,2)-MU(j,2)).^2;
                d = [d,dtoi];
            end
            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            [dd,ii] = min(d);
            if dd > la
                K = K+1;
                Z(i) = K;
                MU = [MU;DATA(i,:)];
            else
                Z(i) = ii;
            end
            
        end

        
        %% CODE 3 - Recompute means per cluster
        % Write code below here
        clusters = zeros(K,2);
        for z = 1:K
            c = zeros(K);
            for j = 1:length(Z)
                if Z(j) == z
                    clusters(z,1) = clusters(z,1)+DATA(j,1);
                    clusters(z,2) = clusters(z,2)+DATA(j,2);
                    c(z) = c(z)+1;

                end
            end
            clusters(z,1) = clusters(z,1)/c(z);
            clusters(z,2) = clusters(z,2)/c(z);
        end
        MU = clusters;
        
        %% CODE 4 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:

        if K_prev == K && (max(max(MU-MU_prev))) < convergence_threshold
            converged = 1;
        end


        %% CODE 5 - Plot final clusters after convergence 
        % Write code below here:

        if (converged)
            %%%%
            disp('Converged');
            figure
            gscatter(DATA(:,1),DATA(:,2),Z(:)');
            xlabel('Feature X');
            ylabel('Feature Y');
            title(['DP-Clustering Lamda = ',num2str(la),' ']);
        end    
    end
end

%% 4.4c
NBA_data = [MPG,PPG];
LAMBDA = [44,100,450];
convergence_threshold = 0.1;
num_points = length(NBA_data);
total_indices = [1:num_points];
for ll = 1:3
    la = LAMBDA(ll);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% DP Means - Initializations for algorithm %%%
    % cluster count
    K = 1;
    K_prev = K;
    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points);

    % means
    MU = [];
    MU = [MU; mean(NBA_data,1)];
    MU_prev = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;
    t = 0;

    while (converged == 0)
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)
        K_prev = K;
        MU_prev = MU;
        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            d = [];
            for j = 1:K
                dtoi = (NBA_data(i,1)-MU(j,1)).^2 + (NBA_data(i,2)-MU(j,2)).^2;
                d = [d,dtoi];
            end
            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            [dd,ii] = min(d);
            if dd > la
                K = K+1;
                Z(i) = K;
                MU = [MU;NBA_data(i,:)];
            else
                Z(i) = ii;
            end
            
        end

        
        %% CODE 3 - Recompute means per cluster
        % Write code below here
        clusters = zeros(K,2);
        for z = 1:K
            c = zeros(K);
            for j = 1:length(Z)
                if Z(j) == z
                    clusters(z,1) = clusters(z,1)+NBA_data(j,1);
                    clusters(z,2) = clusters(z,2)+NBA_data(j,2);
                    c(z) = c(z)+1;

                end
            end
            clusters(z,1) = clusters(z,1)/c(z);
            clusters(z,2) = clusters(z,2)/c(z);
        end
        MU = clusters;
        
        %% CODE 4 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:

        if K_prev == K && (max(max(MU-MU_prev))) < convergence_threshold
            converged = 1;
        end


        %% CODE 5 - Plot final clusters after convergence 
        % Write code below here:

        if (converged)
            %%%%
            disp('Converged');
            figure
            gscatter(NBA_data(:,1),NBA_data(:,2),Z(:)');
            xlabel('MPG');
            ylabel('PPG');
            title(['DP-Clustering NBA DATA Lamda = ',num2str(la),' ']);
        end    
    end
end
