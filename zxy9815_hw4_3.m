% EC 414 - HW 4 - Spring 2020
%problem 4.3
%Xinyuan Zhao

clear, clc, close all,

%% Generate Gaussian data:

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

%% Choose K (4.2d)
DATA = [R1;R2;R3];
K = [2,3,4,5,6,7,8,9,10];
lamda = [1,3,5,7];
for l = 1:4
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
        WCSS_k(ik) = min(WCSS_random)+lamda(l)*K_val;
    end
    figure
    plot(K,WCSS_k);
    xlabel('Value of K');
    ylabel('f(k,lamda)');
    title(['Choose K Plot for lamda = ',num2str(lamda(l)),' ']);
end


%% Observation:
%As lamda increases, there is a more and more obvious 'elbow' in the plot
%of f(k,lamda) against k. Since the panelty increases as either lamda or k
%increases, it is much more easier for us to observe the elbow and choose
%that value of k if we set a higher lamda.