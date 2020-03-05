%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2020
% HW 5
% Xinyuan Zhao, zxy9815@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta11 = 0*pi/6;
[X11, Y11] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta11);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1_1 = X11(:, Y11==1);
X2_1 = X11(:, Y11==2);

figure(1);subplot(2,2,1);
scatter(X1_1(1,:),X1_1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2_1(1,:),X2_1(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
%% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,2);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%
%iii
theta3 = pi/3;
[X33, Y33] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta3);
X1_3 = X33(:, Y33==1);
X2_3 = X33(:, Y33==2);

figure(1);subplot(2,2,3);
scatter(X1_3(1,:),X1_3(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(2),'\times \pi/6']);
scatter(X2_3(1,:),X2_3(2,:),'^','fill','r');
axis equal;

%iv
[X44, Y44] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda2, lambda1, theta);
X1_4 = X44(:, Y44==1);
X2_4 = X44(:, Y44==2);

figure(1);subplot(2,2,4);
scatter(X1_4(1,:),X1_4(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2_4(1,:),X2_4(2,:),'^','fill','r');
axis equal;

%% Observation:
%theta = directon, eigenvalue = spread
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m ii] = max(signal_power_array);
phi_max = phi_array(ii);
fprintf('5.3 c results\n');
fprintf('phi that maximizes the squared distance btw class means: %f \n', phi_max);
[m ii] = min(noise_power_array);
phi_max = phi_array(ii);
fprintf('phi that minimizes the within class variance: %f \n', phi_max);
[m ii] = max(snr_array);
phi_max = phi_array(ii);
fprintf('phi that maximizes the SNR: %f \n', phi_max);
figure
plot(phi_array,signal_power_array,'r-');
hold on
plot(phi_array,noise_power_array,'g-');
hold on
plot(phi_array,snr_array,'b-');
hold off
xlabel('Phi');ylabel('Signal, Noise and SNR');
title('Signal, Noise and SNR against Phi');
legend('Signal','Noise','SNR');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = [0,pi/6,pi/3]
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, j, true);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 
% Generate dataset (i) 
w_LDA = LDA(X,Y);


% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);
mu1x = mean(X1,2);
mu2x = mean(X2,2);
v_diff = [mu1x,(mu2x-mu1x)];
v_w = [mu1x,w_LDA];

figure
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
hold on
plot(v_diff(1,1),v_diff(2,1),'k*');
quiver(v_diff(1,1),v_diff(2,1),v_diff(1,2),v_diff(2,2),'LineWidth',2,'Color','g');
quiver(v_w(1,1),v_w(2,1),v_w(1,2),v_w(2,2),'LineWidth',2,'Color','y');
hold off
legend('Class 1','Class 2','mu1x','Difference Vector','W_LDA Vector');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot
n=length(X);
X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
figure
plot(b_array,ccr_array);
xlabel('Value of b');ylabel('CCR');
title('Plot of CCR vs. b');

[m ii] = max(ccr_array);
ccr_max = ccr_array(ii);
fprintf('5.3d results\n');
fprintf('b that maximizes the CCR: %f, Max_CCR: %f \n', b_array(ii),ccr_max);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2


%%%%%%%%%%%%%%%%%%%%%%
%Insert your code here
%%%%%%%%%%%%%%%%%%%%%%
ei_value = [lambda1,0;0,lambda2];
ei_vector = [cos(theta),sin(theta);sin(theta),-cos(theta)];
cov = ei_vector* ei_value* ei_vector';
x1 = mvnrnd(mu1,cov,n1);
x2 = mvnrnd(mu2,cov,n2);
X = [x1;x2]';
y1 = ones(1,n1);
y2 = 2*ones(1,n2);
Y = [y1,y2];
end
%%
function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1 = X(:, Y==1);
x2 = X(:, Y==2);
n = length(x1)+length(x2);
w = [cos(phi),sin(phi)];
mu1z = mean(w*x1);
mu2z = mean(w*x2);
sigma1 = var(w*x1);
sigma2 = var(w*x2);

signal = (mu2z-mu1z)^2;
noise = (length(x1)/n)*sigma1 + (length(x2)/n)*sigma2;
snr = signal/noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity((w*x1)');
    plot(z1,pdf1)
    hold on;
    [pdf2,z2] = ksdensity((w*x2)');
    plot(z2,pdf2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title(['Estimated class density estimates of data projected along \phi = ',num2str(phi/(pi/6)),' \times \pi/6. Ground-truth \phi = \pi/6'])
end

end
%%
function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
x1 = X(:, Y==1);
x2 = X(:, Y==2);
n = length(x1)+length(x2);
mu1x = mean(x1,2);
mu2x = mean(x2,2);
s1 = zeros(2,2);
s2 = zeros(2,2);
for i = 1:length(x1)
    s1 = s1 + ((x1(:,i) - mu1x)*(x1(:,i) - mu1x)');
end
s1 = s1/length(x1);
for j = 1:length(x2)
    s2 = s2 + ((x2(:,j) - mu2x)*(x2(:,j) - mu2x)');
end
s2 = s2/length(x2);
s_avg = (length(x1)/n)*s1 + (length(x2)/n)*s2;
w_LDA = inv(s_avg)*(mu2x-mu1x);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
y_pred = zeros(1,length(X));
for i = 1:length(X)
    if (w_LDA'*X(:,i) + b) <= 0
        y_pred(i) = 1;
    else
        y_pred(i) = 2;
    end
end
c = confusionmat(Y,y_pred);
ccr = trace(c)/length(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end