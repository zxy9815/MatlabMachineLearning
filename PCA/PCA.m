
function [lambda5, k_] = zxy9815_hw6_2()
%% Q6.2
%% Load AT&T Cambridge, Face images data set
    img_size = [112,92];   % image size (rows,columns)
    % Load the ATT Face data set using load_faces()
    %%%%% TODO
    faces = load_faces();
    [n,d] = size(faces);

    %% Compute mean face and the Auto Covariance Matrix
    % compute X_tilde
    %%%%% TODO
    mean_face = (mean(faces))';
    % Compute covariance using X_tilde
    %%%%% TODO
    cov = (1/n) * ((faces'-mean_face)*(faces'-mean_face)');
    %% Find Eigen Value Decomposition of auto covariance matrix
    %%%%% TODO
    [V,D] = eig(cov);
       
    %% Sort eigen values and corresponding eigen vectors and obtain U, Lambda
    %%%%% TODO
    dia = diag(D);
    [sorted_d,index] = sort(dia,'descend');
    U = zeros(d,d);
    for i = 1:d
        U(:,i) = V(:,index(i));
    end
    Lambda = sorted_d.*eye(d);
    
    %% Find principle components: Y
    %%%%% TODO

%% Q6.2 a) Visualize loaded images and mean face
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image 120 in the dataset
    % practise using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(faces(120,:), img_size)));
    title('image 120 in the dataset');
    % Visualize the Average face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(mean_face, img_size)));
    title('the Average face image');

%% Q6.2 b) Analysing computed eigen values
    warning('off')
    
    % Report first 5 eigen values
    lambda5 = sorted_d(1:5,:); 
    fprintf('6.2b: first 5 eigenvalues:\n');
    disp(lambda5);
    
    % Plot trends in Eigen values and k
    k = 1:d;
    figure(2)
    sgtitle('Eigen Value trends w.r.t k')

    % Plot the eigen values w.r.t to k
    subplot(1,2,1)
    %%%%% TODO
    plot(k(1:450),sorted_d(1:450));
    title('first 450 eigenvalues vs. k');
    xlabel('K');
    ylabel('eigenvalues');
    % Plot running sum of eigen vals over total sum of eigen values w.r.t k
    %%%%% TODO: Compute eigen fractions
    rho_k = zeros(450,1);
    sum_d = sum(sorted_d);
    for j = 1:450
        rho_k(j) = sum(sorted_d(1:j))/sum_d;
    end
        
    subplot(1,2,2)
    %%%%% TODO
    plot(k(1:450),rho_k);
    title('eigen fractions vs. k');
    xlabel('K');
    ylabel('eigen fractions');
    % find & report k for which Eig fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    %%%%% TODO (Hint: ismember())
    % k_ = ?; %%%%% TODO
    k_ = zeros(5,1);
    [is_member,ef_ind] = ismember(ef,round(rho_k,2));
    fprintf('6.2b: k for which Eig fraction:\n');
    for j = 1:length(ef_ind)
        fprintf('k for Eig fraction %.2f is: %d\n',ef(j),k(ef_ind(j)));
        k_(j) = k(ef_ind(j));
    end
   
%% Q6.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = faces(test_img_idx,:);    
    % Computing eigen face coefficients
    %%%% TODO
    K = [0,1,2,k_',400,d];
    eigen_coe = zeros(d,length(K));
    for i = 2:length(K)
        coe = zeros(d,1);
        for j = 1:K(i)
            coe = coe + ((U(:,j)'*(test_img'-mean_face))*U(:,j));
        end
        eigen_coe(:,i) = coe;
    end
            
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to mean face

    % plot the resulatant images from progress of adding eigen faces to the 
    % mean face in a single figure using subplots.

    %%%% TODO 
    %%
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    for i = 1:10
        subplot(5,2,i)
        imshow(uint8(reshape((mean_face+eigen_coe(:,i)), img_size)));
        title(['eigen face of k = ',num2str(K(i)),' ']);
    end
    

%% Q6.2 d) Principle components and corresponding properties in images
    %% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of quantile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    m = load('mnist256.mat');
    mnist = m.mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,d2] = size(X);
    
    %% Compute mean face and the Auto Covariance Matrix
    % compute X_tilde
    %%%%% TODO
    mu_image = (mean(X))';
    % Compute covariance using X_tilde
    %%%%% TODO
    cov_image = (1/n) * ((X'-mu_image)*(X'-mu_image)');
    %% Find Eigen Value Decomposition
    %%%%% TODO
    [V2,D2] = eig(cov_image);
    %% Sort eigen values and corresponding eigen vectors
    %%%%% TODO
    dia2 = diag(D2);
    [sorted_d2,ind] = sort(dia2,'descend');
    %U2 = zeros(d2,d2);
    U2 = V2(:,ind);
    %{
    for i = 1:d2
        U2(:,i) = V2(:,ind(i));
    end
    %}
    Lambda2 = sorted_d2.*eye(d2);
   
    %% Computing first 2 priciple components
    %%%%% TODO
    %y2 is ﬁrst two principal components for all images (2 x 658)
    y2 = U2(:,1:2)'*(X'-mu_image);
    % finding quantile points
    quantile_vals = [0.05, .25, .5, .75, .95];
    %%%%% TODO (Hint: Use the provided fucntion - quantile_points())
    percent_1 = percentile_values(y2(1,:)',quantile_vals*100);
    percent_2 = percentile_values(y2(2,:)',quantile_vals*100);
    fprintf('6.2d: first 2 principle components of 5,25,50,75,95 percentile:\n');
    disp([percent_1,percent_2]);
    % Finding the cartesian product of quantile points to find grid corners
    %%%%% TODO
    percentiles = zeros(25,2);
    corners = zeros(25,2);
    for i = 1:5
        for j = 1:5
            percentiles((i-1)*5+j,2) = quantile_vals(j)*100;
            corners((i-1)*5+j,2) = percent_2(j);
            percentiles((i-1)*5+j,1) = quantile_vals(i)*100;
            corners((i-1)*5+j,1) = percent_1(i);
        end
    end
        
            
    
    %% find closest coordinates to grid corner coordinates    
    % and  Find the actual images with closest coordinate index 
    closest = zeros(25,2); %closest coordinates to grid corner coordinates
    coor_ind = zeros(25,1); %closest coordinate index
    close_image = zeros(25,256); %Actual images of closest coordinate index
    for i = 1:length(corners(:,1))
        dist2 = sum((y2' - corners(i,:)) .^ 2, 2);
        [~,coor_ind(i)] = min(dist2);
        closest(i,:) = y2(:,coor_ind(i))';
        close_image(i,:) = X(coor_ind(i),:);
    end
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 120th image
    subplot(1,2,1)
    %%%%% TODO
    imshow(reshape(X(120,:), img_size));
    title('image 120 in the dataset');
    % Average face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(reshape(mu_image, img_size));
    title('the Average image');
    
    %% Image Projections on Principle components and the corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the Principle component 1 vs 2, Principle component. Draw the
    % grid formed by the quantile points and highlight points closest to the 
    % quantile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)
    scatter(y2(1,:),y2(2,:));
    xticks(percent_1);
    xticklabels({'5^t^h','25^t^h','50^t^h','75^t^h','95^t^h'})
    yticks(percent_2);
    yticklabels({'5^t^h','25^t^h','50^t^h','75^t^h','95^t^h'})
    %plot points closest to the quantile grid corners
    scatter(closest(:,1),closest(:,2),'filled','r');
    xlabel('Principle component 1')
    ylabel('Principle component 2')
    title('Closest points to quantile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images at corresponding red dots')
    hold on
    % Plot the images corresponding to points closest to the quantile grid 
    % corners. Use subplot to put all images in a single figure in a grid
    
    %%%%% TODO
    for i = 1:length(coor_ind)
        subplot(5,5,i)
        imshow(reshape(close_image(i,:), img_size));
        title(['percentile1 = ',num2str(percentiles(i,1)),', percentile2 = ',num2str(percentiles(i,2))]);
    end
    hold off 

end
