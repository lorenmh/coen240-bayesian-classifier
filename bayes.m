clear all; close all; clc;
data_raw = dlmread('corrupted_iris_dataset.dat');

N = 150;  % total number of samples
D = 4; % num features
NC = 50;  % size of each class
K = 10;  % K-fold
NK = N/K; % Number of examples per fold

% Randomly shuffle data
index = randperm(N);
data = data_raw(index,:);

%for k=1:K
k=1
  test_start_index = (k-1)*NK + 1;
  test_end_index = (test_start_index+NK) - 1;

  test_data = data(test_start_index:test_end_index,:);

  train_data = vertcat(
    data(1:test_start_index-1,:),
    data(test_end_index+1:N,:)
  );

  class_1_data = train_data(train_data(:,D+1) == 1,:);
  class_2_data = train_data(train_data(:,D+1) == 2,:);
  class_3_data = train_data(train_data(:,D+1) == 3,:);

  class_1_features = class_1_data(:,1:D);
  class_2_features = class_2_data(:,1:D);
  class_3_features = class_3_data(:,1:D);

  class_1_mean = sum(class_1_features)/length(class_1_features);
  class_2_mean = sum(class_2_features)/length(class_2_features);
  class_3_mean = sum(class_3_features)/length(class_3_features);

  % for some reason the covmle fn wants the mean as a col vec, so transposed
  class_1_cov = covmle(class_1_features, class_1_mean.');
  class_2_cov = covmle(class_2_features, class_2_mean.');
  class_3_cov = covmle(class_3_features, class_3_mean.');

  for i=1:NK
    x = test_data(i,1:D);
    y = test_data(i,D+1);

    c1_xm = x-class_1_mean;
    c1_g = -0.5 * c1_xm * class_1_cov * c1_xm.';

    c2_xm = x-class_2_mean;
    c2_g = -0.5 * c2_xm * class_2_cov * c2_xm.';

    c3_xm = x-class_3_mean;
    c3_g = -0.5 * c3_xm * class_3_cov * c3_xm.';


    disp([c1_g, c2_g, c3_g, y]);
  end

  % disp(train_data);

  % TRAINING:
  % Separate training data and test data
  % Using training data from each class, find mean_mle (u1, u2, u3)
  % and cov_mle (cov1, cov2, cov3)
  % Do not use MATLAB cov function, use covmle provided with
  % this assignment
  % TESTING:
  % Using u1, u2, u3, and cov1, cov2, cov3 found in the training phase,
  % and test data (x), compute the discriminant function for each class,
  % g1, g2, g3.  Assume prior=1/3 for each class.
  % You can use MATLAB inv function for matrix inversion
  % Predicted class label is the largest of g1, g2, g3
  % Check predicted label against the given label in the test data set
%end
% Evaluate classification accuracy
%   Accuracy per iteration = no of correct classification / 15
%   Average accuracy for all 10-fold CV
%   fprintf('Accuracy = %5.4f\n', ...) generates nice format
