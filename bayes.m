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

total_correct_sum = 0;

it_correct_vec = [];

for k=1:K
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

  u1 = (sum(class_1_features)/length(class_1_features)).';
  u2 = (sum(class_2_features)/length(class_2_features)).';
  u3 = (sum(class_3_features)/length(class_3_features)).';

  % for some reason the covmle fn wants the mean as a col vec, so transposed
  cov1 = covmle(class_1_features, u1);
  cov2 = covmle(class_2_features, u2);
  cov3 = covmle(class_3_features, u3);

  it_correct_sum = 0;

  for i=1:NK
    x = test_data(i,1:D);
    class = test_data(i,D+1);

    xm1 = x.' - u1;
    g1 = -0.5 * xm1.' * inv(cov1) * xm1;

    xm2 = x.' - u2;
    g2 = -0.5 * xm2.' * inv(cov2) * xm2;

    xm3 = x.' - u3;
    g3 = -0.5 * xm3.' * inv(cov3) * xm3;

    [_,predicted_class] = max([g1, g2, g3]);

    if predicted_class == class
      it_correct_sum += 1;
      total_correct_sum += 1;
    end
  end

  it_correct_vec = vertcat(it_correct_vec, it_correct_sum/NK);

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
end

disp(it_correct_vec);
disp(total_correct_sum/N);

% Evaluate classification accuracy
%   Accuracy per iteration = no of correct classification / 15
%   Average accuracy for all 10-fold CV
%   fprintf('Accuracy = %5.4f\n', ...) generates nice format
