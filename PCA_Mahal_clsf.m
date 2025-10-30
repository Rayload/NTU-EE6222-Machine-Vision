clear;
close all;
clc;

num_A_train = 21;
num_B_train = 21;

num_A_test = 9;
num_B_test = 9;


%image path
path_a_train = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Train\Apple";
path_b_train = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\\Train\Banana";

path_a_test = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Test\Apple";
path_b_test = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\\Test\Banana";


%load train set
for i = 1:21
    filename = fullfile(path_a_train, sprintf('A%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_a_train(i,:) = img(:)';
end

for i = 1:21
    filename = fullfile(path_b_train, sprintf('B%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_b_train(i,:) = img(:)';
end

i = 0;

%load test set
for i = 22:30
    filename = fullfile(path_a_test, sprintf('A%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_a_test(i-21,:) = img(:)';
end

for i = 22:30
    filename = fullfile(path_b_test, sprintf('B%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_b_test(i-21,:) = img(:)';
end

data_train = [data_a_train; data_b_train];
data_test = [data_a_test; data_b_test];


%labels for trainset and test set
label_train = [ones(num_A_train, 1); 2*ones(num_B_train, 1)];
label_test = [ones(num_A_test, 1); 2*ones(num_B_test, 1)]; 

%apply PCA for train set
mu_train = mean(data_train, 1);
data_train_ctrl = data_train - mu_train;
[coeff, score, latent] = pca(data_train_ctrl);

%transfer test set
d = 10; %number of principal component can be selected
data_pca_train = score(:, 1:d);
data_pca_test = (data_test - mu_train) * coeff (:, 1:d);

data_pca_train_a = data_pca_train(1:21, :);
data_pca_train_b = data_pca_train(22:42, :);
data_pca_test_a = data_pca_test(1:9, :);
data_pca_test_b = data_pca_test(10:18, :);

% train data mean and cov calculation
mu_pca_train_a = mean(data_pca_train_a);
mu_pca_train_b = mean(data_pca_train_b);

cov_pca_train_a = cov(data_pca_train_a);
cov_pca_train_b = cov(data_pca_train_b);

label_pred = zeros(size(label_test));

for i = 1:size(data_test, 1)
    x = data_pca_test(i,:);
    dA = (x - mu_pca_train_a) * pinv(cov_pca_train_a) * (x - mu_pca_train_a)';
    dB = (x - mu_pca_train_b) * pinv(cov_pca_train_b) * (x - mu_pca_train_b)';
    if dA < dB
        label_pred(i) = 1; %classified as apple
    else
        label_pred(i) = 2; %classified as banana
    end
end

%claculate classification accuracy
accuracy = mean(label_pred == label_test) * 100;
fprintf('Classification Accuracy = %.2f%%\n', accuracy);
















