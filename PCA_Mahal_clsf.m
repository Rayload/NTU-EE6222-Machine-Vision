clear;
close all;
clc;

num_A = 30;
num_B = 30;

path_a = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Apple";
path_b = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Banana";

%load images
for i = 1:num_A
    filename = fullfile(path_a, sprintf('A%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_A(i,:) = img(:)';
end

for i = 1:num_B
    filename = fullfile(path_b, sprintf('B%d.jpg', i));
    img = imread(filename);
    img = im2double(img);
    data_B(i,:) = img(:)';
end

data = [data_A; data_B];
labels = [ones(num_A,1); 2*ones(num_B,1)];  % 1 = apple，2 = banana

%apply PCA
mu_data = mean(data,1);
data_ct = data - mu_data;
[coeff, score, latent] = pca(data_ct);

%dimension can be selected
d = 55;
data_pca = score(:, 1:d);

%dataset division: train set and test set
trainIdx = 1:21;
testIdx  = 22:30;

A_train = data_pca(trainIdx, :);
A_test = data_pca(testIdx, :);
B_train= data_pca(30 + trainIdx, :);
B_test = data_pca(30 + testIdx, :);

data_train = [A_train; B_train];
data_test = [A_test; B_test];

label_train = [ones(21,1); 2*ones(21,1)];
label_test = [ones(9,1); 2*ones(9,1)];

%statistical properties of train data
muA_train = mean(A_train);
muB_train = mean(B_train);

covA_train = cov(A_train);
covB_train = cov(B_train);

%Classification Using Mahalanobis Distance Classifier
predLabel = zeros(size(label_test));

for i = 1:size(data_test, 1)
    x = data_test(i,:);
    dA = (x - muA_train) * pinv(covA_train) * (x - muA_train)';
    dB = (x - muB_train) * pinv(covB_train) * (x - muB_train)';
    if dA < dB
        predLabel(i) = 1; %classified as apple
    else
        predLabel(i) = 2; %classified as banana
    end
end

%claculate classification accuracy
accuracy = mean(predLabel == label_test) * 100;
fprintf('Classification Accuracy = %.2f%%\n', accuracy);














