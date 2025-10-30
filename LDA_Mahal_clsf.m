clear;
close all;
clc;

num_A = 30;
num_B = 30;

data = [];
labels = [];

path_a = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Apple";
path_b = "D:\Johnson\NTU\Msc\Sem1\EE6222 Machine Vision\Assignment 6222\Assignment 1\Dataset\Banana";

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

ldaModel = fitcdiscr(data, labels, 'DiscrimType', 'linear');

% linear direction of class 1 and class 2
W = ldaModel.Coeffs(1,2).Linear;

% project data onto LDA direction 将数据投影到LDA方向上
dataLDA = data * W;

% train and test
trainIdx = [1:21, 31:51]; % first 10 images for training
testIdx  = [22:30, 52:60]; % last 10 images for testing

trainData = dataLDA(trainIdx,:);
trainLabel = labels(trainIdx);
testData = dataLDA(testIdx,:);
testLabel = labels(testIdx);


class1 = trainData(trainLabel==1,:);
class2 = trainData(trainLabel==2,:);

mu1 = mean(class1);
mu2 = mean(class2);

cov1 = cov(class1);
cov2 = cov(class2);


predLabel = zeros(size(testLabel));

for i = 1:length(testLabel)
    x = testData(i,:);

    % Mahalanobis Distance
    d1 = sqrt((x - mu1) / cov1 * (x - mu1)');
    d2 = sqrt((x - mu2) / cov2 * (x - mu2)');
    
    if d1 < d2
        predLabel(i) = 1;
    else
        predLabel(i) = 2;
    end
end


accuracy = sum(predLabel == testLabel) / length(testLabel) * 100;
fprintf('Classification accuracy = %.2f%%\n', accuracy);






