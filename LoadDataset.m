function [Xtrain, Xtest, ytrain, ytest] = LoadDataset(name)
% load available dataset
% @param
% name: 'rcv1', 'sido', 'toy'

seed = RandStream.create('mcg16807', 'seed', 5);

if strcmp(name, 'sido')
    %sido dataset
    test_size = 0.33;
    load('../data/sido2_matlab/sido2_train.mat');
    load('../data/sido2_matlab/sido2_train.targets');
    [n, ~] = size(X);
    y = sido2_train;

    ordering = randperm(seed, n);
    y = y(ordering, :);
    X = X(ordering, :);
    split = round(n * (1 - test_size));
    Xtrain = X(1:split, :);
    Xtest = X(split+1:end, :);
    ytrain = y(1:split);
    ytest = y(split+1:end);

    % sido dataset is ready

elseif strcmp(name, 'covtype')
    % covtype dataset
    test_size = 0.33;
    [y, X] = libsvmread('../data/covtype/covtype.libsvm.binary.scale');
    y(y == 2) = -1;
    [n, ~] = size(X);
    ordering = randperm(seed, n);
    y = y(ordering, :);
    X = X(ordering, :);
    split = round(n * (1 - test_size));
    Xtrain = X(1:split, :);
    Xtest = X(split+1:end, :);
    ytrain = y(1:split);
    ytest = y(split+1:end);
    % covtype dataset is ready

elseif strcmp(name, 'rcv1')
    % rcv1 dataset
    [ytrain, Xtrain] = libsvmread('../data/RCV1/rcv1_train.binary');
    [ytest, Xtest] = libsvmread('../data/RCV1/rcv1_test.binary');

    [n, ~] = size(Xtrain);
    ordering = randperm(seed, n);
    Xtrain = Xtrain(ordering, :);
    ytrain = ytrain(ordering);
    % rcv1 dataset is ready

elseif strcmp(name, 'avazu')
    test_size = 0.33;
    % avazu dataset
    [y, X] = libsvmread('../data/avazu/avazu-app');
    [n, ~] = size(X);
    y(y == 0) = -1;

    ordering = randperm(seed, n);
    y = y(ordering, :);
    X = X(ordering, :);
    split = round(n * (1 - test_size));
    Xtrain = X(1:split, :);
    Xtest = X(split+1:end, :);
    ytrain = y(1:split);
    ytest = y(split+1:end);
    % avazu dataset is ready

elseif strcmp(name, 'MNIST')
    % MNIST dataset
    Xtrain = loadMNISTImages('../data/MNIST/train-images.idx3-ubyte')';
    ytrain = loadMNISTLabels('../data/MNIST/train-labels.idx1-ubyte');

    Xtest = loadMNISTImages('../data/MNIST/t10k-images.idx3-ubyte')';
    ytest = loadMNISTLabels('../data/MNIST/t10k-labels.idx1-ubyte');
    trainValid = (ytrain <=  1);
    Xtrain = Xtrain(trainValid, :);
    ytrain = ytrain(trainValid);
    ytrain(ytrain == 0) = -1;

    [n, ~] = size(Xtrain);
    ordering = randperm(seed, n);
    Xtrain = Xtrain(ordering, :);
    ytrain = ytrain(ordering);

    testValid = (ytest <= 1);
    Xtest = Xtest(testValid, :);
    ytest = ytest(testValid, :);
    ytest(ytest == 0) = -1;
    % MNIST dataset is ready

elseif strcmp(name, 'newtoy')
    %prepare synthetic data set
    n = 500;
    d = 300;
    X = randn(n,d);

    % planting eigenvalues
    [u,~,v] = svd(X,'econ');
    s =  1./(1:min(n,d));
    s = s / sqrt(sum(s.^2)) * sqrt(2*n); % planting ||K||_F^2 / lambda / gamma = n
    X = u * diag(s) * v';

    % preparing data
    xpred = randn(d,1);
    noise = randn(n,1);
    y =   sign(X * xpred + .1 * noise * norm(X * xpred,'fro') / norm( noise, 'fro'));

    ytrain = y(1:n/2);
    ytest = y(n/2+1:end);
    Xtrain = X(1:n/2,:);
    Xtest = X(n/2+1:end,:);
    save('../data/toy_dataset.mat', 'Xtrain', 'Xtest', 'ytrain', 'ytest');
    %end data set creation

elseif strcmp(name, 'toy')
    load('../data/toy/toy_dataset.mat');
end

Xtrain = Xtrain';
Xtest = Xtest';
