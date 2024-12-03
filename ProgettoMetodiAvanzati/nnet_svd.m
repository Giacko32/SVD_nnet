file1 = readmatrix("optdigits.tes", FileType="text");
file2 = readmatrix("optdigits.tra", FileType="text");

dataset = [file2; file1];
%feature extraction

labels = dataset(:, 65);       % Prima colonna come etichette
features = dataset(:, 1:64); % Tutte le colonne dalla seconda in poi come features

[U, S, V] = svd(features, "econ");

n_columns = 5;

%calcolo della accuracy della approssimazione
sing_values = diag(S^2);
information_taken = sum(sing_values(1:n_columns));
total_information = sum(sing_values);
accuracy = information_taken/total_information


features = features * V(:,1:n_columns);

% Percentuale di dati da usare per il training
train_ratio = 0.7; 

% Numero di campioni
num_samples = size(features, 1);

% Indici casuali per mescolare i dati
random_indices = randperm(num_samples);

% Indici per training e test
num_train = round(train_ratio * num_samples);
train_indices = random_indices(1:num_train);
test_indices = random_indices(num_train+1:end);

% Suddivisione in train e test set
X_train = features(train_indices, :); % Features del training set
y_train = labels(train_indices);      % Labels del training set
X_test = features(test_indices, :);   % Features del test set
y_test = labels(test_indices);        % Labels del test set

num_classes = 10; % Per classificazione binaria (ad esempio 0 e 1)
y_train = full(ind2vec(y_train' + 1, num_classes))'; % Converti 0 → [1, 0], 1 → [0, 1]
y_test = full(ind2vec(y_test' + 1, num_classes))';


net = dlnetwork;
tempNet = [
    featureInputLayer(n_columns,"Name","featureinput")
    fullyConnectedLayer(40,"Name","fc")
    reluLayer("Name","relu")
    batchNormalizationLayer("Name","batchnorm")
    dropoutLayer(0.1,"Name","dropout")
    fullyConnectedLayer(20,"Name","fc_1")
    reluLayer("Name","relu_1")
    batchNormalizationLayer("Name","batchnorm_1")
    dropoutLayer(0.1,"Name","dropout_1")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")];
net = addLayers(net,tempNet);

% clean up helper variable
clear tempNet;
net = initialize(net);
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    InitialLearnRate=0.0005, ...
    GradientThreshold=1, ...
    ValidationData={X_test,y_test}, ...
    Shuffle = "every-epoch", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    MiniBatchSize=1000,...
    Verbose=false);

trainnet(X_train, y_train, net, "crossentropy", options)
