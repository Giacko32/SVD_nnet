dataset = readmatrix("breast-cancer.csv");
%feature extraction
dataset = dataset(2:end, 3:end);

labels = dataset(:, 1);       % Prima colonna come etichette
features = dataset(:, 2:end); % Tutte le colonne dalla seconda in poi come features

% Percentuale di dati da usare per il training
train_ratio = 0.8; 

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

num_classes = 2; % Per classificazione binaria (ad esempio 0 e 1)
y_train = full(ind2vec(y_train' + 1, num_classes))'; % Converti 0 → [1, 0], 1 → [0, 1]
y_test = full(ind2vec(y_test' + 1, num_classes))';



net = dlnetwork;
tempNet = [
    featureInputLayer(30,"Name","featureinput")
    fullyConnectedLayer(40,"Name","fc")
    reluLayer("Name","relu")
    batchNormalizationLayer("Name","batchnorm")
    dropoutLayer(0.2,"Name","dropout")
    fullyConnectedLayer(20,"Name","fc_1")
    reluLayer("Name","relu_1")
    batchNormalizationLayer("Name","batchnorm_1")
    dropoutLayer(0.2,"Name","dropout_1")
    fullyConnectedLayer(2,"Name","fc_2")
    softmaxLayer("Name","softmax")];
net = addLayers(net,tempNet);

% clean up helper variable
clear tempNet;
net = initialize(net);
options = trainingOptions("adam", ...
    MaxEpochs=300, ...
    InitialLearnRate=0.0005, ...
    GradientThreshold=1, ...
    ValidationData={X_test,y_test}, ...
    Shuffle = "every-epoch", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

trainnet(X_train, y_train, net, "crossentropy", options)
