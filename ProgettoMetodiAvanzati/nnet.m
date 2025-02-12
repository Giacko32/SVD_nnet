% Leggere i file di dati
file1 = readmatrix("optdigits.tes", FileType="text"); % Legge il file di test
file2 = readmatrix("optdigits.tra", FileType="text"); % Legge il file di training

% Unire i dataset
dataset = [file2; file1]; % Combina i due dataset in un unico array

% Estrarre etichette e caratteristiche
labels = dataset(:, 65);       % L'ultima colonna contiene le etichette (valori target)
features = dataset(:, 1:64);   % Le prime 64 colonne sono le feature (valori input)

% Percentuale di dati da usare per il training
train_ratio = 0.8; % 80% dei dati sarà usato per il training

% Numero totale di campioni
num_samples = size(features, 1); % Numero totale di righe nel dataset

% Indici casuali per il training e il test
random_indices = randperm(num_samples); % Mescola casualmente gli indici dei dati
num_train = round(train_ratio * num_samples); % Calcola il numero di campioni per il training
train_indices = random_indices(1:num_train); % Indici per i dati di training
test_indices = random_indices(num_train+1:end); % Indici per i dati di test

% Dividere dati in training e test
X_train = features(train_indices, :); % Feature per il training
y_train = labels(train_indices);      % Etichette per il training
X_test = features(test_indices, :);   % Feature per il test
y_test = labels(test_indices);        % Etichette per il test

% Creazione della rete neurale
layers = [
    featureInputLayer(size(X_train, 2), "Name", "featureinput") % Livello di input per 64 feature
    fullyConnectedLayer(40, "Name", "fc1")                     % Primo livello fully-connected con 40 neuroni
    reluLayer("Name", "relu1")                                 % Livello ReLU
    batchNormalizationLayer("Name", "batchnorm1")             % Batch normalization
    dropoutLayer(0.1, "Name", "dropout1")                     % Dropout con probabilità 0.1
    fullyConnectedLayer(20, "Name", "fc2")                    % Secondo livello fully-connected con 20 neuroni
    reluLayer("Name", "relu2")                                % Livello ReLU
    batchNormalizationLayer("Name", "batchnorm2")            % Batch normalization
    dropoutLayer(0.1, "Name", "dropout2")                    % Dropout con probabilità 0.1
    fullyConnectedLayer(10, "Name", "fc3")                   % Terzo livello fully-connected con 10 neuroni (numero di classi)
    softmaxLayer("Name", "softmax")                          % Livello softmax per calcolare probabilità per ciascuna classe
    classificationLayer("Name", "classification")];          % Livello di classificazione per calcolare l'errore

% Opzioni di allenamento
options = trainingOptions("adam", ...
    MaxEpochs=300, ...                             % Numero massimo di epoche
    InitialLearnRate=0.0005, ...                   % Velocità di apprendimento iniziale
    GradientThreshold=1, ...                       % Soglia per il gradiente (per evitare esplosioni del gradiente)
    ValidationData={X_test, categorical(y_test)}, ... % Dati di validazione (feature e etichette)
    Shuffle="every-epoch", ...                     % Mescola i dati a ogni epoca
    Plots="training-progress", ...                 % Mostra il grafico dei progressi di allenamento
    MiniBatchSize=1000, ...                        % Dimensione del mini-batch
    Verbose=false);                                % Disattiva output dettagliati nella console

% Allenare la rete
trainedNet = trainNetwork(X_train, categorical(y_train), layers, options);