function [U, B, V] = householder_bidiagonalization(A)
    % Ottieni la dimensione di A
    [n, m] = size(A);

    % Inizializza le matrici ortogonali U e V come identità
    U = eye(n);
    V = eye(m);

    % Applica le trasformazioni di Householder per ottenere la matrice bidiagonale
    for k = 1:min(n, m-1)
        % Prendi la k-esima colonna a partire dalla riga k
        x = A(k:n, k);
        
        % Calcola il vettore di Householder v per annullare le componenti sotto la diagonale
        e1 = zeros(length(x), 1);
        e1(1) = 1;
        alpha = norm(x);
        v = x + sign(x(1)) * alpha * e1;
        if norm(v) > 1e-10  % Evita problemi numerici
            v = v / norm(v);
        else
            v = zeros(size(v));
        end

        % Calcola la matrice di Householder Hk per righe
        Hk = eye(n-k+1) - 2 * (v * v');

        % Applicare Hk a sinistra (aggiornare le righe di A)
        A(k:n, k:m) = Hk * A(k:n, k:m);

        % Aggiorna la matrice U
        z = zeros(k-1, n-k+1);
        Hk = [eye(k-1), z; z.', Hk];
        U = U * Hk;

        % Ora riflettiamo le colonne a partire dalla colonna k+1
        if k < m-1
            x = A(k, k+1:m)';
            e1 = zeros(length(x), 1);
            e1(1) = 1;
            alpha = norm(x);
            v = x + sign(x(1)) * alpha * e1;
            if norm(v) > 1e-10  % Evita problemi numerici
                v = v / norm(v);
            else
                v = zeros(size(v));
            end

            % Calcola la matrice di Householder Hk2 per colonne
            Hk2 = eye(m-k) - 2 * (v * v');

            % Applicare Hk2 a destra (aggiornare le colonne di A)
            A(k:n, k+1:m) = A(k:n, k+1:m) * Hk2;

            % Aggiorna la matrice V
            z = zeros(k, m-k);
            Hk2 = [eye(k), z; z.', Hk2];
            V = V * Hk2;
        end
    end

    % Gestione del caso n > m (matrici rettangolari)
    if n > m
        k = m;
        x = A(k:n, k);
        e1 = zeros(length(x), 1);
        e1(1) = 1;
        alpha = norm(x);
        v = x + sign(x(1)) * alpha * e1;
        if norm(v) > 1e-10
            v = v / norm(v);
        else
            v = zeros(size(v));
        end

        % Calcola la matrice di Householder Hk per righe
        Hk = eye(n-k+1) - 2 * (v * v');

        % Applicare Hk a sinistra (aggiornare le righe di A)
        A(k:n, k:m) = Hk * A(k:n, k:m);
        z = zeros(k-1, n-k+1);
        Hk = [eye(k-1), z; z.', Hk];
        U = U * Hk;
    end

    % La matrice A è ora in forma bidiagonale
    B = A;
end
function [U, S, V] = qr_iteration_bidiag(X, tol)
    [M, N] = size(X);
    loopmax = 100 * max(size(X));
    loopcount = 0;
    U = eye(M); % A = IAI
    S = X';
    V = eye(N);
    Err = realmax;
    while (Err > tol && loopcount < loopmax)
        % QR decomposition
        [Q, S] = qr(S'); 
        U = U * Q;
        [Q, S] = qr(S'); 
        V = V * Q;

        % Calcolo dell'errore
        e = triu(S, 1);
        E = norm(e(:));
        F = norm(diag(S));
        if F > tol
            Err = E / F;
        else
            Err = 0;
        end
        loopcount = loopcount + 1;
    end

    % Stabilisce i segni in S
    SS = diag(S);
    S = zeros(M, N);
    for n = 1:length(SS)
        SSN = SS(n);
        S(n, n) = abs(SSN);
        if SSN < 0
            U(:, n) = -U(:, n);
        end
    end
end
function [U, S, V] = svd_algorithm(X)
    [UU, B, VV] = householder_bidiagonalization(X);
    [W, S, Z] = qr_iteration_bidiag(B, 1e-10);
    U = UU * W;
    V = VV * Z;
end

% Leggere i file di dati
file1 = readmatrix("optdigits.tes", FileType="text"); % Legge il file di test
file2 = readmatrix("optdigits.tra", FileType="text"); % Legge il file di training

% Unire i dataset
dataset = [file2; file1]; % Combina i due dataset in un unico array

% Estrarre etichette e caratteristiche
labels = dataset(:, 65);       % L'ultima colonna contiene le etichette (valori target)
features = dataset(:, 1:64);   % Le prime 64 colonne sono le feature (valori input)

[U, S, V] = svd_algorithm(features);
k = 4; %numero di valori singolari da selezionare
V_k = V(:, 1:k);
features = features * V_k; %riduzione della dimensionalità
eigenvalues = diag(S).^2; %calcolo degli autovalori
quality = sum(eigenvalues(1:k))/sum(eigenvalues) %valutazione della qualità rispetto ai dati originali
size(features)
 
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