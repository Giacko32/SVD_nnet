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

function [compressedImg, quality] = compressWithSVD(img, k)

    img = im2double(img);

    [rows, cols, ~] = size(img);
    compressedImg = gpuArray.zeros(rows, cols, 3);
    quality = 0;
    S_mean = zeros(rows, 1);
    for c = 1:3 
        channel = gpuArray(img(:, :, c));
        [U, S, V] = svd_algorithm(channel);
        U_k = U(:, 1:k); 
        S_k = S(1:k, 1:k);
        V_k = V(:, 1:k);
        S_mean = S_mean + diag(S);
        channel = U_k*S_k*V_k';
        quality = quality + sum(diag(S_k).^2)/sum(diag(S).^2);
        compressedImg(:, :, c) = channel;
    end
    S_mean = S_mean/rows
    plot(S_mean)
    quality = quality/3
    compressedImg = gather(im2uint8(compressedImg));
end

folder = "resized_dataset\fire";
allFiles = dir(folder); % Elenco di tutti i file nella cartella
fileList = allFiles(contains({allFiles.name}, {'.png', '.jpg'}));
randomIndex = randi(length(fileList), 1, 1); % Get random number.;
fullFileName = fullfile(folder, fileList(randomIndex).name);
img = imread(fullFileName);

[compressedImg, q] = compressWithSVD(img, 14);

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(compressedImg);
title(['Compressed Image (Energy: ', num2str(q * 100, '%.2f'), '%)']);



