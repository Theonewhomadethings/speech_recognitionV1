function logAlpha = forward_algorithm(O, logPi, logA, mu, Sigma)
    % O: observation sequence matrix (each row represents a frame)
    % logPi: log initial state probabilities
    % logA: log state transition probabilities
    % mu: mean parameters for Gaussian emission model
    % Sigma: covariance matrix parameters for Gaussian emission model

    N = size(logA, 1);   % Number of states
    T = size(O, 1);   % Number of frames

    % Initialize the log forward likelihood matrix
    logAlpha = zeros(N, T);

    % Step 1: Initialization
    logAlpha(:, 1) = logPi + log(mvnpdf(O(1, :), mu(1,:), Sigma(:,:,1)))';

    % Step 2: Recursion
    for t = 2:T
        for j = 1:N
            logAlpha(j, t) = logsumexp(logAlpha(:, t-1) + logA(:, j), 1) + log(mvnpdf(O(t, :), mu(j, :), Sigma(:,:,j)));
        end
    end
end

function result = logsumexp(x, dim)
    % Log-sum-exp trick to prevent numerical instability

    % Find the maximum value along the specified dimension
    max_x = max(x, [], dim);

    % Compute the log-sum-exp
    result = max_x + log(sum(exp(x - max_x), dim));
end
