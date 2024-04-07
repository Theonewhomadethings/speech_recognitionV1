function logBeta = backward_algorithm(obsSeq, trans, mu, Sigma)
    % obsSeq: Observation sequence matrix (each row represents a frame)
    % trans: Transition probabilities matrix
    % mu: Mean parameters for Gaussian emission model
    % Sigma: Covariance matrix parameters for Gaussian emission model

    N = size(trans, 1);   % Number of states
    T = size(obsSeq, 1);  % Number of frames

    % Initialize the log backward matrix
    logBeta = zeros(N, T);

    % Step 1: Initialization
    for i = 1:N
        logBeta(i, T) = 0;  % Log(1) = 0
    end

    % Step 2: Recursion
    for t = T-1:-1:1
        for i = 1:N
            emissionLogProb = log(mvnpdf(obsSeq(t+1, :), mu(i,:), Sigma(:,:,i)));
            tempLogBeta = log(trans(i, :))' + emissionLogProb + logBeta(:, t+1);
            logBeta(i, t) = logsumexp(tempLogBeta, 1);
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