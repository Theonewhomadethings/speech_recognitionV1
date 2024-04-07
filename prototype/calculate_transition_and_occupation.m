function [logTransProb, logOccupationLikelihood] = calculate_transition_and_occupation(trans, mu, Sigma, logAlpha, logBeta, obsSeq)
    % trans: Transition probabilities matrix
    % mu: Mean parameters for Gaussian emission model
    % Sigma: Covariance matrix parameters for Gaussian emission model
    % logAlpha: Log forward likelihood matrix
    % logBeta: Log backward matrix
    % obsSeq: Observation sequence matrix (each row represents a frame)

    N = size(trans, 1);  % Number of states
    T = size(obsSeq, 1); % Number of frames

    % Calculate log occupation likelihoods
    logOccupationLikelihood = logAlpha + logBeta;
    % Normalize log occupation likelihoods
    logOccupationLikelihood = logOccupationLikelihood - logsumexp(logOccupationLikelihood, 1);

    % Calculate log transition probabilities
    logTransProb = zeros(N, N, T-1);
    smallLogValue = log(1e-20); % A small positive value to prevent -Inf

    for t = 1:T-1
        for i = 1:N
            for j = 1:N
                logProb = logAlpha(i, t) + log(trans(i, j)) + log(mvnpdf(obsSeq(t+1, :), mu(j,:), Sigma(:,:,j))) + logBeta(j, t+1);

                % Check if the computed log probability is -Inf
                if isinf(logProb) && logProb < 0
                    logTransProb(i, j, t) = smallLogValue;
                else
                    logTransProb(i, j, t) = logProb;
                end
            end
        end
    end
    % Normalize log transition probabilities
    logTransProb = logTransProb - logsumexp(logsumexp(logTransProb, 1), 2);
end

function result = logsumexp(x, dim)
    % Log-sum-exp trick to prevent numerical instability

    % Find the maximum value along the specified dimension
    max_x = max(x, [], dim);

    % Compute the log-sum-exp
    result = max_x + log(sum(exp(x - max_x), dim));
end