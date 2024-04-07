function [max_cum_log_likelihood, modelIndex] = viterbi_algorithm(obs_seq, HMM, N)
    % Viterbi algorithm for continuous-density hidden Markov models (CD-HMMs)
    
    % Extract parameters from HMM structure
    transmat = HMM.updated_Trans;
    emissionCovs = HMM.updated_emis.Sigma;
    means = repmat(HMM.updated_emis.mu, N, 1);

    % Number of states
    num_states = size(transmat, 1);

    % Initialize delta and psi matrices
    delta = zeros(length(obs_seq), num_states);

    % Initialize at t = 1
    delta(1, :) = log(HMM.pi) + log(mvnpdf(obs_seq(1, :), means, emissionCovs));
    psi(1, :) = zeros(1, num_states);

    % Forward recursion
    for t = 2:length(obs_seq)
        for j = 1:num_states
            % Compute delta using the Viterbi recursion equation
            [max_log_prob, max_prev_state] = max(delta(t - 1, :) + log(transmat(:, j)') + log(mvnpdf(obs_seq(t, :), means(j, :), emissionCovs(:,:,j))));

            delta(t, j) = max_log_prob;
            psi(t, j) = max_prev_state;
        end
    end

    % Finalize
    [max_cum_log_likelihood, most_likely_state] = max(delta(end, :));

    % Trace back
    modelIndex = zeros(1, length(obs_seq));
    modelIndex(end) = most_likely_state;

    for t = length(obs_seq) - 1:-1:1
        modelIndex(t) = psi(t + 1, modelIndex(t + 1));
    end
end
