function errorRate = computeErrorRate(HMM, obsSeq, g_mean, emissionCovariances,File_names,gamma)
    % Forward algorithm to get the predicted labels
    logAlpha = forward_algorithm(obsSeq, gamma, HMM.Trans, repmat(g_mean, 8, 1), emissionCovariances);
    predictedLabels = getPredictedLabels(logAlpha);

    % Ground truth labels (modify this based on your data)
    groundTruthLabels = getGTLabels(File_names);

    % Compare predicted labels with ground truth labels
    errors = sum(predictedLabels ~= groundTruthLabels);

    % Calculate error rate
    errorRate = errors / length(groundTruthLabels);
end
