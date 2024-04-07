function accuracy = computeAccuracy(trueLabels, predictedLabels)
    % Assuming trueLabels and predictedLabels are column vectors
    correctPredictions = sum(trueLabels == predictedLabels);
    totalSamples = numel(trueLabels);
    accuracy = correctPredictions / totalSamples;
end


    