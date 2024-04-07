function matrix = confusionMatrix(predictedWords, groundTruthLabels)
    % Unique labels in the dataset
    uniqueLabels = unique([groundTruthLabels, predictedWords]);

    % Create a map for label indices
    labelMap = containers.Map(uniqueLabels, 1:length(uniqueLabels));

    % Initialize the confusion matrix
    matrix = zeros(length(uniqueLabels));

    % Populate the confusion matrix
    for i = 1:length(predictedWords)
        trueLabelIndex = labelMap(groundTruthLabels{i});
        predictedLabelIndex = labelMap(predictedWords{i});
        matrix(trueLabelIndex, predictedLabelIndex) = matrix(trueLabelIndex, predictedLabelIndex) + 1;
    end

    % Display the confusion matrix with labels
    confusionchart(matrix, uniqueLabels);
end
