function labels = getPredictedLabels(File_names)
    % Initialize an array to store unique labels
    unique_labels = {};
    
    % Iterate through each file
    for i = 1:numel(File_names)
        % Split the file name by underscores
        Fname_parts = strsplit(File_names(i).name, '_');
        
        % Extract the label from the third part (after the second underscore)
        label = Fname_parts{3};
        
        % Remove the file extension (.mp3)
        label = strrep(label, '.mp3', '');
        
        % Add the label to the array if it's not already present
        if ~ismember(label, unique_labels)
            unique_labels = [unique_labels,label];
        end
    end
    end
    