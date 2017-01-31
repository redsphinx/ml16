function [matching_table, T] = ohe(T)
%create one hot encoding for T
    classes = unique(T);
    ohe_T = zeros(length(T), length(classes));
    matching_table = [[1;0] classes ]; % for future reference
    
    for i=1:length(T)
        if T(i) == classes(1)
            ohe_T(i,1) = 1;
        else
            ohe_T(i,2) = 1;
        end
    end
    T = ohe_T;
end
