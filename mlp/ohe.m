function [T] = ohe(T,a,b)
    classes = unique(T);
    ohe_T = zeros(size(T));
    for i=1:length(T)
        if T(i) == classes(1)
            ohe_T(i) = a;
        else
            ohe_T(i) = b;
        end
    end
    T = ohe_T;
end
