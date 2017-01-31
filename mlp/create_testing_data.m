function [X_test,T_test] = create_testing_data(a,b)
    load('mnistAll.mat')
    X = mnist.test_images;
    T = mnist.test_labels;
    hits = double(T == a) + double(T == b);
    size_new_data = sum(hits);
    new_X = zeros(size(X,1), size(X,2), size_new_data);
    new_T = zeros(size_new_data ,1);

    alt_counter = 1;
    for i=1:length(hits)
        if hits(i)
            new_X(:,:,alt_counter) = X(:,:,i);
            new_T(alt_counter) = T(i);
            alt_counter = alt_counter + 1;
        end
    end
    X_test = new_X;
    T_test = new_T;
end