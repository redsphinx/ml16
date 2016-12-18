function alpha = cgdlinesearch(X, t, W, b, Wdir, bdir, eta, ftol)
% CGDLINESEARCH Performs a line search, minimising the square error of
% mlpforward(X, W', b') to t w.r.t. alpha, where
% W' = W + alpha*Wdir, b' = b + alpha*bdir.
% eta is a typical step size, and ftol the precision requested on alpha.
% Uses golden section search, as described on
% https://en.wikipedia.org/wiki/Golden_section_search

phi = (sqrt(5)-1)/2;
eval = @(alpha) cgdlinesearchevaluate(X, t, W, b, Wdir, bdir, alpha);

[lbound, rbound] = cgdlinesearchfindbounds(eval, eta);

midl = rbound - phi*(rbound-lbound);
midlerror = eval(midl);
midr = lbound + phi*(rbound-lbound);
midrerror = eval(midr);
while abs(midl-midr) > ftol
    if midlerror < midrerror
        rbound = midr;
        midr = midl;
        midrerror = midlerror;
        midl = rbound - phi*(rbound-lbound);
        midlerror = eval(midl);
    else
        lbound = midl;
        midl = midr;
        midlerror = midrerror;
        midr = lbound + phi*(rbound-lbound);
        midrerror = eval(midr);
    end
end

alpha = (lbound+rbound)/2;

end

function [lbound, rbound] = cgdlinesearchfindbounds(eval, eta)

lbound = 0;
rbound = eta;

lerror = eval(lbound);
rerror = eval(rbound);

% Identify upper bound for alpha using exponential search-like strategy
prevrerror = lerror;
while rerror < prevrerror
    rbound = 2*rbound;
    rerror = eval(rbound);
end


end

function err = cgdlinesearchevaluate(X, t, W, b, Wdir, bdir, alpha)

W = adjustby(W, Wdir, alpha);
b = adjustby(b, bdir, alpha);
err = sqerror(mlpforward(X, W, b), t);

end

function err = sqerror(Y, t)

diff = (Y - t) .^ 2;
err = sum(diff(:));

end