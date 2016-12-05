rand('state',0)
n=100;
p=n;
w=sprandsym(n,p);
% w=(w>0)-(w<0); % this choice defines a frustrated system
% w=(w>0); % this choice defines a ferromagnetic (easy) system
% w=w-diag(diag(w));
w = ones(n) - eye(n);