function w=makedata(n, COUPLING)
p=n;
w=sprandsym(n,p);
switch COUPLING,
    case 'frustrated' % hard
        w=(w>=0)-(w<0);
    case 'ferromagnetic' % easy
        w=(w>=0);
end

w=w-diag(diag(w));
