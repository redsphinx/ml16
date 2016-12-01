% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal
clear
clc
makedata
METHOD='iter';
NEIGHBORHOODSIZE=2;
n_restart =100;

switch METHOD,
case 'iter'
E_a = 1:n_restart;
	E_min = 1000;
	parfor t=1:n_restart
		% initialize
		x = 2*(rand(1,n)>0.5)-1;
% 		E1 = E(x,w);
		flag = 1;
		while flag == 1
			flag = 0;
			switch NEIGHBORHOODSIZE,
			case 1
                % choose new x by flipping one bit i
				% compute dE directly instead of subtracting E's of
				% different states because of efficiency
                for i=1:length(x)
                    fx = x(i) * ( w(i,:)*x' + w(:,i)*x);
                    if fx > 0
                        x(i) = -x(i);
                    end
                end		
			case 2
				% choose new x by flipping bits i,j
                for i=1:length(x)
                    for j=1:length(x)
                        fx = x(i) * ( w(i,:)*x' + w(:,i)*x) + x(j) * ( w(j,:)*x' + w(:,j)*x)...
                            - w(i,j)*x(i)*x(j) - w(j,i)*x(i)*x(j);
                        if fx > 0
                            x(i) = -x(i);
                            x(j) = -x(j);
                        end
                    end
                end
			end;
		end;
        E1 = E(x,w);
% 		E_min = min(E_min,E1);
        E_a(t) = E1;
	end;
    E_min = intmax;
    for i = 1:size(E_a, 2)
        E_min = min(E_min, E_a(i));
        E_a(i) = E_min;
    end
    plot(E_a)
case 'sa'
	% initialize
	x = 2*(rand(1,n)>0.5)-1;
	E1 = E(x,w);
	E_outer=zeros(1,100);	%stores mean energy at each temperature
	E_bar=zeros(1,100);		% stores std energy at each temperature

	% initialize temperature
	max_dE=0;
	switch NEIGHBORHOODSIZE,
        case 1,
			% estimate maximum dE in single spin flip
        case 2,
			% estimate maximum dE in pair spin flip
        end;
	beta_init=1/max_dE;	% sets initial temperature
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain

	beta=beta_init;
	E_bar(1)=1;
	t2=1;
	while E_bar(t2) > 0,
		t2=t2+1;
		beta=beta*factor;
		E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one random bit i
				% perform Metropolis Hasting step
			case 2,
				% choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
			end;
			% E1 is energy of new state
			E_all(t1)=E1;
		end;
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
	end;
	E_min=E_all(1) % minimal energy 
end;
% plot(1:t2,E_outer(1:t2),1:t2,E_bar(1:t2))
