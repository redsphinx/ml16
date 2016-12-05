% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal
rand('state', 123);
clear
clc
makedata
METHOD='sa';
NEIGHBORHOODSIZE = 2;
n_restart = 100;

switch METHOD,
case 'iter'
    E_a = zeros(1,n_restart);
    E_min = 1000;
    parfor t=1:n_restart
        % initialize
        x = 2*(rand(1,n)>0.5)-1;
        x0 =x;
%         negs = (n-sum(x))/2;
%         negs
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
                    de = -2 * x(i) * w(i,:) * x';
                    if de > 0
                        x(i) = -x(i);
                        flag = 1;
                    end
                end		
            case 2
                % choose new x by flipping bits i,j
                for i=1:length(x)
                    for j=i+1:length(x)
                        a = x;
                        a(i) = -x(i);
                        a(j) = -x(j);
                        % de = E(x,w) - E(a,w);
                        de = 2 * (-x(i) * w(i,:) * x' - x(j) * w(j,:) * x' + 2*x(i)*x(j)*w(i,j));
                        if de > 0
                            x(i) = -x(i);
                            x(j) = -x(j);
                            flag = 1;
                        end
                    end
                end
            end
        end
        E1 = E(x,w);
%         E1
        E_a(t) = E1;
        
    end
    E_min = intmax;
    for i = 1:size(E_a, 2)
        E_min = min(E_min, E_a(i));
        E_a(i) = E_min;
    end
    plot(E_a)
    xlabel('Number of Restarts')
    ylabel('E(x,w)')
    title(strcat({'Frustrated with neighborhood '},num2str(NEIGHBORHOODSIZE)))
%     title('Ferromagnetic with neighborhood 2')
    %only for ferromagnetic:
    %optimal x is all ones or -ones. Compare this to our optimized x.
    x_opt_p = ones(1,n);
    x_opt_m = -1*ones(1,n);
    E_opt_p = E(x_opt_p,w)
    E_opt_m = E(x_opt_m,w)
    
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
            max_dE = 2*n;
        case 2,
			% estimate maximum dE in pair spin flip
            max_dE = 4*n;
        end;
	beta_init=1/max_dE;	% sets initial temperature
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain
    betas = [beta_init];
	beta=beta_init;
	E_bar(1)=1;
	t2=1;
    convergence_point = -1;
	while E_bar(t2) > 0 || t2 < (1.3 * convergence_point),
		t2=t2+1;
		beta=beta*factor;
        betas = [betas beta];
    	E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
                % choose new x by flipping one random bit i
				% perform Metropolis Hasting step
                i = randi(n);
                de = 2 * x(i) * w(i,:) * x';
                a = exp(-beta*de);
                if a > rand
                    x(i)=-x(i);
                end
			case 2,
                % choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
                i = randi(n);
                j = randi(n);
                de = 2 * (x(i) * w(i,:) * x' + x(j) * w(j,:) * x' - 2*x(i)*x(j)*w(i,j));
                a = exp(-beta*de);
                if a > rand
                    x(i)=-x(i);
                    x(j)=-x(j);
                end
			end;
			% E1 is energy of new state
            E1 = E(x, w);
			E_all(t1)=E1;
		end;
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
        if (E_bar(t2) == 0 && convergence_point == -1) 
            convergence_point = t2;
        end
	end;
	E_min=E_all(1) % minimal energy 
end;
semilogx(1 ./ betas, E_outer(1:t2),...
         1 ./ betas, E_bar(1:t2))
legend('Mean Energy', 'std of Energy')
xlabel('Temperature')
ylabel('Energy')
title(strcat('Neighborhood=', int2str(NEIGHBORHOODSIZE),', T1=',int2str(T1)));

