% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal
clear
clc
COUPLINGS = {'frustrated' 'ferromagnetic'};
METHODS = {'iter', 'sa'};
n = 50;
n_restart = 100;
num_runs = 1;
NEIGHBORHOODSIZE = 2;
J= -1;

cnt = 1;
for coupl = 1:2
    for meth = 1:2
        COUPLING = COUPLINGS{coupl}
        METHOD = METHODS{meth}

        durations = zeros(1, num_runs);
        num_retries = zeros(1, num_runs);
        min_energies = zeros(1, num_runs);

        E_all_all = zeros(num_runs,n_restart);

        for run_id=1:num_runs
            seed = floor(run_id - 1/8) + 1;
            rng(seed);
%             config_id = mod(run_id - 1, 8) + 1;
%             config = configs(config_id,:);
        %     COUPLING = config{2};
        %     NEIGHBORHOODSIZE = config{3};
        %     METHOD = config{4};

            w=makedata(n, COUPLING);
            if strcmp(COUPLING, 'ferromagnetic')
                w = w * J;
            end
            
            start_time=cputime;
            switch METHOD,
            case 'iter'
                E_a = zeros(1,n_restart);
                E_min = 1000;
                for t=1:n_restart
                    % initialize
                    x = 2*(rand(1,n)>0.5)-1;
                    x0 = x;
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
                E_all_all(run_id, :) = E_a;
                figure(cnt)
                col=hsv(num_runs);
                hold on
                for a=1:num_runs
                    plot(E_all_all(a,:), 'color', col(a,:))
                end
                xlabel('Number of Restarts')
                ylabel('E(x,w)')
                title(strcat(COUPLING,' with neighborhood ',num2str(NEIGHBORHOODSIZE)))
                %only for ferromagnetic:
                %optimal x is all ones or -ones. Compare this to our optimized x.
                x_opt_p = ones(1,n);
                x_opt_m = -1*ones(1,n);
                E_opt_p = E(x_opt_p,w);
                E_opt_m = E(x_opt_m,w);
                [min_energy, required_restarts] = min(E_a);
                num_retries(run_id) = required_restarts;
                min_energies(run_id) = min_energy;
            case 'sa'
                % initialize
                x = 2*(rand(1,n)>0.5)-1;
                E1 = E(x,w);
                E_outer=zeros(1,100);	%stores mean energy at each temperature
                E_outer_per_spin=zeros(1,100); %stores mean energy per spin at each temperature
                E_bar=zeros(1,100);		% stores std energy at each temperature
                E_bar_per_spin=zeros(1,100);
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
                while E_bar(t2) > 0 || t2 < convergence_point,
                    t2=t2+1;
                    beta=beta*factor;
                    betas = [betas beta];
                    E_all=zeros(1,T1);
                    E_all_per_spin=zeros(1,T1);

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
                        E_all_per_spin(t1)=E1/n;
                    end;
                    E_outer(t2)=mean(E_all);
                    E_outer_per_spin(t2)=mean(E_all_per_spin);
                    E_bar(t2)=std(E_all);
                    E_bar_per_spin(t2)=std(E_all_per_spin);
        %             [t2 beta E_outer(t2) E_bar(t2)] % observe convergence
                    if (E_bar(t2) == 0 && convergence_point == -1) 
                        convergence_point = t2;
                    end
                end;
                E_min=E_all(1); % minimal energy 
                num_retries(run_id)=convergence_point;
                min_energies(run_id)=E_min;
                figure(cnt)
%                 subplot(4,1,cnt)
%                 hold on
        %         semilogx(1 ./ betas, E_outer(1:t2),...
        %                  1 ./ betas, E_bar(1:t2))
                semilogx(1 ./ betas, E_outer_per_spin(1:t2)/n,...
                         1 ./ betas, E_bar_per_spin(1:t2)/n)
                legend('Mean Energy', 'std of Energy')
                xlabel('Temperature')
                ylabel('Energy per spin')
                title(strcat(COUPLING, ' neighborhood=', int2str(NEIGHBORHOODSIZE),', T1=',int2str(T1)));
            end;
            cnt = cnt + 1;
            durations(run_id)=cputime - start_time;
        end;
    end
end

