clear 
figure

% Initialise
dt = 0.2:0.2:2;
nus = 1:.1:10;
x = 1;
clr = jet(numel(dt));
legends = cell(1, numel(dt));
hold on

for d_idx = 1:numel(dt)
    % Compute control strategy
    d = dt(d_idx);
    th = tanh(x ./ (nus*d) - x)/d;
    
    % Plot it
    plot(nus, th, 'Color', clr(d_idx, :));
    legends{d_idx} = strcat('T-t=',num2str(d));
end

legend(legends)
xlabel('\nu');
ylabel('u(x=1,t)');
title('Control strategies for different noise levels and time horizons')

hold off
