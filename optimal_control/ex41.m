clear 
figure

% Initialise
dt = 0:0.01:2;
nus = 1:10;
x = 0.5;
clr = jet(numel(nus));
legends = cell(1, numel(nus));
hold on

for inu = 1:numel(nus)
    % Compute control strategy
    nu = nus(inu);
    th = tanh(x ./ (nu*dt) - x) ./ dt;
    
    % Plot it
    plot(dt, th, 'Color', clr(inu, :));
    legends{inu} = sprintf('\\nu = %i', nu);
end

legend(legends)
xlabel('T-t');
set(gca, 'XDir', 'reverse');
ylabel('u(x=0.5,t)');
title('Control strategies for different noise levels')

hold off
