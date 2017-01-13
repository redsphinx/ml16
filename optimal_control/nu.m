clear 
clc

dt = 0.2:0.2:2;
nus = 1:.1:10;
x = 1;
clr = jet(numel(dt));
legends = cell(1, numel(dt));
hold on
for d_idx = 1:numel(dt)
    d = dt(d_idx);
    th = tanh(x ./ (nus*d) - x)/d;
    plot(nus, th, 'Color', clr(d_idx, :));
    legends{d_idx} = strcat('T-t=',num2str(d));
end

legend(legends)
xlabel('\nu');
ylabel('u(x=1,t)');

hold off
