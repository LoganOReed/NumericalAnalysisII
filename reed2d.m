
a_values = linspace(0.01, 0.399, 100);
dt = 0.01;
Tmax = 20;
T_values = zeros(size(a_values));

for i = 1:length(a_values)
    u = [a_values(i); 0.25];
    t = 0;
    min_norm = inf;
    min_t = NaN;
    prev_norm = inf;
    min_count = 0;

    while t < Tmax
        f = [u(1) - 4*u(1)*u(2);
             -u(2) + 5*u(1)*u(2)];
        J = [1 - 4*u(2), -4*u(1);
             5*u(2), -1 + 5*u(1)];
        Df_f = J * f;

        u = u + dt * f + (dt^2 / 2) * Df_f;
        t = t + dt;

        % find first real local min and take it as one full loop
        current_norm = norm(u - [a_values(i); 0.25]);
        if prev_norm < min_norm && prev_norm < current_norm
            min_count = min_count + 1;
            if min_count == 2
                min_t = t - dt;
                break;
            end
            min_norm = prev_norm;
        end
        prev_norm = current_norm;
    end
    T_values(i) = min_t;
end


figure;
plot(a_values, T_values, 'b-o', 'MarkerSize', 4, 'LineWidth', 1.5);
xlabel('a');
ylabel('T(a)');
title('Return Time T(a) for Lotka-Volterra Model');
grid on;
