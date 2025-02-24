dt = 0.01;
T = 8;
N = T/dt + 1;
t = linspace(0, T, N);
u_approx = zeros(1, N);
v_approx = zeros(1, N);
u_taylor = zeros(1, N);
v_taylor = zeros(1, N);
% ICs
u_approx(1) = 0.2;
v_approx(1) = 0.8;
u_taylor(1) = u_approx(1);
v_taylor(1) = v_approx(1);

% Euler's Method
for n = 1:N-1
    u_approx(n+1) = u_approx(n) + dt * (u_approx(n) - 4 * u_approx(n) * v_approx(n));
    v_approx(n+1) = v_approx(n) + dt * (-v_approx(n) + 5 * u_approx(n) * v_approx(n));
end

% 2nd Order Taylor Series Method
for n = 1:N-1
    f1 = u_taylor(n) - 4 * u_taylor(n) * v_taylor(n);
    f2 = -v_taylor(n) + 5 * u_taylor(n) * v_taylor(n);

    Df_f1 = (-4*u_taylor(n)*(5*u_taylor(n)*v_taylor(n) - v_taylor(n)) + (1 - 4*v_taylor(n))*(-4*u_taylor(n)*v_taylor(n) + u_taylor(n)));
    Df_f2 = (5*v_taylor(n)*(-4*u_taylor(n)*v_taylor(n) + u_taylor(n)) + (5*u_taylor(n) - 1)*(5*u_taylor(n)*v_taylor(n) - v_taylor(n)));

    u_taylor(n+1) = u_taylor(n) + dt * f1 + 0.5 * dt^2 * Df_f1;
    v_taylor(n+1) = v_taylor(n) + dt * f2 + 0.5 * dt^2 * Df_f2;
end

[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
H = u .* v .* exp(-5 .* u - 4 .* v);
du = u - 4*u.*v;
dv = -v + 5*u.*v;

magnitude = sqrt(du.^2 + dv.^2);
du = du ./ magnitude;
dv = dv ./ magnitude;




f = figure;
quiver(u, v, du, dv, 'k');
hold on;

plot(u_approx, v_approx, 'r', 'LineWidth', 1.5, 'DisplayName', 'Euler');
plot(u_approx(1), v_approx(1), 'ro', 'MarkerFaceColor', 'r');

plot(u_taylor, v_taylor, 'g', 'LineWidth', 1.5, 'DisplayName', 'Taylor');
plot(u_taylor(1), v_taylor(1), 'go', 'MarkerFaceColor', 'g');

xlabel('u'); ylabel('v');
title('Part (cd): Eulers Method vs. Taylors Method for t \in [0,8]');


contour(u, v, H, 10, 'b');

hold off;

exportgraphics(f, 'reed2cd.png', 'Resolution', 300);
