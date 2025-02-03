dt = 0.01;
T = 8;
N = T/dt + 1;
t = linspace(0, T, N);
u_approx = zeros(1, N);
v_approx = zeros(1, N);
% ICs
u_approx(1) = 0.2;
v_approx(1) = 0.8;

% Euler's Method
for n = 1:N-1
    u_approx(n+1) = u_approx(n) + dt * (u_approx(n) - 4 * u_approx(n) * v_approx(n));
    v_approx(n+1) = v_approx(n) + dt * (-v_approx(n) + 5 * u_approx(n) * v_approx(n));
end

[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
H = u .* v .* exp(-5 .* u - 4 .* v);
du = u - 4*u.*v;
dv = -v + 5*u.*v;

magnitude = sqrt(du.^2 + dv.^2);
du = du ./ magnitude;
dv = dv ./ magnitude;




figure;
quiver(u, v, du, dv, 'k'); hold on;

plot(u_approx, v_approx, 'r', 'LineWidth', 1.5);
plot(u_approx(1), v_approx(1), 'ro', 'MarkerFaceColor', 'r'); % Initial point

xlabel('u'); ylabel('v');
title('Part (d): Eulers Method for t \in [0,8]');


contour(u, v, H, 10, 'b');

hold off;

