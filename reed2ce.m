dt = [0.04, 0.02, 0.01, 0.005, 0.0025];
colors = {"r", "g", "m", "c", "k"};
labels = {"dt=0.04", "dt=0.02", "dt=0.01", "dt=0.005", "dt=0.0025"};

[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
du = u - 4*u.*v;
dv = -v + 5*u.*v;
magnitude = sqrt(du.^2 + dv.^2);
du = du ./ magnitude;
dv = dv ./ magnitude;

for i = 1:length(dt)
    figure;
    quiver(u, v, du, dv, 'k'); hold on;
    contour(u, v, u.*v.*exp(-5.*u - 4.*v), 10, 'b');
    T = 8;
    N = T/dt(i) + 1;
    t = linspace(0, T, N);
    u_euler = zeros(1, N);
    v_euler = zeros(1, N);
    u_taylor = zeros(1, N);
    v_taylor = zeros(1, N);
    u_euler(1) = 0.2;
    v_euler(1) = 0.8;
    u_taylor(1) = u_euler(1);
    v_taylor(1) = v_euler(1);
    for n = 1:N-1
        f1 = u_euler(n) - 4 * u_euler(n) * v_euler(n);
        f2 = -v_euler(n) + 5 * u_euler(n) * v_euler(n);
        u_euler(n+1) = u_euler(n) + dt(i) * f1;
        v_euler(n+1) = v_euler(n) + dt(i) * f2;
    end
    for n = 1:N-1
        f1 = u_taylor(n) - 4 * u_taylor(n) * v_taylor(n);
        f2 = -v_taylor(n) + 5 * u_taylor(n) * v_taylor(n);

        Df_f1 = (-4*u_taylor(n)*(5*u_taylor(n)*v_taylor(n) - v_taylor(n)) + (1 - 4*v_taylor(n))*(-4*u_taylor(n)*v_taylor(n) + u_taylor(n)));
        Df_f2 = (5*v_taylor(n)*(-4*u_taylor(n)*v_taylor(n) + u_taylor(n)) + (5*u_taylor(n) - 1)*(5*u_taylor(n)*v_taylor(n) - v_taylor(n)));

        u_taylor(n+1) = u_taylor(n) + dt(i) * f1 + 0.5 * dt(i)^2 * Df_f1;
        v_taylor(n+1) = v_taylor(n) + dt(i) * f2 + 0.5 * dt(i)^2 * Df_f2;
    end

    plot(u_euler, v_euler, 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', ['Euler ' labels{i}]);
    plot(u_taylor, v_taylor, '--', 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', ['Taylor ' labels{i}]);
    plot(u_euler(1), v_euler(1), 'ro', 'MarkerFaceColor', 'r');
    xlabel('u'); ylabel('v');
    title(['Euler vs. Taylor Series for ' labels{i}]);
    hold off;
end

