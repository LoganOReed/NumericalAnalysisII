dt = [0.04, 0.02, 0.01, 0.005, 0.0025];
colors = {"r", "g", "m", "c", "k"};
labels = {"dt=0.04", "dt=0.02", "dt=0.01", "dt=0.005", "dt=0.0025"};


[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
H = u .* v .* exp(-5 .* u - 4 .* v);
du = u - 4*u.*v;
dv = -v + 5*u.*v;

magnitude = sqrt(du.^2 + dv.^2);
du = du ./ magnitude;
dv = dv ./ magnitude;


for i = 1:length(dt)
  figure;
  quiver(u, v, du, dv, 'k'); hold on;
  contour(u, v, H, 10, 'b');
  T = 8;
  N = T/dt(i) + 1;
  t = linspace(0, T, N);
  u_approx = zeros(1, N);
  v_approx = zeros(1, N);
  u_approx(1) = 0.2;
  v_approx(1) = 0.8;
  for n = 1:N-1
      u_approx(n+1) = u_approx(n) + dt(i) * (u_approx(n) - 4 * u_approx(n) * v_approx(n));
      v_approx(n+1) = v_approx(n) + dt(i) * (-v_approx(n) + 5 * u_approx(n) * v_approx(n));
  end
  plot(u_approx, v_approx, 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', labels{i});
  plot(u_approx(1), v_approx(1), 'ro', 'MarkerFaceColor', 'r');
  xlabel('u'); ylabel('v');
  title(['Part (e): Eulers Method for t \in [0,8] with ', labels{i}]);
  hold off;
end
