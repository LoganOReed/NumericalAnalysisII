[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
H = u .* v .* exp(-5 .* u - 4 .* v);
du = u - 4*u.*v;
dv = -v + 5*u.*v;

magnitude = sqrt(du.^2 + dv.^2);
du = du ./ magnitude;
dv = dv ./ magnitude;

figure;
quiver(u, v, du, dv, 'k'); hold on;
xlabel('u'); ylabel('v');
title('Part (c): Lotka-Volterra Velocity Field with H isocontours');
axis([0 1 0 1]);

contour(u, v, H, 10, 'b');

hold off;

