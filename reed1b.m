[u, v] = meshgrid(linspace(0,1,100), linspace(0,1,100));
H = u .* v .* exp(-5 .* u - 4 .* v);

figure;
mesh(u, v, H);
xlabel('u');
ylabel('v');
zlabel('H(u,v)');
title('Part (b): H(u, v) = uv exp(-5u - 4v)');
view(30, 30);

