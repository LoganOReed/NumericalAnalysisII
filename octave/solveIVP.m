function [u,t] = solveIVP(f, u0, tspan, h, solver)
  % define u,t output arrays
  t = (tspan(1) : h : tspan(2));
  u = zeros(length(t), length(u0));
  u(1,:) = u0;

  % loop through steps
  for k = 1 : length(t) - 1
    u(k+1,:) = solver(f, u(k,:), t(k), h);
  end
end
