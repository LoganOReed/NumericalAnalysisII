function unew = forwardEuler(f, u, t, h)
unew = u + h * f(u, t);
end
