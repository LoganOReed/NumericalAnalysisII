function temple5044_linear_ode_2x2(example)
%TEMPLE5044_LINEAR_ODE_2X2
%   Vector field and phase flow for linear 2x2 ODE systems.
%   Visualizes the different cases that can arise with
%   2x2 systems of ODE, via quiver plot, eigen-directions,
%   and deformation of the unit circle.
%   Examples are: (1) saddle; (2) repulsor; (3) attractor;
%   (4) locus; (5) repelling locus; (6) attracting locus;
%   and Jordan block attractor (7), neutral (8), and
%   repulsor (9).
%
% (C) 2019/01/15 by Benjamin Seibold
%                   http://www.math.temple.edu/~seibold/

% Problem definition
if nargin<1, example = 1; end
switch example
case 1, A = [ 0 1; 1  0]; name = 'saddle (unstable)';
case 2, A = [ 2 1; 1  2]; name = 'repulsor (unstable)';
case 3, A = [-2 1; 1 -2]; name = 'attractor (asympt. stable)';
case 4, A = [ 0 1;-1  0]; name = 'locus (neutrally stable)';
case 5, A = [ 1 1;-1  1]; name = 'repelling locus (unstable)';
case 6, A = [-1 1;-1 -1]; name = 'attracting locus (asympt. stable)';
case 7, A = [-1 1; 0 -1]; name = 'attractor Jordan block (asympt. stable)';
case 8, A = [ 0 1; 0  0]; name = 'neutral Jordan block (unstable)';
case 9, A = [ 1 1; 0  1]; name = 'repulsor Jordan block (unstable)';
end
t = .1; % time of evolution

% Plot direction field
x = linspace(-2,2,51);
y = x;
[X,Y] = meshgrid(x,y);
U = A(1,1)*X+A(1,2)*Y;
V = A(2,1)*X+A(2,2)*Y;
clf
quiver(x,y,U,V,2)
axis equal
axis([x(1),x(end),y(1),y(end)])
title(sprintf('Linear 2x2 ODE system: %s',name))
xlabel('x'), ylabel('y')
hold on

% Plot deformed circle
p = linspace(0,2*pi,100);
cx = cos(p); cy = sin(p);
plot(cx,cy,'r-') % unit circle
M = expm(t*A); % solution matrix at time t
plot(M(1,1)*cx+M(1,2)*cy,M(2,1)*cx+M(2,2)*cy,'m-') % deformed circle
leg = {'direction field','unit circle',...
    sprintf('demormed circle at t=%g',t)};

% Plot eigen-directions
[V,D] = eig(A); lambda = diag(D);
if isreal(D)
    plot(V(1,1)*[-1 1]*1.5,V(2,1)*[-1 1]*1.5,'k-')
    plot(V(1,2)*[-1 1]*1.5,V(2,2)*[-1 1]*1.5,'k-')
    leg{end+1} = 'eigen-directions';
    fprintf('eigenvalues:   %7.2f ,%7.2f\n',lambda)
    fprintf('eigenvectors: [%7.2f ,%7.2f]\n',V(1,:))
    fprintf('              [%7.2f ,%7.2f]\n',V(2,:))
else
    fprintf('eigenvalues:   %7.2f+%7.2fi,%7.2f+%7.2fi\n',...
        real(lambda(1)),imag(lambda(1)),real(lambda(2)),imag(lambda(2)))
    fprintf('eigenvectors: [%7.2f+%7.2fi,%7.2f+%7.2fi]\n',...
        real(V(1,1)),imag(V(1,1)),real(V(1,2)),imag(V(1,2)))
    fprintf('              [%7.2f+%7.2fi,%7.2f+%7.2fi]\n',...
        real(V(2,1)),imag(V(2,1)),real(V(2,2)),imag(V(2,2)))
end
hold off
legend(leg)
