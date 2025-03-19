function temple5044_voyager
%TEMPLE5044_VOYAGER
%   Simulation of an N-body problem in celestial mechanics.
%   The example contains the sun, earth, jupiter, and the
%   Voyager 1 probe, shot off from earth towards jupiter.
%   The probe gets very close to jupiter, which results
%   in a dramatic change of its velocity. The default
%   integration is done with a fixed time step RK4, which
%   for the close approach is absolutely insufficient in
%   terms of accuracy. The challenge here is to modify the
%   time integrator to adaptive time stepping, so that the
%   gravity assist swing-by is computed sufficiently
%   accurately.
%
%   (C) 2019/02/10 by Benjamin Seibold
%       http://www.math.temple.edu/~seibold/

% Parameters and initialization
name = {'sun','earth','jupiter','voyager'};
mass = [1.9891e30;5.9736e24;1.8986e27;721.9]; % mass in kg
dist = [0;1.49598261e11;7.78547200e11;0]; % semi-major axis in m
period = [0;365.256363;4332.59;0]*24*3600; % orbital period
radius = [6.96342e8;6.371e6;6.9911e7;13];
vel = 2*pi*dist./period; % average velocity
u = [0, 0,   0, -dist(2),   dist(3),0,  radius(2)*10,-dist(2),...
     0,0,  vel(2),  0,  0, vel(3),  vel(2)+1.503e4,0];
N = length(mass); % number of bodies
dim = length(u)/2/N; % spacial dimension of problem
phi = linspace(0,2*pi,20); cx = cos(phi); cy = sin(phi);

% Time integration
tf = 550*24*3600; % final time in seconds
dt = 1e5; % time step in seconds
nt = ceil(tf/dt); % number of time steps
rtol = 1e-9;

options = odeset('RelTol', rtol, 'Stats', 'on', 'OutputFcn', @plotfun);
    [t, u] = ode45(@(t, u) N_body_forces(t, u, mass), [0 tf], u, options);


function status = plotfun(t,u,flag)
    clf, hold on
tf = 550*24*3600; % final time in seconds
if isempty(u)
  return;
end
name = {'sun','earth','jupiter','voyager'};
mass = [1.9891e30;5.9736e24;1.8986e27;721.9]; % mass in kg
dist = [0;1.49598261e11;7.78547200e11;0]; % semi-major axis in m
period = [0;365.256363;4332.59;0]*24*3600; % orbital period
radius = [6.96342e8;6.371e6;6.9911e7;13];
vel = 2*pi*dist./period; % average velocity
N = length(mass); % number of bodies
dim = length(u)/2/N; % spacial dimension of problem
phi = linspace(0,2*pi,20); cx = cos(phi); cy = sin(phi);

% Time integration
tf = 550*24*3600; % final time in seconds
dt = 1e5; % time step in seconds
nt = ceil(tf/dt); % number of time steps
rtol = 1e-9;
    for i = 1:N
        ux = u(dim*(i-1)+1); uy = u(dim*(i-1)+2);
        plot(ux,uy,'r.','markersize',12)
        plot(ux+radius(i)*cx,uy+radius(i)*cy,'k-')
        text(ux,uy,[' ',name{i}])
    end
    hold off, axis equal, axis([-1 1 -1 1]*max(dist)*1.2)
    title(sprintf('t=%0.0f days',t/3600/24)), drawnow
    status = 0;

##for k = 1:20:length(t)
    % visualization
##        clf, hold on
##        for i = 1:N
##            ux = u(dim*(i-1)+1); uy = u(dim*(i-1)+2);
##            plot(ux,uy,'r.','markersize',12)
##            plot(ux+radius(i)*cx,uy+radius(i)*cy,'k-')
##            text(ux,uy,[' ',name{i}])
##        end
##        hold off, axis equal, axis([-1 1 -1 1]*max(dist)*1.2)
##        title(sprintf('t=%0.0f days',k*dt/3600/24)), drawnow
##end

%======================================================================
function v = N_body_forces(t, u, mass)
% Forces acting in an N body problem.
% mass = vector of N masses
% u = state vector of N position vectors, and N velocity vectors
N = length(mass); % number of bodies
dim = length(u)/2/N; % spacial dimension of problem
v = u*0;
v(1:dim*N) = u(dim*N+1:2*dim*N); % position changes by velocities
ind = -dim+1:0;
for i = 1:N % loop over all bodies to sum forces
    for j = i+1:N % loop over all other bodies with higher index
        f = two_body_force(mass(i),mass(j),u(i*dim+ind),u(j*dim+ind));
        v(dim*(N+i)+ind) = v(dim*(N+i)+ind)+f; % add force from j to i
        v(dim*(N+j)+ind) = v(dim*(N+j)+ind)-f; % subst. force i from j
    end
end
v(dim*N+1:2*dim*N) = v(dim*N+1:2*dim*N)./... % divide forces by masses
    mass(reshape(ones(dim,1)*(1:N),[],1));

function f = two_body_force(m1,m2,x1,x2)
% Force between two celetial bodies (pointing from 1 to 2).
G = 6.67384e-11; % graviational constant
d = x2-x1; % distance vector from x1 to x2
r = norm(d); % distance
f = (G*m1*m2/r^3)*d; % force from 1, mass to 2
