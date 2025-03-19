function u_true = lotka_volterra_rk4()
    % Define parameters
    h = 1e-4;  % Step size
    t_final = 31;  % Final time
    u0 = [0.73; 0.25];  % Initial condition
    
    % Time integration using RK4
    t = 0;
    u = u0;
    while t < t_final
        if t + h > t_final
            h = t_final - t; % Adjust step size for last step
        end
        
        k1 = h * f(u);
        k2 = h * f(u + 0.5 * k1);
        k3 = h * f(u + 0.5 * k2);
        k4 = h * f(u + k3);
        
        u = u + (k1 + 2*k2 + 2*k3 + k4) / 6;
        t = t + h;
    end
    
    % Store the final solution
    u_true = u;
    
    % Display the result
    fprintf('Reference solution u_true(31) = [%f, %f]\n', u_true(1), u_true(2));
end

function du = f(u)
    % Define the Lotka-Volterra system
    du = [
        u(1) - 4 * u(1) * u(2);
        -u(2) + 5 * u(1) * u(2);
    ];
end
