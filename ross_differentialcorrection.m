global FORWARD tol mu
FORWARD = 1;
tol = 1e-8;
mu = 0.0121;  % mass ratio for Earth-Moon system

% Lagrange Point L1 (Find gamma as in your original code)
poly1 = [1 -1*(3-mu) (3-2*mu) -mu 2*mu -mu]; 
rt1 = roots(poly1);
for k = 1:5
    if isreal(rt1(k))
        gamma = rt1(k);  % Real root corresponds to L1 location
    end
end
xL = (1-mu) - gamma;  % Location of L1

% Calculate mubar, lam, nu
mubar = mu / abs(xL - 1 + mu)^3 + (1 - mu) / abs(xL + mu)^3;
a = 1 + 2 * mubar;
b = mubar - 1;
lam = sqrt(0.5 * (mubar - 2 + sqrt(9 * mubar^2 - 8 * mubar)));
nu = sqrt(-0.5 * (mubar - 2 - sqrt(9 * mubar^2 - 8 * mubar)));

% Initial guess for orbit
sigma = 2 * lam / (lam^2 + b);
tau = -(nu^2 + a) / (2 * nu);

u1 = [1; -sigma; lam; -lam * sigma];
xL_vec = [xL; 0; 0; 0];  % Vector form of L1 position

% Displacement for initial guess (small perturbation)
displacement = 1e-3;
X0 = xL_vec + displacement * u1;

% Apply differential correction to find Lyapunov orbit
[x0_corr, T_half] = diff_correction(X0, mu, tol);

% Time span for the full period
T_full = 2 * T_half;  % Full period is double the half period
tspan = 0:0.001*T_full:T_full;  % Fine time grid

% ODE solver options
opts = odeset('AbsTol', 1e-8, 'RelTol', 1e-6);

% Integrate to get the corrected orbit
[t, X] = ode78(@(t, X) prtbp(t, X), tspan, x0_corr, opts);

% Plot the orbit
x = X(:, 1);
y = X(:, 2);
figure;
plot(x, y, 'k');  % Plot in black
hold on;
plot(1 - mu, 0, 'ro');  % Plot secondary body (Moon in this case)
title('Lyapunov Orbit around L1 in Rotating Frame');
xlabel('x'); ylabel('y');
axis equal; grid on;


function [x0_corr, T_half] = diff_correction(x0, mu, tol)
    opts = odeset('AbsTol', 1e-8, 'RelTol', 1e-6);
    
    % Initial guess for half-period
    T_half = 2*pi ;
    
    max_iter = 10;  % Maximum number of iterations for correction
    for iter = 1:max_iter
        % Integrate for half the period
        tspan = [0 T_half];
        [~, X] = ode78(@(t, X) prtbp(t, X), tspan, x0, opts);
        
        % Check if the orbit intersects the x-axis (y = 0) at T_half
        y_half = X(end, 2);  % y-component at T_half
        vy_half = X(end, 4);  % vy-component at T_half
        
        % If close enough, stop the iteration
        if abs(y_half) < tol
            break;
        end
        
        % Correct the initial velocity in the y-direction (vy)
        x0(4) = x0(4) - y_half / vy_half;  % Adjust vy based on error
    end
    
    x0_corr = x0;  % Corrected initial condition
end

% Function: Equations of motion for the CR3BP in rotating frame
function dXdt = prtbp(t, X)
    global mu  % mass ratio (Earth-Moon system)

    % Extract state variables
    x = X(1);   % position in x
    y = X(2);   % position in y
    vx = X(3);  % velocity in x
    vy = X(4);  % velocity in y

    % Distances to the primary (Earth) and secondary (Moon) bodies
    r1 = sqrt((x + mu)^2 + y^2);  % Distance to primary
    r2 = sqrt((x - (1 - mu))^2 + y^2);  % Distance to secondary

    % Equations of motion for the CR3BP in rotating frame
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - (1 - mu)) / r2^3;
    ay = -2 * vx + y - (1 - mu) * y / r1^3 - mu * y / r2^3;

    % Return the derivatives of state variables
    dXdt = [vx; vy; ax; ay];
end


