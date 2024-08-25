
G = 6.67430e-11;  % Gravitational constant, m^3 kg^-1 s^-2
M1 = 5.972e24;    % Mass of Earth (kg)
M2 = 7.348e22;    % Mass of Moon (kg)
R = 3844e5;       % Distance between Earth and Moon (m)

% Standard gravitational parameters
mu1 = G * M1;
mu2 = G * M2;
mu = mu2 / (mu1 + mu2);

% Defining the equation for the collinear Lagrange points
lagrange_eq = @(x) (1 - mu) * (x + mu) ./ abs(x + mu).^3 + mu * (x - 1 + mu) ./ abs(x - 1 + mu).^3 - x;

% Solve for L1, L2, and L3 with initial guesses
L1 = fsolve(lagrange_eq, 0.5);
L2 = fsolve(lagrange_eq, 1.5);
L3 = fsolve(lagrange_eq, -1);

% Equations for L4 and L5 (equilateral points)
lagrange_eq4_5 = @(xy) [
    (1 - mu) * (xy(1) + mu) ./ sqrt((xy(1) + mu).^2 + xy(2).^2).^3 + ...
    mu * (xy(1) - 1 + mu) ./ sqrt((xy(1) - 1 + mu).^2 + xy(2).^2).^3 - xy(1);
    (1 - mu) * xy(2) ./ sqrt((xy(1) + mu).^2 + xy(2).^2).^3 + ...
    mu * xy(2) ./ sqrt((xy(1) - 1 + mu).^2 + xy(2).^2).^3 - xy(2)
];

% Initial guesses for L4 and L5
L4_guess = [0.5, sqrt(3)/2];
L5_guess = [0.5, -sqrt(3)/2];

% Solving L4 and L5
L4 = fsolve(lagrange_eq4_5, L4_guess);
L5 = fsolve(lagrange_eq4_5, L5_guess);

% Differential Correction Parameters
max_iter = 100;
tol = 1e-9;

% ODE options
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);

% Period estimation (adjust as needed)
T = 2 * pi; 

% Lagrange Points
L_points = [L1, 0, 0; L2, 0, 0; L3, 0, 0; L4(1), L4(2), 0; L5(1), L5(2), 0];

for i = 1:size(L_points, 1)
    % Initial condition near the Lagrange point
    init_cond = [L_points(i, :) + 1e-6, 0, 0, 0]; % Slightly perturbed initial guess

    for iter = 1:max_iter
        % Integrate the trajectory using ODE45
        [t, sol] = ode45(@(t, state) three_body_eqns(t, state, mu), [0, T], init_cond, options);

        % Calculate the deviation after one period
        delta = sol(end, 1:6) - sol(1, 1:6);

        % Check for convergence
        if norm(delta) < tol
            fprintf('Converged after %d iterations at L%d.\n', iter, i);
            break;
        end

        % Compute the Jacobian and apply correction (Newton-Raphson method)
        J_eval = numerical_jacobian(@(x) state_after_period(x, mu, T), init_cond);
        correction = -J_eval \ delta';
        init_cond = init_cond + correction';
    end

    % Plot the refined periodic orbit
    figure;
    plot3(sol(:, 1), sol(:, 2), sol(:, 3));
    title(sprintf('Differentially Corrected Small Amplitude Periodic Orbit around L%d', i));
    xlabel('x'); ylabel('y'); zlabel('z');
    grid on;
end

% Function to compute the state after one period
function state = state_after_period(state, mu, T)
    [~, sol] = ode45(@(t, state) three_body_eqns(t, state, mu), [0, T], state);
    state = sol(end, :) - sol(1, :);
end

% Function to compute the numerical Jacobian
function J = numerical_jacobian(func, x)
    n = length(x);
    J = zeros(n, n);
    delta = 1e-8;
    for i = 1:n
        x1 = x;
        x2 = x;
        x1(i) = x1(i) + delta;
        x2(i) = x2(i) - delta;
        J(:, i) = (func(x1) - func(x2))' / (2 * delta);
    end
end

% Three-body equations of motion
function dstate_dt = three_body_eqns(~, state, mu)
    x = state(1); y = state(2); z = state(3);
    vx = state(4); vy = state(5); vz = state(6);

    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    dx_dt = vx;
    dy_dt = vy;
    dz_dt = vz;
    dvx_dt = 2*vy + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3;
    dvy_dt = -2*vx + y - (1 - mu) * y / r1^3 - mu * y / r2^3;
    dvz_dt = -(1 - mu) * z / r1^3 - mu * z / r2^3;

    dstate_dt = [dx_dt; dy_dt; dz_dt; dvx_dt; dvy_dt; dvz_dt];
end
