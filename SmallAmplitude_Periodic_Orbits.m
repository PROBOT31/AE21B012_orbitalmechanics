% Constants
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
L4 = [0.5 - mu, sqrt(3)/2];
L5 = [0.5 - mu, -sqrt(3)/2];

% Lagrange Points
L_points = [L1, 0, 0; L2, 0, 0; L3, 0, 0; L4(1), L4(2), 0; L5(1), L5(2), 0];

% Differential Correction Parameters
max_iter = 100;
tol = 1e-9;

% ODE options
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);

for i = 1:size(L_points, 1)
    % Initial condition near the Lagrange point
    equilibrium = [L_points(i, :)'; 0; 0; 0];
    
    % Define the linearized system matrix A at the Lagrange point
    x = equilibrium(1);
    y = equilibrium(2);
    z = equilibrium(3);

    d = sqrt((x + mu)^2 + y^2 + z^2);
    r = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    % Compute the second partial derivatives of the potential function U
    Uxx = 1 - (1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * (x + mu)^2 / d^5 + 3 * mu * (x - 1 + mu)^2 / r^5;
    Uyy = 1 - (1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * y^2 / d^5 + 3 * mu * y^2 / r^5;
    Uzz = -(1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * z^2 / d^5 + 3 * mu * z^2 / r^5;
    Uxy = 3 * (1 - mu) * (x + mu) * y / d^5 + 3 * mu * (x - 1 + mu) * y / r^5;
    Uxz = 3 * (1 - mu) * (x + mu) * z / d^5 + 3 * mu * (x - 1 + mu) * z / r^5;
    Uyz = 3 * (1 - mu) * y * z / d^5 + 3 * mu * y * z / r^5;

    A = [
        0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 1, 0;
        0, 0, 0, 0, 0, 1;
        Uxx, Uxy, Uxz, 0, 2, 0;
        Uxy, Uyy, Uyz, -2, 0, 0;
        Uxz, Uyz, Uzz, 0, 0, 0;
    ];

    % Compute eigenvalues and eigenvectors
    [V, D] = eig(A);
    eigenvalues = diag(D);

    % Print eigenvalues and eigenvectors
    fprintf('Eigenvalues for L%d:\n', i);
    disp(eigenvalues);
    fprintf('Eigenvectors for L%d:\n', i);
    disp(V);

    % Identify the purely imaginary eigenvalues and corresponding eigenvectors
    imag_indices = abs(real(eigenvalues)) < 1e-6 & imag(eigenvalues) ~= 0;
    relevant_eigenvectors = V(:, imag_indices);
    relevant_eigenvalues = eigenvalues(imag_indices);

    % Choose a small perturbation amplitude
    epsilon = 1e-8;

    % Differential correction loop
    for j = 1:length(relevant_eigenvalues)
        % Initial perturbation in the direction of the relevant eigenvector
        perturbed_initial_conditions = equilibrium + epsilon * imag(relevant_eigenvectors(:, j));
        converged = false;
        
        for iter = 1:max_iter
            % Integrate the actual CR3BP system
            omega = abs(imag(relevant_eigenvalues(j)));
            T_span = [0, 2*pi/omega]; % Period of the oscillation
            [~, state] = ode45(@(t, state) cr3bp_ode(t, state, mu), T_span, perturbed_initial_conditions, options);

            % Deviation from periodicity
            delta_state = state(end, :)' - state(1, :)';
            if norm(delta_state) < tol
                converged = true;
                break;
            end

            % Compute numerical Jacobian (approximation)
            J = numerical_jacobian(@(state) cr3bp_ode(0, state, mu), state(1, :));
            correction = -J \ delta_state;
            perturbed_initial_conditions = perturbed_initial_conditions + correction;
        end
        
        % If converged, plot the periodic orbit
        if converged
            fprintf('Converged periodic orbit found at L%d for eigenvector %d\n', i, j);
            figure;
            plot3(state(:, 1), state(:, 2), state(:, 3));
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title(['Periodic Orbit at L', num2str(i), ' (Eigenvector ' num2str(j) ')']);
            grid on;
        else
            fprintf('Failed to converge to a periodic orbit at L%d for eigenvector %d\n', i, j);
        end
    end
end

% Three-body equations of motion
function dstate_dt = cr3bp_ode(~, state, mu)
    x = state(1); y = state(2); z = state(3);
    u = state(4); v = state(5); w = state(6);

    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    dx_dt = u;
    dy_dt = v;
    dz_dt = w;
    du_dt = 2*v + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3;
    dv_dt = -2*u + y - (1 - mu) * y / r1^3 - mu * y / r2^3;
    dw_dt = -(1 - mu) * z / r1^3 - mu * z / r2^3;

    dstate_dt = [dx_dt; dy_dt; dz_dt; du_dt; dv_dt; dw_dt];
end

% Numerical Jacobian (central difference approximation)
function J = numerical_jacobian(f, x)
    n = length(x);
    J = zeros(n, n);
    h = 1e-8; % Perturbation step size
    fx = f(x);
    for i = 1:n
        x_forward = x;
        x_backward = x;
        x_forward(i) = x_forward(i) + h;
        x_backward(i) = x_backward(i) - h;
        J(:, i) = (f(x_forward) - f(x_backward)) / (2*h);
    end
end
