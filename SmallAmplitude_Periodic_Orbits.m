function test
    % Define constants
    G = 6.67430e-11;  % Gravitational constant, m^3 kg^-1 s^-2
    M1 = 5.972e24;    % Mass of Earth (kg)
    M2 = 7.348e22;    % Mass of Moon (kg)
    R = 3844e5;       % Distance between Earth and Moon (m)
    
    % Standard gravitational parameters
    mu1 = G * M1;
    mu2 = G * M2;
    mu = mu2 / (mu1 + mu2);

    % Dimensionless units: x and time in units of R and sqrt(R^3 / (G * (M1 + M2)))
    % Calculate L1 position using an iterative method
    x_L1 = fzero(@(x) (1 - mu) * (x + mu) ./ abs(x + mu).^3 + mu * (x - 1 + mu) ./ abs(x - 1 + mu).^3 - x,0.8); %edit the intital guess to check other libration points as well
    
    % Initial guess for L1
    equilibrium = [x_L1; 0; 0; 0; 0; 0];
    initial_conditions = equilibrium;

    % Linearized system matrix A at L1
    state = equilibrium;
    x = state(1);
    y = state(2);
    z = state(3);
    u = state(4);
    v = state(5);
    w = state(6);

    % Compute distances d and r
    d = sqrt((x + mu)^2 + y^2 + z^2);
    r = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    % Compute the second partial derivatives of the potential function U
    Uxx = 1 - (1 - mu) / d^3 - mu / r^3 ...
          + 3 * (1 - mu) * (x + mu)^2 / d^5 + 3 * mu * (x - 1 + mu)^2 / r^5;

    Uyy = 1 - (1 - mu) / d^3 - mu / r^3 ...
          + 3 * (1 - mu) * y^2 / d^5 + 3 * mu * y^2 / r^5;

    Uzz = -(1 - mu) / d^3 - mu / r^3 ...
          + 3 * (1 - mu) * z^2 / d^5 + 3 * mu * z^2 / r^5;

    Uxy = 3 * (1 - mu) * (x + mu) * y / d^5 + 3 * mu * (x - 1 + mu) * y / r^5;

    Uxz = 3 * (1 - mu) * (x + mu) * z / d^5 + 3 * mu * (x - 1 + mu) * z / r^5;

    Uyz = 3 * (1 - mu) * y * z / d^5 + 3 * mu * y * z / r^5;

    % Construct the system matrix A
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

    % Identify the purely imaginary eigenvalues and corresponding eigenvectors
    imag_indices = abs(real(eigenvalues)) < 1e-9 & imag(eigenvalues) ~= 0;
    relevant_eigenvectors = V(:, imag_indices);
    relevant_eigenvalues = eigenvalues(imag_indices);

    % Choose a small perturbation amplitude
    epsilon = 1e-10;

    % Perturb initial conditions in the direction of the relevant eigenvectors
    for i = 1:length(relevant_eigenvalues)
        omega = abs(imag(relevant_eigenvalues(i)));
        T_span = [0, 2 * pi / omega];  % Period of the oscillation

        perturbed_initial_conditions = initial_conditions + epsilon * imag(relevant_eigenvectors(:, i));

        % Integrate the actual CR3BP system
        options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
        [~, state] = ode45(@(t, state) cr3bp_ode(t, state, mu), T_span, perturbed_initial_conditions, options);

        % Plot the results
        figure;
        plot3(state(:, 1), state(:, 2), state(:, 3));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        title(['Perturbed Trajectory (Eigenvector ' num2str(i) ')']);
        grid on;
    end
end

function dstate_dt = cr3bp_ode(~, state, mu)
    % Defining the state vector
    x = state(1);
    y = state(2);
    z = state(3);
    u = state(4);
    v = state(5);
    w = state(6);

    % Compute distances d and r
    d = sqrt((x + mu)^2 + y^2 + z^2);
    r = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    % Compute accelerations
    ax = 2*v + x - (1 - mu) / d^3 * (x + mu) - mu / r^3 * (x - 1 + mu);
    ay = -2*u + y - (1 - mu) / d^3 * y - mu / r^3 * y;
    az = -(1 - mu) / d^3 * z - mu / r^3 * z;

    dstate_dt = [u; v; w; ax; ay; az];
end
