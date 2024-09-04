function lagrange_points_and_jacobian()
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

    % Display the results
    disp('Lagrange Points (in standardized coordinates):');
    fprintf('L1: (%.6f, 0.000, 0.000)\n', L1);
    fprintf('L2: (%.6f, 0.000, 0.000)\n', L2);
    fprintf('L3: (%.6f, 0.000, 0.000)\n', L3);
    fprintf('L4: (%.6f, %.6f, 0.000)\n', L4(1), L4(2));
    fprintf('L5: (%.6f, %.6f, 0.000)\n', L5(1), L5(2));

    % Define the system of equations for the 3-body problem
    syms x y z vx vy vz
    % Reference Coordinates
    x1 = -mu; y1 = 0; z1 = 0; % Earth
    x2 = 1-mu; y2 = 0; z2 = 0; % Moon (normalized distance R to 1)

    % First-order differential equations
    dx_dt = vx;
    dy_dt = vy;
    dz_dt = vz;
    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);
    dvx_dt = 2*vy + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3;
    dvy_dt = -2*vx + y - (1 - mu) * y / r1^3 - mu * y / r2^3;
    dvz_dt = -(1 - mu) * z / r1^3 - mu * z / r2^3;

    % System of equations
    F = [dx_dt; dy_dt; dz_dt; dvx_dt; dvy_dt; dvz_dt];
    V = [x; y; z; vx; vy; vz];

    % Jacobian matrix
    J = jacobian(F, V);

    % Equilibrium points
    L_points = [L1, 0, 0; L2, 0, 0; L3, 0, 0; L4(1), L4(2), 0; L5(1), L5(2), 0];

    % Evaluate the Jacobian at each equilibrium point
    for i = 1:size(L_points, 1)
        L_point = L_points(i, :);
        J_eval = double(subs(J, {x, y, z, vx, vy, vz}, {L_point(1), L_point(2), L_point(3), 0, 0, 0}));
        fprintf('Jacobian at L%d:\n', i);
        disp(J_eval);
        
        % Compute eigenvalues and eigenvectors
        [V_eig, D] = eig(J_eval);
        fprintf('Eigenvalues at L%d:\n', i);
        disp(diag(D));
        fprintf('Eigenvectors at L%d:\n', i);
        disp(V_eig);
        
        % Identify purely imaginary eigenvalues
        imaginary_indices = find(imag(diag(D)) ~= 0 & real(diag(D)) == 0);
        if isempty(imaginary_indices)
            fprintf('No purely imaginary eigenvalues at L%d.\n', i);
            continue;
        end
       
        center_eigenvectors = V_eig(:, imaginary_indices);
        
        % Initial condition near the Lagrange point
        init_cond = [L_point, 0, 0, 0] + 1e-6 * real(center_eigenvectors(:, 1))';
        
        % Integrate the system for a longer period
        options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
        [t, sol] = ode45(@(t, state) three_body_eqns(t, state, mu), [0, 100], init_cond, options);
        
        % Plot the periodic orbit
        figure;
        plot3(sol(:, 1), sol(:, 2), sol(:, 3));
        title(sprintf('Small Amplitude Periodic Orbit around L%d', i));
        xlabel('x'); ylabel('y'); zlabel('z');
        grid on;
        
        % Check for periodicity
        if norm(sol(end, 1:3) - sol(1, 1:3)) < 1e-6
            fprintf('Periodic orbit found around L%d.\n', i);
        else
            fprintf('The orbit around L%d may not be perfectly periodic; consider refining the initial conditions.\n', i);
        end
    end
end

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
