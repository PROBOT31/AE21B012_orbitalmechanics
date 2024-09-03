function test(~)
    % Define the mass ratio
    mu = 0.012145; 

    % Initial guess for L1
    equilibrium = [  1.1557 ; 0 ; 0; 0; 0; 0];
    initial_conditions = equilibrium;

    % Define the linearized system matrix A at L1
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
    Uxx = 1 - (1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * (x + mu)^2 / d^5 + 3 * mu * (x - 1 + mu)^2 / r^5;
    Uyy = 1 - (1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * y^2 / d^5 + 3 * mu * y^2 / r^5;
    Uzz = -(1 - mu) / d^3 - mu / r^3 + 3 * (1 - mu) * z^2 / d^5 + 3 * mu * z^2 / r^5;
    Uxy = 3 * (1 - mu) * (x + mu) * y / d^5 + 3 * mu * (x - 1 + mu) * y / r^5;
    Uyx = Uxy;
    Uxz = 3 * (1 - mu) * (x + mu) * z / d^5 + 3 * mu * (x - 1 + mu) * z / r^5;
    Uzx = Uxz;
    Uyz =  3 * (1 - mu) * y * z / d^5 + 3 * mu * y * z / r^5;
    Uzy = Uyz;
  
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
    imag_indices = abs(real(eigenvalues)) < 1e-6 & imag(eigenvalues) ~= 0;
    relevant_eigenvectors = V(:, imag_indices);
    relevant_eigenvalues = eigenvalues(imag_indices);
    
    % Choose a small perturbation amplitude
    epsilon = 1e-8;

    % Perturb initial conditions in the direction of the relevant eigenvectors
    for i = 1:length(relevant_eigenvalues)
        omega = abs(imag(relevant_eigenvalues(i)));
        T_span = [0, 20*pi/omega] ; % Period of the oscillation

        perturbed_initial_conditions = initial_conditions + epsilon * imag(relevant_eigenvectors(:, i));
        
        % Integrate the actual CR3BP system
        options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
        [~, state] = ode78(@(t, state) cr3bp_ode(t, state, mu), T_span, perturbed_initial_conditions, options);

        
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
    
    % Inputting d and r relations
    d = sqrt((x + mu)^2 + y^2 + z^2);
    r = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    % Inputting governing equations
    ax = 2*v + x - (1 - mu) / d^3 * (x + mu) - mu / r^3 * (x - 1 + mu);
    ay = -2*u + y - (1 - mu) / d^3 * y - mu / r^3 * y;
    az = -(1 - mu) / d^3 * z - mu / r^3 * z;

    dstate_dt = [u; v; w; ax; ay; az];
end
