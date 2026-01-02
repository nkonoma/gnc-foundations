function dx = lorenz(t, x, sigma, beta, rho)
% x is a 3 dimensional state vector
dx = [
    sigma*(x(2) - x(1));
    x(1)*(rho - x(3)) - x(2);
    x(1)*x(2) - beta*x(3);
    ];
end