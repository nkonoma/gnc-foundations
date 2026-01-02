clear all

% Lorenz's parameters (chaotic)
sigma = 10;
beta = 8/3;
rho = 28;

% initial condition
x0 = [-8; 8; 27];

% compute trajectory
dt = 0.01;
tspan = [0:dt:4];

X(:,1) = x0;
xin = x0;
for i = 1:tspan(end)/dt
    time = i*dt;
    xout = rk4singlestep(@(t,x)lorenz(t, x, sigma, beta, rho), dt, time, xin);
    X = [X xout];
    xin = xout;
end
plot3(X(1,:), X(2,:), X(3,:), 'k', 'LineWidth', 3)
hold on
set(gca, 'FontSize', 24)
view(20,40)
grid on
% Compare with Build in Rk4 adaptive step
[t,x] = ode45(@(t,x)lorenz(t,x,sigma,beta,rho),tspan,x0);
plot3(x(:,1),x(:,2),x(:,3),'c','LineWidth',3)
%set(gca, 'Position', [1500 500 800 600])
comet3(x(:,1),x(:,2),x(:,3))