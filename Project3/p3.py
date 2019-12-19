import numpy as np
import matplotlib.pyplot as plt
from pro1_functions import *

# returns solution vector for given end time t
def analytic_sol(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def forward_scheme(u0, T):
    n_x = u0.size
    dx = 1 / (n_x - 1)
    dt = 0.5 * dx**2
    m_T = int(np.ceil(T/dt))

    dt = T/m_T  # Highest possible dt that ensures landing on the exact ending time value satisfying the stability criterion.
    xi = dt/dx**2
    
    u = u0
    u1 = np.zeros_like(u)
    for _ in range(m_T):
        u1[0] = 0
        for i in range(1,n_x-1):
            u1[i] = (1-2*xi)*u[i] + xi*(u[i+1]+u[i-1])

        u1[-1] = 0
        temp = u
        u = u1
        u1 = temp
    return u

# plot function
Nx = 10; Nt = 50
x = np.linspace(0, 1, Nx)
t = np.linspace(0, 1, Nt)
xx, tt = np.meshgrid(x,t)
surface = analytic_sol(xx, tt)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution, Temperature of rod over time')
ax.plot_surface(tt, xx, surface, linewidth=0, antialiased=False, cmap='coolwarm')
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$')
plt.savefig("results/surface_plot.jpg")


T = [0.05, 0.5]
N = [11, 101]

fig, axs = plt.subplots(2,2)

for idx in ((0,0), (0,1), (1,0), (1,1)):
    ax = axs[idx]
    x = np.linspace(0, 1, N[idx[1]])
    u0 = np.sin(np.pi * x)
    u_analytic = analytic_sol(x, T[idx[0]])
    u_forward = forward_scheme(u0, T[idx[0]])
    ax.plot(x, u_analytic)
    ax.plot(x, u_forward, linestyle='--')
    ax.set_title("dx = 1/" + str(N[idx[1]]-1) + ",  T = " + str(T[idx[0]]))
    print("dx = 1/" + str(N[idx[1]]-1) + ",  T = " + str(T[idx[0]]), " -- MSE: ", MSE(u_analytic, u_forward))
    box = ax.get_position()
    ax.set_position([box.x0 + box.height * 0.02, box.y0 + box.height * 0.1,
                 box.width *0.98, box.height * 0.9])


fig.suptitle("1-d heat equation u(x, T), analytical(solid) vs. explicit scheme(dashed)")
plt.savefig("results/forward_plot.jpg")


