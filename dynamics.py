import taichi as ti
from bs_taichi import d2B_3, dB_3, open_BS_3, open_basis_3
import numpy as np
from quadrature import quad, quad_closure
ti.init(arch = ti.x64, default_fp= ti.f32)

n_quad = 2
n = 8
order_k = 3
n_samples = 1024
dim = 2
n_elements = n - 2 * (order_k - 1)
alpha, beta = 35, 10
mu = 30
gravity = 9.8
dt = 8e-3
p = ti.Vector.field(dim, float)
container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10)
container.place(p)
element = ti.root.dense(ti.i, n_elements)
x = ti.Vector.field(2, float)
v = ti.Vector.field(2, float)
J = ti.Matrix.field(order_k, order_k, float)
M = ti.Matrix.field(order_k, order_k, float)
K = ti.Matrix.field(order_k, order_k, float)
element.place(x, v, alpha, beta, mu, J)

# TODO: for node with multiplicity, add up coeffcients
@ti.func
def J_ii(u, i):
    ret = 0.0
    if i == 0:
        for j in ti.static(range(-order_k + 1, 1)):
            ret += open_basis_3(j, u, n)
    elif i == n:
        for j in ti.static(range(n, n + order_k - 2)):
            ret += open_basis_3(j, u, n)
    else :
        ret = open_basis_3(i, u, n) 
    return ret
@ti.func
def JTJ_ii(u, i):
    return J_ii(u, i) ** 2

@ti.func
def JuTJu_ii(u, i):
    ret = 0.0
    if i == 0:
        for j in ti.static(range(-order_k + 1, 1)):
            ret += dB_3(j, u, n)
    elif i == n:
        for j in ti.static(range(n, n + order_k - 2)):
            ret += dB_3(j, u, n)
    else :
        ret = dB_3(i, u, n) 
    return ret ** 2

@ti.func
def JuuTJuu_ii(u, i):
    ret = 0.0
    if i == 0:
        for j in ti.static(range(-order_k + 1, 1)):
            ret += d2B_3(j, u, n)
    elif i == n:
        for j in ti.static(range(n, n + order_k - 2)):
            ret += d2B_3(j, u, n)
    else :
        ret = d2B_3(i, u, n) 
    return ret ** 2


quad_J_ii = quad_closure(J_ii, n_quad, n_args = 2)
quad_JTJ_ii = quad_closure(JTJ_ii, n_quad, n_args = 2)
quad_JuTJu_ii = quad_closure(JuTJu_ii, n_quad, n_args = 2)
quad_JuuTJuu_ii = quad_closure(JuuTJuu_ii, n_quad, n_args = 2)
    
@ti.func
def jacobian(e):
    for i in range(e, e + order_k):
        I = ti.Vector([i-e, i-e])
        J[e][I] = quad_J_ii(i)
        # jacobian 
        M[e][I] = quad_JTJ_ii(i) 
        # mass
        K[e][I] = alpha * quad_JuTJu_ii(i) + beta * quad_JuuTJuu_ii(i)
        # stiffness
        
    # return J[e]

@ti.func
def mass(e):
    M[e] = J[e].transpose() @ J[e]
    return M[e]

@ti.func
def stiffness(e):
    pass

# n+1 nodes : u0 ,..., un
@ti.kernel
def iga():
    for e in range(n_elements):
        J = jacobian(e)
        # M = mass(e)
        # D = damping(e)
        # K = stiffness(e)
