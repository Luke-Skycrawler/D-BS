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
gravity = ti.Vector([0.0, -9.8])
dt = 8e-3
p_like = lambda: ti.Vector.field(dim, float) 
p = ti.Vector.field(dim, float)
r, d, b, v, old_p = [p_like() for _ in range(5)]
# for cg

# container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10)
container = ti.dense(ti.i, n)
container.place(p, r, d, b, v, old_p)

element = ti.root.dense(ti.i, n_elements).dense(ti.j, order_k)

scalar = lambda: ti.field(float)
J_diag = scalar()
Ju_diag = scalar()
Juu_diag = scalar()
# remember those are 3 * 3 matrix, i.e., identity multiplying the scalar value 
tmp = ti.Vector.field(dim, float)
Ap_ret = ti.Vector.field(dim, float)
element.place(J_diag, Ju_diag, Juu_diag, tmp, Ap_ret)

# TODO: for node with multiplicity, add up coeffcients
@ti.func
def J_ii(u, i):
    ret = 0.0
    if i == 0:
        for j in ti.static(range(-order_k + 1, 1)):
            ret += open_basis_3(j, u, n)
    elif i == n - order_k:
        for j in ti.static(range(n - order_k, n)):
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
    elif i == n - order_k:
        for j in ti.static(range(n - order_k, n)):
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
    elif i == n - order_k:
        for j in ti.static(range(n - order_k, n)):
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
    
@ti.kernel
def compute_b():
    '''
    b = 4\triangle t ^ 2 f_p + 8Mv - (3M - 2\triangle t D)p^{(t-\triangle t)}-\int \mu J^Tc^{(t - \triangle t)} du

    '''
    for i in range(n_elements):
        4 * dt ** 2 * gravity + 8 * Mp(v) - 3 * Mp(p_m1) - mu * quad()
    pass

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

@ti.func
def Ap(p):
    '''
    returns matrix-vector product with A
    A = 4M + 2 \triangle t D + 4(\triangle t) ^ 2 K
    '''
    ret = ti.Vector([0.0] * n * dim)
    # FIXME: probably need a field
    for e in range(n_elements):
        # for each element, integrate M = J^T Jp, K = alpha Ju^T Ju p + beta Juu^T Juu p
        
        # compute v = J(u)p
        for k in ti.static(range(order_k)):
            J_diag[e, k] = J_ii(k + e, u) 
            tmp[e, k] = J_ii(k + e, u) * p[i + k]
            # compute J^T v
            Ap_ret[e, k] += J_diag(e, k) * tmp[e, k]
        # integrate 

        pass

@ti.func
def Mp(p):
    '''
    product with mass matrix M
    '''
    pass
@ti.kernel
def cg(p):
    # informally written
    d = r = b - Ap(p)
    for i in range(n):
        alpha = r.dot(r) / d.dot(Ap(d))
        p += alpha * d
        tmp_r = r
        r -= alpha * Ap(d)
        beta = r.dot(r) / tmp_r.dot(tmp_r)
        d = r + beta * d
        
    
def step():
    compute_b()
    _b = b.to_numpy()
    # solver = ti.linalg.SparseSolver(solver_type="LDLT")
    # solver.analyze_pattern(A)
    # solver.factorize(A)
    # _p = solver.solve(_b)
    p.from_numpy(_p.reshape((dim, n))) # update p
    
