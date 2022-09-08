# import taichi as ti
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# ti.init(ti.x64, default_fp=ti.f32)

n = 8
order_k = 4
n_samples = 64
dim = 2
n_extended = n + 1 + 2 *(order_k - 1)
# t = ti.field(float, (n+1)) # 0...n+1
# p = ti.Vector.field(dim, float, (n_extended), offset=order_k -1) # 0...n+1
# # the end knots are repeated with multiplicity k in order to interpolate the initial and final control points p0 and pn
# sample_points = ti.Vector.field(dim, float, (n_samples))
# @ti.kernel
# def initialize():
#     for i in p:
#         j, k = i % 2, i % 4
#         p[i] = ti.cast(ti.Vector([j, k]), float)

t = [0.0 for _ in range(order_k-1)] + [_ for _ in np.linspace(0.0, 1.0, num = n)] + [1.0 for _ in range(order_k)]
B = lambda i,k,u: (t[i] <= u < t[i+1]) * 1.0 if k <= 1 else B(i, k-1, u) *(u -t[i])/(t[i+k-1]-t[i]) + B(i+1, k-1, u) *  (t[i+k] - u)/(t[i+k]-t[i+1])

def draw_basis():
    for i in range(n):
        x = np.linspace(0.0, 1.0, n_samples)
        y = [B(i, order_k, _x) for _x in x]
        plt.figure('data')
        plt.plot(x, y, '.')
        plt.plot(x, y)
    plt.show()
        
# @ti.func
# def B(i, k, u):
#     assert i < n-1
#     ret = 0.0
#     if k == 1:
#         ret = ti.cast(t[i] < u < t[i+1], float)
#     else:
#         ret = B(i, k-1, u) *(u -t[i])/(t[i+k-1]-t[i]) + B(i+1, k-1, u) *  (t[i+k] - u)/(t[i+k]-t[i+1])
#     # FIXME: recursive func, make it close-form
#     # FIXME: final control points divided by zero
#     return ret

# @ti.func
# def do_nothing():
#     pass

# basis_funcs = [do_nothing]

# def closure_basis(k):
#     basis = basis_funcs[k-1]
#     @ti.func
#     def B(i, u):
#         assert i < n-1
#         ret = 0.0
#         if ti.static(k == 1):
#             ret = ti.cast(t[i] < u < t[i+1], float)
#         else:
#             ret = basis(i, u) *(u -t[i])/(t[i+k-1]-t[i]) + basis(i+1, k-1, u) *  (t[i+k] - u)/(t[i+k]-t[i+1])
#         # FIXME: final control points divided by zero
#         return ret
#     return B

# for _ in range(order_k):
#     basis_funcs.append(closure_basis(_+1)) 
# basis_chosen = basis_funcs[-1]


# @ti.func
# def BS(u):    
#     up = ti.Vector.zero(float ,dim)
#     down = 0.0
#     for i in range(n_extended):
#         _i = ti.min(ti.max(i, 0), n)
#         t = basis_chosen(_i, u)
#         up += p[i] * t
#         down += t
#     return 

# @ti.kernel
# def sample():
#     for i in sample_points:
#         u = i * 1.0 / n_samples
#         sample_points[i] = ti.Vector([u, BS(u)])

# gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
# initialize()

# def render():
#     sample()
#     gui.circles(p.to_numpy(), radius = 3.0, palette=[0xEEEEF0]) 
#     gui.circles(sample_points.to_numpy(), radius = 1.5, 
#                     palette=[0x068587])
                    
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
#     render()
draw_basis()