import taichi as ti
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
ti.init(ti.x64, default_fp=ti.f32)

n = 8
order_k = 3
n_samples = 1024
dim = 2

p = ti.Vector.field(dim, float)
container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10)
container.place(p)

sample_points = ti.Vector.field(dim, float, (n_samples))
basis_field = ti.field(float, n_samples)


# n+1 nodes : u0 ,..., un

@ti.func
def uniform_basis_1(i, u, n):
    u_i = i * 1.0 / n
    u_i1 = (i + 1) * 1.0 / n
    return ti.cast(u_i <= u < u_i1, float)

@ti.func
def uniform_basis_2(i, u, n):
    du = 1.0 / n
    k = 2
    term1 = (u - du * i) / (du * (k-1))
    term2 = (du *(i+k) - u) / (du * (k-1))
    return uniform_basis_1(i,u,n) * term1 + uniform_basis_1(i + 1, u, n) * term2

@ti.func
def uniform_basis_3(i, u, n):
    du = 1.0 / n
    k = 3
    term1 = (u - du * i) / (du * (k-1))
    term2 = (du *(i+k) - u) / (du * (k-1))
    return uniform_basis_2(i,u,n) * term1 + uniform_basis_2(i + 1, u, n) * term2
    # FIXME: prune the 0 part

@ti.func
def BS_3(u, n):
    ret = ti.Vector.zero(float, dim)
    for i in range(n):
        ret += p[i] * uniform_basis_3(i, u, n + order_k - 1)
    return ret

@ti.func
def open_BS_3(u, n):
    ret = ti.Vector.zero(float, dim)
    for i in range(- order_k + 1, n):
        _i = clip(0, n-1, i)
        ret += p[_i] * open_basis_3(i, u, n)
    return ret

@ti.kernel
def sample(n: ti.i32):
    for i in sample_points:
        u = i * 1.0 / n_samples
        sample_points[i] = open_BS_3(u, n)

gui = ti.GUI("Dynamic B-Spline", res=(512, 512), background_color=0x112F41)
# initialize()

# n+1 nodes : u0 ,..., un

@ti.func
def clip(a, b, x):
    return ti.min(b, ti.max(a, x))

@ti.func
def open_basis_1(i, u, n):
    # i can be negative integer
    du = 1.0 / n
    u_i = clip(0, n, i) * du
    u_i1 = clip(0, n, i + 1) * du
    return ti.cast(u_i <= u < u_i1, float)

@ti.func
def open_basis_2(i, u, n):
    du = 1.0 / n
    k = 2
    denominator1 = clip(0,n,i + k -1) - clip(0, n, i)
    denominator2 = clip(0,n,i + k) - clip(0, n, i + 1)
    term1 = 0.0 if denominator1 == 0 else (u - du * clip(0, n, i)) / (du * denominator1)
    term2 = 0.0 if denominator2 == 0 else (du *clip(0, n, i+k) - u) / (du * denominator2)
    return open_basis_1(i,u,n) * term1 + open_basis_1(i + 1, u, n) * term2

@ti.func
def open_basis_3(i, u, n):
    du = 1.0 / n
    k = 3
    denominator1 = clip(0,n,i + k -1) - clip(0, n, i)
    denominator2 = clip(0,n,i + k) - clip(0, n, i + 1)
    term1 = 0.0 if denominator1 == 0 else (u - du * clip(0,n,i)) / (du * denominator1)
    term2 = 0.0 if denominator2 == 0 else (du *clip(0,n,i+k) - u) / (du * denominator2)
    return open_basis_2(i,u,n) * term1 + open_basis_2(i + 1, u, n) * term2

@ti.func
def dB_3(i, u, n):
    '''
    First derivative of basis function B with order 3
    '''
    du = 1.0 / n
    k = 3
    denominator1 = clip(0,n,i + k -1) - clip(0, n, i)
    denominator2 = clip(0,n,i + k) - clip(0, n, i + 1)
    term1 = 0.0 if denominator1 == 0 else (k-1) / (du * denominator1) 
    term2 = 0.0 if denominator2 == 0 else (k-1) / (du * denominator2)
    return open_basis_2(i, u, n) * term1 - open_basis_2(i+1, u, n) * term2

@ti.func
def dB_2(i, u, n):
    '''
    First derivative of order 2 basis function
    '''
    du = 1.0 / n
    k = 2
    denominator1 = clip(0,n,i + k -1) - clip(0, n, i)
    denominator2 = clip(0,n,i + k) - clip(0, n, i + 1)
    term1 = 0.0 if denominator1 == 0 else (k-1) / (du * denominator1) 
    term2 = 0.0 if denominator2 == 0 else (k-1) / (du * denominator2)
    return open_basis_1(i, u, n) * term1 - open_basis_1(i+1, u, n) * term2

@ti.func
def d2B_3(i, u, n):
    '''
    Second derivative of order 3 basis function
    '''
    du = 1.0 / n
    k = 3
    denominator1 = clip(0,n,i + k -1) - clip(0, n, i)
    denominator2 = clip(0,n,i + k) - clip(0, n, i + 1)
    term1 = 0.0 if denominator1 == 0 else (k-1) / (du * denominator1) 
    term2 = 0.0 if denominator2 == 0 else (k-1) / (du * denominator2)
    return term1 * dB_2(i,u,n) - term2 * dB_2(i+1, u, n)

def render(cnt):
    # sample(cnt)
    
    gui.circles(p.to_numpy()[:cnt], radius = 3.0, color = 0xEE0000) 
    gui.circles(sample_points.to_numpy(), radius = 1.0, color=0xEEFFE0)
    gui.show()

def draw_basis(n, derivative = 1):
    assert derivative <= 2
    dBs = [open_basis_3, dB_3, d2B_3]
    dB = dBs[derivative]
    x = np.linspace(0.0, 1.0, n_samples)
    @ti.kernel
    def map(i: ti.i32):
        for j in sample_points:
            u = j * 1.0 / n_samples
            basis_field[j] = dB(i, u, n)
    for i in range(- order_k + 1, n):
        map(i)
        y = basis_field.to_numpy()
        plt.figure('data')
        plt.plot(x, y, '.')
        plt.plot(x, y)
    plt.show()
        
if __name__ == '__main__':
    cnt = 0
    derivative = 0
    plot_basis_only = True
    if plot_basis_only:
        draw_basis(n, derivative)
        quit()

    while gui.running:
        if cnt >= 1:
            render(cnt)
        else:
            gui.show()
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            print(e.key)
            if e.key == ti.GUI.LMB:
                mxy = np.array(gui.get_cursor_pos(), dtype = np.float32)
                p[cnt] = ti.Vector(mxy)
                cnt += 1
                print(cnt)
                sample(cnt)
            if e.key == 'r':
                container.deactivate_all()
                cnt = 0
