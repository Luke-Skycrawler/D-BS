import taichi as ti
import numpy as np
from scipy.special.orthogonal import p_roots

def quad_closure(f, n):
    [x, w] = p_roots(n + 1)
    X, W = ti.Vector(x, dt = ti.f32), ti.Vector(w, dt = ti.f32)
    # @ti.func
    # def gauss1(n):
    #     # [x,w] = p_roots(n+1)
    #     G=sum(w*f(x))
    #     return G
    @ti.func
    def gauss(a,b):
        G=0.5*(b-a)*W.dot(f(0.5*(b-a)*X+0.5*(b+a)))
        return G
    return gauss


if __name__ == "__main__":
    # f = lambda x: x** 2 + 3 * x + 1
    @ti.func
    def f(x):
        return x** 2 + 3 * x + 1
    F = lambda x: x ** 3 / 3 + x ** 2 / 2 * 3 + x
    n = 3
    gauss = quad_closure(f, n)

    @ti.kernel
    def quad():
        t = gauss(0.0, 1.0)
        print(t)

    ti.init(arch = ti.x64, default_fp= ti.f32)
    quad()
    print(F(1))
    
    
    