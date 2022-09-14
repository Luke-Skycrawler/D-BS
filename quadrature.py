import taichi as ti
import numpy as np
from scipy.special.orthogonal import p_roots


[x, w] = p_roots(n + 1)
X, W = ti.Vector(x, dt = ti.f32), ti.Vector(w, dt = ti.f32)

def fused_integrate_sum_closure(f, dst, src, n, order_k):
    '''
    e: element index
    f: basis function to be integrated with two arguments i, u and returns a dim * (n - k + 1) vector

    dst: ti.field accumulator
    src: ti.field to receive the return value of f
    a, b: integral region
    '''
    
    @ti.func
    def _gauss(e, coeff, p):
        '''
        coeff: coefficients to time before adding to the accumulator

        ret: dst field containing the integral result
        '''
        du = 1.0 / n
        a = du * e
        b = du * (e + order_k)
        for q in range(n):
            u = 0.5 * (b-a) * X[q] + 0.5 * (b+a)
            f(e, u, p)
            # result stored in src
            for i in src:
                dst[i] += src[i] * W[q] * coeff * 0.5 * (b - a)

    return _gauss

def quad_closure(f, n, n_args = 1):
    # @ti.func
    # def gauss1(n):
    #     # [x,w] = p_roots(n+1)
    #     G=sum(w*f(x))
    #     return G
    ret = None
    if n_args == 1:
        @ti.func
        def _gauss(a,b):
            G=0.5*(b-a)*W.dot(f(0.5*(b-a)*X+0.5*(b+a)))
            return G
        ret = _gauss
    elif n_args == 2:
        @ti.func
        def _gauss01(i):
            G=0.5 * W.dot(f(i, 0.5 * X + 0.5))
            return G
        ret = _gauss01
    elif n_args == 3:
        @ti.func
        def _gauss01(i, n):
            G=0.5 * W.dot(f(i, 0.5 * X + 0.5, n))
            return G
        ret = _gauss01
    return ret
    


if __name__ == "__main__":
    # f = lambda x: x** 2 + 3 * x + 1
    @ti.func
    def f(x):
        return x** 2 + 3 * x + 1
    @ti.func
    def f2(i, x, n):
        return x** 2 + 3 * x + 1
    F = lambda x: x ** 3 / 3 + x ** 2 / 2 * 3 + x
    n = 3
    select = 2
    gauss = quad_closure(f2, n, 3) if select == 2 else quad_closure(f, n)

    @ti.kernel
    def quad():
        t = gauss(0.0, 1.0)
        print(t)
    
    @ti.kernel
    def quad2():
        t = gauss(0, 0)
        print(t)

    ti.init(arch = ti.x64, default_fp= ti.f32)
    if select == 1:
        quad()
    else:
        quad2()
    print(F(1))
    