import taichi as ti

@ti.func
def plus(f1, f2, res):
    for i in f1:
        res[i] = f1[i] + f2[i]

@ti.func
def minus(f1, f2, res):
    for i in f1:
        res[i] = f1[i] - f2[i]

@ti.func
def dot(f1, f2):
    ret = 0.0
    for i in f1:
        ret += f1[i] * f2[i] 
    return ret

@ti.func
def fma(f1, k, f2):
    '''
    fused multiply and add
    f2 += k f1
    '''
    for i in f2:
        f1[i] += k * f2[i]
    