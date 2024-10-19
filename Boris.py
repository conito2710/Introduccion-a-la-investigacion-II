import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy as sp
from numba import jit

# Algoritmo de Boris

@jit(nopython=True)
def boris(x, y, t, v_i, gamma0, r0, ex, ey, ez, bx, by, bz, NT, Nx, Ny, dt, q, fact=1):

    "Dados los campos electromagneticos ex,ey,ez,bx,by,bz, el algoritmo de Boris resuelve numericamente la ecuación"
    "de fuerza de Lorentz relativista, este codigo aplica el algoritmo de Boris en una particula individial "


    t = np.linspace(0, NT, int(NT * 155 * fact))  # Tiempo de ejecución
    Nt = len(t)

    # Inicialización de p, v, gamma, r y r_plot

    p = np.zeros((Nt, 3))
    p[0, :] = v_i * gamma0 # Condición inical del momento

    v = np.zeros_like(p,dtype=np.float32)
    v[0, :] = v_i # Condición inicial de la velocidad

    gamma = np.zeros(Nt,dtype=np.float32)
    gamma[0] = gamma0 # Factor gamma inicial

    r = np.zeros_like(p,dtype=np.float32)
    r[0, :] = r0 # Posición inicial

    # Crear los interpoladores para los campos E y B
    b_interp_funcs = [
        RegularGridInterpolator((x, y), bx),
        RegularGridInterpolator((x, y), by),
        RegularGridInterpolator((x, y), bz)
    ]
    e_interp_funcs = [
        RegularGridInterpolator((x, y), ex),
        RegularGridInterpolator((x, y), ey),
        RegularGridInterpolator((x, y), ez)
    ]

    r_plot = np.zeros_like(p,dtype=np.float32) # Array para plotear la posición, teniendo en cuenta las CB periodicas
    r_plot[0, :] = r[0, :]

    # Bucle principal para cada paso temporal
    for i in range(Nt - 1):
        # Condiciones de borde periódicas
        r[i, 0] = np.mod(r[i, 0], Nx - 1)
        r[i, 1] = np.mod(r[i, 1], Ny - 1)

        # Preparamos puntos para la interpolación
        point = np.array([r[i, 0], r[i, 1]])

        # Interpolación de campos E y B para la partícula
        b_interp = np.array([func(point) for func in b_interp_funcs])
        e_interp = np.array([func(point) for func in e_interp_funcs])

        # Factor auxiliar a_act
        a_act = q * 0.5 * b_interp / fact

        # Primer paso del método de Boris
        p0 = p[i, :] + q * e_interp / (2 * fact)
        gammap0 = np.sqrt(1 + np.dot(p0, p0))

        # Actualización de p usando el método de Boris
        p1 = p0 + 2 * np.cross((gammap0 * p0 + np.cross(p0, a_act)), a_act) / (gammap0**2 + np.dot(a_act, a_act))
        p[i + 1, :] = p1 + q * e_interp / (2 * fact)

        # Actualización de gamma y velocidad
        gamma[i + 1] = np.sqrt(1 + np.dot(p[i + 1, :], p[i + 1, :]))
        v[i + 1, :] = p[i + 1, :] / gamma[i + 1]

        # Actualización de la posición
        r[i + 1, :] = r[i, :] + v[i, :] * dt / fact

        # Actualización del array para graficar
        r_plot[i + 1, :] = r[i + 1, :]

    return r_plot, v, gamma



# Aqui intento hacerlo para varias particulas la lesera, se me ocurre que podria hacer r un tensor

@jit(nopython=True)
def varias_particulas(x, y, v_i, gamma0, r0, ex, ey, ez, bx, by, bz, NT, Nx, Ny, dt, q, fact=1):
    
    "Dados los campos electromagneticos ex,ey,ez,bx,by,bz, el algoritmo de Boris resuelve numericamente la ecuación"
    "de fuerza de Lorentz relativista, este codigo aplica el algoritmo de Boris en varias particulas"

    # Número de partículas
    Np = r0.shape[1]

    t = np.linspace(0, NT, int(NT * 155 * fact)) # Tiempo de ejecución
    Nt = len(t)

    # Inicialización de p, v, gamma, r y r_plot

    p = np.zeros((Nt, 3, Np))
    p[0, :, :] = v_i * gamma0

    v = np.zeros_like(p)
    v[0, :, :] = v_i

    gamma = np.zeros((Nt, Np))
    gamma[0, :] = gamma0

    r = np.zeros_like(p)
    r[0, :, :] = r0

    r_plot = np.zeros_like(p)
    r_plot[0, :, :] = r0

    # Crear los interpoladores para los campos E y B
    b_interp_funcs = [
        RegularGridInterpolator((x, y), bx),
        RegularGridInterpolator((x, y), by),
        RegularGridInterpolator((x, y), bz)
    ]
    e_interp_funcs = [
        RegularGridInterpolator((x, y), ex),
        RegularGridInterpolator((x, y), ey),
        RegularGridInterpolator((x, y), ez)
    ]

    # Bucle principal para cada paso temporal
    for i in range(Nt - 1):
        # Condiciones de borde periódicas
        r[i, 0, :] = np.mod(r[i, 0, :], Nx - 1)
        r[i, 1, :] = np.mod(r[i, 1, :], Ny - 1)

        # Preparamos puntos para la interpolación 
        points = np.stack((r[i, 0, :], r[i, 1, :]), axis=-1)

        # Interpolación de campos E y B para todas las partículas (vectorizado)
        b_interp = np.vstack([func(points) for func in b_interp_funcs])
        e_interp = np.vstack([func(points) for func in e_interp_funcs])

        # Factor auxiliar a_act 
        a_act = q * 0.5 * b_interp / fact

        # Primer paso del método de Boris
        p0 = p[i, :, :] + q * e_interp / (2 * fact)
        gammap0 = np.sqrt(1 + np.sum(p0 ** 2, axis=0))  # Vectorizado

        # Actualización de p usando el método de Boris
        p1 = p0 + 2 * np.cross((gammap0 * p0 + np.cross(p0, a_act, axis=0)), a_act, axis=0) / (gammap0**2 + np.sum(a_act**2, axis=0))
        p[i + 1, :, :] = p1 + q * e_interp / (2 * fact)

        # Actualización de gamma y velocidad (vectorizado)
        gamma[i + 1, :] = np.sqrt(1 + np.sum(p[i + 1, :, :]**2, axis=0))
        v[i + 1, :, :] = p[i + 1, :, :] / gamma[i + 1, :]

        # Actualización de la posición (vectorizado)
        r[i + 1, :, :] = r[i, :, :] + v[i, :, :] * dt / fact

        # Almacenar los valores de la posición para graficar
        r_plot[i + 1, :, :] = r[i + 1, :, :]

    return r_plot, v, gamma

@jit(nopython=True)
def varias_particulas_potencia(x, y, v_i, gamma0, r0, ex, ey, ez, bx, by, bz, NT, Nx, Ny, dt, q, fact=1):

    "Dados los campos electromagneticos ex,ey,ez,bx,by,bz, el algoritmo de Boris resuelve numericamente la ecuación"
    "de fuerza de Lorentz relativista, este codigo aplica el algoritmo de Boris en varias particulas, ademas calcula"
    "la potencia entregada por los campos"

    # Número de partículas
    Np = r0.shape[1]

    t = np.linspace(0, NT, int(NT * 155 * fact))  # Tiempo de ejecución
    Nt = len(t)

    # Inicialización de p, v, gamma, r, r_plot y P (potencia)
    p = np.zeros((Nt, 3, Np))
    p[0, :, :] = v_i * gamma0

    v = np.zeros_like(p)
    v[0, :, :] = v_i

    gamma = np.zeros((Nt, Np))
    gamma[0, :] = gamma0

    r = np.zeros_like(p)
    r[0, :, :] = r0

    P = np.zeros((Nt, Np))

    # Crear interpoladores para los campos E y B (fuera del bucle para eficiencia)
    b_interp_funcs = [
        RegularGridInterpolator((x, y), bx),
        RegularGridInterpolator((x, y), by),
        RegularGridInterpolator((x, y), bz)
    ]
    e_interp_funcs = [
        RegularGridInterpolator((x, y), ex),
        RegularGridInterpolator((x, y), ey),
        RegularGridInterpolator((x, y), ez)
    ]

    # Inicialización de la potencia en t=0
    points_initial = np.stack((r0[0, :], r0[1, :]), axis=-1)
    e_interp_in = np.array([func(points_initial) for func in e_interp_funcs]).reshape(3, Np)
    P[0, :] = q*np.einsum('ij,ij->j', v_i, e_interp_in)

    # Bucle principal para la simulación
    for i in range(Nt - 1):
        # Condiciones de borde periódicas
        r[i, 0, :] = np.mod(r[i, 0, :], Nx - 1)
        r[i, 1, :] = np.mod(r[i, 1, :], Ny - 1)

        # Preparamos puntos para la interpolación
        points = np.stack((r[i, 0, :], r[i, 1, :]), axis=-1)

        # Interpolación de campos E y B
        b_interp = np.array([func(points) for func in b_interp_funcs]).reshape(3, Np)
        e_interp = np.array([func(points) for func in e_interp_funcs]).reshape(3, Np)

        # Factor auxiliar a_act
        a_act = q * 0.5 * b_interp / fact

        # Primer paso del método de Boris
        p0 = p[i, :, :] + q * e_interp / (2 * fact)
        gammap0 = np.sqrt(1 + np.sum(p0 ** 2, axis=0))  # Vectorizado para todas las partículas

        # Actualización de p usando el método de Boris
        p1 = p0 + 2 * np.cross(gammap0 * p0 + np.cross(p0, a_act, axis=0), a_act, axis=0) / (gammap0**2 + np.sum(a_act**2, axis=0))
        p[i + 1, :, :] = p1 + q * e_interp / (2 * fact)

        # Actualización de gamma y velocidad, todas vectorizadas
        gamma[i + 1, :] = np.sqrt(1 + np.sum(p[i + 1, :, :]**2, axis=0))
        v[i + 1, :, :] = p[i + 1, :, :] / gamma[i + 1, :]

        # Actualización de la posición
        r[i + 1, :, :] = r[i, :, :] + v[i, :, :] * dt / fact

        # Cálculo de la potencia, vectorizado
        P[i + 1, :] = q*np.einsum('ij,ij->j', v[i + 1, :, :], e_interp)

    return r, v, gamma, P

@jit(nopython=True)
def varias_particulas_trabajo_separados(x, y, v_i, gamma0, r0, v, ex, ey, ez, bx, by, bz, NT, Nx, Ny, dt, q, fact=1):

    "Dados los campos electromagneticos ex,ey,ez,bx,by,bz, el algoritmo de Boris resuelve numericamente la ecuación"
    "de fuerza de Lorentz relativista, este codigo aplica el algoritmo de Boris en varias particulas, ademas calcula"
    "el trabajo entegado por los campos, esto para el campo electrico ideal y no ideal"

    b=np.array((bx,by,bz))

    e=np.array((ex,ey,ez))

    e_i = -np.cross(v, b, axis=0) # -v * B

    e_ni=e-e_i  # e_i + e_ni = e_tot

    # Número de partículas
    Np = r0.shape[1]

    t = np.linspace(0, NT, int(NT * 155 * fact))  # Tiempo de ejecución
    Nt = len(t)

    # Inicialización de p, v, gamma, r, r_plot y P (potencia)
    p = np.zeros((Nt, 3, Np))
    p[0, :, :] = v_i * gamma0

    v = np.zeros_like(p)
    v[0, :, :] = v_i

    gamma = np.zeros((Nt, Np))
    gamma[0, :] = gamma0

    r = np.zeros_like(p)
    r[0, :, :] = r0

    P_ideal = np.zeros((Nt, Np))
    P_no_ideal = np.zeros((Nt, Np))

    W_ideal = np.zeros((Nt, Np))

    W_no_ideal = np.zeros((Nt, Np))

    # Crear interpoladores para los campos E y B (fuera del bucle para eficiencia)
    b_interp_funcs = [
        RegularGridInterpolator((x, y), bx),
        RegularGridInterpolator((x, y), by),
        RegularGridInterpolator((x, y), bz)
    ]
    e_interp_funcs = [
        RegularGridInterpolator((x, y), ex),
        RegularGridInterpolator((x, y), ey),
        RegularGridInterpolator((x, y), ez)
    ]

    e_interp_funcs_ideal = [
        RegularGridInterpolator((x, y), e_i[0]),
        RegularGridInterpolator((x, y), e_i[1]),
        RegularGridInterpolator((x, y), e_i[2])
    ]

    e_interp_funcs_no_ideal = [
        RegularGridInterpolator((x, y), e_ni[0]),
        RegularGridInterpolator((x, y), e_ni[1]),
        RegularGridInterpolator((x, y), e_ni[2])
    ]

    # Inicialización de la potencia en t=0
    points_initial = np.stack((r0[0, :], r0[1, :]), axis=-1)
    e_interp_in_ideal = np.array([func(points_initial) for func in e_interp_funcs_ideal]).reshape(3, Np)
    P_ideal[0, :] = q*np.einsum('ij,ij->j', v_i, e_interp_in_ideal)


    # Inicialización de la potencia en t=0
    points_initial = np.stack((r0[0, :], r0[1, :]), axis=-1)
    e_interp_in_no_ideal = np.array([func(points_initial) for func in e_interp_funcs_no_ideal]).reshape(3, Np)
    P_no_ideal[0, :] = q*np.einsum('ij,ij->j', v_i, e_interp_in_no_ideal)

    # Bucle principal para la simulación
    for i in range(Nt - 1):
        # Condiciones de borde periódicas
        r[i, 0, :] = np.mod(r[i, 0, :], Nx - 1)
        r[i, 1, :] = np.mod(r[i, 1, :], Ny - 1)

        # Preparamos puntos para la interpolación
        points = np.stack((r[i, 0, :], r[i, 1, :]), axis=-1)

        # Interpolación de campos E y B
        b_interp = np.array([func(points) for func in b_interp_funcs]).reshape(3, Np)
        e_interp = np.array([func(points) for func in e_interp_funcs]).reshape(3, Np)

        # Interpolación de E ideal y E no ideal
        
        e_interp_ideal = np.array([func(points) for func in e_interp_funcs_ideal]).reshape(3, Np)
        e_interp_no_ideal = np.array([func(points) for func in e_interp_funcs_no_ideal]).reshape(3, Np)


        # Factor auxiliar a_act
        a_act = q * 0.5 * b_interp / fact

        # Primer paso del método de Boris
        p0 = p[i, :, :] + q * e_interp / (2 * fact)
        gammap0 = np.sqrt(1 + np.sum(p0 ** 2, axis=0))  # Vectorizado para todas las partículas

        # Actualización de p usando el método de Boris
        p1 = p0 + 2 * np.cross(gammap0 * p0 + np.cross(p0, a_act, axis=0), a_act, axis=0) / (gammap0**2 + np.sum(a_act**2, axis=0))
        p[i + 1, :, :] = p1 + q * e_interp / (2 * fact)

        # Actualización de gamma y velocidad, todas vectorizadas
        gamma[i + 1, :] = np.sqrt(1 + np.sum(p[i + 1, :, :]**2, axis=0))
        v[i + 1, :, :] = p[i + 1, :, :] / gamma[i + 1, :]

        # Actualización de la posición
        r[i + 1, :, :] = r[i, :, :] + v[i, :, :] * dt / fact

        # Cálculo de la potencia, vectorizado
        P_ideal[i + 1, :] = q*np.einsum('ij,ij->j', v[i + 1, :, :], e_interp_ideal)
        P_no_ideal[i + 1, :] = q*np.einsum('ij,ij->j', v[i + 1, :, :], e_interp_no_ideal)

    W_ideal = sp.integrate.cumtrapz(P_ideal, axis=0, initial=0)
    W_no_ideal = sp.integrate.cumtrapz(P_no_ideal, axis=0, initial=0)
            
    return r, v, gamma, W_ideal, W_no_ideal