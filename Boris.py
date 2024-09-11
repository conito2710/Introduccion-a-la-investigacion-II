import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def boris(x,y,t,v_i,gamma0,r0,ex,ey,ez,bx,by,bz,NT,Nx,Ny,dt,fact=1):

    t=np.linspace(0,NT,int(NT*155*fact)) # Defino mi Delta t como se obtuvo en en analisis

    Nt=int(np.size(t))

    p = np.zeros((Nt, 3))
    p[0, :] = v_i * gamma0  # Condición inicial de p

    v = np.zeros_like(p)
    v[0, :] = v_i  # Condición inicial de v

    gamma = np.zeros((Nt))
    gamma[0] = gamma0

    r = np.zeros_like(p)

    r[0,:] = r0

    b_interp_func_x = RegularGridInterpolator((x, y), bx)
    b_interp_func_y = RegularGridInterpolator((x, y), by)
    b_interp_func_z = RegularGridInterpolator((x, y), bz)

    e_interp_func_x = RegularGridInterpolator((x, y), ex)
    e_interp_func_y = RegularGridInterpolator((x, y), ey)
    e_interp_func_z = RegularGridInterpolator((x, y), ez)

    # Array auxiliar para graficar
    r_plot = np.zeros_like(p)
    r_plot[0, :] = r[0, :]

    for i in range(0, Nt - 1):
        # Condiciones de Borde periódicas
        crossed_boundary = False

        if r[i, 0] < 0:
            r[i, 0] += (Nx - 1)
            crossed_boundary = True
        elif r[i, 0] > Nx - 1:
            r[i, 0] -= (Nx - 1)
            crossed_boundary = True

        if r[i, 1] < 0:
            r[i, 1] += (Ny - 1)
            crossed_boundary = True
        elif r[i, 1] > Ny - 1:
            r[i, 1] -= (Ny - 1)
            crossed_boundary = True

        point = np.array((r[i, 0], r[i, 1]))

        b_interp_x = b_interp_func_x(point)
        b_interp_y = b_interp_func_y(point)
        b_interp_z = b_interp_func_z(point)
        b_interp = np.array((b_interp_x, b_interp_y, b_interp_z))[:, 0]

        e_interp_x = e_interp_func_x(point)
        e_interp_y = e_interp_func_y(point)
        e_interp_z = e_interp_func_z(point)
        e_interp = np.array((e_interp_x, e_interp_y, e_interp_z))[:, 0]

        a_act = -1/2 * b_interp/fact

        p0 = p[i, :] - e_interp / (2*fact)

        gammap0=np.sqrt(1+np.linalg.norm(p0)**2)

        p1 = p0 + 2 / (gammap0**2 + np.linalg.norm(a_act)**2) * np.cross((gammap0* p0 + np.cross(p0, a_act)), a_act)
        p[i + 1, :] = p1 - e_interp / (2*fact)

        gamma[i + 1] = np.sqrt((1 + np.linalg.norm(p[i + 1,:])**2))

        v[i+1, :] = p[i+1 , :] / gamma[i+1] # Obtenemos v de vuelta

        # Actualizamos la posición r_i+1
        r[i + 1, :] = r[i, :] + v[i, :] * dt/fact # r_i+1 = r_i + v_i delta t*0.45/istep

        # Actualizar el array auxiliar para graficar
        if crossed_boundary:
            r_plot[i, :] = [np.nan, np.nan, np.nan]  # Insertar NaN para romper la línea en el gráfico
        r_plot[i+1, :] = r[i+1, :]

    return r_plot,v,gamma

# Aqui intento hacerlo para varias particulas la lesera, se me ocurre que podria hacer r un tensor

def varias_particulas(x,y,v_i,gamma0,r0,ex,ey,ez,bx,by,bz,NT,Nx,Ny,dt,fact=1):

    #r0 es un vector con las condiciones iniciales de todas las particulas, tiene dimensión Np

    Np=int(np.size(r0)/3) # Divido en 3 por la cant de coordenadas

    t=np.linspace(0,NT,int(NT*155*fact)) # Defino mi Delta t como se obtuvo en en analisis

    Nt=int(np.size(t))

    p = np.zeros((Nt, 3, Np))
    p[0, :, :] = v_i * gamma0  # Condición inicial de p

    v = np.zeros_like(p)
    v[0, :, :] = v_i  # Condición inicial de v

    gamma = np.zeros((Nt,Np))
    gamma[0,:] = gamma0

    r = np.zeros_like(p)

    r[0,: ,:] = r0

    b_interp_func_x = RegularGridInterpolator((x, y), bx)
    b_interp_func_y = RegularGridInterpolator((x, y), by)
    b_interp_func_z = RegularGridInterpolator((x, y), bz)

    e_interp_func_x = RegularGridInterpolator((x, y), ex)
    e_interp_func_y = RegularGridInterpolator((x, y), ey)
    e_interp_func_z = RegularGridInterpolator((x, y), ez)

    # Array auxiliar para graficar
    r_plot = np.zeros_like(p)
    r_plot[0, :, :] = r[0, : ,: ]

    for i in range(0, Nt - 1):
        for j in range(0,Np):
            # Condiciones de Borde periódicas
            crossed_boundary = False

            if r[i, 0, j] < 0:
                r[i, 0, j] += (Nx - 1)
                crossed_boundary = True
            elif r[i, 0, j] > Nx - 1:
                r[i, 0, j] -= (Nx - 1)
                crossed_boundary = True

            if r[i, 1, j] < 0:
                r[i, 1, j] += (Ny - 1)
                crossed_boundary = True
            elif r[i, 1, j] > Ny - 1:
                r[i, 1, j] -= (Ny - 1)
                crossed_boundary = True

            points = np.array((r[i, 0, j], r[i, 1, j]))

            # Points es un array de todos los puntos donde estan las particulas

            b_interp_x = b_interp_func_x(points)
            b_interp_y = b_interp_func_y(points)
            b_interp_z = b_interp_func_z(points)
            b_interp = np.array((b_interp_x, b_interp_y, b_interp_z))[:, 0]

            e_interp_x = e_interp_func_x(points)
            e_interp_y = e_interp_func_y(points)
            e_interp_z = e_interp_func_z(points)
            e_interp = np.array((e_interp_x, e_interp_y, e_interp_z))[:, 0]

            a_act = -1/2 * b_interp/fact

            p0 = p[i, : ,j] - e_interp / (2*fact)

            gammap0=np.sqrt(1+np.linalg.norm(p0)**2)

            p1 = p0 + 2 / (gammap0**2 + np.linalg.norm(a_act)**2) * np.cross((gammap0* p0 + np.cross(p0, a_act)), a_act)
            p[i + 1, :, j] = p1 - e_interp / (2*fact)

            gamma[i + 1,j] = np.sqrt((1 + np.linalg.norm(p[i + 1,:, j])**2))

            v[i+1, :, j] = p[i+1 , :, j] / gamma[i+1, j] # Obtenemos v de vuelta

            # Actualizamos la posición r_i+1
            r[i + 1, :, j] = r[i, :, j] + v[i, :, j] * dt/fact # r_i+1 = r_i + v_i delta t

            # Actualizar el array auxiliar para graficar
            if crossed_boundary:
                r_plot[i, :, j] = [np.nan, np.nan, np.nan]  # Insertar NaN para romper la línea en el gráfico
            r_plot[i+1, :, j] = r[i+1, :, j]

    return r_plot,v,gamma


def varias_particulas_potencia(x,y,v_i,gamma0,r0,ex,ey,ez,bx,by,bz,NT,Nx,Ny,dt,fact=1):

    #r0 es un vector con las condiciones iniciales de todas las particulas, tiene dimensión Np

    Np=int(np.size(r0)/3) # Divido en 3 por la cant de coordenadas

    t=np.linspace(0,NT,int(NT*155*fact)) # Defino mi Delta t como se obtuvo en en analisis

    Nt=int(np.size(t))

    p = np.zeros((Nt, 3, Np))
    p[0, :, :] = v_i * gamma0  # Condición inicial de p

    v = np.zeros_like(p)
    v[0, :, :] = v_i  # Condición inicial de v

    gamma = np.zeros((Nt,Np))
    gamma[0,:]=gamma0
    r = np.zeros_like(p)

    r[0,: ,:] = r0

    P = np.zeros_like(gamma)

    b_interp_func_x = RegularGridInterpolator((x, y), bx)
    b_interp_func_y = RegularGridInterpolator((x, y), by)
    b_interp_func_z = RegularGridInterpolator((x, y), bz)

    e_interp_func_x = RegularGridInterpolator((x, y), ex)
    e_interp_func_y = RegularGridInterpolator((x, y), ey)
    e_interp_func_z = RegularGridInterpolator((x, y), ez)

    # Array auxiliar para graficar
    r_plot = np.zeros_like(p)
    r_plot[0, :, :] = r[0, : ,: ]

    for i in range(0,Np):
        point_i=np.array((r0[0,i],r0[1,i]))

        e_interp_x = e_interp_func_x(point_i)
        e_interp_y = e_interp_func_y(point_i)
        e_interp_z = e_interp_func_z(point_i)

        e_interp_in = np.array((e_interp_x, e_interp_y, e_interp_z))[:, 0]

        P[0,i]=-1*v_i[:,i]@e_interp_in

    for i in range(0, Nt - 1):
        for j in range(0,Np):
            # Condiciones de Borde periódicas
            crossed_boundary = False

            if r[i, 0, j] < 0:
                r[i, 0, j] += (Nx - 1)
                crossed_boundary = True
            elif r[i, 0, j] > Nx - 1:
                r[i, 0, j] -= (Nx - 1)
                crossed_boundary = True

            if r[i, 1, j] < 0:
                r[i, 1, j] += (Ny - 1)
                crossed_boundary = True
            elif r[i, 1, j] > Ny - 1:
                r[i, 1, j] -= (Ny - 1)
                crossed_boundary = True

            points = np.array((r[i, 0, j], r[i, 1, j]))

            # Points es un array de todos los puntos donde estan las particulas

            b_interp_x = b_interp_func_x(points)
            b_interp_y = b_interp_func_y(points)
            b_interp_z = b_interp_func_z(points)
            b_interp = np.array((b_interp_x, b_interp_y, b_interp_z))[:, 0]

            e_interp_x = e_interp_func_x(points)
            e_interp_y = e_interp_func_y(points)
            e_interp_z = e_interp_func_z(points)
            e_interp = np.array((e_interp_x, e_interp_y, e_interp_z))[:, 0]

            a_act = -1/2 * b_interp/fact

            p0 = p[i, : ,j] - e_interp / (2*fact)

            gammap0=np.sqrt(1+np.linalg.norm(p0)**2)

            p1 = p0 + 2 / (gammap0**2 + np.linalg.norm(a_act)**2) * np.cross((gammap0* p0 + np.cross(p0, a_act)), a_act)
            p[i + 1, :, j] = p1 - e_interp / (2*fact)

            gamma[i + 1,j] = np.sqrt((1 + np.linalg.norm(p[i + 1,:, j])**2))

            v[i+1, :, j] = p[i+1 , :, j] / gamma[i+1, j] # Obtenemos v de vuelta

            # Actualizamos la posición r_i+1
            r[i + 1, :, j] = r[i, :, j] + v[i, :, j] * dt/fact # r_i+1 = r_i + v_i delta t

            P[i + 1, j] = -1*np.dot(e_interp,v[i+1, :, j])# r_i+1 = r_i + v_i delta t

            # Actualizar el array auxiliar para graficar
            if crossed_boundary:
                r_plot[i, :, j] = [np.nan, np.nan, np.nan]  # Insertar NaN para romper la línea en el gráfico
            r_plot[i+1, :, j] = r[i+1, :, j]

    return r_plot,v,gamma,P



