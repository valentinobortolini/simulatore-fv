import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from scipy.special import lambertw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Simulatore FV Completo", layout="wide")

# =========================================================================
# INITIALIZE SESSION STATE PER LA CRONOLOGIA SIMULAZIONI
# =========================================================================
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.title("☀️ Modello Termo-Elettrico Modulo Fotovoltaico")
st.markdown("Modello Termo-Elettrico per il calcolo della distribuzione di temperatura in un modulo fotovoltaico, con possibilità di simulare ombreggiamenti multipli e confrontare più esecuzioni.")

# =========================================================================
# 1. FUNZIONI MATEMATICHE DI BACKGROUND
# =========================================================================

def prepara_termini_fourier(Sole, xg, yg, cellex, celley, xm, ym, n, m, Ut, kT, dx_m, dy_m):
    alfa_vals = np.arange(1, n + 1) * np.pi / xm
    beta_vals = np.arange(1, m + 1) * np.pi / ym
    l_idx = np.arange(0, len(cellex) - 1, 2)
    v_idx = np.arange(0, len(celley) - 1, 2)

    somma_x = np.array([
        np.sum(np.cos(alfa * cellex[l_idx + 1]) - np.cos(alfa * cellex[l_idx]))
        for alfa in alfa_vals
    ])
    somma_y = np.array([
        np.sum(np.cos(beta * celley[v_idx + 1]) - np.cos(beta * celley[v_idx]))
        for beta in beta_vals
    ])

    den = np.zeros((n, m))
    den2 = np.zeros((n, m))
    fattore_sol = np.zeros((n, m))

    for ii, alfa in enumerate(alfa_vals):
        sin_alfa_xg = np.sin(alfa * xg)
        for jj, beta in enumerate(beta_vals):
            matrice_spaziale = sin_alfa_xg * np.sin(beta * yg)
            den[ii, jj] = (Ut + kT * (alfa**2 + beta**2)) * alfa * beta
            den2[ii, jj] = Ut + kT * (alfa**2 + beta**2)
            fattore_sol[ii, jj] = dx_m * dy_m * np.sum(Sole * matrice_spaziale)

    return {
        "alfa": alfa_vals,
        "beta": beta_vals,
        "somma_x": somma_x,
        "somma_y": somma_y,
        "den": den,
        "den2": den2,
        "fattore_sol": fattore_sol,
    }


def calcola_bordi_cella_analitici(x_c, y_c, termini, xm, ym, Uk, h, Ta, qohm, kTT):
    x1, x2 = x_c[0], x_c[-1]
    y1, y2 = y_c[0], y_c[-1]

    tansupsol = np.zeros_like(x_c)
    taninfsol = np.zeros_like(x_c)
    tansxsol = np.zeros_like(y_c)
    tandxsol = np.zeros_like(y_c)

    tansupohm = np.zeros_like(x_c)
    taninfohm = np.zeros_like(x_c)
    tansxohm = np.zeros_like(y_c)
    tandxohm = np.zeros_like(y_c)

    tansup_rob = np.zeros_like(x_c)
    taninf_rob = np.zeros_like(x_c)
    tansx_rob = np.zeros_like(y_c)
    tandx_rob = np.zeros_like(y_c)

    costante_molt = 4 / (xm * ym)
    alfa_vals = termini["alfa"]
    beta_vals = termini["beta"]

    for ii, alfa in enumerate(alfa_vals):
        sin_alfa_x = np.sin(alfa * x_c)
        sin_alfa_x1 = np.sin(alfa * x1)
        sin_alfa_x2 = np.sin(alfa * x2)

        for jj, beta in enumerate(beta_vals):
            sin_beta_y2 = np.sin(beta * y2)
            sin_beta_y1 = np.sin(beta * y1)
            sin_beta_y = np.sin(beta * y_c)

            molt_sol = (costante_molt / termini["den2"][ii, jj]) * termini["fattore_sol"][ii, jj]
            soluzioneintegrale = termini["somma_x"][ii] * termini["somma_y"][jj]
            molt_ohm = (costante_molt / termini["den"][ii, jj]) * soluzioneintegrale * qohm

            tansupsol += molt_sol * (sin_alfa_x * sin_beta_y2)
            taninfsol += molt_sol * (sin_alfa_x * sin_beta_y1)
            tansxsol += molt_sol * (sin_alfa_x1 * sin_beta_y)
            tandxsol += molt_sol * (sin_alfa_x2 * sin_beta_y)

            tansupohm += molt_ohm * (sin_alfa_x * sin_beta_y2)
            taninfohm += molt_ohm * (sin_alfa_x * sin_beta_y1)
            tansxohm += molt_ohm * (sin_alfa_x1 * sin_beta_y)
            tandxohm += molt_ohm * (sin_alfa_x2 * sin_beta_y)

    for i in range(1, len(alfa_vals) + 1):
        if i % 2 != 0:
            lam_n = i * np.pi / ym
            mu_n = np.sqrt(lam_n**2 + Uk)
            num_n = (2 / (i * np.pi)) * (1 - (-1)**i) * h * Ta

            C1n = num_n / (kTT * mu_n * np.cosh(mu_n * xm) + h * np.sinh(mu_n * xm))
            C2n = num_n / (-kTT * mu_n * np.cosh(-mu_n * xm) + h * np.sinh(-mu_n * xm))

            fun_x = lambda x_val: C1n * np.sinh(mu_n * x_val) + C2n * np.sinh(mu_n * (x_val - xm))

            tansup_rob += fun_x(x_c) * np.sin(lam_n * y2)
            taninf_rob += fun_x(x_c) * np.sin(lam_n * y1)
            tansx_rob += fun_x(x1) * np.sin(lam_n * y_c)
            tandx_rob += fun_x(x2) * np.sin(lam_n * y_c)

    for j in range(1, len(beta_vals) + 1):
        if j % 2 != 0:
            lam_m = j * np.pi / xm
            gam_m = np.sqrt(lam_m**2 + Uk)
            num_m = (2 / (j * np.pi)) * (1 - (-1)**j) * h * Ta

            C3m = num_m / (kTT * gam_m * np.cosh(gam_m * ym) + h * np.sinh(gam_m * ym))
            C4m = num_m / (-kTT * gam_m * np.cosh(-gam_m * ym) + h * np.sinh(-gam_m * ym))

            fun_y = lambda y_val: C3m * np.sinh(gam_m * y_val) + C4m * np.sinh(gam_m * (y_val - ym))

            tansup_rob += fun_y(y2) * np.sin(lam_m * x_c)
            taninf_rob += fun_y(y1) * np.sin(lam_m * x_c)
            tansx_rob += fun_y(y_c) * np.sin(lam_m * x1)
            tandx_rob += fun_y(y_c) * np.sin(lam_m * x2)

    tansup = tansupsol + tansupohm + tansup_rob
    taninf = taninfsol + taninfohm + taninf_rob
    tansx = tansxsol + tansxohm + tansx_rob
    tandx = tandxsol + tandxohm + tandx_rob

    return tansup, taninf, tansx, tandx

@st.cache_data
def calcola_soluzione_analitica_globale(xg, yg, Sole, cellex, celley, xm, ym, n, m, Uk, Ut, h, Ta, kT, qohm, dx_m, dy_m, kTT):
    Tsol = np.zeros_like(xg)
    Tohm = np.zeros_like(xg)
    Trobin = np.zeros_like(xg)
    
    costante_molt = 4 / (xm * ym)
    l_idx = np.arange(0, len(cellex)-1, 2)
    v_idx = np.arange(0, len(celley)-1, 2)
    
    # 1. Sole e Ohm
    for i in range(1, n + 1):
        alfa = i * np.pi / xm
        somma_x = np.sum(np.cos(alfa * cellex[l_idx+1]) - np.cos(alfa * cellex[l_idx]))
        sin_alfa_xg = np.sin(alfa * xg)
        
        for j in range(1, m + 1):
            beta = j * np.pi / ym
            den = (Ut + kT * (alfa**2 + beta**2)) * alfa * beta
            den2 = Ut + kT * (alfa**2 + beta**2)
            somma_y = np.sum(np.cos(beta * celley[v_idx+1]) - np.cos(beta * celley[v_idx]))
            soluzioneintegrale = somma_x * somma_y
            matrice_spaziale = sin_alfa_xg * np.sin(beta * yg)
            
            Tohm += (costante_molt / den) * matrice_spaziale * soluzioneintegrale * qohm
            fattore = dx_m * dy_m * np.sum(Sole * matrice_spaziale)
            Tsol += (costante_molt / den2) * matrice_spaziale * fattore
            
    # 2. Robin
    for i in range(1, n + 1):
        if i % 2 != 0:
            lam_n = i * np.pi / ym
            mu_n = np.sqrt(lam_n**2 + Uk)
            num_n = (2 / (i * np.pi)) * (1 - (-1)**i) * h * Ta
            
            # Sostituito kT con kTT nei coefficienti
            C1n = num_n / (kTT * mu_n * np.cosh(mu_n * xm) + h * np.sinh(mu_n * xm))
            C2n = num_n / (-kTT * mu_n * np.cosh(-mu_n * xm) + h * np.sinh(-mu_n * xm))
            
            Trobin += (C1n * np.sinh(mu_n * xg) + C2n * np.sinh(mu_n * (xg - xm))) * np.sin(lam_n * yg)
            
    for j in range(1, m + 1):
        if j % 2 != 0:
            lam_m = j * np.pi / xm
            gam_m = np.sqrt(lam_m**2 + Uk)
            num_m = (2 / (j * np.pi)) * (1 - (-1)**j) * h * Ta
            
            # Sostituito kT con kTT nei coefficienti
            C3m = num_m / (kTT * gam_m * np.cosh(gam_m * ym) + h * np.sinh(gam_m * ym))
            C4m = num_m / (-kTT * gam_m * np.cosh(-gam_m * ym) + h * np.sinh(-gam_m * ym))
            
            Trobin += (C3m * np.sinh(gam_m * yg) + C4m * np.sinh(gam_m * (yg - ym))) * np.sin(lam_m * xg)
            
    Tan = Tsol + Tohm + Trobin + Ta - 273.15
    return Tan


def interpola_univoca(x, y, x_new, left=np.nan, right=np.nan):
    ordine = np.argsort(x)
    x_ord = np.asarray(x)[ordine]
    y_ord = np.asarray(y)[ordine]
    x_unique, idx_unique = np.unique(x_ord, return_index=True)
    y_unique = y_ord[idx_unique]

    if len(x_unique) < 2:
        return np.full_like(x_new, y_unique[0] if len(y_unique) else 0.0, dtype=float)

    return np.interp(x_new, x_unique, y_unique, left=left, right=right)


def calcola_corrente_modulo_iv(T_celle_C, irradianza, ncx, ncy, celle_oscurate, fattore_sole):
    q = 1.60218e-19
    k_b = 1.3800e-23
    T_STC = 25.0
    G_STC = 1e3
    Voc = 40.3 / 60
    alpha = 0.06
    Ns = 1

    energy_gap_STC = 1.121 * q
    coeff_Iph = 9.0
    coeff_I0 = 1.97e-9
    coeff_n = 1.49
    coeff_n_G = 1.13e-4
    coeff_n_T = -1.39e-3
    coeff_Rs = 3.667e-3
    coeff_Rs_beta = 0.07
    coeff_Rsh = 17.2

    n_points = 1000
    T_new = np.asarray(T_celle_C, dtype=float).reshape(-1)
    n_cells_total = T_new.size
    if n_cells_total == 0:
        return 0.0, np.array([]), np.array([])

    irradiance = np.ones(n_cells_total) * irradianza
    if celle_oscurate:
        for riga, colonna in celle_oscurate:
            idx = (int(riga) - 1) * int(ncx) + (int(colonna) - 1)
            if 0 <= idx < n_cells_total:
                irradiance[idx] = fattore_sole * irradianza

    stringhe = [idx for idx in np.array_split(np.arange(n_cells_total), 3) if len(idx) > 0]
    I_cells = np.zeros((n_points, n_cells_total))
    V_cells = np.zeros((n_points, n_cells_total))

    energy_gap = lambda T: energy_gap_STC * (1 - 2.677e-4 * (T - 273.15 - T_STC))
    Iph_all = coeff_Iph * (1 + alpha / 100 * (T_new - T_STC)) * irradiance / G_STC
    Iph_limit = [np.min(Iph_all[indici]) for indici in stringhe]
    V_cell = np.linspace(0, Voc * 1.1, n_points)

    for numero_stringa, indici_stringa in enumerate(stringhe):
        for cell_idx in indici_stringa:
            irr_model = max(float(irradiance[cell_idx]), 1e-9)
            temperatura = T_new[cell_idx] + 273.15
            Vt = k_b * temperatura / q

            Iph = coeff_Iph * (1 + alpha / 100 * (temperatura - 273.15 - T_STC)) * irr_model / G_STC
            Iph = min(Iph, Iph_limit[numero_stringa])
            I0 = coeff_I0 * (temperatura / (T_STC + 273.15)) ** (1 / 3) * np.exp(
                ((energy_gap_STC / (T_STC + 273.15)) - energy_gap(temperatura) / temperatura) / k_b
            )
            n = coeff_n + coeff_n_T * temperatura + coeff_n_G * irr_model
            Rs = coeff_Rs * temperatura / (T_STC + 273.15) * (1 - coeff_Rs_beta * np.log(irr_model / G_STC))
            Rh = coeff_Rsh * G_STC / irr_model

            arg = Rs * Rh * I0 / Ns**2 * np.exp(
                Rh / Ns * (Rs * Iph + Rs * I0 + V_cell) / Ns / (n * Vt * (Rs + Rh) / Ns)
            )
            W = np.real(lambertw(arg))
            I_cell = (
                -V_cell / Rs
                + n * Vt * Ns / Rs
                * (
                    -W / n / Vt / (Rs + Rh) * Ns
                    + Rh / Ns * (Rs * Iph + Rs * I0 + V_cell) / Ns / (n * Vt * (Rs + Rh) / Ns)
                )
            )

            V_cells[:, cell_idx] = V_cell
            I_cells[:, cell_idx] = np.real(I_cell)

    I_stringhe = []
    V_stringhe = []

    for indici_stringa in stringhe:
        Isc_cells = np.max(I_cells[:, indici_stringa], axis=0)
        idx_ref = indici_stringa[int(np.argmin(Isc_cells))]
        I_ref = I_cells[:, idx_ref]
        V_tot = np.zeros_like(I_ref)

        for cell_idx in indici_stringa:
            V_interp = interpola_univoca(I_cells[:, cell_idx], V_cells[:, cell_idx], I_ref)
            V_tot += np.nan_to_num(V_interp, nan=0.0)

        I_stringhe.append(I_ref)
        V_stringhe.append(V_tot)

    max_corrente = max(float(np.nanmax(I_str)) for I_str in I_stringhe)
    I_mod_clean = np.linspace(0, max_corrente, 2000)
    V_mod_clean = np.zeros_like(I_mod_clean)

    for I_str, V_str in zip(I_stringhe, V_stringhe):
        V_mod_clean += np.nan_to_num(interpola_univoca(I_str, V_str, I_mod_clean, left=0.0, right=0.0), nan=0.0)

    sort_idx = np.argsort(V_mod_clean)
    V_mod_clean = V_mod_clean[sort_idx]
    I_mod_clean = I_mod_clean[sort_idx]
    P_module = I_mod_clean * V_mod_clean
    idx_mpp = int(np.nanargmax(P_module))
    Impp_module = float(I_mod_clean[idx_mpp])

    return Impp_module, V_mod_clean, I_mod_clean


def vettore_sorgente(Nx, Ny, S, kT):
    dim = (Nx - 1) * (Ny - 1)
    qv = np.zeros(dim)
    for i in range(1, Nx):
        for j in range(1, Ny):
            k_idx = (i - 1) * (Ny - 1) + (j - 1)
            qv[k_idx] = S[i - 1, j - 1]
    return qv / kT

@st.cache_data
def esegui_simulazione_completa(p):
    xm, ym = p['xm'], p['ym']
    ncx, ncy = int(p['ncx']), int(p['ncy'])
    
    xc = (xm - 2*p['s_bord_lr'] - (ncx-1)*p['s_cell_x']) / ncx
    yc = (ym - p['s_bord_t'] - p['s_bord_b'] - (ncy-1)*p['s_cell_y']) / ncy
    Ac = xc * yc
    
    ESSE = np.array([p['sglass'], p['seva'], p['spv'], p['sted']])
    KAPPA = np.array([p['kglass'], p['keva'], p['kpv'], p['kted']])
    kT = np.sum(ESSE * KAPPA) / np.sum(ESSE)
    
    mod_thick = p['mod_thick']
    Ta = p['Ta_C'] + 273.15
    Ut = (p['U0'] + p['vento'] * p['U1']) / mod_thick
    
    Uk = Ut / kT
    
    # Rv = p['Rs'] * ncy * ncx
    Rv = p['Rs']
    QSOL = p['GSTC'] / mod_thick
    
    iniziocx = np.arange(p['s_bord_lr'], xm - xc + 1e-5, xc + p['s_cell_x'])
    finecx = iniziocx + xc
    iniziocy = np.arange(p['s_bord_b'], ym - yc + 1e-5, yc + p['s_cell_y'])
    finecy = iniziocy + yc
    cellex = np.sort(np.concatenate((iniziocx, finecx)))
    celley = np.sort(np.concatenate((iniziocy, finecy)))
    
    x_g_vec = np.linspace(0, xm, p['nx_res'] + 1)
    y_g_vec = np.linspace(0, ym, p['ny_res'] + 1)
    xg, yg = np.meshgrid(x_g_vec, y_g_vec)
    
    Nx, Ny = int(p['Nx_cella']), int(p['Ny_cella'])
    dx, dy = xc / Nx, yc / Ny
    dim = (Nx - 1) * (Ny - 1)
    
    e = np.ones(dim)
    ds = e / dy**2
    di = e / dy**2
    ds[0::Ny-1] = 0
    di[Ny-2::Ny-1] = 0
    A = sp.diags([e/dx**2, di, (-2/dx**2 - 2/dy**2 - Ut/kT)*e, ds, e/dx**2], [-(Ny-1), -1, 0, 1, (Ny-1)], shape=(dim, dim), format='csc')
    risolvi_A = spla.factorized(A)
    
    dx_m = x_g_vec[1] - x_g_vec[0]
    dy_m = y_g_vec[1] - y_g_vec[0]

    x_celle = [np.linspace(cellex[2 * c], cellex[2 * c + 1], Nx + 1) for c in range(ncx)]
    y_celle = [np.linspace(celley[2 * r], celley[2 * r + 1], Ny + 1) for r in range(ncy)]

    eta = p['eta_STC']
    err, iii = 1.0, 0
    Impr = p['Imp']
    qohm = 0.0
    Sole = QSOL * np.ones_like(xg) * (1 - eta)

    num_celle = ncx * ncy
    Tsup_prec = np.ones((num_celle, Nx + 1))
    Tinf_prec = np.ones((num_celle, Nx + 1))
    Tsx_prec = np.ones((num_celle, Ny + 1))
    Tdx_prec = np.ones((num_celle, Ny + 1))

    def errore_relativo(nuovo, vecchio):
        denom = np.maximum(np.abs(vecchio), 1e-12)
        return np.max(np.abs(vecchio - nuovo) / denom)

    while err > p['toll'] and iii < p['itmax']:
        Sole = QSOL * np.ones_like(xg) * (1 - eta)

        if p['ombra_attiva'] and p['celle_oscurate']:
            for (r_o, c_o) in p['celle_oscurate']:
                ixi = np.searchsorted(x_g_vec, iniziocx[c_o - 1])
                fxi = np.searchsorted(x_g_vec, finecx[c_o - 1])
                iyi = np.searchsorted(y_g_vec, iniziocy[r_o - 1])
                fyi = np.searchsorted(y_g_vec, finecy[r_o - 1])
                Sole[iyi:fyi, ixi:fxi] = p['fattore_sole'] * QSOL * (1 - eta)

        termini_bordi = prepara_termini_fourier(
            Sole, xg, yg, cellex, celley, xm, ym, int(p['n_bordi']), int(p['m_bordi']), Ut, kT, dx_m, dy_m
        )
        interpolator_sole = RegularGridInterpolator((y_g_vec, x_g_vec), Sole, bounds_error=False, fill_value=None)

        Tansup = np.zeros_like(Tsup_prec)
        Taninf = np.zeros_like(Tinf_prec)
        Tansx = np.zeros_like(Tsx_prec)
        Tandx = np.zeros_like(Tdx_prec)

        for i_cell in range(num_celle):
            riga, colonna = int(i_cell / ncx), i_cell % ncx
            Tansup[i_cell], Taninf[i_cell], Tansx[i_cell], Tandx[i_cell] = calcola_bordi_cella_analitici(
                x_celle[colonna], y_celle[riga], termini_bordi, xm, ym, Uk, p['h'], Ta, qohm, p['kTT']
            )

        err = max(
            errore_relativo(Tansup, Tsup_prec),
            errore_relativo(Taninf, Tinf_prec),
            errore_relativo(Tansx, Tsx_prec),
            errore_relativo(Tandx, Tdx_prec),
        )
        Tsup_prec, Tinf_prec, Tsx_prec, Tdx_prec = Tansup, Taninf, Tansx, Tandx

        Tm = np.zeros(num_celle)
        Tmedio = np.zeros(num_celle)
        for i_cell in range(num_celle):
            riga, colonna = int(i_cell / ncx), i_cell % ncx
            x_c = x_celle[colonna]
            y_c = y_celle[riga]

            TFSUP = Tansup[i_cell]
            TFINF = Taninf[i_cell]
            TFSX = Tansx[i_cell]
            TFDX = Tandx[i_cell]

            b = np.zeros(dim)
            b[Ny - 2::Ny - 1] += -(TFSUP[1:-1]) / dy**2
            b[0::Ny - 1] += -(TFINF[1:-1]) / dy**2
            b[(Nx - 2) * (Ny - 1):] += -(TFDX[1:-1]) / dx**2
            b[:Ny - 1] += -(TFSX[1:-1]) / dx**2

            X_sing, Y_sing = np.meshgrid(x_c, y_c)
            pts = np.array([Y_sing.ravel(), X_sing.ravel()]).T
            Ssol = interpolator_sole(pts).reshape(Ny + 1, Nx + 1)[1:-1, 1:-1].T

            qv = vettore_sorgente(Nx, Ny, Ssol + qohm, kT)
            T_interna = risolvi_A(b - qv).reshape((Ny - 1, Nx - 1), order='F')
            T_full = np.vstack((TFINF[1:-1], T_interna, TFSUP[1:-1]))
            T_full = np.column_stack((TFSX, T_full, TFDX))

            Tm[i_cell] = np.max(T_full)
            Tmedio[i_cell] = np.mean(T_full)

        deltaT = np.max(Tmedio) + p['Ta_C'] - 25.0
        Impr, _, _ = calcola_corrente_modulo_iv(
            Tmedio + p['Ta_C'], p['GSTC'], ncx, ncy, p['celle_oscurate'], p['fattore_sole']
        )
        eta = p['eta_STC'] * (1 + p['gamma'] * deltaT)
        qohm = Rv * Impr**2 / (Ac * mod_thick)
        iii += 1
    
    # Passaggio di kTT alla funzione
    Tan = calcola_soluzione_analitica_globale(xg, yg, Sole, cellex, celley, xm, ym, p['n_f'], p['m_f'], Uk, Ut, p['h'], Ta, kT, qohm, dx_m, dy_m, p['kTT'])
    Tmaxan = np.max(Tan)
    
    return xg, yg, Tan, Tmaxan, eta, Impr, iii, kT

# =========================================================================
# 2. SIDEBAR DI CONFIGURAZIONE (CON MENU A TENDINA)
# =========================================================================

with st.sidebar.expander("📐 Geometria Modulo", expanded=False):
    x_m = st.number_input("xm: Larghezza modulo [m]", value=0.991)
    y_m = st.number_input("ym: Altezza modulo [m]", value=1.65)
    sb_b = st.number_input("s_bord_b: Spazio bordo inferiore [m]", value=0.04, format="%.4f")
    sb_t = st.number_input("s_bord_t: Spazio bordo superiore [m]", value=0.033, format="%.4f")
    sb_lr = st.number_input("s_bord_lr: Spazio bordi laterali [m]", value=0.02, format="%.4f")
    sc_y = st.number_input("s_cell_y: Spazio verticale tra celle [m]", value=0.002, format="%.4f")
    sc_x = st.number_input("s_cell_x: Spazio orizzontale tra celle [m]", value=0.0035, format="%.4f")
    ncx = st.number_input("ncx: Celle su X", value=6, step=1)
    ncy = st.number_input("ncy: Celle su Y", value=10, step=1)

with st.sidebar.expander("🧱 Layer Stratigrafia", expanded=False):
    kglass = st.number_input("kglass: Conducibilità Vetro [W/mK]", value=1.8)
    sglass = st.number_input("sglass: Spessore Vetro [mm]", value=3.2)
    keva = st.number_input("keva: Conducibilità EVA [W/mK]", value=0.35)
    seva = st.number_input("seva: Spessore EVA [mm]", value=3.0)
    kpv = st.number_input("kpv: Conducibilità Silicio PV [W/mK]", value=148.0)
    spv = st.number_input("spv: Spessore Silicio PV [mm]", value=3.0)
    kted = st.number_input("kted: Conducibilità Tedlar [W/mK]", value=0.2)
    sted = st.number_input("sted: Spessore Tedlar [mm]", value=2.0)
    kTT = st.number_input("kTT: Conducibilità Equivalente Bordi [W/mK]", value=200.0) # NUOVO PARAMETRO

with st.sidebar.expander("🔌 Parametri Elettrici", expanded=False):
    Isc = st.number_input("Isc: Corrente cc [A]", value=9.91)
    Rs = st.number_input("Rs: Resistenza serie [Ohm]", value=3.8e-3, format="%.5f")
    eta_STC = st.number_input("eta_STC", value=0.19)
    Imp = st.number_input("Imp: [A]", value=9.91)
    alfaimp = st.number_input("alfaimp", value=0.06/100, format="%.6f")
    gamma = st.number_input("gamma", value=-0.0038, format="%.5f")

with st.sidebar.expander("🌍 Condizioni Operative", expanded=False):
    vento = st.number_input("Vento [m/s]", value=4.0)
    Ta_C = st.number_input("Ta Ambiente [°C]", value=31.0)
    GSTC = st.number_input("GSTC [W/m²]", value=1000)
    mod_thick = st.number_input("Spessore totale [m]", value=0.01, format="%.3f")
    U0 = st.number_input("U0", value=8.0)
    U1 = st.number_input("U1", value=5.0)

with st.sidebar.expander("🌫️ Ombreggiamento MULTIPLO", expanded=False):
    ombra_attiva = st.checkbox("Attiva Ombra", value=True)
    celle_oscurate = []
    
    if ombra_attiva:
        opzioni_celle = [f"R{r}-C{c}" for r in range(1, int(ncy) + 1) for c in range(1, int(ncx) + 1)]
        
        if "R1-C2" in opzioni_celle:
            default_cella = ["R1-C2"]
        elif opzioni_celle:
            default_cella = [opzioni_celle[0]]
        else:
            default_cella = []
            
        celle_selezionate = st.multiselect(
            f"Seleziona le celle da oscurare (Totale pannello: {int(ncx*ncy)}):", 
            opzioni_celle, 
            default=default_cella
        )
        fattore_sole = st.slider("Trasmissione (0.0 = buio, 1.0 = luce totale)", 0.0, 1.0, 0.05)
        
        for cella in celle_selezionate:
            parti = cella.replace("R","").split("-C")
            celle_oscurate.append((int(parti[0]), int(parti[1])))
    else:
        celle_selezionate = []
        fattore_sole = 1.0

with st.sidebar.expander("🔄 Risoluzione", expanded=False):
    n_bordi = st.number_input("n_fourier_bordi", value=25, step=1)
    m_bordi = st.number_input("m_fourier_bordi", value=25, step=1)
    n_fourier = st.number_input("n_fourier_globale", value=100, step=1)
    m_fourier = st.number_input("m_fourier_globale", value=100, step=1)
    itmax = st.number_input("Iterazioni max", value=20)
    toll = st.number_input("Tolleranza", value=1e-6, format="%.1e")
    nx_res = st.number_input("Mesh X", value=60)
    ny_res = st.number_input("Mesh Y", value=70)
    Nx_cella = st.number_input("Mesh cella X", value=80, step=1)
    Ny_cella = st.number_input("Mesh cella Y", value=90, step=1)

with st.sidebar.expander("🎨 Aspetto Grafici", expanded=False):
    tema_scuro = st.checkbox("Usa sfondo scuro per i grafici", value=False)

# =========================================================================
# 3. INTERFACCIA DI CALCOLO E PLOT
# =========================================================================

if tema_scuro:
    plt.style.use('dark_background')
else:
    plt.style.use('default')

st.markdown("---")
col_input, col_btn = st.columns([2, 1])
with col_input:
    nome_simulazione = st.text_input("📝 Assegna un nome a questa simulazione (Opzionale):", placeholder="Es. Test Ombra 50% - Vento 10 m/s")
with col_btn:
    st.write("")
    esegui = st.button("🚀 Esegui Modello Termo-Elettrico", type="primary", use_container_width=True)

if esegui:
    with st.spinner("Calcolo integrale Analitico su tutto il pannello in corso..."):
        
        p = {
            'xm': x_m, 'ym': y_m,
            's_bord_b': sb_b, 's_bord_t': sb_t, 's_bord_lr': sb_lr,
            's_cell_y': sc_y, 's_cell_x': sc_x,
            'ncx': ncx, 'ncy': ncy,
            'kglass': kglass, 'sglass': sglass,
            'keva': keva, 'seva': seva,
            'kpv': kpv, 'spv': spv,
            'kted': kted, 'sted': sted,
            'kTT': kTT,
            'Isc': Isc, 'Rs': Rs, 'eta_STC': eta_STC,
            'Imp': Imp, 'alfaimp': alfaimp, 'gamma': gamma,
            'vento': vento, 'Ta_C': Ta_C, 'GSTC': GSTC,
            'mod_thick': mod_thick, 'U0': U0, 'U1': U1,
            'ombra_attiva': ombra_attiva,
            'celle_oscurate': celle_oscurate,
            'fattore_sole': fattore_sole,
            'n_bordi': n_bordi, 'm_bordi': m_bordi,
            'n_f': n_fourier, 'm_f': m_fourier,
            'itmax': itmax, 'toll': toll,
            'Nx_cella': Nx_cella, 'Ny_cella': Ny_cella,
            'nx_res': nx_res, 'ny_res': ny_res,
        }
        p['h'] = 2.8 + 3 * vento
        
        xg, yg, Tan, Tmaxan, eta, Impr, iterazioni, kT_calc = esegui_simulazione_completa(p)
        
        geo_plot = {'x_m': x_m, 'y_m': y_m, 'sb_lr': sb_lr, 'sc_x': sc_x, 'ncx': ncx, 'sb_t': sb_t, 'sb_b': sb_b, 'sc_y': sc_y, 'ncy': ncy}
        str_celle_ombra = ", ".join(celle_selezionate) if ombra_attiva and celle_selezionate else "Nessuna"
        run_id = nome_simulazione.strip() if nome_simulazione.strip() else f"Run #{len(st.session_state['history']) + 1}"
        
        risultato_corrente = {
            "ID": run_id,
            "Tmax [°C]": round(Tmaxan, 2),
            "Efficienza [%]": round(eta * 100, 2),
            "Corrente [A]": round(Impr, 2),
            "Vento [m/s]": vento,
            "Celle d'Ombra": str_celle_ombra,
            
            "xg": xg, "yg": yg, "Tan": Tan, "Tmaxan_full": Tmaxan,
            "eta": eta, "Impr": Impr, "eta_STC": eta_STC, "Imp": Imp,
            "geo": geo_plot, "kT_calc": kT_calc, "iter": iterazioni
        }
        
        st.session_state['history'].append(risultato_corrente)
        
        if len(st.session_state['history']) > 5:
            st.session_state['history'].pop(0)
            
        st.rerun()

if st.session_state['history']:
    st.markdown("---")
    st.markdown("### 📋 Tabella Riassuntiva Simulazioni")
    
    tabella_visiva = [{k: v for k, v in run.items() if k not in ['xg', 'yg', 'Tan', 'Tmaxan_full', 'eta', 'Impr', 'eta_STC', 'Imp', 'geo', 'kT_calc', 'iter']} for run in st.session_state['history']]
    st.dataframe(tabella_visiva, use_container_width=True)
    
    col_btn_clear, col_vuota = st.columns([1, 5])
    with col_btn_clear:
        if st.button("🗑️ Cancella Cronologia", use_container_width=True):
            st.session_state['history'] = []
            st.rerun()

    st.markdown("### 🔍 Confronto Dettagliato (Split-Screen)")
    
    nomi_run = [run["ID"] for run in st.session_state['history']]
    
    col_sel_sx, col_sel_dx = st.columns(2)
    with col_sel_sx:
        idx_sx = max(0, len(nomi_run) - 2)
        run_sx_nome = st.selectbox("⬅️ Simulazione SINISTRA:", nomi_run, index=idx_sx, key="sel_sx")
    with col_sel_dx:
        idx_dx = len(nomi_run) - 1
        run_dx_nome = st.selectbox("➡️ Simulazione DESTRA:", nomi_run, index=idx_dx, key="sel_dx")

    run_sx_dati = next(run for run in st.session_state['history'] if run["ID"] == run_sx_nome)
    run_dx_dati = next(run for run in st.session_state['history'] if run["ID"] == run_dx_nome)

    def disegna_colonna_grafici(dati_plot):
        st.success(f"Dettagli: **{dati_plot['ID']}** | {dati_plot['iter']} iterazioni | kT = {dati_plot['kT_calc']:.4f}")
        
        xg_p, yg_p, Tan_p = dati_plot['xg'], dati_plot['yg'], dati_plot['Tan']
        gp = dati_plot['geo']
        
        st.markdown("#### Mappa Analitica (pcolor)")
        fig1, ax1 = plt.subplots(figsize=(5, 6))
        c = ax1.pcolormesh(xg_p, yg_p, Tan_p, shading='gouraud', cmap='viridis')
        fig1.colorbar(c, ax=ax1, label="Temperatura [°C]")
        
        xc_f = (gp['x_m'] - 2*gp['sb_lr'] - (gp['ncx']-1)*gp['sc_x']) / gp['ncx']
        yc_f = (gp['y_m'] - gp['sb_t'] - gp['sb_b'] - (gp['ncy']-1)*gp['sc_y']) / gp['ncy']
        iniziocx_f = np.arange(gp['sb_lr'], gp['x_m'] - xc_f + 1e-5, xc_f + gp['sc_x'])
        finecx_f = iniziocx_f + xc_f
        iniziocy_f = np.arange(gp['sb_b'], gp['y_m'] - yc_f + 1e-5, yc_f + gp['sc_y'])
        finecy_f = iniziocy_f + yc_f
        cellex_f = np.sort(np.concatenate((iniziocx_f, finecx_f)))
        celley_f = np.sort(np.concatenate((iniziocy_f, finecy_f)))
        
        ax1.hlines(celley_f, 0, gp['x_m'], colors='k', linewidth=1.5)
        ax1.vlines(cellex_f, 0, gp['y_m'], colors='k', linewidth=1.5)
        ax1.set_xlabel('$x_c$')
        ax1.set_ylabel('$y_c$')
        ax1.set_xlim([0, gp['x_m']])
        ax1.set_ylim([0, gp['y_m']])
        st.pyplot(fig1)
        
        st.markdown("#### Superficie Analitica (surf)")
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        surf = ax2.plot_surface(xg_p, yg_p, Tan_p, cmap='viridis', edgecolor='none', antialiased=True)
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_zlabel('Temperatura [°C]')
        fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
        st.pyplot(fig2)
        
        st.markdown("#### Impatto Elettrico")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            ax3.bar(['Ideale', 'Reale'], [dati_plot['eta_STC'] * 100, dati_plot['eta'] * 100], color=['#0072BD', '#D95319'])
            ax3.set_ylabel("Efficienza [%]")
            st.pyplot(fig3)
        with col_b2:
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            ax4.bar(['Ideale', 'Reale'], [dati_plot['Imp'], dati_plot['Impr']], color=['#0072BD', '#EDB120'])
            ax4.set_ylabel("Corrente [A]")
            st.pyplot(fig4)
            
        st.markdown("#### 📊 Dati Convettivi (Robin)")
        T_bordo_inf = np.mean(Tan_p[0, :])
        T_bordo_sup = np.mean(Tan_p[-1, :])
        T_bordo_sx  = np.mean(Tan_p[:, 0])
        T_bordo_dx  = np.mean(Tan_p[:, -1])
        T_media_perimetro = np.mean([T_bordo_inf, T_bordo_sup, T_bordo_sx, T_bordo_dx])
        delta_centro_bordi = dati_plot['Tmaxan_full'] - T_media_perimetro
        
        st.info(f"**Temperatura Massima:** {dati_plot['Tmaxan_full']:.2f} °C")
        st.warning(f"**Delta Termico (Centro - Bordi):** {delta_centro_bordi:.2f} °C")
        st.write(f"**T. Bordo Sup:** {T_bordo_sup:.2f} °C | **T. Bordo Inf:** {T_bordo_inf:.2f} °C")
        st.write(f"**T. Bordo Sx:** {T_bordo_sx:.2f} °C | **T. Bordo Dx:** {T_bordo_dx:.2f} °C")

    col_vis_sx, col_vis_dx = st.columns(2)
    
    with col_vis_sx:
        disegna_colonna_grafici(run_sx_dati)
        
    with col_vis_dx:
        disegna_colonna_grafici(run_dx_dati)
