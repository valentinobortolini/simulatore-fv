import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Forza l'estetica scura per i grafici
plt.style.use('dark_background')
st.set_page_config(page_title="Simulatore FV Completo", layout="wide")

st.title("☀️ Modello Termo-Elettrico Modulo Fotovoltaico")
st.markdown("Simulatore avanzato fedele al modello MATLAB originale con risoluzione integrale di Fourier e condizioni di Robin.")

# =========================================================================
# 1. FUNZIONI MATEMATICHE DI BACKGROUND
# =========================================================================

@st.cache_data
def calcola_somme_fourier(x_g_vec, y_g_vec, cellex, celley, xm, ym, n, m, Uk, Ut, h, Ta, kT, qohm):
    Tansup = np.zeros_like(x_g_vec)
    Taninf = np.zeros_like(x_g_vec)
    Tansx = np.zeros_like(y_g_vec)
    Tandx = np.zeros_like(y_g_vec)
    
    costante_molt = 4 / (xm * ym)
    l_idx = np.arange(0, len(cellex)-1, 2)
    v_idx = np.arange(0, len(celley)-1, 2)
    
    # 1. Contributi Robin Y
    for i in range(1, n + 1):
        if i % 2 != 0:
            lam_n = i * np.pi / ym
            mu_n = np.sqrt(lam_n**2 + Uk)
            num_n = (2 / (i * np.pi)) * (1 - (-1)**i) * h * Ta
            C1n = num_n / (kT * mu_n * np.cosh(mu_n * xm) + h * np.sinh(mu_n * xm))
            C2n = num_n / (-kT * mu_n * np.cosh(-mu_n * xm) + h * np.sinh(-mu_n * xm))
            fun_x = lambda x_val: C1n * np.sinh(mu_n * x_val) + C2n * np.sinh(mu_n * (x_val - xm))
            Tansx += fun_x(0) * np.sin(lam_n * y_g_vec)
            Tandx += fun_x(xm) * np.sin(lam_n * y_g_vec)
            Tansup += fun_x(x_g_vec) * np.sin(lam_n * ym)
            Taninf += fun_x(x_g_vec) * np.sin(lam_n * 0)

    # 2. Contributi Robin X
    for j in range(1, m + 1):
        if j % 2 != 0:
            lam_m = j * np.pi / xm
            gam_m = np.sqrt(lam_m**2 + Uk)
            num_m = (2 / (j * np.pi)) * (1 - (-1)**j) * h * Ta
            C3m = num_m / (kT * gam_m * np.cosh(gam_m * ym) + h * np.sinh(gam_m * ym))
            C4m = num_m / (-kT * gam_m * np.cosh(-gam_m * ym) + h * np.sinh(-gam_m * ym))
            fun_y = lambda y_val: C3m * np.sinh(gam_m * y_val) + C4m * np.sinh(gam_m * (y_val - ym))
            Tansup += fun_y(ym) * np.sin(lam_m * x_g_vec)
            Taninf += fun_y(0) * np.sin(lam_m * x_g_vec)
            Tansx += fun_y(y_g_vec) * np.sin(lam_m * 0)
            Tandx += fun_y(y_g_vec) * np.sin(lam_m * xm)
            
    # 3. Contributo Ohmico
    for i in range(1, n + 1):
        alfa = i * np.pi / xm
        somma_x = np.sum(np.cos(alfa * cellex[l_idx+1]) - np.cos(alfa * cellex[l_idx]))
        for j in range(1, m + 1):
            beta = j * np.pi / ym
            somma_y = np.sum(np.cos(beta * celley[v_idx+1]) - np.cos(beta * celley[v_idx]))
            soluzioneintegrale = somma_x * somma_y
            den = (Ut + kT * (alfa**2 + beta**2)) * alfa * beta
            molt_ohm = (costante_molt / den) * soluzioneintegrale * qohm
            
            Tansup += molt_ohm * (np.sin(alfa * x_g_vec) * np.sin(beta * ym))
            Taninf += molt_ohm * (np.sin(alfa * x_g_vec) * np.sin(beta * 0))
            Tansx += molt_ohm * (np.sin(alfa * 0) * np.sin(beta * y_g_vec))
            Tandx += molt_ohm * (np.sin(alfa * xm) * np.sin(beta * y_g_vec))
            
    return Tansup, Taninf, Tansx, Tandx

@st.cache_data
def calcola_soluzione_analitica_globale(xg, yg, Sole, cellex, celley, xm, ym, n, m, Uk, Ut, h, Ta, kT, qohm, dx_m, dy_m):
    # Traduzione della funzione MATLAB "soluzione_analitica_globale"
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
            C1n = num_n / (kT * mu_n * np.cosh(mu_n * xm) + h * np.sinh(mu_n * xm))
            C2n = num_n / (-kT * mu_n * np.cosh(-mu_n * xm) + h * np.sinh(-mu_n * xm))
            Trobin += (C1n * np.sinh(mu_n * xg) + C2n * np.sinh(mu_n * (xg - xm))) * np.sin(lam_n * yg)
            
    for j in range(1, m + 1):
        if j % 2 != 0:
            lam_m = j * np.pi / xm
            gam_m = np.sqrt(lam_m**2 + Uk)
            num_m = (2 / (j * np.pi)) * (1 - (-1)**j) * h * Ta
            C3m = num_m / (kT * gam_m * np.cosh(gam_m * ym) + h * np.sinh(gam_m * ym))
            C4m = num_m / (-kT * gam_m * np.cosh(-gam_m * ym) + h * np.sinh(-gam_m * ym))
            Trobin += (C3m * np.sinh(gam_m * yg) + C4m * np.sinh(gam_m * (yg - ym))) * np.sin(lam_m * xg)
            
    Tan = Tsol + Tohm + Trobin + Ta - 273.15
    return Tan

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
    ncx, ncy = p['ncx'], p['ncy']
    
    xc = (xm - 2*p['s_bord_lr'] - (ncx-1)*p['s_cell_x']) / ncx
    yc = (ym - p['s_bord_t'] - p['s_bord_b'] - (ncy-1)*p['s_cell_y']) / ncy
    Ac = xc * yc
    
    ESSE = np.array([p['sglass'], p['seva'], p['spv'], p['sted']])
    KAPPA = np.array([p['kglass'], p['keva'], p['kpv'], p['kted']])
    kT = np.sum(ESSE * KAPPA) / np.sum(ESSE)
    
    mod_thick = p['mod_thick']
    Ta = p['Ta_C'] + 273.15
    Ut = (p['U0'] + p['vento'] * p['U1']) / mod_thick
    Uk = Ut / 200.0
    
    Rv = p['Rs'] * ncy * ncx
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
    
    Nx, Ny = p['Nx_cella'], p['Ny_cella']
    dx, dy = xc / Nx, yc / Ny
    dim = (Nx - 1) * (Ny - 1)
    
    e = np.ones(dim)
    ds = e / dy**2
    di = e / dy**2
    ds[0::Ny-1] = 0
    di[Ny-2::Ny-1] = 0
    A = sp.diags([e/dx**2, di, (-2/dx**2 - 2/dy**2 - Ut/kT)*e, ds, e/dx**2], [-(Ny-1), -1, 0, 1, (Ny-1)], shape=(dim, dim), format='csc')
    
    eta = p['eta_STC']
    err, iii = 1, 0
    qohm = 0
    
    while err > p['toll'] and iii < p['itmax']:
        Sole = QSOL * np.ones_like(xg) * (1 - eta)
        if p['ombra_attiva']:
            ixi, fxi = np.searchsorted(x_g_vec, iniziocx[p['col_ombra']-1]), np.searchsorted(x_g_vec, finecx[p['col_ombra']-1])
            iyi, fyi = np.searchsorted(y_g_vec, iniziocy[p['riga_ombra']-1]), np.searchsorted(y_g_vec, finecy[p['riga_ombra']-1])
            Sole[iyi:fyi, ixi:fxi] = p['fattore_sole'] * QSOL * (1 - eta)
            
        interpolator_sole = RegularGridInterpolator((y_g_vec, x_g_vec), Sole, bounds_error=False, fill_value=None)
        
        Tansup_g, Taninf_g, Tansx_g, Tandx_g = calcola_somme_fourier(
            x_g_vec, y_g_vec, cellex, celley, xm, ym, p['n_f'], p['m_f'], Uk, Ut, p['h'], Ta, kT, qohm)
        
        Tm_max_iter = 0
        for i_cell in range(ncx * ncy):
            riga, colonna = int(i_cell / ncx), i_cell % ncx
            x_c = np.linspace(cellex[2*colonna], cellex[2*colonna+1], Nx+1)
            y_c = np.linspace(celley[2*riga], celley[2*riga+1], Ny+1)
            
            TFSUP = np.interp(x_c, x_g_vec, Tansup_g)
            TFINF = np.interp(x_c, x_g_vec, Taninf_g)
            TFSX  = np.interp(y_c, y_g_vec, Tansx_g)
            TFDX  = np.interp(y_c, y_g_vec, Tandx_g)
            
            b = np.zeros(dim)
            b[Ny-2::Ny-1] += -(TFSUP[1:-1]) / dy**2
            b[0::Ny-1] += -(TFINF[1:-1]) / dy**2
            b[(Nx-2)*(Ny-1):] += -(TFDX[1:-1]) / dx**2
            b[:Ny-1] += -(TFSX[1:-1]) / dx**2
            
            X_sing, Y_sing = np.meshgrid(x_c, y_c)
            pts = np.array([Y_sing.ravel(), X_sing.ravel()]).T
            Ssol = interpolator_sole(pts).reshape(Ny+1, Nx+1)[1:-1, 1:-1]
            
            qv = vettore_sorgente(Nx, Ny, Ssol + qohm, kT)
            T_interna = spla.spsolve(A, (b - qv)).reshape((Ny-1, Nx-1))
            Tm_max_iter = max(Tm_max_iter, np.max(T_interna) + p['Ta_C'])
            
        deltaT = Tm_max_iter - 25.0
        eta_new = p['eta_STC'] * (1 + p['gamma'] * deltaT)
        Impr = p['Imp'] * (1 + p['alfaimp'] * deltaT)
        qohm = Rv * Impr**2 / (Ac * mod_thick)
        err = abs(eta_new - eta)
        eta = eta_new
        iii += 1
        
    # --- CALCOLO FINALE ANALITICO SU TUTTA LA GRIGLIA COME IN MATLAB ---
    dx_m = x_g_vec[1] - x_g_vec[0]
    dy_m = y_g_vec[1] - y_g_vec[0]
    Tan = calcola_soluzione_analitica_globale(xg, yg, Sole, cellex, celley, xm, ym, p['n_f'], p['m_f'], Uk, Ut, p['h'], Ta, kT, qohm, dx_m, dy_m)
    
    Tmaxan = np.max(Tan)
    
    return xg, yg, Tan, Tmaxan, eta, Impr, iii, kT

# =========================================================================
# 2. SIDEBAR DI CONFIGURAZIONE COMPLETA (DATI_N)
# =========================================================================
with st.sidebar.expander("📐 Geometria Modulo", expanded=True):
    x_m = st.number_input("xm: Larghezza modulo [m]", value=0.991)
    y_m = st.number_input("ym: Altezza modulo [m]", value=1.65)
    sb_b = st.number_input("s_bord_b: Spazio bordo inferiore [m]", value=0.04, format="%.4f")
    sb_t = st.number_input("s_bord_t: Spazio bordo superiore [m]", value=0.033, format="%.4f")
    sb_lr = st.number_input("s_bord_lr: Spazio bordi laterali [m]", value=0.02, format="%.4f")
    sc_y = st.number_input("s_cell_y: Spazio verticale tra celle [m]", value=0.002, format="%.4f")
    sc_x = st.number_input("s_cell_x: Spazio orizzontale tra celle [m]", value=0.0035, format="%.4f")
    ncx = st.number_input("ncx: Celle su X", value=6, step=1)
    ncy = st.number_input("ncy: Celle su Y", value=10, step=1)

# Le altre sezioni sono chiuse all'avvio (expanded=False)
with st.sidebar.expander("🧱 Layer Stratigrafia", expanded=False):
    kglass = st.number_input("kglass: Conducibilità Vetro [W/mK]", value=1.8)
    sglass = st.number_input("sglass: Spessore Vetro [mm]", value=3.2)
    keva = st.number_input("keva: Conducibilità EVA [W/mK]", value=0.35)
    seva = st.number_input("seva: Spessore EVA [mm]", value=3.0)
    kpv = st.number_input("kpv: Conducibilità Silicio PV [W/mK]", value=148.0)
    spv = st.number_input("spv: Spessore Silicio PV [mm]", value=3.0)
    kted = st.number_input("kted: Conducibilità Tedlar [W/mK]", value=0.2)
    sted = st.number_input("sted: Spessore Tedlar [mm]", value=2.0)

with st.sidebar.expander("🔌 Parametri Elettrici", expanded=False):
    Isc = st.number_input("Isc: Corrente cc [A]", value=11.05)
    Rs = st.number_input("Rs: Resistenza serie [Ohm]", value=3.8e-3, format="%.5f")
    eta_STC = st.number_input("eta_STC", value=0.19)
    Imp = st.number_input("Imp: [A]", value=11.05)
    alfaimp = st.number_input("alfaimp", value=-0.03/100, format="%.6f")
    gamma = st.number_input("gamma", value=-0.005, format="%.5f")

with st.sidebar.expander("🌍 Condizioni Operative", expanded=False):
    vento = st.number_input("Vento [m/s]", value=5.0)
    Ta_C = st.number_input("Ta Ambiente [°C]", value=31.0)
    GSTC = st.number_input("GSTC [W/m²]", value=1000)
    mod_thick = st.number_input("Spessore totale [m]", value=0.01, format="%.3f")
    U0 = st.number_input("U0", value=33.0)
    U1 = st.number_input("U1", value=7.0)

with st.sidebar.expander("🌫️ Ombreggiamento", expanded=False):
    ombra_attiva = st.checkbox("Attiva Ombra", value=True)
    if ombra_attiva:
        riga_ombra = st.number_input(f"Riga (1-{ncy})", 1, ncy, 1)
        col_ombra = st.number_input(f"Colonna (1-{ncx})", 1, ncx, 2)
        fattore_sole = st.slider("Trasmissione", 0.0, 1.0, 0.05)
    else:
        riga_ombra, col_ombra, fattore_sole = 1, 1, 1.0

with st.sidebar.expander("🔄 Risoluzione", expanded=False):
    n_fourier = st.number_input("n_fourier", value=100)
    m_fourier = st.number_input("m_fourier", value=100)
    itmax = st.number_input("Iterazioni max", value=20)
    toll = st.number_input("Tolleranza", value=1e-6, format="%.1e")
    nx_res = st.number_input("Mesh X", value=80)
    ny_res = st.number_input("Mesh Y", value=70)

# =========================================================================
# 3. INTERFACCIA DI CALCOLO E PLOT
# =========================================================================
if st.button("🚀 Esegui Modello Termo-Elettrico", type="primary"):
    with st.spinner("Calcolo integrale Analitico su tutto il pannello in corso..."):
        p = {'xm':x_m,'ym':y_m,'s_bord_b':sb_b,'s_bord_t':sb_t,'s_bord_lr':sb_lr,'s_cell_y':sc_y,'s_cell_x':sc_x,'ncx':ncx,'ncy':ncy,'kglass':kglass,'sglass':sglass,'keva':keva,'seva':seva,'kpv':kpv,'spv':spv,'kted':kted,'sted':sted,'Isc':Isc,'Rs':Rs,'eta_STC':eta_STC,'Imp':Imp,'alfaimp':alfaimp,'gamma':gamma,'vento':vento,'Ta_C':Ta_C,'GSTC':GSTC,'mod_thick':mod_thick,'U0':U0,'U1':U1,'ombra_attiva':ombra_attiva,'riga_ombra':riga_ombra,'col_ombra':col_ombra,'fattore_sole':fattore_sole,'n_f':n_fourier,'m_f':m_fourier,'itmax':itmax,'toll':toll,'Nx_cella':20,'Ny_cella':20,'nx_res':nx_res,'ny_res':ny_res}
        p['h'] = 2.8 + 3 * vento
        
        xg, yg, Tan, Tmaxan, eta, Impr, iterazioni, kT_calc = esegui_simulazione_completa(p)
        
        st.success(f"Convergenza in {iterazioni} iterazioni. kT calcolato = {kT_calc:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mappa Analitica (pcolor)")
            fig1, ax1 = plt.subplots(figsize=(6, 7))
            c = ax1.pcolormesh(xg, yg, Tan, shading='gouraud', cmap='turbo')
            fig1.colorbar(c, ax=ax1, label="Temperatura [°C]")
            
            xc_f = (x_m - 2*sb_lr - (ncx-1)*sc_x) / ncx
            yc_f = (y_m - sb_t - sb_b - (ncy-1)*sc_y) / ncy
            iniziocx_f = np.arange(sb_lr, x_m - xc_f + 1e-5, xc_f + sc_x)
            finecx_f = iniziocx_f + xc_f
            iniziocy_f = np.arange(sb_b, y_m - yc_f + 1e-5, yc_f + sc_y)
            finecy_f = iniziocy_f + yc_f
            cellex_f = np.sort(np.concatenate((iniziocx_f, finecx_f)))
            celley_f = np.sort(np.concatenate((iniziocy_f, finecy_f)))
            
            ax1.hlines(celley_f, 0, x_m, colors='k', linewidth=1.5)
            ax1.vlines(cellex_f, 0, y_m, colors='k', linewidth=1.5)
            ax1.set_xlabel('$x_c$')
            ax1.set_ylabel('$y_c$')
            ax1.set_xlim([0, x_m])
            ax1.set_ylim([0, y_m])
            st.pyplot(fig1)
            
        with col2:
            st.subheader("Superficie Analitica (surf)")
            fig2 = plt.figure(figsize=(7, 7))
            ax2 = fig2.add_subplot(111, projection='3d')
            surf = ax2.plot_surface(xg, yg, Tan, cmap='turbo', edgecolor='none', antialiased=True)
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            ax2.set_zlabel('Temperatura [°C]')
            fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
            st.pyplot(fig2)
            
        st.subheader("Impatto Elettrico")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.bar(['Ideale', 'Reale'], [eta_STC * 100, eta * 100], color=['#0072BD', '#D95319'])
            ax3.set_ylabel("Efficienza [%]")
            st.pyplot(fig3)
        with col_b2:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.bar(['Ideale', 'Reale'], [Imp, Impr], color=['#0072BD', '#EDB120'])
            ax4.set_ylabel("Corrente [A]")
            st.pyplot(fig4)
            
        st.markdown("---")
        st.subheader("📊 Analisi Effetto Convettivo di Bordo (Robin)")
        
        # Le temperature dei bordi vengono calcolate direttamente da Tan, esattamente come su MATLAB
        T_bordo_inf = np.mean(Tan[0, :])
        T_bordo_sup = np.mean(Tan[-1, :])
        T_bordo_sx  = np.mean(Tan[:, 0])
        T_bordo_dx  = np.mean(Tan[:, -1])
        T_media_perimetro = np.mean([T_bordo_inf, T_bordo_sup, T_bordo_sx, T_bordo_dx])
        delta_centro_bordi = Tmaxan - T_media_perimetro
        
        c_disp1, c_disp2 = st.columns(2)
        with c_disp1:
            st.info(f"**Temperatura Massima Analitica:** {Tmaxan:.2f} °C")
            st.metric("Temperatura Media Bordo Inferiore", f"{T_bordo_inf:.2f} °C")
            st.metric("Temperatura Media Bordo Superiore", f"{T_bordo_sup:.2f} °C")
        with c_disp2:
            st.warning(f"**Delta Termico (Tmax - Tperimetro):** {delta_centro_bordi:.2f} °C")
            st.metric("Temperatura Media Bordo Sinistro", f"{T_bordo_sx:.2f} °C")
            st.metric("Temperatura Media Bordo Destro", f"{T_bordo_dx:.2f} °C")