import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Calculateur VC", layout="wide")
st.title("Calculateur de Vitesse Critique")

# --------------------------------------------------------
# SECTION : Entrée des performances (2 à 6 tests)
# --------------------------------------------------------
st.header("Entrez vos performances terrain (2 à 6 tests)")

performances = []

# Deux tests minimum
for i in range(1, 3):
    st.subheader(f"Test {i}")
    D = st.number_input(f"Distance {i} (m)", min_value=1.0, step=10.0, key=f"D{i}")
    T = st.number_input(f"Temps {i} (s)", min_value=1.0, step=10.0, key=f"T{i}")
    performances.append((D, T))

# Possibilité d'ajouter jusqu'à 4 tests supplémentaires
st.subheader("Tests supplémentaires")
nb_extra = st.number_input("Nombre de tests en plus (0 à 4)", min_value=0, max_value=4, step=1)

for i in range(3, 3 + nb_extra):
    st.subheader(f"Test {i}")
    D = st.number_input(f"Distance {i} (m)", min_value=1.0, step=10.0, key=f"D{i}")
    T = st.number_input(f"Temps {i} (s)", min_value=1.0, step=10.0, key=f"T{i}")
    performances.append((D, T))

# --------------------------------------------------------
# BOUTON
# --------------------------------------------------------
if st.button("Calculer VC"):

    # --------------------------------------------------------
    # Vérification des données
    # --------------------------------------------------------
    if len(performances) < 2:
        st.error("Veuillez entrer au moins deux tests.")
        st.stop()

    # --------------------------------------------------------
    # EXTRACTION des deux points pour VC / D′
    # --------------------------------------------------------
    D1, T1 = performances[0]
    D2, T2 = performances[1]

    if T1 == T2:
        st.error("Les deux premiers temps doivent être différents.")
        st.stop()

    # Calcul VC / D′
    CS = (D2 - D1) / (T2 - T1)
    D_prime = D1 - CS * T1
    V_kmh = CS * 3.6

    if V_kmh <= 0 or not math.isfinite(V_kmh):
        st.error("Vitesse critique non calculable (données invalides).")
        st.stop()

    # --------------------------------------------------------
    # Conversion des performances en liste pour le modèle log
    # --------------------------------------------------------
    # V = vitesse moyenne (m/s), T = temps (s)
    V_list = []
    T_list = []
    for D, T in performances:
        if D > 0 and T > 0:
            V = D / T
            V_list.append(V)
            T_list.append(T)

    if len(V_list) < 2:
        st.error("Il faut au moins deux tests valides pour le modèle log.")
        st.stop()

    # --------------------------------------------------------
    # RÉGRESSION LOG-LOG
    # T = A * V^{-k}
    # ln(T) = ln(A) - k ln(V)
    # --------------------------------------------------------
    X = np.log(1 / np.array(V_list))
    Y = np.log(np.array(T_list))

    k, lnA = np.polyfit(X, Y, 1)
    A = np.exp(lnA)

    # --------------------------------------------------------
    # AFFICHAGE DES RÉSULTATS VC
    # --------------------------------------------------------
    min_per_km = 60.0 / V_kmh
    total_seconds = int(round(min_per_km * 60.0))
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    st.subheader("Résultats VC / D′")
    st.write(f"Vitesse critique = {CS:.2f} m/s")
    st.write(f"Vitesse critique = {V_kmh:.2f} km/h")
    st.write(f"Vitesse critique = {minutes}:{seconds:02d} min/km  ({min_per_km:.2f} min/km)")
    st.write(f"Capacité anaérobie (D′) = {D_prime:.2f} m")

    # --------------------------------------------------------
    # TABLEAU DES ALLURES (80% → 130%)
    # --------------------------------------------------------
    st.subheader("Temps limites et allures (modèle log & D′)")

    pourcentages = list(range(80, 100, 2)) + list(range(102, 132, 2))
    rows = []

    for p in pourcentages:
        v = V_kmh * (p / 100.0)  # km/h
        v_ms = v / 3.6

        # -------------------------
        # Zone <100% → modèle log
        # -------------------------
        if p < 100:
            Tlim = A * (v_ms ** (-k))

        # -------------------------
        # Zone >100% → modèle D′
        # -------------------------
        else:
            denom = v_ms - CS
            if denom <= 0:
                continue
            Tlim = D_prime / denom

        if Tlim > 0 and math.isfinite(Tlim):
            # Format mm:ss
            m = int(Tlim // 60)
            s = int(Tlim % 60)
            T_format = f"{m}:{s:02d}"

            # Allure min/km
            pace_min = 60 / v
            total_pace_sec = int(round(pace_min * 60))
            pm = total_pace_sec // 60
            ps = total_pace_sec % 60
            pace_format = f"{pm}:{ps:02d}"

            rows.append({
                "% VC": f"{p}%",
                "Temps limite (mm:ss)": T_format,
                "Allure (min/km)": pace_format,
                "Modèle": "Log" if p < 100 else "D′"
            })

    df = pd.DataFrame(rows)
    st.table(df)
