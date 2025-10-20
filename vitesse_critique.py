import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Calculateur VC", layout="wide")
st.title("Calculateur de Vitesse Critique")

# Entrées utilisateur
st.header("Entrez vos performances")
D1 = st.number_input("Distance 1 (m)", value=1460.0, min_value=1.0, step=10.0)
T1 = st.number_input("Temps 1 (s)", value=360.0, min_value=1.0, step=10.0)
D2 = st.number_input("Distance 2 (m)", value=2690.0, min_value=1.0, step=10.0)
T2 = st.number_input("Temps 2 (s)", value=720.0, min_value=1.0, step=10.0)

if st.button("Calculer VC"):
    if T2 == T1:
        st.error("Erreur : Temps 1 et Temps 2 sont identiques (division par zéro).")
    elif D1 <= 0 or D2 <= 0 or T1 <= 0 or T2 <= 0:
        st.error("Les distances et temps doivent être strictement positifs.")
    else:
        CS = (D2 - D1) / (T2 - T1)
        D_prime = D1 - CS * T1
        V_kmh = CS * 3.6

        if V_kmh <= 0 or not math.isfinite(V_kmh):
            st.error("Vitesse critique non calculable (vérifiez les données).")
        else:
            min_per_km = 60.0 / V_kmh
            total_seconds = int(round(min_per_km * 60.0))
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            st.subheader("Résultats")
            st.write(f"Vitesse critique = {CS:.2f} m/s")
            st.write(f"Vitesse critique = {V_kmh:.2f} km/h")
            st.write(f"Vitesse critique = {minutes}:{seconds:02d} min/km  ({min_per_km:.2f} min/km)")
            st.write(f"Capacité anaérobie (D′) = {D_prime:.2f} m")

            T_max = max(T1, T2) * 2.0
            T_plot = np.linspace(max(5.0, T_max * 0.01), T_max, 500)
            D_plot = CS * T_plot + D_prime * (1.0 - np.exp(-T_plot / 500.0))
            V_plot = np.gradient(D_plot, T_plot) * 3.6

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(T_plot / 60.0, D_plot / 1000.0, label="Distance (km)", color="tab:blue")
            ax1.scatter([T1 / 60.0, T2 / 60.0], [D1 / 1000.0, D2 / 1000.0],
                        color="tab:green", label="Performances", zorder=5, s=40)
            ax1.set_xlabel("Temps (min)")
            ax1.set_ylabel("Distance (km)")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(T_plot / 60.0, V_plot, label="Vitesse (km/h)", color="tab:orange")
            ax2.axhline(y=V_kmh, color="red", linestyle="--", label=f"VC = {V_kmh:.2f} km/h")
            ax2.set_ylabel("Vitesse (km/h)")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")

            st.pyplot(fig)

            # Tableau des intensités
            pourcentages = list(range(102, 132, 2))
            rows = []
            for p in pourcentages:
                vitesse = V_kmh * (p / 100.0)
                temps_limite = D_prime / ((vitesse / 3.6) - CS)
                if temps_limite > 0 and math.isfinite(temps_limite):
                    minutes_lim = int(temps_limite // 60)
                    seconds_lim = int(temps_limite % 60)
                    temps_format = f"{minutes_lim}:{seconds_lim:02d}"

                    min_per_km = 60 / vitesse
                    total_sec = int(round(min_per_km * 60))
                    minutes_allure = total_sec // 60
                    seconds_allure = total_sec % 60
                    allure_format = f"{minutes_allure}:{seconds_allure:02d}"

                    rows.append({
                        "% VC": f"{p}%",
                        "Temps limite (mm:ss)": temps_format,
                        "Allure (min/km)": allure_format
                    })

            df = pd.DataFrame(rows)
            st.subheader("Temps limites et allures selon intensité")
            st.table(df)
