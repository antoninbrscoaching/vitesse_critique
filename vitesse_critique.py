# analyse_course_v8.py
# Application Streamlit unifiée — 3 onglets
# NOUVEAUTÉS v8.5 :
#   - Modèle de fatigue à DEUX phases empilées (fatigue_multiplier_dual) :
#     Phase 1 (parcours) reste pilotée par le profil terrain / JSON de
#     cohorte comme avant. Phase 2 (optionnelle, décochée par défaut)
#     AJOUTE une fatigue personnelle plus tardive, détectée sur tes
#     propres références (section 2), par-dessus la phase 1 — au lieu
#     de la remplacer comme le faisait la v8.4. Le R² du point de rupture
#     est affiché à côté du réglage pour juger de sa fiabilité.
#
# NOUVEAUTÉS v8.4 :
#   - Signature de fatigue personnelle (seuil/taux) désormais calculée
#     AUTOMATIQUEMENT depuis tes propres courses de référence (section 2
#     de l'onglet Prédiction, FIT/TCX importés) — plus besoin de passer par
#     l'onglet 🧪 Tests d'endurance + VC ni de cliquer sur un bouton
#     séparé. La détection tourne dans le même passage de script que le
#     reste de la section 4, donc pas de décalage d'un run à l'autre.
#     L'onglet Vitesse Critique reste dédié à l'analyse des tests
#     d'effort/entraînement, sans lien avec la prédiction de course.
# NOUVEAUTÉS v8.2 :
#   - Modèle de marche en forte pente (VAM) : au-delà d'un seuil de pente
#     (défaut 25%, réglable), le temps est gouverné par la VAM (Vitesse
#     d'Ascension Moyenne, m de D+/heure) plutôt que par l'allure horizontale.
#     Corrige une sous-estimation majeure du temps sur les passages très
#     raides (>25-30%) : le plafond de pente calibré (max_cap) reflète des
#     pentes MOYENNES de segments de plusieurs km dans les données de
#     calibration v8.1, pas les pointes instantanées les plus raides d'un
#     parcours réel — sans ce patch, un parcours type UTMB pouvait être
#     largement sous-estimé en temps.
#   - Détection du point de rupture sur tests à effort maximal (onglet VC,
#     section 4) : pour chaque test importé en FIT/TCX, détecte le moment où
#     l'allure décroche le plus nettement, en % de la durée totale. Comparer
#     ce % entre tests de durées différentes permet de repérer une signature
#     de fatigue propre à l'athlète, utilisable comme point de départ pour le
#     seuil de fatigue en course longue.
#   - Import direct de profils calibrés (JSON) dans l'onglet Prédiction,
#     section 4 : plus besoin de repasser par l'onglet Cohorte pour
#     appliquer un profil déjà calibré — le fichier profils_calibres.json
#     (exporté depuis l'onglet Cohorte ou depuis l'artefact de calibration)
#     se dépose directement ici et les coefficients s'appliquent aussitôt.
#
# NOUVEAUTÉS v8.1 (calibration empirique) :
#   - Courbe de pente en montée recalibrée par profil sur 1245 segments coureurs
#     (top10 réel de 12 ultra-trails : TDS, UTMB, OCC, Templiers, Sainte-Lyon,
#     Saint-Jacques, Trans Gran Canaria, Transvulcania, Chianti, MCC, EcoTrail,
#     Nantes-Montaigu). g0_up=5.0 pour tous les profils trail (au lieu de 3.0) ;
# NOUVEAUTÉS v8.3 :
#   - BUG CRITIQUE corrigé : le plafond haut de l'exposant de fatigue Riegel
#     (K) est relevé de 1.25 à 1.60. La borne 1.25 reflétait des exposants
#     typiques de course sur route (5km→marathon) ; elle était bien trop
#     basse pour extrapoler des références ultra-trail (ex : marathon →
#     66km → 135km) vers une distance encore plus longue (UTMB, 100 miles).
#     Le K brut (non plafonné) est maintenant affiché à côté du K utilisé
#     dans l'onglet Prédiction pour repérer ce genre de troncature.
#
#     k_up et max_cap (=max_up) recalibrés par type de course.
#   - Nouveau profil "🏞️ Trail roulant / peu technique" (R²=0.918, le mieux
#     identifié des 3 clusters) pour les ultra-trails à faible D+/km.
#   - Seuil de fatigue (fatigue_threshold/fatigue_rate) rendu dépendant du
#     profil de terrain : dégradation continue dès les premiers km pour les
#     ultras roulants/montagneux, "plateau puis chute à 60%" conservé pour le
#     profil technique moyen (seul cas où l'hypothèse d'origine est confirmée).
#   - cold_quad 0.0015→0.0003 (le froid a un effet quasi nul sur l'allure
#     élite) ; hot_quad 0.0020→0.0055 (la chaleur pénalise ~3x plus que prévu).
#   - Cap descente -6% (max_down) validé tel quel par la donnée — inchangé.
#   - Vent : non recalibré (aucune donnée de direction de vent fiable trouvée
#     pour les 12 courses de référence).
#   Voir calibration_facteurs_v8.md pour la méthodologie complète.
#
# NOUVEAUTÉS v8 (par rapport à v7) :
#   - Filtre bruit GPS : compute_gpx_distance_filtered (max_step=50m)
#     → évite l'inflation de distance (ex: 10km GPX → 10.6km affiché)
#   - Lissage altitude : padding miroir 'reflect' au lieu de mode='same'
#     → supprime l'artefact de faux dénivelé sur le 1er km
#   - Allure moy. correcte quand distance forcée (_dist_simulated_km)
#   - Profil Route recalibré : k_up=4, grade_power=0.75, lissage=25pts
#     → moins réactif aux micro-variations GPS sur route plane
#   - Onglet VC : import FIT/TCX avec segmentation par heure début/fin
#
# CONSERVÉ de v7 :
#   - Suppression des noms d'élites — outil standardisable
#   - Modèle de fatigue avancé : seuil de dégradation personnalisé par athlète
#   - Fix get_avg_weather() pour le chargement des références FIT/TCX
#   - Prédiction de zone FC cible basée sur les données personnelles de l'athlète
#   - Régression FC log(durée) → FC_moy spécifique à l'athlète
#
# pip install streamlit gpxpy fitparse fitdecode pandas numpy pydeck matplotlib requests scipy

import streamlit as st
import math
import re
import gpxpy
from fitparse import FitFile
try:
    import fitdecode
    HAS_FITDECODE = True
except ImportError:
    HAS_FITDECODE = False

from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xml.etree.ElementTree as ET
import requests
import io
import os
import glob
import sqlite3
import hashlib
import secrets
import warnings
import json as _json
from scipy import stats as sp_stats
from scipy.optimize import least_squares

st.set_page_config(page_title="Coach Running — Suite complète", layout="wide", page_icon="🏃")
TZ_NAME_DEFAULT = "Europe/Paris"

# ══════════════════════════════════════════════════════════════
# v8.8 — CHARTE GRAPHIQUE (fond noir, texte blanc, accents rouges)
# Uniquement du rendu : aucun calcul, aucun seuil, aucune fonctionnalité
# n'est modifié par ce bloc.
# ══════════════════════════════════════════════════════════════
C_BG        = "#0B0B0E"   # fond de page
C_SURFACE   = "#111116"   # fond des graphiques et des cartes
C_SURFACE_2 = "#17171E"   # fond des blocs secondaires (sidebar, expanders)
C_LINE      = "#23232C"   # bordures / grille
C_TEXT      = "#F5F5F7"   # texte principal
C_TEXT_MUT  = "#A5AAB3"   # texte secondaire
C_RED       = "#E63946"   # accent principal
C_RED_SOFT  = "#FF7A83"   # accent secondaire (seuil n°2, bandes)
C_WHITE     = "#D9D4CB"   # série neutre claire
C_GREY      = "#7F8C99"   # série neutre froide
C_DIM       = "#4A4F58"   # traces brutes / éléments très en retrait
# Ordre FIXE des séries catégorielles (jamais recyclé au-delà de 8) :
# séparation CVD vérifiée (ΔE ≥ 19 sur toutes les paires adjacentes, fond #111116).
CHART_CYCLE = [C_RED, C_WHITE, C_GREY, C_RED_SOFT, "#B9B2A6", "#5C6875", "#FFB3B8", "#8E8579"]

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE, "axes.edgecolor": C_LINE, "axes.linewidth": 1.0,
    "axes.labelcolor": C_TEXT_MUT, "axes.titlecolor": C_TEXT,
    "axes.titlesize": 11.5, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 9.5, "axes.labelpad": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": C_LINE, "grid.linewidth": 0.8, "grid.alpha": 0.9,
    "text.color": C_TEXT, "xtick.color": C_TEXT_MUT, "ytick.color": C_TEXT_MUT,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.facecolor": C_SURFACE_2, "legend.edgecolor": C_LINE, "legend.framealpha": 0.95,
    "legend.labelcolor": C_TEXT, "legend.fontsize": 8.5, "legend.borderpad": 0.6,
    "figure.autolayout": False, "font.size": 9.5,
    "axes.prop_cycle": plt.cycler(color=CHART_CYCLE),
})

def df_arrow_safe(df):
    """Streamlit convertit les tableaux en Arrow, qui refuse une colonne mélangeant
    des nombres et du texte (ex. la colonne « Km » où le dernier kilomètre partiel
    s'écrit « 34 (974m) », ou une colonne où les valeurs absentes sont notées « — »).
    On force ces colonnes-là en texte : l'affichage est identique, mais le tableau
    ne fait plus planter le rendu de la page."""
    try:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                types = {type(v) for v in out[c].dropna().head(500)}
                if len(types) > 1:
                    out[c] = out[c].astype(str)
        return out
    except Exception:
        return df

# Tous les tableaux de l'app passent par ce filtre, y compris ceux ajoutés plus tard.
_st_dataframe_original = st.dataframe
def _st_dataframe_safe(data=None, *args, **kwargs):
    if isinstance(data, pd.DataFrame):
        data = df_arrow_safe(data)
    return _st_dataframe_original(data, *args, **kwargs)
st.dataframe = _st_dataframe_safe

def chart_title(ax, title, subtitle=None):
    """Titre de graphique en deux niveaux (titre blanc + sous-titre gris)."""
    ax.set_title(title, color=C_TEXT, fontsize=11.5, fontweight="600", loc="left", pad=(22 if subtitle else 12))
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 12),
                    textcoords="offset points", fontsize=8.5, color=C_TEXT_MUT, ha="left", va="bottom")

def kpi_row(items):
    """Rangée de cartes chiffre-clé : libellé discret, grande valeur blanche,
    filet rouge. items = [(libellé, valeur, précision|None), ...]"""
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub or "&nbsp;"}</div></div>'
        for lbl, val, sub in items)
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)

st.markdown(f"""
<style>
:root {{
  --bg:{C_BG}; --surface:{C_SURFACE}; --surface2:{C_SURFACE_2}; --line:{C_LINE};
  --text:{C_TEXT}; --muted:{C_TEXT_MUT}; --red:{C_RED};
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background:var(--bg);}}
[data-testid="stSidebar"] {{background:var(--surface); border-right:1px solid var(--line);}}
html, body, .stApp, p, li, label, span, div {{color:var(--text);}}
h1, h2, h3, h4, h5 {{color:var(--text); font-weight:650; letter-spacing:-0.01em;}}
h1 {{font-size:1.85rem;}} h2 {{font-size:1.25rem;}} h3 {{font-size:1.05rem;}}
h2::after {{content:""; display:block; width:44px; height:3px; background:var(--red);
            border-radius:2px; margin-top:8px;}}
[data-testid="stCaptionContainer"], .stCaption, small {{color:var(--muted) !important;}}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {{gap:2px; border-bottom:1px solid var(--line);}}
.stTabs [data-baseweb="tab"] {{background:transparent; color:var(--muted); border-radius:8px 8px 0 0;
    padding:10px 16px; font-size:0.92rem;}}
.stTabs [aria-selected="true"] {{background:var(--surface); color:var(--text) !important;
    box-shadow:inset 0 -2px 0 var(--red);}}

/* Cartes chiffres-clés */
.kpi-row {{display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px 0;}}
.kpi-card {{flex:1 1 150px; background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:13px 16px 12px 16px; position:relative; overflow:hidden;}}
.kpi-card::before {{content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--red);}}
.kpi-label {{font-size:0.68rem; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted);
    margin-bottom:6px;}}
.kpi-value {{font-size:1.55rem; font-weight:700; color:var(--text); line-height:1.15;
    font-variant-numeric:tabular-nums;}}
.kpi-sub {{font-size:0.72rem; color:var(--muted); margin-top:3px;}}

/* Métriques natives Streamlit */
[data-testid="stMetric"] {{background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:12px 16px;}}
[data-testid="stMetricLabel"] p {{color:var(--muted) !important; font-size:0.72rem !important;
    text-transform:uppercase; letter-spacing:0.07em;}}
[data-testid="stMetricValue"] {{color:var(--text); font-variant-numeric:tabular-nums;}}

/* Blocs de texte */
.param-box{{background:var(--surface); border-left:3px solid var(--red); border-radius:8px;
    padding:9px 13px; margin-bottom:8px; font-size:0.86rem; color:var(--muted);}}
.param-up{{color:{C_RED}; font-weight:600;}} .param-down{{color:{C_WHITE}; font-weight:600;}}
.highlight-box{{background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--red);
    border-radius:10px; padding:13px 17px; margin:10px 0; color:var(--text); font-size:0.90rem;}}
.test-card{{background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:15px 17px; margin-bottom:12px;}}
.test-card h4{{margin:0 0 10px 0; color:var(--text); font-size:1rem;}}
.result-metric{{text-align:center; font-size:1.4rem; font-weight:700; color:var(--text);}}
.sidebar-label{{background:var(--surface2); border-left:3px solid var(--red); border-radius:8px;
    padding:8px 11px; font-size:0.78rem; color:var(--muted); margin-bottom:12px;}}
.interval-card{{background:var(--surface); border:1px solid var(--line); border-radius:10px;
    padding:11px 15px; margin-bottom:8px;}}
.terrain-badge{{display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.76rem;
    font-weight:600; margin-bottom:6px;}}
.note-box{{background:var(--surface); border:1px solid var(--line); border-radius:10px;
    padding:11px 15px; margin:8px 0; font-size:0.86rem; color:var(--muted);}}
.note-box b, .note-box strong {{color:var(--text);}}
.note-red{{border-left:3px solid var(--red);}}

/* Contrôles */
.stButton>button, .stDownloadButton>button {{background:var(--surface); color:var(--text);
    border:1px solid var(--line); border-radius:9px; padding:6px 15px; font-weight:550;}}
.stButton>button:hover, .stDownloadButton>button:hover {{border-color:var(--red); color:var(--text);}}
.stButton>button[kind="primary"] {{background:var(--red); border-color:var(--red); color:#fff;}}
[data-testid="stExpander"] {{background:var(--surface); border:1px solid var(--line); border-radius:12px;}}
[data-testid="stExpander"] summary {{color:var(--text);}}
[data-testid="stFileUploaderDropzone"] {{background:var(--surface); border:1px dashed var(--line);}}
[data-testid="stDataFrame"] {{border:1px solid var(--line); border-radius:10px;}}
hr {{border-color:var(--line);}}

/* Encarts st.info / st.warning / st.success / st.error */
[data-testid="stAlert"], [data-testid="stAlertContainer"], [data-testid="stNotification"] {{
    background:var(--surface) !important; border:1px solid var(--line) !important;
    border-left:3px solid var(--red) !important; border-radius:10px; color:var(--text) !important;}}
[data-testid="stAlert"] p, [data-testid="stAlertContainer"] p {{color:var(--text) !important;}}
[data-testid="stAlert"] svg {{fill:var(--red);}}
</style>
""", unsafe_allow_html=True)

def param_help(text_up,text_down,note=""):
    note_html=f"<br><em>{note}</em>" if note else ""
    st.markdown(f'<div class="param-box"><span class="param-up">⬆️ Augmenter</span>:{text_up}<br><span class="param-down">⬇️ Diminuer</span>:{text_down}{note_html}</div>',unsafe_allow_html=True)

def safe_float(val,default=0.0):
    try:
        if val is None:return float(default)
        if isinstance(val,str):
            s=val.strip()
            if s in("","nan","none"):return float(default)
            return float(s.replace(",","."))
        if isinstance(val,(float,int,np.number)):
            if np.isnan(val) or np.isinf(val):return float(default)
            return float(val)
        return float(val)
    except:return float(default)

def hms_to_seconds(hms):
    if hms is None:return 0
    try:
        parts=[int(p) for p in str(hms).strip().split(":")]
        if len(parts)==3:h,m,s=parts
        elif len(parts)==2:h,m,s=0,parts[0],parts[1]
        elif len(parts)==1:h,m,s=0,0,parts[0]
        else:return 0
        if not(0<=m<=59 and 0<=s<=59):return 0
        return max(0,h*3600+m*60+s)
    except:return 0

def seconds_to_hms(s):
    s=int(round(s))
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def hms_to_timedelta(hms):return timedelta(seconds=hms_to_seconds(hms))

def validate_hms(val):
    try:
        parts=[int(p) for p in val.strip().split(":")]
        if len(parts)==3:return 0<=parts[1]<=59 and 0<=parts[2]<=59
        if len(parts)==2:return 0<=parts[0]<=59 and 0<=parts[1]<=59
        return False
    except:return False

def hms_input(label,default="0:00:00",key=None,help=None,compact=False):
    val=st.text_input(label,value=str(st.session_state.get(key,default)) if key else default,key=key,help=help or "Format : hh:mm:ss",placeholder="hh:mm:ss")
    if val and not validate_hms(val):st.warning(f"⚠️ Format invalide : **{val}**")
    return val

def pace_str(secs_per_km):
    if secs_per_km is None or secs_per_km<=0 or not math.isfinite(secs_per_km):return "0:00"
    t=int(round(float(secs_per_km)))
    return f"{t//60}:{t%60:02d}"

def haversine_m(lat1,lon1,lat2,lon2):
    R=6371000.0
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=math.radians(lat2-lat1);dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

# ── v8 PATCH 1 : filtre bruit GPS ──────────────────────────────────────────
def compute_gpx_distance_filtered(points, max_step_m=None):
    """
    Distance GPX filtrée — ignore les sauts GPS aberrants.
    Le seuil est adaptatif : max(100m, médiane_des_pas × 10).
    - Trace activité 1Hz dense (médiane ~5m)  → seuil ~100m
    - Parcours course peu dense (médiane ~30m) → seuil ~300m
    Cela évite de couper les GPX de parcours où les pas légitimes
    peuvent dépasser 200m (ex : UTSJ 137km avec seuil fixe 50m = 78km).
    """
    import statistics as _stats
    steps = []
    for i in range(1, len(points)):
        d = haversine_m(
            points[i-1].latitude, points[i-1].longitude,
            points[i].latitude,   points[i].longitude
        )
        steps.append(d)
    if not steps:
        return 0.0
    if max_step_m is None:
        med = _stats.median(steps)
        max_step_m = max(100.0, min(2000.0, med * 10))
    return sum(s for s in steps if s <= max_step_m)
# ───────────────────────────────────────────────────────────────────────────

def bearing_deg(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2);dl=math.radians(lon2-lon1)
    y=math.sin(dl)*math.cos(p2);x=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return(math.degrees(math.atan2(y,x))+360.0)%360.0

def compute_dplus_dminus(elevs):
    arr=np.array([safe_float(e,np.nan) for e in elevs],dtype=float);arr=arr[~np.isnan(arr)]
    if arr.size<2:return 0.0,0.0
    diffs=np.diff(arr)
    return float(np.sum(np.clip(diffs,0,None))),float(-np.sum(np.clip(diffs,None,0)))

def wbgt_simplified(T_c,RH):
    try:
        RH_c=max(0.0,min(100.0,float(RH)));T=float(T_c)
        Tw=(T*math.atan(0.151977*(RH_c+8.313659)**0.5)+math.atan(T+RH_c)-math.atan(RH_c-1.676331)+0.00391838*RH_c**1.5*math.atan(0.023101*RH_c)-4.686035)
        return 0.7*Tw+0.2*(T+2.0)+0.1*T
    except:return float(T_c)

def effective_temp(T_c,RH,use_wbgt):return wbgt_simplified(T_c,RH) if use_wbgt else float(T_c)

def altitude_vo2_multiplier(altitude_m,altitude_ref_m=0.0):
    alt=max(0.0,float(altitude_m));alt_ref=max(0.0,float(altitude_ref_m))
    effective_alt=max(0.0,alt-max(1500.0,alt_ref))
    return 1.0+min(0.25,0.01*(effective_alt/100.0))

def minetti_cost(grade_fraction):
    g=max(-0.45,min(0.45,float(grade_fraction)))
    return max(0.1,float(155.4*g**5-30.4*g**4-43.3*g**3+46.3*g**2+19.5*g+3.6))

def minetti_multiplier(grade_pct):
    return float(max(0.92,min(1.35,minetti_cost(float(grade_pct)/100.0)/3.6)))

def grade_multiplier_heuristic(grade_pct,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down):
    try:
        g=float(grade_pct)/100.0;g0u=max(1e-6,float(g0_up)/100.0);g0d=max(1e-6,float(g0_down)/100.0)
        if g>=0:
            g_eff=math.tanh(g/g0u)*g0u;mult=1.0+float(k_up)*g_eff
        else:
            g_eff=math.tanh((-g)/g0d)*g0d;bonus=min(float(k_down)*g_eff,abs(float(down_cap)))
            mult=1.0-bonus
        mult=min(mult,1.0+float(max_up));mult=max(mult,1.0+float(max_down))
        return max(0.01,float(mult))
    except:return 1.0

def combined_grade_multiplier(grade_pct,use_minetti,minetti_weight,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down):
    if not use_minetti:return grade_multiplier_heuristic(grade_pct,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    m_min=minetti_multiplier(grade_pct)
    m_heu=grade_multiplier_heuristic(grade_pct,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    w=max(0.0,min(1.0,float(minetti_weight)))
    return w*m_min+(1.0-w)*m_heu

# ── v8.2 PATCH : marche en forte pente, gouvernée par la VAM ──────────────
# Le modèle de pente ci-dessus (combined_grade_multiplier) a été calibré sur
# des SEGMENTS DE PLUSIEURS KM (moyennés), pas sur la pente instantanée GPX.
# Son plafond (max_up/max_cap) reflète donc "jusqu'où une pente MOYENNE de
# segment a pu pénaliser l'allure dans les données réelles" — pas le coût
# physiologique réel d'un passage ponctuel à 35-45% où même les meilleurs
# marchent. Au-delà d'un seuil de pente, ce patch fait gouverner le temps par
# la VAM (Vitesse d'Ascension Moyenne, m de D+ par heure) plutôt que par
# l'allure horizontale — c'est le bon référentiel physiologique pour la
# marche en forte pente, indépendant de la calibration k_up/max_cap.
def grade_multiplier_with_vam(grade_pct, base_s_per_km, vam_threshold_pct, vam_rate_m_per_h, vam_blend_width_pct,
                               use_minetti, minetti_weight, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    g = float(grade_pct)
    mult_smooth = combined_grade_multiplier(g, use_minetti, minetti_weight, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    if g <= 0:
        return mult_smooth
    lo = float(vam_threshold_pct) - float(vam_blend_width_pct) / 2.0
    if g <= lo:
        return mult_smooth
    d_plus_per_km_m = g / 100.0 * 1000.0  # m de D+ pour 1km horizontal à cette pente
    t_vam_s_per_km = d_plus_per_km_m / max(1.0, float(vam_rate_m_per_h)) * 3600.0
    mult_vam = t_vam_s_per_km / max(1.0, float(base_s_per_km))
    hi = float(vam_threshold_pct) + float(vam_blend_width_pct) / 2.0
    w = 1.0 if g >= hi else max(0.0, (g - lo) / max(1e-6, (hi - lo)))
    return (1.0 - w) * mult_smooth + w * max(mult_smooth, mult_vam)
# ───────────────────────────────────────────────────────────────────────────

def temp_multiplier(temp_eff, opt_temp, cold_quad, hot_quad, max_penalty):
    if temp_eff is None: return 1.0
    d = float(temp_eff) - float(opt_temp)
    pen = hot_quad * d**2 if d >= 0 else cold_quad * (-d)**2
    return 1.0 + min(float(max_penalty), float(pen))

def wind_components(wind_speed_ms, wind_dir_from_deg, course_bearing_deg):
    if wind_speed_ms is None or wind_dir_from_deg is None: return 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0: return 0.0, 0.0
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians((wind_to - course_bearing_deg + 540.0) % 360.0 - 180.0)
    along = ws * math.cos(delta)
    cross = ws * abs(math.sin(delta)) * 0.20
    if along >= 0:
        tail_eff = max(0.0, along - cross * 0.3)
        head_eff = 0.0
    else:
        head_eff = abs(along) + cross
        tail_eff = 0.0
    return float(head_eff), float(tail_eff)

def wind_multiplier(head_ms,tail_ms,pace_s_per_km,drag_coeff,tail_credit,cap_head,cap_tail):
    pace=max(150.0,float(pace_s_per_km));v_run=1000.0/pace
    w_along=float(head_ms)-float(tail_ms);v_rel=max(0.0,v_run+w_along)
    base=max(1e-9,v_run**2);extra=(v_rel**2-v_run**2)/base
    if extra<0:extra=float(tail_credit)*extra
    mult=1.0+float(drag_coeff)*extra
    return float(max(1.0+cap_tail,min(1.0+cap_head,mult)))

def wind_gate(grade_pct,g1=2.0,g2=8.0,min_gate=0.25):
    g=max(0.0,float(grade_pct))
    if g<=g1:return 1.0
    if g>=g2:return float(min_gate)
    return float(1.0-(g-g1)/(g2-g1)*(1.0-min_gate))

def cap_combined(mult_total,grade_pct,base_cap,extra_per_pct,max_cap):
    g=max(0.0,float(grade_pct))
    cap=min(float(max_cap),float(base_cap)+float(extra_per_pct)*g)
    return min(float(mult_total),1.0+cap)

def cap_combined_with_vam(mult_total, grade_pct, base_cap, extra_per_pct, max_cap,
                           base_s_per_km, apply_vam, vam_threshold_pct, vam_rate_m_per_h):
    """Comme cap_combined, mais empêche le plafonnement de redescendre sous le
    plancher physiologique imposé par la VAM quand la pente NETTE du km est
    elle-même extrême — sinon ce second plafond (calibré sur des pentes
    nettes de km, pas sur la pente instantanée) annulerait le correctif
    apporté par grade_multiplier_with_vam au niveau point par point."""
    capped = cap_combined(mult_total, grade_pct, base_cap, extra_per_pct, max_cap)
    g = max(0.0, float(grade_pct))
    if apply_vam and g >= float(vam_threshold_pct):
        d_plus_per_km_m = g / 100.0 * 1000.0
        t_vam_s_per_km = d_plus_per_km_m / max(1.0, float(vam_rate_m_per_h)) * 3600.0
        mult_vam_floor = t_vam_s_per_km / max(1.0, float(base_s_per_km))
        return max(capped, mult_vam_floor)
    return capped

def fatigue_multiplier(d_plus_cum,dist_cum,d_plus_total,dist_total,rate_pct,mode):
    """Modèle de fatigue simple (conservé pour compatibilité)."""
    if rate_pct<=0:return 1.0
    rate=rate_pct/100.0
    prog_dist=min(1.0,dist_cum/max(1.0,dist_total))
    prog_dplus=min(1.0,d_plus_cum/max(1.0,d_plus_total))
    dplus_ratio=d_plus_total/max(1.0,dist_total)
    w_dplus=min(0.8,dplus_ratio*10.0)
    if mode=="distance":prog=prog_dist
    elif mode=="d_plus":prog=prog_dplus
    else:prog=w_dplus*prog_dplus+(1.0-w_dplus)*prog_dist
    k=2.0;factor=(math.exp(k*prog)-1.0)/(math.exp(k)-1.0)
    return 1.0+rate*factor


def fatigue_multiplier_dual(d_plus_cum, dist_cum, d_plus_total, dist_total,
                             threshold1_pct, rate1_pct, threshold2_pct, rate2_pct, mode):
    """
    Modèle de fatigue à DEUX phases, empilées :
    - Phase 1 (0 → seuil1) : montée en régime + dynamique propre à CE parcours — calibrée
      sur des coureurs réels de cette course (seuil1/taux1 = fatigue_threshold/fatigue_rate
      du profil terrain, typiquement issus d'un JSON de cohorte).
    - Phase 2 (seuil1 → seuil2) : rythme de croisière, convergence douce vers rate1 (reprend
      la forme du modèle à une phase de fatigue_multiplier_advanced).
    - Phase 3 (seuil2 → 100%) : fatigue personnelle profonde, calibrée sur TES propres
      références longues (seuil2/taux2 = rupture d'allure détectée), qui s'AJOUTE par-dessus
      rate1 plutôt que de le remplacer.
    Si seuil2 <= seuil1, seuil2 est repoussé à seuil1+2% pour éviter une phase 2 dégénérée.
    Si rate2=0 (aucune donnée personnelle), se réduit exactement au modèle à une phase.
    """
    if rate1_pct <= 0 and rate2_pct <= 0:
        return 1.0
    dist_total = max(1e-6, float(dist_total)); d_plus_total = max(1e-6, float(d_plus_total))
    thr1 = max(0.01, min(0.97, float(threshold1_pct) / 100.0))
    thr2 = max(thr1 + 0.02, min(0.99, float(threshold2_pct) / 100.0))
    rate1 = max(0.0, float(rate1_pct)) / 100.0
    rate2 = max(0.0, float(rate2_pct)) / 100.0
    prog_dist = min(1.0, dist_cum / dist_total)
    prog_dplus = min(1.0, d_plus_cum / max(1.0, d_plus_total))
    dplus_ratio = d_plus_total / dist_total
    w_dplus = min(0.8, dplus_ratio * 10.0)
    if mode == "distance": prog = prog_dist
    elif mode == "d_plus": prog = prog_dplus
    else: prog = w_dplus * prog_dplus + (1.0 - w_dplus) * prog_dist
    pre1 = rate1 * 0.10
    k = 2.5
    if prog <= thr1:
        factor = pre1 * (prog / thr1)
    elif prog <= thr2:
        p = (prog - thr1) / (thr2 - thr1)
        factor = pre1 + (rate1 - pre1) * (math.exp(k * p) - 1.0) / (math.exp(k) - 1.0)
    else:
        p = (prog - thr2) / (1.0 - thr2)
        factor = rate1 + rate2 * (math.exp(k * p) - 1.0) / (math.exp(k) - 1.0)
    return 1.0 + min(factor, (rate1 + rate2) * 1.05)

def fatigue_multiplier_advanced(d_plus_cum, dist_cum, d_plus_total, dist_total,
                                 threshold_pct, decay_rate_pct, mode):
    """
    Modèle de fatigue avancé avec seuil de dégradation personnalisé.
    Avant le seuil : fatigue légère (~10% du taux total).
    Après le seuil : accélération exponentielle jusqu'à decay_rate_pct total.
    """
    if decay_rate_pct <= 0:
        return 1.0
    dist_total  = max(1e-6, float(dist_total))
    d_plus_total = max(1e-6, float(d_plus_total))
    rate      = decay_rate_pct / 100.0
    threshold = max(0.01, min(0.99, float(threshold_pct) / 100.0))
    prog_dist  = min(1.0, dist_cum  / dist_total)
    prog_dplus = min(1.0, d_plus_cum / max(1.0, d_plus_total))
    dplus_ratio = d_plus_total / dist_total
    w_dplus = min(0.8, dplus_ratio * 10.0)
    if mode == "distance":
        prog = prog_dist
    elif mode == "d_plus":
        prog = prog_dplus
    else:
        prog = w_dplus * prog_dplus + (1.0 - w_dplus) * prog_dist
    pre_threshold_rate = rate * 0.10
    if prog <= threshold:
        factor = pre_threshold_rate * (prog / threshold)
    else:
        prog_post = (prog - threshold) / (1.0 - threshold)
        k = 2.5
        exp_factor = (math.exp(k * prog_post) - 1.0) / (math.exp(k) - 1.0)
        factor = pre_threshold_rate + (rate - pre_threshold_rate) * exp_factor
    return 1.0 + min(float(factor), float(rate) * 1.05)


# ══════════════════════════════════════════════════════════════
# SYSTÈME MÉTÉO ROBUSTE v7 (inchangé)
# ══════════════════════════════════════════════════════════════

def _diurnal_weather(hour_float, t_base, t_amp, wind_ms, humidity_pct, wind_dir_deg=180.0):
    T = t_base + t_amp * math.sin(math.pi * max(0.0, hour_float - 6.0) / 12.0)
    W = wind_ms * (1.0 + 0.15 * math.sin(math.pi * max(0.0, hour_float - 8.0) / 10.0))
    return {"temp":round(float(T),2),"wind":round(float(max(0.0,W)),2),
            "humidity":float(humidity_pct),"wind_dir":float(wind_dir_deg),"source":"diurnal_model"}

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_openmeteo_forecast(lat, lon, tz_name):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&forecast_days=16&timezone={tz_name}")
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return None
        d = r.json()
        if "hourly" not in d: return None
        return {"times":d["hourly"]["time"],"temps":d["hourly"]["temperature_2m"],
                "winds":d["hourly"]["wind_speed_10m"],"hums":d["hourly"]["relativehumidity_2m"],
                "wdirs":d["hourly"]["wind_direction_10m"]}
    except: return None

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_openmeteo_archive(lat, lon, date_str, tz_name):
    try:
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={date_str}&end_date={date_str}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return None
        d = r.json()
        if "hourly" not in d: return None
        return {"times":d["hourly"]["time"],"temps":d["hourly"]["temperature_2m"],
                "winds":d["hourly"]["wind_speed_10m"],"hums":d["hourly"]["relativehumidity_2m"],
                "wdirs":d["hourly"]["wind_direction_10m"]}
    except: return None

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_openmeteo_histforecast(lat, lon, date_str, tz_name):
    try:
        url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={date_str}&end_date={date_str}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return None
        d = r.json()
        if "hourly" not in d: return None
        return {"times":d["hourly"]["time"],"temps":d["hourly"]["temperature_2m"],
                "winds":d["hourly"]["wind_speed_10m"],"hums":d["hourly"]["relativehumidity_2m"],
                "wdirs":d["hourly"]["wind_direction_10m"]}
    except: return None

def _interp_meteo(md, dt):
    if md is None: return None
    try:
        times = [datetime.fromisoformat(t) for t in md["times"]]
        for i in range(len(times)-1):
            if times[i] <= dt <= times[i+1]:
                r = (dt-times[i]).total_seconds() / max(1.0,(times[i+1]-times[i]).total_seconds())
                a1,a2 = float(md["wdirs"][i])%360, float(md["wdirs"][i+1])%360
                da = (a2-a1+540.0)%360.0-180.0
                return {"temp":round(md["temps"][i]+r*(md["temps"][i+1]-md["temps"][i]),2),
                        "wind":round(md["winds"][i]+r*(md["winds"][i+1]-md["winds"][i]),2),
                        "humidity":round(md["hums"][i]+r*(md["hums"][i+1]-md["hums"][i]),1),
                        "wind_dir":(a1+r*da)%360.0, "source":"api"}
        idx = min(range(len(times)), key=lambda i: abs(times[i]-dt))
        return {"temp":float(md["temps"][idx]),"wind":float(md["winds"][idx]),
                "humidity":float(md["hums"][idx]),"wind_dir":float(md["wdirs"][idx]),"source":"api"}
    except: return None

def _get_session_meteo_cache():
    if "_meteo_api_cache" not in st.session_state:
        st.session_state["_meteo_api_cache"] = {}
    return st.session_state["_meteo_api_cache"]

def get_weather_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT,
                          fallback_temp=12.0, fallback_temp_amp=4.0,
                          fallback_wind=2.0, fallback_humidity=60.0,
                          fallback_wind_dir=180.0):
    hour = dt_local_naive.hour + dt_local_naive.minute / 60.0
    today = datetime.now()
    diff_days = (dt_local_naive.date() - today.date()).days
    date_str = dt_local_naive.strftime("%Y-%m-%d")
    lat_r = round(lat, 2); lon_r = round(lon, 2)
    cache_key = f"{date_str}_{lat_r}_{lon_r}"
    cache = _get_session_meteo_cache()
    if cache_key not in cache:
        md = None
        if 0 <= diff_days <= 15:
            md = _fetch_openmeteo_forecast(lat_r, lon_r, tz_name)
        elif diff_days < 0:
            md = _fetch_openmeteo_archive(lat_r, lon_r, date_str, tz_name)
            if md is None:
                md = _fetch_openmeteo_histforecast(lat_r, lon_r, date_str, tz_name)
        else:
            md = _fetch_openmeteo_forecast(lat_r, lon_r, tz_name)
            if md is None:
                past_date = dt_local_naive.replace(year=dt_local_naive.year - 1)
                md = _fetch_openmeteo_archive(lat_r, lon_r, past_date.strftime("%Y-%m-%d"), tz_name)
        cache[cache_key] = md
    result = _interp_meteo(cache.get(cache_key), dt_local_naive)
    if result is not None:
        return result
    return _diurnal_weather(hour, fallback_temp, fallback_temp_amp,
                            fallback_wind, fallback_humidity, fallback_wind_dir)

def get_avg_weather(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    try:
        duration_s = (end_dt - start_dt).total_seconds()
        n_samples = min(6, max(2, int(duration_s / 1800)))
        temps, winds, hums = [], [], []
        for i in range(n_samples):
            frac = i / max(1, n_samples - 1)
            dt_sample = start_dt + timedelta(seconds=frac * duration_s)
            w = get_weather_minutely(lat, lon, dt_sample, tz_name)
            if w:
                if w.get("temp") is not None:     temps.append(float(w["temp"]))
                if w.get("wind") is not None:     winds.append(float(w["wind"]))
                if w.get("humidity") is not None: hums.append(float(w["humidity"]))
        avg_t = round(float(np.mean(temps)), 1) if temps else None
        avg_w = round(float(np.mean(winds)), 1) if winds else None
        avg_h = round(float(np.mean(hums)), 1)  if hums  else None
        return avg_t, avg_w, avg_h
    except Exception:
        return None, None, None


def predict_hr_zone(refs_with_hr, target_duration_s):
    valid     = [(r["dur_s"], r["hr_avg"]) for r in refs_with_hr
                 if r.get("dur_s", 0) > 0 and r.get("hr_avg") and r["hr_avg"] > 50]
    valid_max = [(r["dur_s"], r["hr_max"]) for r in refs_with_hr
                 if r.get("dur_s", 0) > 0 and r.get("hr_max") and r["hr_max"] > 50]
    if not valid:
        return None
    if len(valid) < 2:
        hr_mean = float(np.mean([v[1] for v in valid]))
        hr_mx   = float(np.mean([v[1] for v in valid_max])) if valid_max else hr_mean * 1.08
        return {"hr_target_avg":round(hr_mean),"hr_target_range":(round(hr_mean-5),round(hr_mean+5)),
                "hr_target_max":round(hr_mx),"r2":None,"n_refs":len(valid),"model":"mean"}
    X = np.array([math.log(max(60, d)) for d, _ in valid])
    Y = np.array([hr for _, hr in valid])
    slope, intercept, r, p, se = sp_stats.linregress(X, Y)
    r2 = float(r ** 2)
    x_target  = math.log(max(60, float(target_duration_s)))
    hr_pred   = float(slope * x_target + intercept)
    residuals = Y - (slope * X + intercept)
    sigma     = float(np.std(residuals)) if len(residuals) > 2 else 5.0
    hr_max_pred = hr_pred * 1.08
    if len(valid_max) >= 2:
        Xm = np.array([math.log(max(60, d)) for d, _ in valid_max])
        Ym = np.array([hm for _, hm in valid_max])
        sm, im, _, _, _ = sp_stats.linregress(Xm, Ym)
        hr_max_pred = float(sm * x_target + im)
    return {"hr_target_avg":round(hr_pred),"hr_target_range":(round(hr_pred-sigma),round(hr_pred+sigma)),
            "hr_target_max":round(hr_max_pred),"r2":round(r2,3),"n_refs":len(valid),
            "model":"regression","slope":round(float(slope),2),"intercept":round(float(intercept),1)}


def correct_elevations_dem(points,max_points=100,dataset="srtm30m"):
    n=len(points)
    if n<2:return np.array([getattr(p,"elevation",0.0) or 0.0 for p in points])
    step=max(1,n//max_points);indices=list(range(0,n,step))
    if indices[-1]!=n-1:indices.append(n-1)
    lats=tuple(points[i].latitude for i in indices);lons=tuple(points[i].longitude for i in indices)
    dem=fetch_dem_elevations(lats,lons,dataset=dataset)
    cum_all=[0.0]
    for i in range(1,n):
        cum_all.append(cum_all[-1]+haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude))
    cum_sub=[cum_all[i] for i in indices]
    valid=[(d,e) for d,e in zip(cum_sub,dem) if e is not None]
    if len(valid)<2:return np.array([getattr(p,"elevation",0.0) or 0.0 for p in points])
    return np.interp(cum_all,[v[0] for v in valid],[v[1] for v in valid])

def analyze_hr_v3(hr_records):
    hrs=[h for h in hr_records if h is not None and 50<=h<=220]
    if len(hrs)<10:return{"hr_max":None,"hr_avg":None,"hr_drift":None,"reliability":"inconnue"}
    arr=np.array(hrs,dtype=float);n=len(arr)
    hr_max=float(np.percentile(arr,95));hr_avg=float(np.mean(arr))
    q1,q3=int(n*0.25),int(n*0.75)
    drift=float(np.mean(arr[q3:]))-float(np.mean(arr[:q1]))
    reliability="haute" if drift<5 else("moyenne" if drift<12 else "basse (dérive cardiaque forte)")
    return{"hr_max":round(hr_max),"hr_avg":round(hr_avg),"hr_drift":round(drift,1),
           "hr_threshold_est":round(hr_max*0.88),"reliability":reliability}

class SimplePoint:
    def __init__(self,lat,lon,elev=0.0,time=None):
        self.latitude=float(lat);self.longitude=float(lon)
        self.elevation=float(elev) if elev is not None else 0.0;self.time=time
    def distance_3d(self,other):
        h=haversine_m(self.latitude,self.longitude,other.latitude,other.longitude)
        v=self.elevation-other.elevation;return math.sqrt(h*h+v*v)

def parse_gpx_points(file):
    try:
        file.seek(0);gpx=gpxpy.parse(file)
        pts=[p for track in gpx.tracks for seg in track.segments for p in seg.points]
        return gpx,pts
    except Exception as e:st.error(f"Erreur GPX:{e}");return None,[]

# ── v8.6 PATCH — lecture FIT tolérante ────────────────────────────────────
# fitparse lève FitParseError "Invalid field size N for type 'uintXX'" dès qu'un
# message de définition déclare un champ dont la taille n'est pas un multiple de
# son type de base (fréquent sur Coros/Suunto/Wahoo et sur les Garmin avec champs
# développeur). fitdecode, lui, se contente d'un warning et lit le fichier.
# parse_fit_ref essaie donc fitparse, puis bascule automatiquement sur fitdecode.
# ──────────────────────────────────────────────────────────────────────────
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)

def _fit_read_bytes(file):
    """Lit le fichier une seule fois en mémoire : permet de le re-parser avec
    un second moteur sans dépendre de la position du curseur."""
    try:
        file.seek(0)
    except Exception:
        pass
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    if isinstance(data, str):
        data = data.encode("latin-1")
    return data

def _fit_extract_fitparse(raw_bytes):
    """Extraction via fitparse (moteur historique de l'app)."""
    fit = FitFile(io.BytesIO(raw_bytes))
    fit.parse()
    records, times_pts, hr_records = [], [], []
    start_global = elapsed_global = None
    for msg in fit.get_messages("session"):
        vals = {d.name: d.value for d in msg}
        if isinstance(vals.get("start_time"), datetime):
            start_global = vals["start_time"].replace(tzinfo=None)
        if isinstance(vals.get("total_elapsed_time"), (int, float)):
            elapsed_global = float(vals["total_elapsed_time"])
    for msg in fit.get_messages("record"):
        vals = {d.name: d.value for d in msg}
        lat_r, lon_r = vals.get("position_lat"), vals.get("position_long")
        if lat_r is None or lon_r is None:
            continue
        ts = vals.get("timestamp")
        dt = ts.replace(tzinfo=None) if isinstance(ts, datetime) else None
        alt = vals.get("enhanced_altitude") or vals.get("altitude") or 0.0
        records.append((lat_r * SEMICIRCLE_TO_DEG, lon_r * SEMICIRCLE_TO_DEG,
                        safe_float(alt, 0.0), safe_float(vals.get("distance"), 0.0)))
        times_pts.append(dt)
        hr = vals.get("heart_rate")
        hr_records.append(int(hr) if hr is not None else None)
    return records, times_pts, hr_records, start_global, elapsed_global

def _fit_extract_fitdecode(raw_bytes):
    """Extraction via fitdecode — tolérant aux champs de taille non standard
    (il émet un simple warning là où fitparse s'arrête sur une exception)."""
    records, times_pts, hr_records = [], [], []
    start_global = elapsed_global = None
    def gv(frame, name):
        try:
            return frame.get_value(name) if frame.has_field(name) else None
        except Exception:
            return None
    kwargs = {}
    if hasattr(fitdecode, "CrcCheck"):
        kwargs["check_crc"] = fitdecode.CrcCheck.WARN   # CRC douteux ≠ fichier inutilisable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # évite d'inonder la console Streamlit
        with fitdecode.FitReader(io.BytesIO(raw_bytes), **kwargs) as fr:
            for frame in fr:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name == "session":
                    stv = gv(frame, "start_time")
                    if isinstance(stv, datetime):
                        start_global = stv.replace(tzinfo=None)
                    ev = gv(frame, "total_elapsed_time")
                    if isinstance(ev, (int, float)):
                        elapsed_global = float(ev)
                    continue
                if frame.name != "record":
                    continue
                lat_r, lon_r = gv(frame, "position_lat"), gv(frame, "position_long")
                if lat_r is None or lon_r is None:
                    continue
                ts = gv(frame, "timestamp")
                dt = ts.replace(tzinfo=None) if isinstance(ts, datetime) else None
                alt = gv(frame, "enhanced_altitude")
                if alt is None:
                    alt = gv(frame, "altitude")
                records.append((lat_r * SEMICIRCLE_TO_DEG, lon_r * SEMICIRCLE_TO_DEG,
                                safe_float(alt, 0.0), safe_float(gv(frame, "distance"), 0.0)))
                times_pts.append(dt)
                hr = gv(frame, "heart_rate")
                hr_records.append(int(hr) if hr is not None else None)
    return records, times_pts, hr_records, start_global, elapsed_global

def parse_fit_ref(file, tz_name=TZ_NAME_DEFAULT):
    """Lit un FIT de référence. Essaie fitparse, puis bascule sur fitdecode si
    fitparse refuse le fichier (champ de taille non standard, CRC, etc.)."""
    try:
        raw_bytes = _fit_read_bytes(file)
    except Exception as e:
        st.error(f"Erreur FIT : fichier illisible ({e})")
        return None

    records = times_pts = hr_records = None
    start_global = elapsed_global = None
    engine = None
    err_fitparse = None
    try:
        records, times_pts, hr_records, start_global, elapsed_global = _fit_extract_fitparse(raw_bytes)
        engine = "fitparse"
    except Exception as e:
        err_fitparse = e          # ex : Invalid field size 1 for type 'uint32'

    if not records:
        if not HAS_FITDECODE:
            st.error(f"Erreur FIT : {err_fitparse}. Ce fichier utilise un encodage que fitparse "
                     f"refuse ; installe fitdecode (`pip install fitdecode`) et recharge l'app — "
                     f"la lecture basculera automatiquement dessus.")
            return None
        try:
            records, times_pts, hr_records, start_global, elapsed_global = _fit_extract_fitdecode(raw_bytes)
            engine = "fitdecode"
        except Exception as e2:
            st.error(f"Erreur FIT : illisible par fitparse ({err_fitparse}) et par fitdecode ({e2}).")
            return None
        if records and err_fitparse is not None:
            st.caption(f"ℹ️ Fichier FIT non standard (fitparse : {err_fitparse}) — "
                       f"lu avec fitdecode : {len(records)} points GPS récupérés.")

    if not records:
        st.error("Erreur FIT : aucun point GPS exploitable dans ce fichier "
                 "(activité sans GPS, ou uniquement des tours/résumés).")
        return None

    df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
    valid_t = [t for t in times_pts if t is not None]
    if len(valid_t) >= 2:
        start_dt, end_dt = min(valid_t), max(valid_t)
    elif start_global and elapsed_global:
        start_dt = start_global
        end_dt = start_global + timedelta(seconds=elapsed_global)
    else:
        start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end_dt = start_dt + timedelta(minutes=5)

    avgT, avgW, avgH = get_avg_weather(records[0][0], records[0][1], start_dt, end_dt, tz_name)
    elev_arr = df["elev"].values
    dup = float(np.sum(np.clip(np.diff(elev_arr), 0, None))) if elev_arr.size >= 2 else 0.0
    ddn = float(-np.sum(np.clip(np.diff(elev_arr), None, 0))) if elev_arr.size >= 2 else 0.0

    # Certains FIT (ou certains capteurs) n'écrivent pas le champ "distance" :
    # on la recalcule alors depuis les coordonnées plutôt que de renvoyer 0 m.
    dist_max = float(df["dist"].max())
    if dist_max <= 1.0 and len(records) >= 2:
        dist_max = sum(haversine_m(records[i-1][0], records[i-1][1], records[i][0], records[i][1])
                       for i in range(1, len(records)))

    return {"points": [{"lat": r[0], "lon": r[1], "elev": r[2], "dist": r[3], "time": t}
                       for r, t in zip(records, times_pts)],
            "distance": dist_max, "D_up": dup, "D_down": ddn,
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT, "avg_wind": avgW, "avg_humidity": avgH,
            "hr_analysis": analyze_hr_v3(hr_records), "parser": engine}

def parse_tcx_ref(file,tz_name=TZ_NAME_DEFAULT):
    try:file.seek(0);root=ET.parse(file).getroot()
    except:return None
    ns={"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    pts,times,elevs,hr_records=[],[],[],[]
    for tp in root.findall(".//tcx:Trackpoint",ns):
        lat=tp.find("tcx:Position/tcx:LatitudeDegrees",ns);lon=tp.find("tcx:Position/tcx:LongitudeDegrees",ns)
        if lat is None or lon is None:continue
        ele=tp.find("tcx:AltitudeMeters",ns);tim=tp.find("tcx:Time",ns)
        hr_el=tp.find(".//tcx:Value",ns)
        elev=float(ele.text) if ele is not None else 0.0
        hr_val=int(hr_el.text) if hr_el is not None else None
        try:t=datetime.fromisoformat(tim.text.replace("Z","+00:00")).replace(tzinfo=None)
        except:t=None
        pts.append(SimplePoint(float(lat.text),float(lon.text),elev,t))
        times.append(t);elevs.append(elev);hr_records.append(hr_val)
    if len(pts)<2:return None
    vt=[t for t in times if t is not None]
    start_dt=vt[0] if vt else datetime.now()-timedelta(days=1)
    end_dt=vt[-1] if len(vt)>1 else start_dt+timedelta(minutes=5)
    avgT,avgW,avgH=get_avg_weather(pts[0].latitude,pts[0].longitude,start_dt,end_dt,tz_name)
    total=sum(pts[i].distance_3d(pts[i-1]) for i in range(1,len(pts)))
    dup,ddn=compute_dplus_dminus(elevs)
    hr_analysis=analyze_hr_v3(hr_records)
    return{"points":pts,"distance":round(total),"D_up":round(dup,1),"D_down":round(ddn,1),
           "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
           "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,"hr_analysis":hr_analysis}

def extract_segment(points,start_td,end_td):
    def get_t(p):return p.get("time") if isinstance(p,dict) else getattr(p,"time",None)
    ts=[get_t(p) for p in points if get_t(p) is not None]
    if len(ts)<2:return points
    t0=min(ts)
    seg=[p for p in points if get_t(p) is not None and t0+start_td<=get_t(p)<=t0+end_td+timedelta(seconds=1)]
    return seg if len(seg)>=2 else points

def load_activity(file):
    if file is None:return None
    fname=file.name.lower();df=None
    if fname.endswith(".fit"):
        if not HAS_FITDECODE:st.error("pip install fitdecode");return None
        rows=[];file.seek(0)
        with fitdecode.FitReader(file) as fit:
            for frame in fit:
                if not isinstance(frame,fitdecode.FitDataMessage):continue
                if frame.name!="record":continue
                def fv(field,default=None):
                    try:return frame.get_value(field)
                    except:return default
                ts=fv("timestamp");hr=fv("heart_rate");spd=fv("speed")
                dist=fv("distance");alt=fv("enhanced_altitude") or fv("altitude")
                # v8.7 : cadence (souvent enregistrée par jambe → normalisée plus tard)
                cad=fv("cadence");frac=fv("fractional_cadence")
                cad_tot=(float(cad)+float(frac or 0.0)) if cad is not None else None
                if ts is None:continue
                rows.append({"timestamp":ts,"heart_rate":hr,"speed_ms":spd,"distance_m":dist,"altitude_m":alt,
                             "cadence_spm":cad_tot})
        if not rows:return None
        df=pd.DataFrame(rows)
        df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df=df.sort_values("timestamp").reset_index(drop=True)
        t0=df["timestamp"].iloc[0];df["elapsed_s"]=(df["timestamp"]-t0).dt.total_seconds()
    elif fname.endswith(".gpx"):
        file.seek(0);gpx=gpxpy.parse(file);rows=[]
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    # v8.7 : FC / cadence éventuellement présentes dans les extensions Garmin
                    _hr_g=_cad_g=None
                    for _ext in (getattr(pt,"extensions",None) or []):
                        for _el in list(_ext.iter()) if hasattr(_ext,"iter") else []:
                            _tag=str(getattr(_el,"tag","")).lower()
                            try:_val=float(_el.text)
                            except:continue
                            if _tag.endswith("hr"):_hr_g=_val
                            elif _tag.endswith("cad"):_cad_g=_val
                    rows.append({"timestamp":pt.time,"heart_rate":_hr_g,"speed_ms":None,"distance_m":None,
                                 "altitude_m":pt.elevation or 0.0,"cadence_spm":_cad_g})
        if not rows:return None
        df=pd.DataFrame(rows);df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df=df.sort_values("timestamp").reset_index(drop=True);t0=df["timestamp"].iloc[0]
        df["elapsed_s"]=(df["timestamp"]-t0).dt.total_seconds()
        lats=[];file.seek(0);gpx2=gpxpy.parse(file)
        for track in gpx2.tracks:
            for seg in track.segments:
                for pt in seg.points:lats.append((pt.latitude,pt.longitude))
        cumd=[0.0]
        for i in range(1,len(lats)):cumd.append(cumd[-1]+haversine_m(lats[i-1][0],lats[i-1][1],lats[i][0],lats[i][1]))
        df["distance_m"]=cumd[:len(df)]
    elif fname.endswith(".tcx"):
        file.seek(0);root=ET.parse(file).getroot()
        ns={"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"};rows=[]
        for tp in root.findall(".//tcx:Trackpoint",ns):
            tim=tp.find("tcx:Time",ns);hr_el=tp.find(".//tcx:Value",ns)
            spd_el=tp.find("tcx:Extensions/tcx:TPX/tcx:Speed",ns) or tp.find(".//tcx:Speed",ns)
            dist_el=tp.find("tcx:DistanceMeters",ns);alt_el=tp.find("tcx:AltitudeMeters",ns)
            # v8.7 : cadence de course (extension Garmin ns3:RunCadence, sinon Cadence)
            cad_el=tp.find(".//{*}RunCadence")
            if cad_el is None:cad_el=tp.find("tcx:Cadence",ns)
            try:ts=datetime.fromisoformat(tim.text.replace("Z","+00:00"))
            except:continue
            rows.append({"timestamp":ts,"heart_rate":int(hr_el.text) if hr_el is not None else None,
                         "speed_ms":float(spd_el.text) if spd_el is not None else None,
                         "distance_m":float(dist_el.text) if dist_el is not None else None,
                         "altitude_m":float(alt_el.text) if alt_el is not None else None,
                         "cadence_spm":float(cad_el.text) if cad_el is not None else None})
        if not rows:return None
        df=pd.DataFrame(rows);df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df=df.sort_values("timestamp").reset_index(drop=True);t0=df["timestamp"].iloc[0]
        df["elapsed_s"]=(df["timestamp"]-t0).dt.total_seconds()
    elif fname.endswith(".csv"):
        file.seek(0);df=pd.read_csv(file);df.columns=[c.lower().strip() for c in df.columns];renames={}
        for c in df.columns:
            if "heart" in c or c in("hr","fc"):renames[c]="heart_rate"
            if "speed" in c and "ms" not in c:renames[c]="speed_ms"
            if "dist" in c:renames[c]="distance_m"
            if "alt" in c or "elev" in c:renames[c]="altitude_m"
            if "time" in c or "elapsed" in c:renames[c]="elapsed_s"
            if "cad" in c:renames[c]="cadence_spm"
        df.rename(columns=renames,inplace=True)
        if "elapsed_s" not in df.columns:df["elapsed_s"]=range(len(df))
    if df is None:return None
    for col in["heart_rate","speed_ms","distance_m","altitude_m","elapsed_s","cadence_spm"]:
        if col not in df.columns:df[col]=None
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["elapsed_s"]).reset_index(drop=True)
    return df if len(df)>=5 else None

def smooth_hr(series,window=None):
    n=len(series)
    if n<5:return series
    if window is None:window=max(3,n//20)
    if window%2==0:window+=1
    return series.rolling(window,center=True,min_periods=1).mean()

def analyze_heart_rate(df):
    col="heart_rate"
    if col not in df.columns:return{}
    hr=df[col].dropna();hr=hr[(hr>=40)&(hr<=220)]
    if len(hr)<10:return{"available":False}
    arr=hr.values;n=len(arr)
    fc_max=float(np.percentile(arr,95));fc_avg=float(np.mean(arr));fc_min=float(np.percentile(arr,5))
    q1,q3=int(n*0.25),int(n*0.75)
    drift_abs=float(np.mean(arr[q3:]))-float(np.mean(arr[:q1]))
    drift_pct=drift_abs/max(1.0,float(np.mean(arr[:q1])))*100.0
    x=np.arange(n,dtype=float);slope,intercept,r,p,_=sp_stats.linregress(x,arr)
    trend_bpm_per_min=slope*60.0
    reliability=("haute" if abs(drift_abs)<5 else "moyenne" if abs(drift_abs)<12 else "basse (dérive élevée)")
    return{"available":True,"fc_max":round(fc_max),"fc_avg":round(fc_avg,1),"fc_min":round(fc_min),
           "drift_abs":round(drift_abs,1),"drift_pct":round(drift_pct,2),"trend_bpm_per_min":round(trend_bpm_per_min,2),
           "r_value":round(r,3),"reliability":reliability,"seuil_estime":round(fc_max*0.88)}

def analyze_speed_kinetics(df):
    col="speed_ms"
    if col not in df.columns:return{}
    spd=df[col].dropna();spd=spd[spd>0]
    if len(spd)<10:return{"available":False}
    arr=spd.values;x=np.arange(len(arr),dtype=float)
    slope,intercept,r,p,_=sp_stats.linregress(x,arr)
    return{"available":True,"speed_avg_ms":round(float(np.mean(arr)),3),
           "speed_max_ms":round(float(np.percentile(arr,95)),3),"slope":round(slope,5),"r_value":round(r,3)}

# ══════════════════════════════════════════════════════════════
# v8.7 — ANALYSE TRAIL DE SÉANCE (onglet ⚙️ Analyse entraînement)
#   A. Biomécanique course / marche : nuage vitesse × pente coloré par la
#      cadence, avec détection de la ZONE DE TRANSITION (bande de pentes où
#      l'on bascule de la course à la marche) et répartition du temps.
#   B. Évolution par terrain : découpe la sortie en portions Montée /
#      Relief roulant / Descente, calcule la VAP (Vitesse Ajustée à la Pente,
#      modèle Minetti déjà utilisé par l'app) de chaque portion, et l'exprime
#      en % de la PREMIÈRE portion qualifiée de la même famille → mesure la
#      perte de performance au fil de la sortie, gradient par gradient.
# ══════════════════════════════════════════════════════════════

def vap_cost_ratio(grade_pct):
    """Rapport de coût énergétique pente / plat (Minetti). Sert à convertir une
    vitesse réelle en VAP : v_plat_équivalente = v_réelle × ratio."""
    return float(minetti_cost(float(grade_pct) / 100.0) / minetti_cost(0.0))

def build_trail_samples(df, window_s=20.0, grid_step_s=2.0,
                        alt_smooth_s=30.0, pause_speed_kmh=1.0):
    """Ré-échantillonne la séance sur une grille temporelle régulière et calcule,
    sur une fenêtre glissante, la vitesse réelle, la pente instantanée, la
    cadence et la FC. Les pauses (vitesse ~nulle : ravito, arrêt photo) sont
    marquées pour être exclues des analyses.
    Retourne (DataFrame, infos) ou (None, {"error": ...})."""
    if df is None or "elapsed_s" not in df.columns:
        return None, {"error": "Pas de base temps exploitable."}
    d = df.dropna(subset=["elapsed_s"]).sort_values("elapsed_s").reset_index(drop=True)
    if len(d) < 20:
        return None, {"error": "Trop peu de points dans le fichier."}
    t = d["elapsed_s"].astype(float).values
    t_max = float(t[-1])
    if t_max < 300:
        return None, {"error": "Séance trop courte (< 5 min) pour cette analyse."}

    # --- distance cumulée : champ distance sinon intégration de la vitesse ---
    dist = d["distance_m"].astype(float).values if "distance_m" in d.columns else np.full(len(d), np.nan)
    if np.isnan(dist).all() or np.nanmax(dist) <= 1.0:
        if "speed_ms" in d.columns and d["speed_ms"].notna().any():
            spd_fill = pd.Series(d["speed_ms"].astype(float)).interpolate().fillna(0.0).values
            dist = np.concatenate([[0.0], np.cumsum(spd_fill[1:] * np.diff(t))])
        else:
            return None, {"error": "Ni distance ni vitesse dans ce fichier — impossible de calculer l'allure."}
    dist = pd.Series(dist).interpolate(limit_direction="both").fillna(0.0).values
    dist = np.maximum.accumulate(dist)  # distance monotone (anti-glitch capteur)

    if "altitude_m" in d.columns and d["altitude_m"].notna().any():
        alt = pd.Series(d["altitude_m"].astype(float)).interpolate(limit_direction="both").fillna(0.0).values
    else:
        return None, {"error": "Pas d'altitude dans ce fichier — analyse de pente impossible."}

    grid = np.arange(0.0, t_max, float(grid_step_s))
    if len(grid) < 30:
        return None, {"error": "Séance trop courte pour cette analyse."}
    dist_g = np.interp(grid, t, dist)
    alt_g = np.interp(grid, t, alt)
    # lissage altitude (médiane puis moyenne) : le bruit baro/GPS crée des pentes fantômes
    w_alt = max(3, int(round(float(alt_smooth_s) / float(grid_step_s))))
    if w_alt % 2 == 0:
        w_alt += 1
    alt_s = pd.Series(alt_g).rolling(w_alt, center=True, min_periods=1).median()
    alt_s = alt_s.rolling(w_alt, center=True, min_periods=1).mean().values

    half = max(1, int(round(float(window_s) / float(grid_step_s) / 2.0)))
    n = len(grid)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    dd = dist_g[i1] - dist_g[i0]
    dt = grid[i1] - grid[i0]
    de = alt_s[i1] - alt_s[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_ms = np.where(dt > 0, dd / np.maximum(dt, 1e-9), 0.0)
        grade = np.where(dd > 2.0, de / np.maximum(dd, 1e-9) * 100.0, 0.0)
    grade = np.clip(np.nan_to_num(grade), -45.0, 45.0)
    speed_kmh = np.clip(np.nan_to_num(speed_ms) * 3.6, 0.0, 35.0)

    out = pd.DataFrame({"t_s": grid, "speed_kmh": speed_kmh, "grade_pct": grade,
                        "dist_m": dist_g, "alt_m": alt_s})

    # --- cadence (spm) ---
    cad_available = False
    cad_doubled = False
    if "cadence_spm" in d.columns and d["cadence_spm"].notna().any():
        cad_raw = pd.to_numeric(d["cadence_spm"], errors="coerce").values
        valid = cad_raw[(~np.isnan(cad_raw)) & (cad_raw > 20)]
        if len(valid) >= max(20, 0.20 * len(d)):
            cad_g = np.interp(grid, t, pd.Series(cad_raw).interpolate(limit_direction="both").fillna(0.0).values)
            med = float(np.median(valid))
            if med < 110.0:          # cadence par jambe (rpm) → pas/min
                cad_g = cad_g * 2.0
                cad_doubled = True
            w_c = max(3, int(round(10.0 / float(grid_step_s))))
            out["cadence_spm"] = pd.Series(cad_g).rolling(w_c, center=True, min_periods=1).median().values
            cad_available = True
    if not cad_available:
        out["cadence_spm"] = np.nan

    if "heart_rate" in d.columns and d["heart_rate"].notna().any():
        hr_raw = pd.to_numeric(d["heart_rate"], errors="coerce")
        out["hr"] = np.interp(grid, t, hr_raw.interpolate(limit_direction="both").fillna(0.0).values)
    else:
        out["hr"] = np.nan

    out["moving"] = out["speed_kmh"] >= float(pause_speed_kmh)
    infos = {"cadence_available": cad_available, "cadence_doubled": cad_doubled,
             "grid_step_s": float(grid_step_s), "window_s": float(window_s),
             "n_samples": int(len(out)), "moving_s": float(out["moving"].sum() * grid_step_s),
             "total_s": float(t_max), "dist_km": float(dist_g[-1] / 1000.0)}
    return out, infos

def classify_walk_run(samples, infos, cadence_threshold=135.0, speed_walk_kmh=7.0,
                      run_speed_override_kmh=13.0):
    """Marque chaque échantillon comme marche probable ou course.
    Méthode 1 (fiable) : cadence < seuil → marche (la cadence chute nettement
    au passage en marche, même en forte pente).
    Méthode 2 (repli, sans cadence) : vitesse < seuil → marche probable."""
    s = samples.copy()
    if infos.get("cadence_available") and s["cadence_spm"].notna().any():
        walk = s["cadence_spm"] < float(cadence_threshold)
        walk &= s["speed_kmh"] < float(run_speed_override_kmh)   # cadence manquante/erronée à vitesse élevée
        method = "cadence"
    else:
        walk = s["speed_kmh"] < float(speed_walk_kmh)
        method = "vitesse"
    s["is_walk"] = walk & s["moving"]
    s["is_run"] = (~walk) & s["moving"]
    return s, method

def fit_walk_logistic(grades, is_walk):
    """Ajuste P(marche) = 1 / (1 + exp(-(a + b·pente)) sur les échantillons en
    montée. Une régression logistique lisse la relation et évite qu'une tranche
    de pente peu peuplée (peu d'échantillons, donc bruitée) ne fasse sauter la
    zone de transition de plusieurs points de pente d'une sortie à l'autre."""
    g = np.asarray(grades, dtype=float)
    y = np.asarray(is_walk, dtype=float)
    if len(g) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        return None
    def resid(p):
        z = np.clip(p[0] + p[1] * g, -30, 30)
        return 1.0 / (1.0 + np.exp(-z)) - y
    try:
        res = least_squares(resid, [-3.0, 0.2], max_nfev=2000)
    except Exception:
        return None
    a, b = float(res.x[0]), float(res.x[1])
    if not np.isfinite(a) or not np.isfinite(b) or b <= 1e-4:
        return None
    return a, b

def _grade_at_p(fit, p):
    if fit is None:
        return None
    a, b = fit
    return float((math.log(p / (1.0 - p)) - a) / b)

def detect_transition_zone(samples, bin_width=1.0, p_lo=0.25, p_hi=0.75, min_pts_per_bin=15):
    """Zone de pentes où l'athlète bascule de la course à la marche.
    Deux lectures complémentaires :
      • la courbe brute : % de temps passé en marche par tranche de pente ;
      • une régression logistique sur les pentes positives, dont on tire les
        pentes correspondant à 25 %, 50 % et 75 % de marche → bornes de la zone.
    Compare aussi le point de bascule (50 %) du 1er tiers et du dernier tiers de
    la sortie : son glissement vers des pentes plus faibles est un marqueur de
    fatigue (on se met à marcher de plus en plus tôt)."""
    s = samples[samples["moving"]].copy()
    if s.empty:
        return None
    bins = np.arange(np.floor(s["grade_pct"].min()), np.ceil(s["grade_pct"].max()) + bin_width, bin_width)
    if len(bins) < 4:
        return None
    s["bin"] = pd.cut(s["grade_pct"], bins, labels=False, include_lowest=True)
    grp = s.groupby("bin").agg(p_walk=("is_walk", "mean"), n=("is_walk", "size"),
                               grade=("grade_pct", "mean"), speed=("speed_kmh", "median")).reset_index()
    min_n = max(int(min_pts_per_bin), int(0.004 * len(s)))
    grp = grp[grp["n"] >= min_n].sort_values("grade").reset_index(drop=True)
    if len(grp) < 4:
        return None
    grp["p_walk_smooth"] = grp["p_walk"].rolling(3, center=True, min_periods=1).mean()

    up = s[s["grade_pct"] >= 0.0]
    fit = fit_walk_logistic(up["grade_pct"].values, up["is_walk"].values)
    lo, mid, hi = (_grade_at_p(fit, p_lo), _grade_at_p(fit, 0.5), _grade_at_p(fit, p_hi))
    g_max = float(s["grade_pct"].max())
    extrapolated = bool(hi is not None and hi > g_max)

    def _mid_supported(sub, m, half=3.0, min_n=25, min_share=0.12):
        """Le point de bascule n'a de sens que si l'on a VRAIMENT couru ET marché
        autour de cette pente. Si le parcours n'offre que du plat couru et des
        raidillons marchés (rien entre les deux), n'importe quelle valeur
        intermédiaire ajuste les données aussi bien : on préfère alors ne rien
        annoncer plutôt qu'un chiffre non contraint par la donnée."""
        if m is None:
            return False
        near = sub[(sub["grade_pct"] >= m - half) & (sub["grade_pct"] <= m + half)]
        if len(near) < min_n:
            return False
        share = float(near["is_walk"].mean())
        return min_share <= share <= (1.0 - min_share)

    mid_supported = _mid_supported(up, mid)

    # glissement du point de bascule entre le début et la fin de la sortie
    mid_start = mid_end = None
    if fit is not None and len(up) >= 400:
        t_split = up["t_s"].quantile([0.333, 0.667]).values
        first = up[up["t_s"] <= t_split[0]]
        last = up[up["t_s"] >= t_split[1]]
        def _mid_of(sub):
            """Point de bascule d'un tiers de sortie, borné à la plage de pentes
            réellement rencontrée dans ce tiers et validé par le test de support."""
            f = fit_walk_logistic(sub["grade_pct"].values, sub["is_walk"].values)
            m = _grade_at_p(f, 0.5)
            if m is None:
                return None
            g_min = float(sub["grade_pct"].quantile(0.02)); g_mx = float(sub["grade_pct"].quantile(0.98))
            m = float(min(max(m, g_min), g_mx))
            return m if _mid_supported(sub, m, min_n=15) else None
        mid_start = _mid_of(first)
        mid_end = _mid_of(last)

    return {"bins": grp, "slope_lo": lo, "slope_mid": mid, "slope_hi": hi,
            "p_lo": p_lo, "p_hi": p_hi, "fit": fit, "extrapolated": extrapolated,
            "grade_max": g_max, "mid_start": mid_start, "mid_end": mid_end,
            "mid_supported": bool(mid_supported)}

def walk_run_summary(samples, grid_step_s):
    """Répartition du temps en mouvement entre course et marche probable."""
    s = samples[samples["moving"]]
    n = len(s)
    if n == 0:
        return None
    walk_s = float(s["is_walk"].sum() * grid_step_s)
    run_s = float(s["is_run"].sum() * grid_step_s)
    tot = max(1e-6, walk_s + run_s)
    d_walk = float(np.sum(np.diff(np.concatenate([[0], s["dist_m"].values]))[s["is_walk"].values]))
    return {"walk_s": walk_s, "run_s": run_s, "pct_walk": walk_s / tot * 100.0,
            "pct_run": run_s / tot * 100.0,
            "walk_speed_med": float(s.loc[s["is_walk"], "speed_kmh"].median()) if s["is_walk"].any() else None,
            "run_speed_med": float(s.loc[s["is_run"], "speed_kmh"].median()) if s["is_run"].any() else None,
            "walk_grade_med": float(s.loc[s["is_walk"], "grade_pct"].median()) if s["is_walk"].any() else None}

def plot_biomecanique(samples, transition, summary, infos, method,
                      cadence_threshold=135.0, speed_walk_kmh=7.0):
    """Deux panneaux partageant l'axe des pentes (jamais deux échelles sur un même
    axe) : en haut le nuage vitesse réelle × pente coloré par la cadence, en bas
    la probabilité de marche et la zone de transition."""
    s = samples[samples["moving"]]
    if len(s) > 6000:                       # allège le rendu sur les grosses sorties
        s = s.sample(6000, random_state=0).sort_values("t_s")
    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(11.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.12})

    lo = hi = mid = None
    if transition is not None:
        lo, hi, mid = transition.get("slope_lo"), transition.get("slope_hi"), transition.get("slope_mid")

    # ── panneau haut : nuage vitesse × pente ──────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "cad", [C_RED, "#C4707A", C_GREY, C_WHITE, "#FFFFFF"])
    if infos.get("cadence_available"):
        c = s["cadence_spm"].values
        vmin = float(np.nanpercentile(c, 2)); vmax = float(np.nanpercentile(c, 98))
        sc = ax.scatter(s["grade_pct"], s["speed_kmh"], c=c, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=7, alpha=0.65, linewidths=0)
        cb = fig.colorbar(sc, ax=[ax, axp], pad=0.015, fraction=0.028, aspect=34)
        cb.set_label("Cadence (pas/min)", fontsize=8, color=C_TEXT_MUT)
        cb.ax.tick_params(labelsize=7, colors=C_TEXT_MUT)
        cb.outline.set_edgecolor(C_LINE)
    else:
        ax.scatter(s.loc[s["is_run"], "grade_pct"], s.loc[s["is_run"], "speed_kmh"],
                   c=C_WHITE, s=7, alpha=0.55, linewidths=0, label="Course")
        ax.scatter(s.loc[s["is_walk"], "grade_pct"], s.loc[s["is_walk"], "speed_kmh"],
                   c=C_RED, s=7, alpha=0.55, linewidths=0, label="Marche probable")

    if transition is not None:
        gb = transition["bins"]
        ax.plot(gb["grade"], gb["speed"], color=C_SURFACE, lw=4.0, zorder=4)      # liseré de fond
        ax.plot(gb["grade"], gb["speed"], color=C_TEXT, lw=2.0, zorder=5,
                label="Vitesse médiane par pente")
    ax.axvline(0, color=C_LINE, lw=1.0)
    ax.set_ylabel("Vitesse réelle (km/h)")
    ax.set_ylim(bottom=0)
    chart_title(ax, "Biomécanique course / marche",
                "Un point toutes les 2 s, hors pauses · couleur = cadence · la zone rouge est la bascule vers la marche")
    if not infos.get("cadence_available"):
        ax.legend(loc="lower left", markerscale=2.4)
    elif transition is not None:
        ax.legend(loc="lower left")

    # ── panneau bas : probabilité de marche ───────────────────────────────
    if transition is not None and transition.get("fit") is not None:
        a, b = transition["fit"]
        gg = np.linspace(0, max(1.0, transition.get("grade_max", 30.0)), 160)
        pw = 100.0 / (1.0 + np.exp(-(a + b * gg)))
        axp.fill_between(gg, 0, pw, color=C_RED, alpha=0.18, linewidth=0)
        axp.plot(gg, pw, color=C_RED, lw=2.0)
    gbins = transition["bins"] if transition is not None else None
    if gbins is not None:
        axp.plot(gbins["grade"], gbins["p_walk"] * 100.0, ls="none", marker="o", ms=3.5,
                 color=C_TEXT_MUT, alpha=0.75, label="Mesuré par tranche de pente")
        axp.legend(loc="upper left", fontsize=7.5, borderpad=0.4)
    axp.axhline(50, color=C_LINE, lw=1.0, ls=":")
    axp.set_ylim(0, 105); axp.set_yticks([0, 50, 100])
    axp.set_yticklabels(["0 %", "50 %", "100 %"])
    axp.set_ylabel("Temps passé\nen marche")
    axp.set_xlabel("Pente (%)")

    # ── zone de transition, reportée sur les deux panneaux ────────────────
    if lo is not None and hi is not None and hi > lo:
        x_lo, x_hi = ax.get_xlim()
        for _a in (ax, axp):
            _a.axvspan(max(lo, x_lo), min(hi, x_hi), color=C_RED, alpha=0.10, zorder=0)
            for xv in (lo, hi):
                if x_lo <= xv <= x_hi:
                    _a.axvline(xv, color=C_RED, lw=1.1, ls="--", alpha=0.75, zorder=2)
        y_top = ax.get_ylim()[1]
        x_mid = min(max((lo + hi) / 2, x_lo + 2), x_hi - 2)
        ax.annotate("ZONE DE TRANSITION", xy=(x_mid, y_top * 0.985), ha="center", va="top",
                    fontsize=8, color=C_RED, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc=C_SURFACE, ec="none"))
        ax.annotate(f"{lo:.0f} %  →  {hi:.0f} %", xy=(x_mid, y_top * 0.90), ha="center", va="top",
                    fontsize=8.5, color=C_TEXT)
        if mid is not None:
            axp.annotate(f"bascule\n{mid:.0f} %", xy=(mid, 50), xytext=(0, 14),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=7.5, color=C_TEXT)
        ax.annotate("je cours", xy=(x_lo + (lo - x_lo) * 0.45, ax.get_ylim()[1] * 0.10),
                    ha="center", fontsize=8.5, color=C_TEXT_MUT)
        if hi < x_hi:
            ax.annotate("je marche", xy=(hi + (x_hi - hi) * 0.5, ax.get_ylim()[1] * 0.10),
                        ha="center", fontsize=8.5, color=C_TEXT_MUT)
    return fig

# ── B. Évolution par terrain (VAP indexée sur la 1re portion de chaque famille) ──

TERRAIN_FAMILIES = [("Montée", C_RED, "M"), ("Relief roulant", C_GREY, "P"), ("Descente", C_WHITE, "D")]

def segment_terrain_portions(samples, up_thr=4.0, down_thr=-4.0,
                             min_dur_s=120.0, min_dist_m=300.0, class_smooth_s=60.0):
    """Découpe la sortie en portions homogènes de terrain et calcule la VAP de
    chacune. Le temps de référence est le temps en MOUVEMENT (pauses exclues),
    pour que deux sorties avec des ravitos différents restent comparables."""
    s = samples.copy()
    step = float(s["t_s"].iloc[1] - s["t_s"].iloc[0]) if len(s) > 1 else 1.0
    cls = np.where(s["grade_pct"] >= up_thr, 1, np.where(s["grade_pct"] <= down_thr, -1, 0)).astype(float)
    w = max(3, int(round(float(class_smooth_s) / step)))
    if w % 2 == 0:
        w += 1
    cls = pd.Series(cls).rolling(w, center=True, min_periods=1).median().round().fillna(0).values
    s["cls"] = cls
    s["moving_s_cum"] = np.cumsum(np.where(s["moving"], step, 0.0))

    portions = []
    _b = np.flatnonzero(np.diff(cls) != 0) + 1
    _starts = np.concatenate([[0], _b]).astype(int)
    _ends = np.concatenate([_b, [len(cls)]]).astype(int)
    for start, i in zip(_starts, _ends):
        seg = s.iloc[start:i]
        seg_mov = seg[seg["moving"]]
        dur = float(len(seg_mov) * step)
        dist = float(seg["dist_m"].iloc[-1] - seg["dist_m"].iloc[0])
        if dur >= float(min_dur_s) and dist >= float(min_dist_m) and len(seg_mov) >= 5:
            ratios = np.array([vap_cost_ratio(g) for g in seg_mov["grade_pct"].values])
            vap_kmh = float(np.mean(seg_mov["speed_kmh"].values * ratios))
            d_elev = np.diff(seg["alt_m"].values)
            fam = {1: "Montée", -1: "Descente", 0: "Relief roulant"}[int(seg["cls"].iloc[0])]
            portions.append({
                "famille": fam,
                "t_start_s": float(seg["moving_s_cum"].iloc[0]),
                "t_mid_s": float((seg_mov["moving_s_cum"].iloc[0] + seg_mov["moving_s_cum"].iloc[-1]) / 2.0),
                "t_end_s": float(seg_mov["moving_s_cum"].iloc[-1]),
                "dur_s": dur, "dist_m": dist,
                "d_plus": float(np.sum(np.clip(d_elev, 0, None))),
                "d_moins": float(-np.sum(np.clip(d_elev, None, 0))),
                "grade_med": float(seg_mov["grade_pct"].median()),
                "speed_kmh": float(dist / max(1.0, dur) * 3.6),
                "vap_kmh": vap_kmh,
                "hr": float(seg_mov["hr"].mean()) if seg_mov["hr"].notna().any() else None,
                "pct_walk": float(seg_mov["is_walk"].mean() * 100.0) if "is_walk" in seg_mov.columns else None,
            })
    # indexation : la 1re portion qualifiée de chaque famille = 100 %
    refs = {}
    for p in portions:
        if p["famille"] not in refs:
            refs[p["famille"]] = p["vap_kmh"]
        p["vap_ref_kmh"] = refs[p["famille"]]
        p["index_pct"] = p["vap_kmh"] / max(1e-6, refs[p["famille"]]) * 100.0
    counters = {}
    for p in portions:
        pref = {"Montée": "M", "Relief roulant": "P", "Descente": "D"}[p["famille"]]
        counters[pref] = counters.get(pref, 0) + 1
        p["label"] = f"{pref}{counters[pref]}"
    return portions

def terrain_trends(portions):
    """Pente de dégradation (points d'index perdus par heure) et dernière portion
    comparable, par famille de terrain."""
    out = {}
    for fam, color, pref in TERRAIN_FAMILIES:
        fam_p = [p for p in portions if p["famille"] == fam]
        if len(fam_p) < 2:
            out[fam] = {"n": len(fam_p), "slope_pts_per_h": None,
                        "last_pct": fam_p[-1]["index_pct"] if fam_p else None,
                        "last_label": fam_p[-1]["label"] if fam_p else None, "r2": None,
                        "vap_ref": fam_p[0]["vap_kmh"] if fam_p else None}
            continue
        x = np.array([p["t_mid_s"] / 3600.0 for p in fam_p])
        y = np.array([p["index_pct"] for p in fam_p])
        slope, intercept, r, _, _ = sp_stats.linregress(x, y)
        out[fam] = {"n": len(fam_p), "slope_pts_per_h": float(slope), "r2": float(r ** 2),
                    "last_pct": float(y[-1]), "last_label": fam_p[-1]["label"],
                    "vap_ref": float(fam_p[0]["vap_kmh"]), "intercept": float(intercept)}
    return out

def plot_terrain_evolution(portions, trends):
    """Index de VAP (% de la 1re portion de la même famille) au fil de la sortie.
    Étiquettes sélectives : uniquement la dernière portion de chaque famille — le
    détail complet est dans le tableau sous le graphique."""
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ends = []
    for fam, color, pref in TERRAIN_FAMILIES:
        fam_p = [p for p in portions if p["famille"] == fam]
        if not fam_p:
            continue
        x = [p["t_mid_s"] / 3600.0 for p in fam_p]
        y = [p["index_pct"] for p in fam_p]
        tr = trends.get(fam, {})
        if tr.get("slope_pts_per_h") is not None and len(fam_p) >= 3:
            xs = np.linspace(min(x), max(x), 20)
            ax.plot(xs, tr["intercept"] + tr["slope_pts_per_h"] * xs, ls="--", lw=1.1,
                    color=color, alpha=0.45, zorder=3)
        ax.plot(x, y, "-", color=color, lw=2.0, zorder=4, label=f"{fam} ({len(fam_p)} portions)")
        ax.plot(x, y, "o", color=color, ms=7, mec=C_SURFACE, mew=1.6, zorder=5)
        if len(fam_p) > 1:
            ends.append((x[-1], y[-1], f"{fam_p[-1]['label']} · {y[-1]:.0f} %", color))

    y_vals = [p["index_pct"] for p in portions]
    y_min = min(y_vals + [100.0]); y_max = max(y_vals + [100.0])
    span = max(6.0, y_max - y_min)
    y0, y1 = y_min - span * 0.22, y_max + span * 0.18
    ax.set_ylim(y0, y1)
    ax.axhspan(y0, 100, color=C_RED, alpha=0.05, zorder=0)          # sous la référence
    ax.axhline(100, color=C_TEXT_MUT, lw=1.2, ls=":", zorder=2)
    ax.annotate("niveau de référence — 1re portion de chaque terrain = 100 %",
                xy=(0.995, 100), xycoords=("axes fraction", "data"), xytext=(0, 7),
                textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=C_TEXT_MUT)

    # étiquettes de fin, écartées verticalement pour rester lisibles
    x_max = max([p["t_mid_s"] / 3600.0 for p in portions])
    ax.set_xlim(left=-0.05, right=x_max * 1.20 + 0.05)
    ends.sort(key=lambda e: e[1], reverse=True)
    offsets = [18, 0, -18][:len(ends)] if len(ends) > 1 else [0]
    for (xe, ye, txt, col), dy in zip(ends, offsets):
        ax.annotate(txt, xy=(xe, ye), xytext=(12, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=col, fontweight="600",
                    bbox=dict(boxstyle="round,pad=0.28", fc=C_SURFACE, ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=col, alpha=0.35, lw=0.9,
                                    shrinkA=2, shrinkB=4))
    ax.set_xlabel("Temps d'effort depuis le départ (h, pauses exclues)")
    ax.set_ylabel("VAP en % de la 1re portion")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.0f} %"))
    chart_title(ax, "Tenue de la performance par type de terrain",
                "VAP = vitesse ramenée à son équivalent sur le plat · chaque terrain comparé à sa propre première portion")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    return fig

def compute_vc(distances_m,durations_s):
    T=np.array(durations_s,dtype=float);D=np.array(distances_m,dtype=float)
    if len(T)<2:return None,None,None
    slope,intercept,r,p,se=sp_stats.linregress(T,D)
    return float(slope),float(intercept),float(r**2)

# ── v8.2 PATCH : détection du point de rupture sur un test à effort maximal ──
def compute_pace_series_from_points(points, window_s=20.0, grid_step_s=5.0):
    """À partir d'une liste de points (dict issus de parse_fit_ref, ou objets
    SimplePoint issus de parse_tcx_ref — les deux formats coexistent dans
    l'app), construit une série temps/allure lissée sur fenêtre glissante.
    Retourne (None, None) si la trace est trop courte ou sans temps exploitable."""
    def _t(p): return p.get("time") if isinstance(p, dict) else getattr(p, "time", None)
    def _lat(p): return p.get("lat") if isinstance(p, dict) else getattr(p, "latitude", None)
    def _lon(p): return p.get("lon") if isinstance(p, dict) else getattr(p, "longitude", None)
    def _dist(p):
        if not isinstance(p, dict): return None
        v = p.get("dist")
        return safe_float(v) if v is not None else None

    pts = [p for p in (points or []) if _t(p) is not None]
    if len(pts) < 10:
        return None, None
    t0 = _t(pts[0])
    try:
        times_s = np.array([(_t(p) - t0).total_seconds() for p in pts])
    except Exception:
        return None, None

    raw_dists = [_dist(p) for p in pts]
    if all(d is not None for d in raw_dists) and raw_dists[-1] and raw_dists[-1] > 1.0:
        dists_m = np.array([d if d is not None else 0.0 for d in raw_dists])
    else:
        d = [0.0]
        for i in range(1, len(pts)):
            d.append(d[-1] + haversine_m(_lat(pts[i-1]), _lon(pts[i-1]), _lat(pts[i]), _lon(pts[i])))
        dists_m = np.array(d)

    order = np.argsort(times_s)
    times_s = times_s[order]; dists_m = dists_m[order]
    t_max = times_s[-1]
    if t_max < 60:
        return None, None
    grid = np.arange(0, t_max, grid_step_s)
    dist_grid = np.interp(grid, times_s, dists_m)
    half = max(1, int(window_s / grid_step_s / 2))
    pace = np.full(len(grid), np.nan)
    for i in range(len(grid)):
        i0, i1 = max(0, i - half), min(len(grid) - 1, i + half)
        dd = dist_grid[i1] - dist_grid[i0]; dt = grid[i1] - grid[i0]
        if dd > 1.0 and dt > 0:
            pace[i] = dt / (dd / 1000.0)
    valid = ~np.isnan(pace)
    if valid.sum() < 10:
        return None, None
    return grid[valid], pace[valid]

def detect_pace_breakpoint(t_array_s, pace_array_s_per_km):
    """Cherche le point unique qui sépare le test en deux régimes d'allure
    les plus distincts possibles (minimise la somme des variances intra-
    segment — détection de rupture classique à un seul point de cassure).
    Retourne le moment, le % de la durée totale, les allures avant/après,
    le % de dégradation et un R² indiquant la qualité de la séparation."""
    n = len(pace_array_s_per_km)
    if n < 8:
        return None
    total_var = float(np.var(pace_array_s_per_km)) * n
    if total_var <= 0:
        return None
    best_idx = None; best_explained = -1.0
    margin = max(2, n // 10)
    for idx in range(margin, n - margin):
        seg1 = pace_array_s_per_km[:idx]; seg2 = pace_array_s_per_km[idx:]
        ss = float(np.var(seg1)) * len(seg1) + float(np.var(seg2)) * len(seg2)
        explained = total_var - ss
        if explained > best_explained:
            best_explained = explained; best_idx = idx
    if best_idx is None:
        return None
    t_break = float(t_array_s[best_idx])
    pct_break = t_break / float(t_array_s[-1]) * 100.0
    pace_before = float(np.mean(pace_array_s_per_km[:best_idx]))
    pace_after = float(np.mean(pace_array_s_per_km[best_idx:]))
    drop_pct = (pace_after - pace_before) / max(1e-6, pace_before) * 100.0
    r2_break = best_explained / total_var
    return {"t_break_s": t_break, "pct_break": pct_break, "pace_before": pace_before,
            "pace_after": pace_after, "drop_pct": drop_pct, "r2": r2_break}
# ───────────────────────────────────────────────────────────────────────────

def build_holding_table(vc_ms,d_prime,refs_fit,K_riegel):
    if vc_ms is None or vc_ms<=0:return pd.DataFrame()
    X,Y=[],[]
    for r in refs_fit:
        d_m=float(r.get("distance",0));t_s=float(r.get("temps",0))
        if d_m>0 and t_s>0:X.append(math.log(d_m/1000.0));Y.append(math.log(t_s))
    if len(X)>=2:
        a_fit,K_fit=clamped_loglog_fit(X,Y)
    elif len(X)==1:K_fit=float(K_riegel);a_fit=math.exp(Y[0])/math.exp(X[0])
    else:K_fit=float(K_riegel);a_fit=240.0
    pct_steps=[pct/100.0 for pct in range(80,121,2)];rows=[]
    for pct in pct_steps:
        v=vc_ms*pct
        if v<=0:continue
        pct_label=f"{round(pct*100):.0f} %";pace=pace_str(1000.0/v)
        if pct<1.0:
            if abs(1.0-K_fit)<1e-6:continue
            try:t=(a_fit*(v/1000.0)**K_fit)**(1.0/(1.0-K_fit))
            except:continue
            modele="Riegel"
        else:
            delta_v=v-vc_ms
            if d_prime is None or delta_v<0.01:continue
            t=d_prime/delta_v;modele="Modèle D'"
        t=max(0.0,min(t,360000.0))
        if t>0:rows.append({"% VC":pct_label,"Vitesse (m/s)":round(v,2),"Allure (/km)":pace,
                             "Temps de maintien":seconds_to_hms(t),"Durée (min)":round(t/60.0,1),"Modèle":modele})
    return pd.DataFrame(rows)

def parse_zonex_csv(file):
    try:
        file.seek(0);df=pd.read_csv(file);df.columns=[c.strip() for c in df.columns];renames={}
        for c in df.columns:
            cl=c.lower().replace(" ","").replace("(","").replace(")","")
            if cl.startswith("vin"):renames[c]="VIn"
            elif cl.startswith("vel/min") or cl=="vel/min":renames[c]="VE"
            elif "vel/breath" in cl:renames[c]="VE_per_breath"
            elif cl=="velmin" or cl=="vel":renames[c]="VE"
            elif cl=="rhythm":renames[c]="RR"
            elif cl=="eqco2":renames[c]="eqCO2"
            elif cl=="vco2":renames[c]="VCO2"
            elif cl=="feco2":renames[c]="FeCO2"
            elif cl=="hr":renames[c]="HR"
            elif cl=="power":renames[c]="Power"
            elif cl=="cadence":renames[c]="Cadence"
            elif cl=="event":renames[c]="palier"
            elif cl=="aux1":renames[c]="aux1"      # v9.4 : dérivé de la ventilation, PAS un équivalent O₂
            elif cl=="aux2":renames[c]="aux2"
            elif cl in("eqo2","veo2","ve/vo2"):renames[c]="eqO2_raw"
            elif cl in("vo2","vo2l/min","vo2lmin"):renames[c]="VO2_Lmin"
        df.rename(columns=renames,inplace=True)
        if "VE" not in df.columns:
            for c in df.columns:
                if "VE" in c and "L/min" in c:df.rename(columns={c:"VE"},inplace=True);break
        # v9.3 — un masque qui ne mesure QUE le CO₂ est le cas normal : seuls
        # le temps, la ventilation, le CO₂ et le n° de palier sont exigés.
        # L'oxygène, s'il est présent, est un bonus.
        df["timestamp"]=pd.to_numeric(df["timestamp"],errors="coerce") if "timestamp" in df.columns else None
        if "timestamp" not in df.columns or df["timestamp"] is None:return None
        for col in["VE","VCO2","eqO2_raw","HR","Cadence","FeCO2","eqCO2","VIn"]:
            if col in df.columns:df[col]=pd.to_numeric(df[col],errors="coerce")
        # VCO₂ reconstruit depuis la fraction expirée si la colonne directe manque
        if "VCO2" not in df.columns and "FeCO2" in df.columns and "VE" in df.columns:
            df["VCO2"]=df["VE"]*df["FeCO2"]/100.0
        required=["timestamp","VE","VCO2","palier"]
        missing=[r for r in required if r not in df.columns]
        if missing:return None
        df=df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"]=df["timestamp"]-df["timestamp"].iloc[0];df["elapsed_min"]=df["elapsed_s"]/60.0
        # O₂ : uniquement si une colonne d'oxygène explicite existe (VO2 ou éq. O₂).
        # Les colonnes aux1/aux2 des masques CO₂ suivent la ventilation (r≈0.99 avec VE)
        # et ne sont surtout pas des équivalents O₂ : les prendre pour tels fabriquerait
        # un VO₂ et un RER entièrement faux.
        _has_o2=False
        if "VO2_Lmin" in df.columns and pd.to_numeric(df["VO2_Lmin"],errors="coerce").notna().mean()>0.5:
            df["VO2_Lmin"]=pd.to_numeric(df["VO2_Lmin"],errors="coerce")
            df["RQ"]=df["VCO2"]/df["VO2_Lmin"].replace(0,np.nan)
            df["eqO2"]=df["VE"]/df["VO2_Lmin"].replace(0,np.nan)
            _has_o2=True
        elif "eqO2_raw" in df.columns:
            _eq=pd.to_numeric(df["eqO2_raw"],errors="coerce").replace(0,np.nan)
            if _eq.notna().mean()>0.5 and 12.0<=float(_eq.median())<=70.0:
                df["VO2_Lmin"]=df["VE"]/_eq
                df["RQ"]=df["VCO2"]/df["VO2_Lmin"].replace(0,np.nan)
                df["eqO2"]=_eq
                _has_o2=True
        if not _has_o2:
            # équivalent ventilatoire du CO₂ : c'est LE signal exploitable ici
            df["eqCO2"]=df.get("eqCO2", df["VE"]/df["VCO2"].replace(0,np.nan))
        if "HR" not in df.columns:df["HR"]=np.nan
        # FC figée (capteur non connecté) : une seule valeur sur tout le test → inutilisable
        _hr_ok=bool(pd.to_numeric(df["HR"],errors="coerce").dropna().nunique()>3)
        if not _hr_ok:df["HR"]=np.nan
        # vitesse : la colonne Cadence de ces masques porte la vitesse tapis en km/h,
        # mais elle vaut 0 quand rien n'est connecté
        _speed_ok=False
        if "Cadence" in df.columns:
            _sp=pd.to_numeric(df["Cadence"],errors="coerce")
            _speed_ok=bool(_sp.notna().any() and float(_sp.max())>1.0)
        # ── lissage respiratoire : en breath-by-breath, le CV brut d'un palier atteint
        # 10 % alors que le signal physiologique est stable à ~3 %. Une médiane glissante
        # (~15 s) enlève ce bruit sans déformer la tendance ; les colonnes brutes restent
        # disponibles pour l'affichage.
        _step=float(np.median(np.diff(df["elapsed_s"].values))) if len(df)>3 else 1.0
        _w=int(max(3,round(15.0/max(0.2,_step))))
        if _w%2==0:_w+=1
        for _c in("VE","VCO2","VO2_Lmin"):
            if _c in df.columns:
                df[_c+"_raw"]=df[_c]
                df[_c]=df[_c].rolling(_w,center=True,min_periods=2).median()
        if "VCO2" in df.columns and "VE" in df.columns:
            df["eqCO2"]=df["VE"]/df["VCO2"].replace(0,np.nan)
        if _has_o2:
            df["RQ"]=df["VCO2"]/df["VO2_Lmin"].replace(0,np.nan)
        df["palier"]=pd.to_numeric(df["palier"],errors="coerce").fillna(0).astype(int)
        df.attrs["has_o2"]=_has_o2
        df.attrs["has_hr"]=_hr_ok
        df.attrs["has_speed"]=_speed_ok
        df.attrs["smooth_window_s"]=round(_w*_step)
        return df
    except:return None

def aggregate_by_palier(df):
    cols=["elapsed_min","HR","VE","VO2_Lmin","VCO2","RQ","eqO2","eqCO2","Cadence"]
    cols=[c for c in cols if c in df.columns]   # en mode CO₂ seul, VO2/RQ/eqO2 sont simplement absents
    agg=df.groupby("palier")[cols].mean().reset_index()
    return agg[agg["palier"]>0].sort_values("palier")

def detect_sv1_sv2(df_pal):
    result={"sv1":None,"sv2":None}
    if df_pal.empty or len(df_pal)<3:return result
    pals=df_pal.sort_values("palier").reset_index(drop=True);n=len(pals);sv1_idx=None
    for i in range(n-1):
        if pals.loc[i,"RQ"]>=0.85 and pals.loc[i+1,"RQ"]>=0.85:sv1_idx=i;break
    if "eqCO2" in pals.columns:
        for i in range(1,n-1):
            deqo2=pals.loc[i,"eqO2"]-pals.loc[i-1,"eqO2"];deqco2=pals.loc[i,"eqCO2"]-pals.loc[i-1,"eqCO2"]
            if deqo2>0.5 and deqco2<=0.3:
                vent_sv1=i
                if sv1_idx is None or vent_sv1<sv1_idx:sv1_idx=vent_sv1
                break
    sv2_idx=None
    for i in range(n):
        if pals.loc[i,"RQ"]>=1.00:sv2_idx=i;break
    if "eqCO2" in pals.columns and sv2_idx is not None:
        nadir_idx=pals.loc[:sv2_idx,"eqCO2"].idxmin()
        for i in range(nadir_idx+1,n-1):
            deqco2=pals.loc[i,"eqCO2"]-pals.loc[i-1,"eqCO2"]
            if deqco2>0.3:sv2_idx=min(sv2_idx,i);break
    for key,idx in[("sv1",sv1_idx),("sv2",sv2_idx)]:
        if idx is not None and 0<=idx<n:
            row=pals.iloc[idx]
            result[key]={"palier":int(row["palier"]),"t_min":round(float(row["elapsed_min"]),1),
                         "HR":round(float(row["HR"])),"VO2":round(float(row.get("VO2_Lmin",0)),3),
                         "VCO2":round(float(row.get("VCO2",0)),3),"RQ":round(float(row["RQ"]),3),
                         "eqO2":round(float(row["eqO2"]),1),"eqCO2":round(float(row.get("eqCO2",0)),1),
                         "VE":round(float(row["VE"]),1),"Cadence":round(float(row.get("Cadence",0)),2)}
    return result

PERF_STANDARDS={
    "5 km":{"dist_m":5000,"H":{"Minima Champ. France (A)":"0:16:00","Minima Champ. France (B)":"0:17:00","Minima Champ. Europe":"0:13:25","Minima Champ. Monde":"0:13:16","Record de France":"0:13:07","Record d'Europe":"0:12:48","Record du Monde":"0:12:35"},
             "F":{"Minima Champ. France (A)":"0:20:15","Minima Champ. France (B)":"0:21:15","Minima Champ. Europe":"0:15:20","Minima Champ. Monde":"0:15:10","Record de France":"0:14:53","Record d'Europe":"0:14:19","Record du Monde":"0:14:05"}},
    "10 km":{"dist_m":10000,"H":{"Minima Champ. France":"0:34:15","Minima Champ. Europe":"0:28:15","Minima Champ. Monde":"0:27:50","Record de France":"0:27:13","Record d'Europe":"0:26:46","Record du Monde":"0:26:11"},
              "F":{"Minima Champ. France":"0:43:00","Minima Champ. Europe":"0:32:00","Minima Champ. Monde":"0:31:25","Record de France":"0:30:39","Record d'Europe":"0:29:56","Record du Monde":"0:29:01"}},
    "Semi-marathon":{"dist_m":21097,"H":{"Minima Champ. France":"1:15:30","Minima Champ. Europe":"1:02:30","Minima Champ. Monde":"1:01:30","Record de France":"1:00:01","Record d'Europe":"0:59:26","Record du Monde":"0:57:31"},
                     "F":{"Minima Champ. France":"1:45:00","Minima Champ. Europe":"1:10:30","Minima Champ. Monde":"1:09:00","Record de France":"1:07:34","Record d'Europe":"1:05:46","Record du Monde":"1:02:52"}},
    "Marathon":{"dist_m":42195,"H":{"Minima Champ. France":"2:48:00","Minima Champ. Europe":"2:14:30","Minima Champ. Monde":"2:11:30","Record de France":"2:06:51","Record d'Europe":"2:04:11","Record du Monde":"2:00:35"},
                "F":{"Minima Champ. France":"3:38:00","Minima Champ. Europe":"2:31:00","Minima Champ. Monde":"2:27:00","Record de France":"2:21:49","Record d'Europe":"2:17:01","Record du Monde":"2:11:53"}},
}
STANDARDS_ORDER=["Minima Champ. France","Minima Champ. France (A)","Minima Champ. France (B)","Minima Champ. Europe","Minima Champ. Monde","Record de France","Record d'Europe","Record du Monde"]
STANDARDS_EMOJI={"Minima Champ. France":"🇫🇷","Minima Champ. France (A)":"🇫🇷⭐","Minima Champ. France (B)":"🇫🇷","Minima Champ. Europe":"🇪🇺","Minima Champ. Monde":"🌍","Record de France":"🇫🇷🏅","Record d'Europe":"🇪🇺🏅","Record du Monde":"🌍🏅"}

def riegel_predict(dist_m,a,K):
    if dist_m<=0 or a<=0:return 0.0
    return float(a)*(float(dist_m)/1000.0)**float(K)

def predict_performances(vc_ms,d_prime,refs_fit,K_riegel,genre="H"):
    X,Y=[],[]
    for r in refs_fit:
        d_m=float(r.get("distance",0));t_s=float(r.get("temps",0))
        if d_m>0 and t_s>0:X.append(math.log(d_m/1000.0));Y.append(math.log(t_s))
    if len(X)>=2:
        a_fit,K_fit=clamped_loglog_fit(X,Y)
    elif len(X)==1:K_fit=float(K_riegel);a_fit=math.exp(Y[0])/math.exp(X[0])
    else:K_fit=float(K_riegel);a_fit=240.0
    results={}
    for dist_label,info in PERF_STANDARDS.items():
        dist_m=info["dist_m"];stds=info.get(genre,{})
        t_pred_s=riegel_predict(dist_m,a_fit,K_fit)
        if t_pred_s<=0:continue
        pace_pred=pace_str(t_pred_s/(dist_m/1000.0));std_rows=[]
        for std_name in STANDARDS_ORDER:
            if std_name not in stds:continue
            t_std_s=hms_to_seconds(stds[std_name])
            if t_std_s<=0:continue
            diff_s=t_pred_s-t_std_s;emoji=STANDARDS_EMOJI.get(std_name,"")
            std_rows.append({"standard":std_name,"emoji":emoji,"temps_std":seconds_to_hms(t_std_s),
                             "diff_s":diff_s,"diff_str":(f"+{seconds_to_hms(int(diff_s))}" if diff_s>0 else f"-{seconds_to_hms(int(-diff_s))}"),"atteint":diff_s<=0})
        closest=min(std_rows,key=lambda r:abs(r["diff_s"])) if std_rows else None
        results[dist_label]={"dist_m":dist_m,"t_pred_s":t_pred_s,"t_pred_hms":seconds_to_hms(t_pred_s),
                              "pace_pred":pace_pred,"standards":std_rows,"closest":closest}
    best_dist=None;best_ratio=float("inf")
    for dist_label,info in results.items():
        if info["closest"]:
            ratio=abs(info["closest"]["diff_s"])/max(1,info["closest"]["diff_s"]+info["t_pred_s"])
            if ratio<best_ratio:best_ratio=ratio;best_dist=dist_label
    return results,best_dist

# v8.2 PATCH — BUG CRITIQUE corrigé : quand la pente K de la régression
# log-log est plafonnée, l'ordonnée à l'origine (a) doit être RECALCULÉE
# pour rester cohérente avec la pente effectivement utilisée. Avant ce
# patch, "a" restait celui de la régression NON plafonnée — (a,K) ne
# formait alors plus une paire valide, et predict_flat() pouvait prédire
# des temps complètement absurdes (ex : 0h51 pour un marathon de référence
# à 2h41) dès qu'on s'éloignait du centre de gravité des références.
#
# v8.3 PATCH — plafond haut de K relevé de 1.25 à 1.60 : la borne 1.25
# reflète des exposants Riegel typiques route (5km→marathon). Elle est
# beaucoup trop basse pour extrapoler des références ultra-trail (ex :
# marathon → 66km → 135km) vers une distance encore plus longue (UTMB
# ~170km+) : la fatigue réelle sur ces distances (déplétion glycogénique,
# nuit, autosuffisance) donne souvent des K observés >1.25, parfois >1.5.
# Écraser ce signal à 1.25 tronque artificiellement la base de temps plat
# extrapolée et peut sous-estimer le temps total de plusieurs heures sur
# un 100 miles. Le plafond bas (0.85) reste inchangé — aucune course ne
# ralentit moins vite que le rythme constant en extrapolant vers le long.
def clamped_loglog_fit(X, Y, k_max=1.60):
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if len(X) < 2:
        return None
    K_raw, loga_raw = np.polyfit(X, Y, 1)
    K = float(max(0.85, min(k_max, K_raw)))
    loga = float(loga_raw) if abs(K - K_raw) < 1e-9 else float(np.mean(Y - K * X))
    return math.exp(loga), K

def raw_loglog_slope(X, Y):
    """K non plafonné de la régression log-log — calculé séparément pour affichage,
    afin que l'utilisateur voie quand le plafond de clamped_loglog_fit tronque un
    signal de fatigue réel plutôt que de filtrer du bruit."""
    if len(X) < 2:
        return None
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    K_raw, _ = np.polyfit(X, Y, 1)
    return float(K_raw)

def fit_loglog(refs):
    X,Y=[],[]
    for r in refs:
        d_m=safe_float(r.get("distance",0));t=r.get("temps")
        secs=float(t) if isinstance(t,(int,float,np.number)) else hms_to_seconds(str(t))
        if d_m<=0 or secs<=0:continue
        X.append(math.log(d_m/1000.0));Y.append(math.log(secs))
    if len(X)>=2:
        a,K=clamped_loglog_fit(X,Y)
        return(a if 0<a<1e7 else 240.0),K
    elif len(X)==1:return math.exp(Y[0])/(math.exp(X[0])),1.0
    return 240.0,1.0

def predict_flat(dist_m,a,K):return float(a)*((dist_m/1000.0)**float(K))

def crossval_loo(refs_prepared):
    n=len(refs_prepared)
    if n<3:return None
    rows=[]
    for i in range(n):
        train=[r for j,r in enumerate(refs_prepared) if j!=i];test=refs_prepared[i]
        a_cv,K_cv=fit_loglog(train);pred_s=predict_flat(test["distance"],a_cv,K_cv)
        actual_s=float(test["temps"])
        rows.append({"Réf":i+1,"Distance (km)":round(test["distance"]/1000.0,2),
                     "Temps réel":seconds_to_hms(actual_s),"Temps prédit":seconds_to_hms(pred_s),
                     "Erreur (s)":round(pred_s-actual_s,0),"Erreur (%)":round((pred_s-actual_s)/actual_s*100.0,2) if actual_s>0 else 0})
    df_cv=pd.DataFrame(rows)
    mae=float(np.mean(np.abs(df_cv["Erreur (s)"].values)));mape=float(np.mean(np.abs(df_cv["Erreur (%)"].values)))
    return df_cv,mae,mape

def elev_factor_global(D_up_m,D_down_m,dist_m,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down):
    dist=max(1e-6,float(dist_m));g_up=float(D_up_m)/dist;g_dn=float(D_down_m)/dist
    g0u=max(1e-6,float(g0_up)/100.0);g0d=max(1e-6,float(g0_down)/100.0)
    up_term=float(k_up)*math.tanh(g_up/g0u)*g0u
    down_bonus=min(float(k_down)*math.tanh(g_dn/g0d)*g0d,abs(float(down_cap)))
    mult=1.0+up_term-down_bonus;mult=min(mult,1.0+float(max_up));mult=max(mult,1.0+float(max_down))
    return max(0.01,float(mult))

def recalibrate_ref_to_ideal(ref,opt_temp,use_wbgt,cold_quad,hot_quad,temp_max_penalty,
                              k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,elev_ref_power,temp_ref_power):
    secs=hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up=safe_float(ref.get("D_up",0.0));D_down=safe_float(ref.get("D_down",0.0))
    dist=max(1.0,safe_float(ref.get("distance",1000.0)))
    f_elev=elev_factor_global(D_up,D_down,dist,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    secs_no_elev=secs/(f_elev**float(elev_ref_power))
    temp_real=ref.get("avg_temp");hum_real=safe_float(ref.get("avg_humidity",50.0),50.0)
    if temp_real is not None:
        temp_eff=effective_temp(temp_real,hum_real,use_wbgt)
        f_temp=temp_multiplier(temp_eff,opt_temp,cold_quad,hot_quad,temp_max_penalty)
        secs_no_temp=secs_no_elev/(max(0.01,f_temp)**float(temp_ref_power))
    else:secs_no_temp=secs_no_elev
    return max(0.0,float(secs_no_temp))

def prepare_refs(refs_input,use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,
                 temp_max_penalty,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,elev_ref_power,temp_ref_power):
    out=[]
    for r in refs_input:
        d=safe_float(r.get("distance",0.0));raw_t=r.get("duration_hms_file") or r.get("temps","0:00:00")
        if use_recalibrated:
            secs=recalibrate_ref_to_ideal(ref={**r,"temps":raw_t},opt_temp=opt_temp,use_wbgt=use_wbgt,
                cold_quad=cold_quad,hot_quad=hot_quad,temp_max_penalty=temp_max_penalty,
                k_up=k_up,k_down=k_down,down_cap=down_cap,g0_up=g0_up,g0_down=g0_down,
                max_up=max_up,max_down=max_down,elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power) if use_recalibrated else float(t_brut)
        else:secs=float(hms_to_seconds(raw_t))
        out.append({"distance":float(d),"temps":float(secs)})
    return out

def apply_ultra_pacing(t_raw,d_end_m,seg_len_m,total_corr_m,amp_pct):
    if len(t_raw)==0 or amp_pct<=0:return t_raw
    total_corr_m=max(1e-9,float(total_corr_m))
    d_mid=np.asarray(d_end_m)-0.5*np.asarray(seg_len_m);prog=np.clip(d_mid/total_corr_m,0.0,1.0)
    A=amp_pct/100.0;mult=1.0+A*(2.0*prog-1.0);t_adj=np.asarray(t_raw)*mult
    s_raw=np.sum(t_raw);s_adj=np.sum(t_adj)
    if s_raw>0 and s_adj>0:t_adj*=s_raw/s_adj
    return t_adj


def run_prediction(distance_cible_km,refs_input,points,date_course,heure_course,
    use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,temp_max_penalty,temp_power,
    elev_ref_power,temp_ref_power,apply_grade,use_minetti,minetti_weight,
    k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,elev_smooth_window,grade_power,
    apply_altitude,altitude_ref_m,apply_wind,wind_mode,wind_smooth_km,
    drag_coeff,tail_credit,wind_cap_head,wind_cap_tail,wind_power,
    wind_gate_g1,wind_gate_g2,wind_gate_min,base_cap,extra_per_pct,max_cap,
    apply_fatigue,fatigue_rate,fatigue_mode,fatigue_threshold=60.0,
    dual_fatigue=False,fatigue_threshold2=None,fatigue_rate2=None,
    apply_ultra=False,ultra_amp=0.0,
    objective_hms=None,show_smooth_pace=True,smooth_window_km=3,dem_elevations=None,
    surface_mult=1.0,tz_name=TZ_NAME_DEFAULT,
    meteo_fallback_temp=12.0, meteo_fallback_amp=4.0,
    meteo_fallback_wind=2.0,  meteo_fallback_humidity=60.0,
    meteo_fallback_wind_dir=180.0,
    pace_sensitivity_ref=360.0,
    apply_vam=True,vam_threshold_pct=25.0,vam_rate_m_per_h=900.0,vam_blend_width_pct=10.0):

    if not points or len(points)<2:raise ValueError("GPX invalide ou trop court.")
    if dem_elevations is not None and len(dem_elevations)==len(points):
        elev_arr=np.array([e if e is not None else 0.0 for e in dem_elevations],dtype=float)
    else:
        elev_arr=np.array([getattr(p,"elevation",0.0) or 0.0 for p in points],dtype=float)

    # ── v8 PATCH 1c : distance filtrée anti-bruit GPS (seuil adaptatif) ──
    import statistics as _stats_rp
    _all_steps_rp = [haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude) for i in range(1,len(points))]
    _med_rp = _stats_rp.median(_all_steps_rp) if _all_steps_rp else 5.0
    _max_step_rp = max(100.0, min(2000.0, _med_rp * 10))
    total_m=0.0;cum=[0.0]
    for d_step in _all_steps_rp:
        if d_step<=_max_step_rp:
            total_m+=d_step
        cum.append(total_m)
    dist_gpx_km=total_m/1000.0
    if not distance_cible_km:distance_cible_km=dist_gpx_km
    fac=distance_cible_km/max(dist_gpx_km,1e-9)
    total_corr=total_m*fac;dists_corr=np.array(cum,dtype=float)*fac

    if elev_arr.size!=dists_corr.size:
        xs=np.linspace(0,total_m,elev_arr.size)
        elev_arr=np.interp(np.linspace(0,total_m,dists_corr.size),xs,elev_arr)

    w=int(elev_smooth_window)
    if w%2==0:w+=1
    # ── v8 PATCH 2 : lissage altitude avec padding miroir (élimine artefact 1er km) ──
    if w>=3 and elev_arr.size>=w:
        pad=w//2
        elev_padded=np.pad(elev_arr,pad,mode='reflect')
        elev_s=np.convolve(elev_padded,np.ones(w)/w,mode='valid')
        if elev_s.size>elev_arr.size:elev_s=elev_s[:elev_arr.size]
        elif elev_s.size<elev_arr.size:elev_s=np.pad(elev_s,(0,elev_arr.size-elev_s.size),mode='edge')
    else:
        elev_s=elev_arr

    diffs_el=np.diff(elev_s);d_plus_total=float(np.sum(np.clip(diffs_el,0,None)))
    avg_alt=float(np.mean(elev_s))

    refs_fit=prepare_refs(refs_input,use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,
                           temp_max_penalty,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                           elev_ref_power,temp_ref_power)
    a,K=fit_loglog(refs_fit)
    _X_kraw=[math.log(max(1e-6,safe_float(r.get("distance",0))/1000.0)) for r in refs_fit if safe_float(r.get("distance",0))>0 and float(r.get("temps",0))>0]
    _Y_kraw=[math.log(float(r.get("temps",0))) for r in refs_fit if safe_float(r.get("distance",0))>0 and float(r.get("temps",0))>0]
    K_raw=raw_loglog_slope(_X_kraw,_Y_kraw)
    base_total_s=predict_flat(int(distance_cible_km*1000),a,K)
    base_s_per_km=base_total_s/max(distance_cible_km,1e-9)*float(surface_mult)

    obj_s_target = hms_to_seconds(objective_hms) if objective_hms else None
    if obj_s_target and obj_s_target > 0:
        a_init = obj_s_target / (distance_cible_km**K)
        base_s_per_km = (a_init * (distance_cible_km ** K) / max(distance_cible_km,1e-9)) * float(surface_mult)

    pace_sens_ref_s = float(pace_sensitivity_ref)
    pace_sens_factor = min(1.0, math.sqrt(base_s_per_km / max(1.0, pace_sens_ref_s)))

    alt_mult=altitude_vo2_multiplier(avg_alt,altitude_ref_m) if apply_altitude else 1.0

    km_marks=[i*1000 for i in range(1,int(total_corr//1000)+1)]
    last=total_corr-int(total_corr//1000)*1000
    if last>1e-6:km_marks.append(total_corr)

    lats_arr=np.array([p.latitude for p in points],dtype=float)
    lons_arr=np.array([p.longitude for p in points],dtype=float)
    dt_dep=datetime.combine(date_course,heure_course)

    # ── Pré-calcul par km : D+ réel + multiplicateur de pente pondéré ──
    #
    # CORRECTION PHYSIQUE FONDAMENTALE :
    # La vitesse dépend de la pente INSTANTANÉE, pas de la pente nette du km.
    # 80m D+ sur 400m à 20% + 600m plat ≠ 80m D+ sur 1000m à 8% en termes de vitesse.
    # Le modèle calcule donc :
    #   - seg_dp : D+ réel par cumul des diffs entre points GPX (pas diff extrémités)
    #   - grade_mult : moyenne pondérée des multiplicateurs Minetti/heuristique
    #                  calculés point par point, pondérés par la longueur de chaque micro-segment
    #
    _km_bounds = [0.0] + list(km_marks)
    _seg_dplus_arr = []
    _seg_grade_mult_arr = []
    _seg_grade_nette_arr = []
    for _ki in range(len(_km_bounds)-1):
        _d0, _d1 = _km_bounds[_ki], _km_bounds[_ki+1]
        _mask = (dists_corr[:-1] >= _d0) & (dists_corr[:-1] < _d1)
        if _mask.sum() < 1:
            _seg_dplus_arr.append(0.0)
            _seg_grade_mult_arr.append(1.0)
            _seg_grade_nette_arr.append(0.0)
            continue
        _idx = np.where(_mask)[0]
        _idx_end = min(_idx[-1]+2, len(elev_s))
        _elev_seg = elev_s[_idx[0]:_idx_end]
        _dist_seg = dists_corr[_idx[0]:_idx_end]
        # D+ réel
        _seg_dplus_arr.append(float(np.sum(np.clip(np.diff(_elev_seg),0,None))))
        # Multiplicateur de pente pondéré point à point
        _total_w = 0.0; _weighted_mult = 0.0; _weighted_grade = 0.0
        for _j in range(len(_elev_seg)-1):
            _dd = float(_dist_seg[_j+1] - _dist_seg[_j]) if _j+1<len(_dist_seg) else 1.0
            if _dd <= 0: continue
            _g_pt = (_elev_seg[_j+1]-_elev_seg[_j])/_dd*100.0
            if apply_vam:
                _m_pt = grade_multiplier_with_vam(_g_pt,base_s_per_km,vam_threshold_pct,vam_rate_m_per_h,vam_blend_width_pct,
                                                   use_minetti,minetti_weight,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
            else:
                _m_pt = combined_grade_multiplier(_g_pt,use_minetti,minetti_weight,
                                                   k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
            _weighted_mult  += _m_pt * _dd
            _weighted_grade += _g_pt * _dd
            _total_w += _dd
        if _total_w > 0:
            _seg_grade_mult_arr.append(_weighted_mult / _total_w)
            _seg_grade_nette_arr.append(_weighted_grade / _total_w)
        else:
            _seg_grade_mult_arr.append(1.0)
            _seg_grade_nette_arr.append(0.0)

    pre=[];cum_t=cum_dp=cum_dist=0.0
    for i,d in enumerate(km_marks):
        seg_len=1000.0
        if i==len(km_marks)-1 and last>1e-6:seg_len=d-(km_marks[-2] if len(km_marks)>=2 else 0)
        # Pente nette du km (pour affichage et cap_combined)
        e_cur=float(np.interp(d,dists_corr,elev_s))
        e_prv=float(np.interp(max(d-seg_len,0),dists_corr,elev_s)) if i>0 else e_cur
        grade=(e_cur-e_prv)/max(1e-6,seg_len)*100.0
        # D+ et multiplicateur pente : valeurs pré-calculées point par point
        seg_dp=_seg_dplus_arr[i] if i<len(_seg_dplus_arr) else max(0.0,e_cur-e_prv)
        cum_dp+=seg_dp;cum_dist+=seg_len;t_flat=base_s_per_km*(seg_len/1000.0)
        if apply_grade:
            # Multiplicateur pondéré point par point (pré-calculé)
            gm=_seg_grade_mult_arr[i] if i<len(_seg_grade_mult_arr) else combined_grade_multiplier(grade,use_minetti,minetti_weight,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
            t1=t_flat*(gm**grade_power)
        else:gm=1.0;t1=t_flat
        t2=t1*alt_mult
        if apply_fatigue and dual_fatigue and fatigue_rate2 and fatigue_rate2>0:
            fm=fatigue_multiplier_dual(cum_dp,cum_dist,d_plus_total,total_corr,
                                        fatigue_threshold,fatigue_rate,
                                        fatigue_threshold2,fatigue_rate2,fatigue_mode)
        elif apply_fatigue and fatigue_rate>0:
            fm=fatigue_multiplier_advanced(cum_dp,cum_dist,d_plus_total,total_corr,
                                            fatigue_threshold,fatigue_rate,fatigue_mode)
        else:
            fm=1.0
        t3=t2*fm
        passage_dt=dt_dep+timedelta(seconds=cum_t+t3/2.0)
        lat_s=float(np.interp(d,dists_corr,lats_arr));lon_s=float(np.interp(d,dists_corr,lons_arr))
        lat0=float(np.interp(max(d-seg_len,0),dists_corr,lats_arr));lon0=float(np.interp(max(d-seg_len,0),dists_corr,lons_arr))
        cap=bearing_deg(lat0,lon0,lat_s,lon_s)
        meteo=get_weather_minutely(lat_s, lon_s, passage_dt, tz_name,
                                    fallback_temp=meteo_fallback_temp,fallback_temp_amp=meteo_fallback_amp,
                                    fallback_wind=meteo_fallback_wind,fallback_humidity=meteo_fallback_humidity,
                                    fallback_wind_dir=meteo_fallback_wind_dir)
        temp_raw=meteo["temp"] if meteo else None;wind_raw=meteo["wind"] if meteo else None
        hum_raw=meteo["humidity"] if meteo else None;wdir_raw=meteo.get("wind_dir") if meteo else None
        temp_eff_val=None
        if temp_raw is not None and hum_raw is not None:temp_eff_val=effective_temp(temp_raw,hum_raw,use_wbgt)
        if temp_eff_val is not None:
            tm_raw=temp_multiplier(temp_eff_val,opt_temp,cold_quad,hot_quad,temp_max_penalty)
            tm=1.0+(tm_raw-1.0)*pace_sens_factor
            t4=t3*(tm**temp_power)
        else:tm=1.0;t4=t3
        pace_local=(t4/seg_len)*1000.0 if seg_len>0 else t4
        head_raw,tail_raw=wind_components(wind_raw,wdir_raw,cap)
        head=head_raw*pace_sens_factor
        tail=tail_raw*pace_sens_factor
        pre.append({"idx":i,"d":d,"seg_len":seg_len,"grade":grade,"grade_mult":gm,"seg_dp":seg_dp,
                    "cum_dp":cum_dp,"fat_mult":fm,"alt_mult":alt_mult,"temp_raw":temp_raw,
                    "temp_eff":temp_eff_val,"hum":hum_raw,"wind":wind_raw,"wdir":wdir_raw,"cap":cap,
                    "head":head,"tail":tail,"temp_mult":tm,"t_flat":t_flat,"t_no_wind":t4,"pace_no_wind":pace_local,
                    "meteo_source":meteo.get("source","—") if meteo else "none"})
        cum_t+=t4

    df_pre=pd.DataFrame(pre)
    if apply_wind and not df_pre.empty:
        if wind_mode=="Global":
            hg=float(np.median(df_pre["head"]));tg=float(np.median(df_pre["tail"]))
            pg=float(np.median(df_pre["pace_no_wind"]))
            wm_raw=wind_multiplier(hg,tg,pg,drag_coeff,tail_credit,wind_cap_head,wind_cap_tail)
            df_pre["wind_mult_raw"]=wm_raw
        else:
            w_s=int(max(1,wind_smooth_km));w_s+=(1 if w_s%2==0 else 0)
            hs=pd.Series(df_pre["head"]).rolling(w_s,center=True,min_periods=1).median()
            ts_=pd.Series(df_pre["tail"]).rolling(w_s,center=True,min_periods=1).median()
            wms=[wind_multiplier(h,t,p,drag_coeff,tail_credit,wind_cap_head,wind_cap_tail)
                 for h,t,p in zip(hs,ts_,df_pre["pace_no_wind"])]
            df_pre["wind_mult_raw"]=wms;df_pre["head_s"]=hs.values;df_pre["tail_s"]=ts_.values
    else:df_pre["wind_mult_raw"]=1.0

    t_raw=[];wm_adj_list=[]
    for _,row in df_pre.iterrows():
        wm=float(row["wind_mult_raw"]);g=float(row["grade"])
        gate=wind_gate(g,wind_gate_g1,wind_gate_g2,wind_gate_min);wm_gated=1.0+gate*(wm-1.0)
        t_w=float(row["t_no_wind"])*(wm_gated**wind_power)
        gm_capped=cap_combined_with_vam(float(row["grade_mult"]),g,base_cap,extra_per_pct,max_cap,
                                        base_s_per_km,apply_vam,vam_threshold_pct,vam_rate_m_per_h)
        gm_raw=float(row["grade_mult"])
        if gm_raw > 0:
            t_final=t_w*(gm_capped/gm_raw)
        else:
            t_final=t_w
        t_raw.append(float(t_final));wm_adj_list.append(wm_gated)
    df_pre["wind_mult_adj"]=wm_adj_list;t_raw=np.array(t_raw,dtype=float)

    if apply_ultra and ultra_amp>0:
        t_raw=apply_ultra_pacing(t_raw,df_pre["d"].values,df_pre["seg_len"].values,total_corr,ultra_amp)

    if obj_s_target and obj_s_target > 0 and float(np.sum(t_raw)) > 0:
        seg_lens_km = df_pre["seg_len"].values / 1000.0
        mult_totaux = t_raw / (base_s_per_km * seg_lens_km + 1e-9)
        sum_weighted_mults = float(np.sum(mult_totaux * seg_lens_km))
        if sum_weighted_mults > 0:
            base_opt = obj_s_target / sum_weighted_mults
            t_raw = base_opt * mult_totaux * seg_lens_km

    rows=[];cum_t2=0.0
    for i in range(len(df_pre)):
        seg=df_pre.iloc[i];ts=float(t_raw[i]);cum_t2+=ts
        pace_val=(ts/float(seg["seg_len"]))*1000.0 if seg["seg_len"]>0 else ts
        rows.append({"Km":(int(seg["idx"])+1) if seg["seg_len"]>=999 else f"{int(seg['idx'])+1} ({seg['seg_len']:.0f}m)",
                     "Pente (%)":round(float(seg["grade"]),2),"Mult Pente":round(float(seg["grade_mult"]),4),
                     "D+ seg (m)":round(float(seg["seg_dp"]),1),"D+ cum (m)":round(float(seg["cum_dp"]),1),
                     "Mult Fatigue":round(float(seg["fat_mult"]),4),"Mult Altitude":round(float(seg["alt_mult"]),4),
                     "Temp (°C)":round(float(seg["temp_raw"]),1) if seg["temp_raw"] is not None else "—",
                     "Temp WBGT (°C)":round(float(seg["temp_eff"]),1) if seg["temp_eff"] is not None else "—",
                     "Mult Temp":round(float(seg["temp_mult"]),4),
                     "Vent (m/s)":round(float(seg["wind"]),1) if seg["wind"] is not None else "—",
                     "Dir. vent (°)":round(float(seg["wdir"]),0) if seg.get("wdir") is not None else "—",
                     "Cap course (°)":round(float(seg["cap"]),0),
                     "Headwind eff. (m/s)":round(float(seg.get("head_s",seg["head"])),2),
                     "Tailwind eff. (m/s)":round(float(seg.get("tail_s",seg["tail"])),2),
                     "Mult Vent":round(float(seg["wind_mult_adj"]),4),
                     "Humidité (%)":round(float(seg["hum"]),1) if seg["hum"] is not None else "—",
                     "Src météo":seg.get("meteo_source","—"),
                     "Temps seg (s)":round(ts,1),"Allure (min/km)":pace_str(pace_val),
                     "Temps cumulé":seconds_to_hms(cum_t2)})
    df_out=pd.DataFrame(rows)
    if show_smooth_pace and not df_out.empty:
        w_p=int(max(1,smooth_window_km));w_p+=(1 if w_p%2==0 else 0)
        s_p=pd.Series(df_out["Temps seg (s)"].astype(float)).rolling(w_p,center=True,min_periods=1).median()
        df_out["Allure lissée (min/km)"]=s_p.apply(pace_str)
    total_s=float(np.sum(t_raw))
    _std_dists = {"10 km":10000,"Semi":21097,"Marathon":42195,"50 km":50000,"100 km":100000}
    _gpx_m = dist_gpx_km * 1000.0
    _below = {k:v for k,v in _std_dists.items() if v <= _gpx_m * 1.05}
    _above = {k:v for k,v in _std_dists.items() if v >= _gpx_m * 0.95}
    _ci_ref_low = max(_below.values()) if _below else None
    _ci_ref_high = min(_above.values()) if _above else None
    _label_low = next((k for k,v in _std_dists.items() if v == _ci_ref_low), None)
    _label_high = next((k for k,v in _std_dists.items() if v == _ci_ref_high), None)
    _avg_pace = total_s / max(_gpx_m, 1.0)
    _ci_low_hms  = seconds_to_hms(_avg_pace * _ci_ref_low)  if _ci_ref_low  else seconds_to_hms(total_s*0.97)
    _ci_high_hms = seconds_to_hms(_avg_pace * _ci_ref_high) if _ci_ref_high else seconds_to_hms(total_s*1.03)
    _ci_low_label  = _label_low  if _label_low  else "−3%"
    _ci_high_label = _label_high if _label_high else "+3%"

    return{"df":df_out,"total_s":total_s,"total_human":seconds_to_hms(total_s),
           "ci_low":_ci_low_hms,"ci_high":_ci_high_hms,
           "ci_low_label":_ci_low_label,"ci_high_label":_ci_high_label,
           "dist_gpx_km":dist_gpx_km,"K":K,"K_raw":K_raw,"avg_alt":avg_alt,"d_plus_total":d_plus_total,
           "refs_fit":refs_fit,"pre_df":df_pre}

def build_checkpoint_schedule(df_out, checkpoints, start_dt=None):
    """Feuille de route : pour chaque ravitaillement, heure/temps d'arrivée, temps
    d'arrêt prévu, temps de départ, et allure du segment précédent (hors arrêts).
    Les arrêts des ravitaillements précédents décalent tous les temps suivants.
    Retourne (lignes, total_arrets_s)."""
    km_vals, t_cum = [], []
    for _, row in df_out.iterrows():
        try:
            km = float(str(row["Km"]).split()[0])
        except (ValueError, IndexError):
            continue
        km_vals.append(km)
        t_cum.append(float(hms_to_seconds(str(row["Temps cumulé"]))))
    if len(km_vals) < 2 or not checkpoints:
        return [], 0.0
    rows = []
    stops_before = 0.0
    prev_km = 0.0
    prev_t_mov = 0.0
    for cp in sorted(checkpoints, key=lambda c: float(c.get("dist_km", 0))):
        km = float(cp.get("dist_km", 0))
        t_mov = float(np.interp(km, km_vals, t_cum))
        arrivee = t_mov + stops_before
        arret = max(0.0, float(cp.get("arret_s", 0) or 0))
        depart = arrivee + arret
        seg_km = max(0.0, km - prev_km)
        seg_s = max(0.0, t_mov - prev_t_mov)
        rows.append({
            "label": cp.get("label", "Ravito"), "type": cp.get("type", ""),
            "dist_km": km, "alt": cp.get("alt"),
            "t_mouvement_s": t_mov, "arrivee_s": arrivee, "arret_s": arret, "depart_s": depart,
            "segment_km": seg_km, "segment_s": seg_s,
            "allure_segment_s_km": (seg_s / seg_km) if seg_km > 0.05 else None,
            "allure_moy_depuis_depart_s_km": (t_mov / km) if km > 0.05 else None,
            "heure_arrivee": (start_dt + timedelta(seconds=arrivee)).strftime("%H:%M:%S") if start_dt else None,
            "heure_depart": (start_dt + timedelta(seconds=depart)).strftime("%H:%M:%S") if start_dt else None,
        })
        stops_before += arret
        prev_km = km
        prev_t_mov = t_mov
    return rows, stops_before

def distribute_stop_budget(checkpoints, budget_s, max_per_stop_s, only_types=("🥤 Ravitaillement",)):
    """Répartit un budget d'arrêt total sur les ravitaillements, sans dépasser le
    maximum autorisé par ravitaillement. Retourne (nb_ravitos_servis, budget_placé,
    budget_non_plaçable)."""
    elig = [c for c in checkpoints if c.get("type") in only_types] or list(checkpoints)
    n = len(elig)
    if n == 0 or budget_s <= 0:
        for c in checkpoints:
            c["arret_s"] = 0.0
        return 0, 0.0, float(budget_s)
    per = min(float(max_per_stop_s), float(budget_s) / n)
    placed = 0.0
    for c in checkpoints:
        if c in elig:
            c["arret_s"] = round(per, 1)
            placed += per
        else:
            c["arret_s"] = 0.0
    return n, placed, max(0.0, float(budget_s) - placed)

def extract_interval_df(df,start_hms,end_hms):
    t_start=float(hms_to_seconds(start_hms));t_end=float(hms_to_seconds(end_hms))
    if t_end<=t_start:return pd.DataFrame()
    sub=df[(df["elapsed_s"]>=t_start)&(df["elapsed_s"]<=t_end)].copy()
    sub["elapsed_s"]=sub["elapsed_s"]-t_start
    return sub.reset_index(drop=True)

def analyze_interval(df_int,name):
    if df_int.empty:return{"name":name,"valid":False}
    dur_s=float(df_int["elapsed_s"].max());dist_m=None
    if "distance_m" in df_int.columns and df_int["distance_m"].notna().any():
        d0=df_int["distance_m"].dropna().iloc[0];d1=df_int["distance_m"].dropna().iloc[-1]
        dist_m=float(d1-d0) if d1>d0 else float(df_int["distance_m"].dropna().max()-df_int["distance_m"].dropna().min())
    hr_stats=analyze_heart_rate(df_int);spd_stats=analyze_speed_kinetics(df_int)
    avg_speed=None
    if dist_m and dur_s>0:avg_speed=dist_m/dur_s
    return{"name":name,"valid":True,"dur_s":dur_s,"dist_m":dist_m,"avg_speed":avg_speed,
           "hr":hr_stats,"spd":spd_stats,"df":df_int}


def detect_technical_terrain(points, dem_elevations=None, seg_len_m=1000, is_trail=True):
    if not is_trail or len(points) < 10:
        return [], {"global_score": 0, "label": "—", "k_up_adj": 1.0,
                    "k_down_adj": 1.0, "surface_mult_adj": 1.0, "skipped": True}
    n = len(points)
    if dem_elevations is not None and len(dem_elevations) == n:
        elevs = [dem_elevations[i] if dem_elevations[i] is not None else 0.0 for i in range(n)]
    else:
        elevs = [getattr(points[i], "elevation", 0.0) or 0.0 for i in range(n)]
    cum = [0.0]
    for i in range(1, n):
        cum.append(cum[-1] + haversine_m(
            points[i-1].latitude, points[i-1].longitude,
            points[i].latitude,   points[i].longitude))
    total_m = cum[-1]
    km_marks = list(range(0, int(total_m), int(seg_len_m))) + [int(total_m)]
    segments = []
    for ki in range(len(km_marks) - 1):
        d_start = km_marks[ki]; d_end = km_marks[ki + 1]
        idx_seg = [i for i in range(n) if d_start <= cum[i] <= d_end]
        if len(idx_seg) < 3: continue
        pts_seg  = [points[i] for i in idx_seg]
        elev_seg = [elevs[i]  for i in idx_seg]
        dist_seg = [cum[i] - d_start for i in idx_seg]
        d_gps  = dist_seg[-1]
        d_eucl = haversine_m(pts_seg[0].latitude, pts_seg[0].longitude,
                             pts_seg[-1].latitude, pts_seg[-1].longitude)
        sinuosity = d_gps / max(1.0, d_eucl)
        bearings = []
        for i in range(1, len(pts_seg)):
            dd = dist_seg[i] - dist_seg[i-1]
            if dd > 2.0:
                b = bearing_deg(pts_seg[i-1].latitude, pts_seg[i-1].longitude,
                                pts_seg[i].latitude,   pts_seg[i].longitude)
                bearings.append(b)
        real_turns_score = 0.0
        if len(bearings) >= 2:
            for i in range(1, len(bearings)):
                delta = abs((bearings[i] - bearings[i-1] + 180) % 360 - 180)
                if delta < 45: pass
                elif delta < 90:  real_turns_score += delta * 0.30
                elif delta < 135: real_turns_score += delta * 0.80
                else:             real_turns_score += delta * 1.20
        real_turns_per_km = real_turns_score / max(0.001, d_gps / 1000.0)
        grades = []
        for i in range(1, len(pts_seg)):
            dd = dist_seg[i] - dist_seg[i-1]
            if dd > 0.5:
                grades.append((elev_seg[i] - elev_seg[i-1]) / dd * 100.0)
        if not grades: grades = [0.0]
        grade_abs  = [abs(g) for g in grades]
        grade_max  = float(np.max(grade_abs))
        grade_std  = float(np.std(grades))
        grade_mean = float(np.mean(grades))
        steep_grades = [g for g in grade_abs if g > 10.0]
        steep_ratio  = len(steep_grades) / max(1, len(grade_abs))
        norm_sinu  = min(1.0, max(0.0, (sinuosity - 1.05) / 0.35))
        norm_turns = min(1.0, real_turns_per_km / 900.0)
        norm_grade = min(1.0, grade_max / 35.0)
        norm_steep = min(1.0, steep_ratio / 0.40)
        norm_std   = min(1.0, grade_std / 12.0)
        synergy_bonus = 0.0
        if grade_max > 15.0 and real_turns_per_km > 300.0:
            synergy_bonus = 0.10
        tech_score = (0.20*norm_sinu + 0.30*norm_turns + 0.20*norm_grade +
                      0.15*norm_steep + 0.15*norm_std + synergy_bonus)
        tech_score = min(1.0, tech_score)
        if tech_score < 0.25:   label = "🟢 Facile"
        elif tech_score < 0.45: label = "🟡 Modéré"
        elif tech_score < 0.70: label = "🟠 Technique"
        else:                   label = "🔴 Très technique"
        k_up_adj_seg   = 1.0 + 0.30 * max(0, tech_score - 0.25)
        k_down_adj_seg = 1.0 - 0.15 * max(0, tech_score - 0.25)
        surf_adj_seg   = 1.0 + 0.12 * max(0, tech_score - 0.25)
        segments.append({
            "km_start": round(d_start/1000.0,1), "km_end": round(d_end/1000.0,1),
            "sinuosity": round(sinuosity,3), "turns_score_km": round(real_turns_per_km,0),
            "grade_max": round(grade_max,1), "grade_std": round(grade_std,1),
            "grade_mean": round(grade_mean,1), "steep_ratio": round(steep_ratio,2),
            "tech_score": round(tech_score,3), "label": label,
            "k_up_adj": round(k_up_adj_seg,3), "k_down_adj": round(k_down_adj_seg,3),
            "surface_adj": round(surf_adj_seg,3),
        })
    if not segments:
        return [], {"global_score": 0, "label": "—", "k_up_adj": 1.0,
                    "k_down_adj": 1.0, "surface_mult_adj": 1.0}
    global_score   = float(np.mean([s["tech_score"] for s in segments]))
    max_score      = float(np.max([s["tech_score"] for s in segments]))
    weighted_score = 0.65 * global_score + 0.35 * max_score
    if weighted_score < 0.25:   global_label = "🟢 Terrain facile / Non technique"
    elif weighted_score < 0.42: global_label = "🟡 Trail modéré — quelques passages techniques"
    elif weighted_score < 0.62: global_label = "🟠 Trail technique — rochers, lacets, pentes"
    else:                       global_label = "🔴 Ultra-trail très technique"
    k_up_g   = float(np.median([s["k_up_adj"]   for s in segments]))
    k_down_g = float(np.median([s["k_down_adj"]  for s in segments]))
    surf_g   = float(np.median([s["surface_adj"] for s in segments]))
    global_info = {
        "global_score":     round(weighted_score,3), "label": global_label,
        "k_up_adj":         round(k_up_g,3), "k_down_adj": round(k_down_g,3),
        "surface_mult_adj": round(surf_g,3),
        "pct_technique":    round(sum(1 for s in segments if s["tech_score"]>0.45)/len(segments)*100,1),
        "sinuosity_mean":   round(float(np.mean([s["sinuosity"] for s in segments])),3),
        "grade_max_all":    round(float(np.max([s["grade_max"] for s in segments])),1),
    }
    return segments, global_info

@st.cache_data(show_spinner="Analyse surface OSM en cours...")
def fetch_osm_surface(lats_tuple, lons_tuple):
    OSM_SURFACE_MULT = {
        "paved":1.00,"asphalt":1.00,"concrete":1.00,"cobblestone":1.08,"sett":1.08,
        "unhewn_cobblestone":1.10,"compacted":1.03,"fine_gravel":1.04,"gravel":1.05,
        "pebblestone":1.07,"unpaved":1.07,"ground":1.08,"dirt":1.08,"earth":1.08,
        "mud":1.18,"sand":1.20,"grass":1.09,"grass_paver":1.07,"rock":1.12,"rocks":1.12,
        "stone":1.10,"wood":1.06,"woodchips":1.07,"snow":1.18,"ice":1.25,"unknown":1.06,
    }
    lats = list(lats_tuple); lons = list(lons_tuple)
    lat_min=min(lats)-0.005; lat_max=max(lats)+0.005
    lon_min=min(lons)-0.005; lon_max=max(lons)+0.005
    query=f"""[out:json][timeout:25];
(
  way["highway"]["surface"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["highway"]["tracktype"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out body geom;"""
    try:
        OVERPASS_ENDPOINTS = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.fr/api/interpreter",
        ]
        data = None; last_err = None
        for ep in OVERPASS_ENDPOINTS:
            try:
                resp=requests.post(ep,data={"data":query},timeout=25); resp.raise_for_status()
                data=resp.json(); break
            except Exception as e: last_err=e; continue
        if data is None: raise Exception(f"Tous les endpoints Overpass injoignables. Dernier: {last_err}")
    except Exception as e:
        return {"error":str(e),"dominant_surface":"unknown","surface_mult_osm":1.06,"surface_counts":{},"detail":[]}
    ways=data.get("elements",[])
    if not ways:
        return {"dominant_surface":"unknown","surface_mult_osm":1.06,"surface_counts":{},"detail":[],"ways_found":0}
    step=max(1,len(lats)//80); sample_lats=lats[::step]; sample_lons=lons[::step]
    surface_hits=[]; detail=[]
    for si,(slat,slon) in enumerate(zip(sample_lats,sample_lons)):
        best_dist=float("inf"); best_surface=None; best_highway=None
        for way in ways:
            geom=way.get("geometry",[]); surface=way.get("tags",{}).get("surface")
            tracktype=way.get("tags",{}).get("tracktype"); highway=way.get("tags",{}).get("highway","")
            if not surface and tracktype:
                surface={"grade1":"compacted","grade2":"fine_gravel","grade3":"gravel","grade4":"unpaved","grade5":"ground"}.get(tracktype,"unpaved")
            if not surface: continue
            for node in geom[:20]:
                nd=haversine_m(slat,slon,node["lat"],node["lon"])
                if nd<best_dist: best_dist=nd; best_surface=surface; best_highway=highway
        if best_surface and best_dist<50:
            surface_hits.append(best_surface.lower())
            detail.append({"km":round(si*step/max(1,len(lats))*(len(lats)/1000.0),1),
                           "surface":best_surface,"highway":best_highway or "—","dist_m":round(best_dist,0)})
    if not surface_hits:
        return {"dominant_surface":"unknown","surface_mult_osm":1.06,"surface_counts":{},"detail":[],"ways_found":len(ways)}
    from collections import Counter
    counts=Counter(surface_hits); dominant=counts.most_common(1)[0][0]
    total_c=sum(counts.values())
    weighted_mult=sum(OSM_SURFACE_MULT.get(s,1.06)*cnt/total_c for s,cnt in counts.items())
    return {"dominant_surface":dominant,"surface_mult_osm":round(weighted_mult,3),
            "surface_counts":dict(counts.most_common(10)),"detail":detail[:30],
            "ways_found":len(ways),"coverage_pct":round(len(surface_hits)/len(sample_lats)*100,1)}


# ══════════════════════════════════════════════════════════════
# NOTE v8 : generate_3d_terrain_html et generate_3d_animation
# sont identiques à v7 — copiez-les telles quelles depuis v7.
# Elles sont omises ici pour ne pas dépasser les limites de taille.
# Remplacez ce bloc par les fonctions complètes de v7.
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# VUE 3D RELIEF RÉEL — generate_3d_terrain_html (inchangé v7)
# ══════════════════════════════════════════════════════════════

def generate_3d_terrain_html(points, cum_d_map, checkpoints,
                               df_prediction=None, tech_segments=None,
                               dem_elevations=None, osm_surface_data=None,
                               height=600, pitch=52):
    import json as _json
    import math as _math

    n_pts = len(points); step = max(1, n_pts // 700)
    lats_s  = [points[i].latitude  for i in range(0, n_pts, step)]
    lons_s  = [points[i].longitude for i in range(0, n_pts, step)]
    dist_s  = [cum_d_map[i] / 1000.0 for i in range(0, n_pts, step)]
    n_sub   = len(lats_s)

    if dem_elevations is not None and len(dem_elevations) == n_pts:
        elevs_s = [dem_elevations[i] if dem_elevations[i] is not None else 0.0
                   for i in range(0, n_pts, step)]
    else:
        elevs_s = [getattr(points[i], "elevation", 0.0) or 0.0
                   for i in range(0, n_pts, step)]

    elev_min = min(elevs_s); elev_max = max(elevs_s)
    center_lat = float(np.mean(lats_s)); center_lon = float(np.mean(lons_s))
    span_km = haversine_m(min(lats_s), min(lons_s), max(lats_s), max(lons_s)) / 1000.0
    zoom = max(10.5, min(13.5, 14.5 - _math.log2(max(1, span_km))))
    bounds = [min(lons_s)-0.03, min(lats_s)-0.03, max(lons_s)+0.03, max(lats_s)+0.03]
    d_plus = float(np.sum(np.clip(np.diff(elevs_s), 0, None)))
    total_km = cum_d_map[-1] / 1000.0

    seg_size = max(1, n_sub // 80); trace_data = []
    for si in range(0, n_sub - 1, seg_size):
        end_i = min(si + seg_size + 1, n_sub)
        path  = [[lons_s[j], lats_s[j], elevs_s[j] + 4] for j in range(si, end_i)]
        if len(path) < 2: continue
        mid_i = min((si + end_i) // 2, n_sub - 1)
        trace_data.append({"path":path,"elevation":elevs_s[mid_i],"dist":round(dist_s[mid_i],2)})

    pace_data = []
    if df_prediction is not None and not df_prediction.empty and "Allure (min/km)" in df_prediction.columns:
        def _p2s(p):
            try: parts=str(p).split(":"); return int(parts[0])*60+int(parts[1])
            except: return 0
        for idx_row, row in df_prediction.iterrows():
            ps = _p2s(row["Allure (min/km)"])
            if ps <= 0: continue
            d_m = (idx_row + 0.5) * 1000.0
            if d_m > cum_d_map[-1]: d_m = cum_d_map[-1]
            seg_lat = float(np.interp(d_m, cum_d_map, [points[i].latitude  for i in range(n_pts)]))
            seg_lon = float(np.interp(d_m, cum_d_map, [points[i].longitude for i in range(n_pts)]))
            seg_el  = float(np.interp(d_m, [i * (cum_d_map[-1]/(n_sub-1) if n_sub>1 else 1) for i in range(n_sub)], elevs_s))
            d_start = idx_row * 1000.0; d_end = min((idx_row + 1) * 1000.0, cum_d_map[-1])
            idx_s2  = [i for i in range(n_sub) if d_start <= (i*(cum_d_map[-1]/(n_sub-1) if n_sub>1 else 1)) <= d_end]
            path_seg = [[lons_s[j], lats_s[j], elevs_s[j]+4] for j in idx_s2] if len(idx_s2)>=2 else [[seg_lon, seg_lat, seg_el+4]]
            pace_data.append({"path":path_seg,"pace_s":ps,"dist":round(d_m/1000,1)})

    osm_segs = []
    if osm_surface_data and osm_surface_data.get("detail"):
        for d_entry in osm_surface_data["detail"]:
            km_k=d_entry.get("km",0); surf=d_entry.get("surface","unknown")
            d_m=km_k*1000.0; d_end_m=min(d_m+1000.0,cum_d_map[-1])
            idx_s3=[i for i in range(n_sub) if d_m<=dist_s[i]*1000<=d_end_m]
            if len(idx_s3)<2: continue
            osm_segs.append({"path":[[lons_s[j],lats_s[j],elevs_s[j]+4] for j in idx_s3],"surface":surf,"dist":round(km_k,1)})

    tech_data_js = []
    if tech_segments:
        for seg in tech_segments:
            if seg.get("tech_score",0)>0.45:
                d_mid=(seg["km_start"]+seg["km_end"])/2.0*1000.0
                sl=float(np.interp(d_mid,cum_d_map,[points[i].latitude  for i in range(n_pts)]))
                so=float(np.interp(d_mid,cum_d_map,[points[i].longitude for i in range(n_pts)]))
                se=float(np.interp(d_mid,cum_d_map,[getattr(points[i],"elevation",0.0) or 0.0 for i in range(n_pts)]))
                ts=seg["tech_score"]
                tech_data_js.append({"position":[so,sl,se+25],"label":f"{seg['label']} {seg['km_start']:.0f}-{seg['km_end']:.0f}km",
                                      "score":round(ts,2),"color":[min(255,int(249*min(1,ts*1.4))),int(80*(1-ts)),30,210],
                                      "radius":int(80+ts*120)})

    CP_COL={"🥤 Ravitaillement":[0,229,255,230],"🏔 Sommet":[255,214,0,230],"🔻 Col":[224,64,251,230],
             "⏱ Point de passage":[64,196,255,230],"🏁 Intermédiaire":[180,180,180,200],"⚠️ Point clé":[255,23,68,230]}
    cp_data_js=[]
    for cp in sorted(checkpoints,key=lambda c:c["dist_km"]):
        cp_el=float(np.interp(cp["dist_km"]*1000,cum_d_map,[getattr(points[i],"elevation",0.0) or 0.0 for i in range(n_pts)]))
        cp_data_js.append({"position":[cp["lon"],cp["lat"],cp_el+18],"label":cp["label"],"dist":cp["dist_km"],"color":CP_COL.get(cp.get("type",""),[249,115,22,230])})

    def js(obj): return _json.dumps(obj, ensure_ascii=False)
    view_state={"longitude":center_lon,"latitude":center_lat,"zoom":round(zoom,1),"pitch":pitch,"bearing":0,"minPitch":0,"maxPitch":85}

    html=f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trail 3D — {total_km:.1f} km</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box}}html,body{{width:100%;height:100%;background:#060c18;font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0;overflow:hidden}}#c{{width:100%;height:100%;position:relative}}#ui{{position:absolute;top:10px;left:10px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:12px;padding:11px 15px;min-width:200px;user-select:none}}#ui h3{{font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;color:#64748b;margin-bottom:9px;font-weight:600}}.tog{{display:flex;align-items:center;gap:8px;margin:4px 0;cursor:pointer;font-size:.76rem;padding:2px 0}}.tog input{{accent-color:#f97316;cursor:pointer;flex-shrink:0}}.tog label{{cursor:pointer;color:#cbd5e1;transition:color .15s}}.tog:hover label,.tog input:checked+label{{color:#f97316}}.sep{{border:none;border-top:1px solid rgba(255,255,255,.07);margin:8px 0}}.btn-grp{{display:flex;gap:5px;margin-top:4px}}.btn{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#94a3b8;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:.68rem;transition:all .15s}}.btn:hover,.btn.on{{background:rgba(249,115,22,.25);border-color:#f97316;color:#fff}}#hud{{position:absolute;bottom:12px;left:12px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:10px;padding:8px 14px;display:flex;gap:18px}}.hud-item{{display:flex;flex-direction:column;align-items:center}}.hud-val{{font-size:.85rem;font-weight:700;color:#f97316;font-family:'Courier New',monospace}}.hud-lbl{{font-size:.58rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-top:1px}}#leg{{position:absolute;bottom:12px;right:12px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:10px;padding:9px 13px;min-width:160px}}#leg-title{{font-size:.68rem;color:#94a3b8;margin-bottom:5px;font-weight:600}}#leg-bar{{width:100%;height:10px;border-radius:3px;margin:3px 0}}#leg-labs{{display:flex;justify-content:space-between;font-size:.60rem;color:#475569}}#tooltip{{position:absolute;pointer-events:none;z-index:300;background:rgba(6,12,24,.92);border:1px solid rgba(249,115,22,.4);border-radius:8px;padding:7px 12px;font-size:.76rem;color:#e2e8f0;max-width:220px;display:none}}</style>
</head><body>
<div id="c"></div>
<div id="ui">
  <h3>🗺 Couches</h3>
  <div class="tog"><input type="checkbox" id="l0" checked><label for="l0">🏔 Relief 3D satellite</label></div>
  <div class="tog"><input type="checkbox" id="l1" checked><label for="l1">🔵 Tracé (altitude)</label></div>
  <div class="tog"><input type="checkbox" id="l2"><label for="l2">⏱ Allure prédite</label></div>
  <div class="tog"><input type="checkbox" id="l3"><label for="l3">🌿 Surfaces OSM</label></div>
  <div class="tog"><input type="checkbox" id="l4" checked><label for="l4">⚠️ Zones techniques</label></div>
  <div class="tog"><input type="checkbox" id="l5" checked><label for="l5">📍 Checkpoints</label></div>
  <hr class="sep">
  <h3>Vue</h3>
  <div class="btn-grp">
    <button class="btn on" id="v0" onclick="setView(52)">Oblique</button>
    <button class="btn" id="v1" onclick="setView(0)">Dessus</button>
    <button class="btn" id="v2" onclick="setView(75)">Plongeant</button>
  </div>
  <hr class="sep">
  <div style="font-size:.63rem;color:#374151">Glisser : rotation · Scroll : zoom</div>
</div>
<div id="hud">
  <div class="hud-item"><div class="hud-val">{total_km:.1f}</div><div class="hud-lbl">km total</div></div>
  <div class="hud-item"><div class="hud-val">{round(d_plus)}</div><div class="hud-lbl">m D+</div></div>
  <div class="hud-item"><div class="hud-val">{round(elev_min)}</div><div class="hud-lbl">m alt min</div></div>
  <div class="hud-item"><div class="hud-val">{round(elev_max)}</div><div class="hud-lbl">m alt max</div></div>
</div>
<div id="leg"><div id="leg-title">Altitude</div><canvas id="leg-bar" width="160" height="10"></canvas><div id="leg-labs"><span>{round(elev_min)} m</span><span>{round(elev_max)} m</span></div></div>
<div id="tooltip"></div>
<script>
const TRACE={js(trace_data)};const PACE={js(pace_data)};const OSM={js(osm_segs)};const TECH={js(tech_data_js)};const CPS={js(cp_data_js)};const BOUNDS={js(bounds)};
const EMIN={round(elev_min,1)},EMAX={round(elev_max,1)};
const {{Deck,TerrainLayer,PathLayer,ScatterplotLayer,TextLayer}}=deck;
function altC(t){{if(t<.33){{const u=t/.33;return[Math.round(20+u*80),Math.round(100+u*120),Math.round(255-u*80)];}}if(t<.66){{const u=(t-.33)/.33;return[Math.round(100+u*140),Math.round(220-u*60),Math.round(175-u*120)];}}const u=(t-.66)/.34;return[240,Math.round(160-u*120),Math.round(55-u*40)];}}
function paceC(t){{return[Math.round(30+t*219),Math.round(200-t*160),Math.round(80-t*60)];}}
const OSMC={{asphalt:[120,120,120],paved:[140,140,140],concrete:[160,160,160],compacted:[200,170,100],gravel:[180,140,80],unpaved:[160,120,60],ground:[139,90,43],dirt:[130,80,30],grass:[60,160,60],rock:[100,100,80],mud:[80,50,20],snow:[220,235,255],ice:[180,220,255],unknown:[150,150,150]}};
function osmC(s){{return OSMC[s]||OSMC.unknown;}}
function drawLeg(title,cfn,lbl0,lbl1,isText){{document.getElementById('leg-title').textContent=title;document.getElementById('leg-labs').innerHTML='<span>'+lbl0+'</span><span>'+lbl1+'</span>';const cv=document.getElementById('leg-bar');const ctx=cv.getContext('2d');if(isText){{ctx.fillStyle='rgba(100,116,139,.3)';ctx.fillRect(0,0,cv.width,cv.height);return;}}const g=ctx.createLinearGradient(0,0,cv.width,0);for(let i=0;i<=12;i++){{const[r,b,c]=cfn(i/12);g.addColorStop(i/12,`rgb(${{r}},${{b}},${{c}})`);}}ctx.fillStyle=g;ctx.fillRect(0,0,cv.width,cv.height);}}
drawLeg('Altitude',altC,Math.round(EMIN)+' m',Math.round(EMAX)+' m',false);
const S={{l0:true,l1:true,l2:false,l3:false,l4:true,l5:true}};let currentPitch={pitch};
const deckInst=new Deck({{container:'c',initialViewState:{js(view_state)},controller:{{touchRotate:true,touchZoom:true,scrollZoom:true,dragRotate:true}},
onHover:({{object,x,y}})=>{{const tt=document.getElementById('tooltip');if(object&&(object.label||object.surface||object.dist!==undefined)){{let html='';if(object.label)html+=`<b>${{object.label}}</b>`;if(object.dist!==undefined)html+=`<br>📍 ${{typeof object.dist==='number'?object.dist.toFixed(1):object.dist}} km`;if(object.score!==undefined)html+=`<br>⚠️ Score tech: ${{object.score}}`;if(object.surface)html+=`<br>🌿 Surface: ${{object.surface}}`;if(object.elevation!==undefined)html+=`<br>🏔 Alt: ${{Math.round(object.elevation)}} m`;tt.innerHTML=html;tt.style.left=(x+14)+'px';tt.style.top=(y-10)+'px';tt.style.display='block';}}else{{tt.style.display='none';}}}},layers:[]}});
function buildLayers(){{const L=[];
if(S.l0)L.push(new TerrainLayer({{id:'terrain',minZoom:0,maxZoom:13,elevationDecoder:{{rScaler:6553.6,gScaler:25.6,bScaler:0.1,offset:-10000}},elevationData:'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer/tile/{{z}}/{{y}}/{{x}}',texture:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',bounds:BOUNDS,color:[255,255,255]}}));
if(S.l1){{let data;if(S.l2&&PACE.length>0){{const ps=PACE.map(p=>p.pace_s),pmn=Math.min(...ps),pmx=Math.max(...ps);data=PACE.map(p=>{{const t=(p.pace_s-pmn)/Math.max(1,pmx-pmn);return{{...p,color:[...paceC(t),220]}}}});drawLeg('⏱ Allure',paceC,Math.floor(Math.min(...ps)/60)+':'+String(Math.round(Math.min(...ps)%60)).padStart(2,'0')+'/km',Math.floor(Math.max(...ps)/60)+':'+String(Math.round(Math.max(...ps)%60)).padStart(2,'0')+'/km',false);}}else if(S.l3&&OSM.length>0){{data=OSM.map(s=>{{return{{...s,color:[...osmC(s.surface),220]}}}});drawLeg('🌿 Surfaces',null,'Béton→Terre→Roche','Herbe→Boue',true);}}else{{data=TRACE.map(s=>{{const t=(s.elevation-EMIN)/Math.max(1,EMAX-EMIN);return{{...s,color:[...altC(t),220]}}}});drawLeg('Altitude',altC,Math.round(EMIN)+' m',Math.round(EMAX)+' m',false);}}L.push(new PathLayer({{id:'trace',data,getPath:d=>d.path,getColor:d=>d.color,getWidth:10,widthMinPixels:3,widthMaxPixels:15,pickable:true}}));}}
if(S.l4&&TECH.length>0){{L.push(new ScatterplotLayer({{id:'tech',data:TECH,getPosition:d=>d.position,getColor:d=>d.color,getRadius:d=>d.radius,radiusMinPixels:10,radiusMaxPixels:30,pickable:true}}));L.push(new TextLayer({{id:'tech_lbl',data:TECH,getPosition:d=>d.position,getText:d=>d.label,getSize:11,getColor:[255,220,50,220],background:true,getBackgroundColor:[0,0,0,150],getPixelOffset:[0,-30],billboard:true}}));}}
if(S.l5&&CPS.length>0){{L.push(new ScatterplotLayer({{id:'cp',data:CPS,getPosition:d=>d.position,getColor:d=>d.color,getRadius:100,radiusMinPixels:8,radiusMaxPixels:22,pickable:true}}));L.push(new TextLayer({{id:'cp_lbl',data:CPS,getPosition:d=>d.position,getText:d=>d.label,getSize:13,getColor:[255,255,255,230],background:true,getBackgroundColor:[0,0,0,160],getPixelOffset:[0,-24],billboard:true}}));}}
if(TRACE.length>0){{const se=[{{position:TRACE[0].path[0],color:[0,255,100,255],r:160,label:'🟢 Départ'}},{{position:TRACE[TRACE.length-1].path[TRACE[TRACE.length-1].path.length-1],color:[255,50,50,255],r:160,label:'🔴 Arrivée'}}];L.push(new ScatterplotLayer({{id:'se',data:se,getPosition:d=>d.position,getColor:d=>d.color,getRadius:d=>d.r,radiusMinPixels:12,radiusMaxPixels:28,pickable:true}}));}}
return L;}}
function refresh(){{deckInst.setProps({{layers:buildLayers()}});}}refresh();
for(let i=0;i<6;i++){{const el=document.getElementById('l'+i);el.addEventListener('change',()=>{{S['l'+i]=el.checked;refresh();}});}}
function setView(p){{currentPitch=p;deckInst.setProps({{initialViewState:{{...{js(view_state)},pitch:p,transitionDuration:600}}}});['v0','v1','v2'].forEach(id=>document.getElementById(id).classList.remove('on'));if(p===52)document.getElementById('v0').classList.add('on');else if(p===0)document.getElementById('v1').classList.add('on');else document.getElementById('v2').classList.add('on');}}
</script></body></html>"""
    return html

def generate_pydeck_terrain(points, cum_d_map, checkpoints, df_prediction=None,
                             tech_segments=None, dem_elevations=None,
                             osm_surface_data=None, active_layers=None,
                             height=600, pitch=52):
    return generate_3d_terrain_html(
        points=points, cum_d_map=cum_d_map, checkpoints=checkpoints,
        df_prediction=df_prediction, tech_segments=tech_segments,
        dem_elevations=dem_elevations, osm_surface_data=osm_surface_data,
        height=height, pitch=pitch,
    )


def generate_3d_animation(points, cum_d_map, checkpoints, total_dist_km, dem_elevations=None):
    n_pts = len(points); step = max(1, n_pts // 500)
    lats_a = [points[i].latitude  for i in range(0, n_pts, step)]
    lons_a = [points[i].longitude for i in range(0, n_pts, step)]
    if dem_elevations is not None and len(dem_elevations) == n_pts:
        elevs_raw = [dem_elevations[i] if dem_elevations[i] is not None else 0.0 for i in range(0, n_pts, step)]
    else:
        elevs_raw = [getattr(points[i], "elevation", 0.0) or 0.0 for i in range(0, n_pts, step)]
    dist_a = [cum_d_map[i] / 1000.0 for i in range(0, n_pts, step)]

    lat_min, lat_max = min(lats_a), max(lats_a); lon_min, lon_max = min(lons_a), max(lons_a)
    elev_min, elev_max = min(elevs_raw), max(elevs_raw); elev_range = max(1.0, elev_max - elev_min)
    lat_center=(lat_min+lat_max)/2; lon_center=(lon_min+lon_max)/2; span=max(lat_max-lat_min,lon_max-lon_min) or 1.0

    def norm_lon(v): return (v-lon_center)/span*2
    def norm_lat(v): return (v-lat_center)/span*2
    def norm_elev(v): return (v-elev_min)/elev_range

    coords_3d=[]
    for la,lo,el in zip(lats_a,lons_a,elevs_raw):
        coords_3d.append([round(norm_lon(lo),5),round(norm_elev(el)*0.6,5),round(-norm_lat(la),5)])

    cp_3d=[]; color_map_cp={"🥤 Ravitaillement":"#00e5ff","🏔 Sommet":"#ffd600","🔻 Col":"#e040fb","⏱ Point de passage":"#40c4ff","🏁 Intermédiaire":"#b0bec5","⚠️ Point clé":"#ff1744"}
    for cp in sorted(checkpoints,key=lambda c:c["dist_km"]):
        cp_elev=float(np.interp(cp["dist_km"]*1000,cum_d_map,[getattr(points[i],"elevation",0.0) or 0.0 for i in range(n_pts)]))
        cp_3d.append({"x":round(norm_lon(cp["lon"]),5),"y":round(norm_elev(cp_elev)*0.6+0.04,5),
                      "z":round(-norm_lat(cp["lat"]),5),"label":cp["label"],"dist":cp["dist_km"],
                      "color":color_map_cp.get(cp.get("type",""),"#f97316")})

    GRID=40; grid_x=np.linspace(-1.2,1.2,GRID); grid_z=np.linspace(-1.2,1.2,GRID); grid_pts=[]
    for gx in grid_x:
        for gz in grid_z:
            dists_sq=[(gx-c[0])**2+(gz-c[2])**2 for c in coords_3d]
            nearest=min(range(len(coords_3d)),key=lambda i:dists_sq[i]); dist_sq=dists_sq[nearest]
            base_y=coords_3d[nearest][1]; falloff=math.exp(-dist_sq*8.0)
            noise=(math.sin(gx*7.3+gz*5.1)*0.04+math.sin(gx*3.1-gz*8.7)*0.03+math.sin(gx*13.0+gz*2.3)*0.015)
            y_terrain=base_y*falloff+noise*(1.0-falloff*0.5)
            grid_pts.append(round(max(-0.05,y_terrain),4))

    COORDS_JS=_json.dumps(coords_3d); DIST_JS=_json.dumps([round(d,3) for d in dist_a])
    ELEV_JS=_json.dumps([round(e,1) for e in elevs_raw]); CP_JS=_json.dumps(cp_3d); GRID_JS=_json.dumps(grid_pts)
    TOT=str(round(total_dist_km,1)); EMIN=str(round(elev_min)); EMAX=str(round(elev_max))
    DPLUS=str(int(sum(max(0,elevs_raw[i]-elevs_raw[i-1]) for i in range(1,len(elevs_raw)))))
    GN=str(GRID); N=str(len(coords_3d))

    html="""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"/><title>Trail 3D</title>
<style>*{margin:0;padding:0;box-sizing:border-box;}body{background:#050510;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#e2e8f0;}#c{width:100%;height:100%;display:block;}#hud{position:fixed;top:0;left:0;right:0;height:44px;background:linear-gradient(180deg,rgba(5,5,20,.97) 0%,transparent 100%);display:flex;align-items:center;padding:0 16px;gap:6px;z-index:10;}#hud h1{font-size:.70rem;font-weight:800;letter-spacing:.1em;background:linear-gradient(90deg,#a78bfa,#e879f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;flex-shrink:0;}.st{display:flex;flex-direction:column;align-items:center;background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);border-radius:6px;padding:2px 9px;min-width:50px;}.sv{font-size:.80rem;font-weight:800;color:#a78bfa;font-variant-numeric:tabular-nums;}.sl{font-size:.40rem;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;}#ctrls{margin-left:auto;display:flex;gap:4px;}.btn{background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.25);color:#c4b5fd;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:.60rem;font-weight:700;transition:all .15s;}.btn:hover,.btn.on{background:rgba(167,139,250,.3);border-color:#a78bfa;color:#fff;}.btn.pl{background:#7c3aed;border-color:#7c3aed;color:#fff;}#prog{position:fixed;bottom:0;left:0;right:0;height:2px;background:rgba(167,139,250,.1);z-index:10;}#progf{height:100%;width:0;background:linear-gradient(90deg,#7c3aed,#e879f9);box-shadow:0 0 8px #a78bfa;transition:width .1s;}#toast{display:none;position:fixed;top:52px;left:50%;transform:translateX(-50%);background:rgba(5,5,20,.95);border:1px solid #a78bfa;border-radius:10px;padding:5px 14px;z-index:20;text-align:center;pointer-events:none;}#toast .tn{font-size:.75rem;font-weight:700;color:#a78bfa;}#toast .tm{font-size:.55rem;color:#6b7280;}</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><h1>⛰ TRAIL __TOT__ KM — D+__DPLUS__M — __EMIN__–__EMAX__M</h1>
<div class="st"><div class="sv" id="hd">0.0</div><div class="sl">km</div></div>
<div class="st"><div class="sv" id="he">—</div><div class="sl">m alt</div></div>
<div class="st"><div class="sv" id="hdp">0</div><div class="sl">d+</div></div>
<div id="ctrls"><button class="btn on" id="s1" onclick="spd(1)">1×</button><button class="btn" id="s2" onclick="spd(2)">2×</button><button class="btn" id="s4" onclick="spd(4)">4×</button><button class="btn on" id="bf" onclick="tgCam()">CAM</button><button class="btn pl" id="bp" onclick="tgPlay()">⏸</button><button class="btn" onclick="rst()">↺</button></div></div>
<div id="prog"><div id="progf"></div></div>
<div id="toast"><div class="tn" id="tn"></div><div class="tm" id="tm"></div></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
var COORDS=__COORDS__,DIST=__DIST__,ELEV=__ELEV__,CP=__CP__;
var GRID_Y=__GRID__,GN=__GN__,TOT=__TOT2__,EMIN=__EMIN2__,EMAX=__EMAX2__;
var N=COORDS.length;
var renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
var scene=new THREE.Scene();scene.background=new THREE.Color(0x050510);scene.fog=new THREE.FogExp2(0x0a0520,0.32);
var camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.001,20);camera.position.set(0,1.2,1.8);camera.lookAt(0,0,0);
scene.add(new THREE.AmbientLight(0x1a1040,2.0));var dlight=new THREE.DirectionalLight(0xff6040,1.8);dlight.position.set(1,2,-1);scene.add(dlight);
var pl1=new THREE.PointLight(0xa78bfa,3.0,4.0);pl1.position.set(0,0.8,0);scene.add(pl1);var pl2=new THREE.PointLight(0xe879f9,1.5,3.0);pl2.position.set(-0.5,0.6,0.5);scene.add(pl2);
(function(){var geom=new THREE.PlaneGeometry(2.8,2.8,GN-1,GN-1);geom.rotateX(-Math.PI/2);var pos=geom.attributes.position,cols=[];for(var i=0;i<pos.count;i++){pos.setY(i,GRID_Y[i]||0);}geom.computeVertexNormals();for(var i=0;i<pos.count;i++){var yv=pos.array[i*3+1],t=Math.max(0,Math.min(1,(yv+0.05)/0.65)),c=new THREE.Color();if(t<0.3)c.setRGB(0.10+t*0.3,0.06+t*0.15,0.20+t*0.2);else if(t<0.7)c.setRGB(0.28+t*0.25,0.12+t*0.20,0.25);else c.setRGB(0.55+t*0.35,0.50+t*0.40,0.55+t*0.35);cols.push(c.r,c.g,c.b);}geom.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));var mat=new THREE.MeshStandardMaterial({roughness:0.85,metalness:0.05,vertexColors:true});scene.add(new THREE.Mesh(geom,mat));})();
var trailPts=COORDS.map(function(c){return new THREE.Vector3(c[0],c[1]+0.008,c[2]);});var curve=new THREE.CatmullRomCurve3(trailPts,false,'catmullrom',0.3);var gG=new THREE.TubeGeometry(curve,N*2,0.009,6,false);scene.add(new THREE.Mesh(gG,new THREE.MeshBasicMaterial({color:0x7c3aed,transparent:true,opacity:0.10})));
function mkSph(x,y,z,col,r){var g=new THREE.SphereGeometry(r,12,8),m=new THREE.MeshBasicMaterial({color:col});var mesh=new THREE.Mesh(g,m);mesh.position.set(x,y,z);scene.add(mesh);var gh=new THREE.SphereGeometry(r*2.5,12,8),mh=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.15});var mhesh=new THREE.Mesh(gh,mh);mhesh.position.set(x,y,z);scene.add(mhesh);}
mkSph(COORDS[0][0],COORDS[0][1]+0.015,COORDS[0][2],0x00ff88,0.012);mkSph(COORDS[N-1][0],COORDS[N-1][1]+0.015,COORDS[N-1][2],0xff4040,0.012);CP.forEach(function(cp){mkSph(cp.x,cp.y,cp.z,parseInt(cp.color.replace('#','0x')),0.010);});
var dotG=new THREE.SphereGeometry(0.018,16,12),dotM=new THREE.MeshBasicMaterial({color:0xffffff});var dot=new THREE.Mesh(dotG,dotM);scene.add(dot);var haloG=new THREE.SphereGeometry(0.038,16,12),haloM=new THREE.MeshBasicMaterial({color:0xa78bfa,transparent:true,opacity:0.35});var halo=new THREE.Mesh(haloG,haloM);scene.add(halo);
var sg=new THREE.BufferGeometry(),sv=[];for(var i=0;i<1200;i++)sv.push((Math.random()-0.5)*12,Math.random()*6+1,(Math.random()-0.5)*12);sg.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xffffff,size:0.008,transparent:true,opacity:0.7})));
function neon(t){if(t<0.5)return new THREE.Color().setHSL(0.75+t*0.08,1.0,0.55+t*0.2);return new THREE.Color().setHSL(0.83-t*0.05,0.9,0.75+t*0.15);}
function lerp3(arr,f){var i=Math.min(Math.floor(f),arr.length-2),r=f-i;return new THREE.Vector3(arr[i][0]*(1-r)+arr[i+1][0]*r,arr[i][1]*(1-r)+arr[i+1][1]*r,arr[i][2]*(1-r)+arr[i+1][2]*r);}
function lerpA(arr,f){var i=Math.min(Math.floor(f),arr.length-2),r=f-i;return arr[i]*(1-r)+arr[i+1]*r;}
var segs=[],drawn=-1,frac=0,SPD=1,playing=true,followCam=true,lcp=-1,cumDP=0,lastSI=-1;var BASE=N/(TOT*60);var clock=new THREE.Clock();
function addSeg(a,b,t){var dir=new THREE.Vector3().subVectors(b,a),len=dir.length();if(len<0.0001)return;var col=neon(t);[0.003,0.009].forEach(function(r,ri){var g=new THREE.CylinderGeometry(r,r,len,5);g.translate(0,len/2,0);g.rotateX(Math.PI/2);var m=new THREE.MeshBasicMaterial({color:col,transparent:ri>0,opacity:ri>0?0.18:1.0});var mesh=new THREE.Mesh(g,m);mesh.position.copy(a);mesh.lookAt(b);scene.add(mesh);segs.push(mesh);});}
function animate(){requestAnimationFrame(animate);var dt=Math.min(clock.getDelta(),0.1);if(playing&&frac<N-1){frac=Math.min(frac+BASE*SPD*dt*60,N-1);var fi=Math.floor(frac);for(var si=drawn+1;si<=fi&&si<N-1;si++){addSeg(new THREE.Vector3(COORDS[si][0],COORDS[si][1]+0.008,COORDS[si][2]),new THREE.Vector3(COORDS[si+1][0],COORDS[si+1][1]+0.008,COORDS[si+1][2]),si/(N-1));drawn=si;}var pos=lerp3(COORDS,frac);pos.y+=0.022;dot.position.copy(pos);halo.position.copy(pos);var pulse=0.8+0.2*Math.sin(Date.now()*0.006);halo.scale.setScalar(pulse);haloM.opacity=0.25+0.15*pulse;pl1.position.set(pos.x,pos.y+0.3,pos.z);if(fi>0&&fi>lastSI){cumDP+=Math.max(0,(COORDS[fi][1]-COORDS[Math.max(0,fi-1)][1])*(parseFloat(EMAX)-parseFloat(EMIN)));lastSI=fi;}var d=lerpA(DIST,frac),e=lerpA(ELEV,frac);document.getElementById('hd').textContent=d.toFixed(2);document.getElementById('he').textContent=Math.round(e);document.getElementById('hdp').textContent=Math.round(cumDP);document.getElementById('progf').style.width=(frac/(N-1)*100)+'%';CP.forEach(function(cp,ci){if(ci!==lcp&&Math.abs(cp.dist-d)<0.3){lcp=ci;document.getElementById('tn').textContent=cp.label;document.getElementById('tm').textContent=cp.dist.toFixed(1)+' km · '+Math.round(e)+' m';var t2=document.getElementById('toast');t2.style.borderColor=cp.color;t2.style.display='block';clearTimeout(window._ct);window._ct=setTimeout(function(){t2.style.display='none';},2800);}});if(followCam){var ah=Math.min(frac+20,N-1),ap=lerp3(COORDS,ah);var ct=new THREE.Vector3(pos.x*0.3+ap.x*0.7,pos.y+0.55,pos.z*0.3+ap.z*0.7);camera.position.lerp(ct,0.025);camera.lookAt(pos.x,pos.y+0.05,pos.z);}if(frac>=N-1){playing=false;document.getElementById('bp').textContent='▶';}}var t3=Date.now()*0.001;pl2.position.x=Math.sin(t3*0.4)*0.3;pl2.position.z=Math.cos(t3*0.3)*0.3;renderer.render(scene,camera);}
animate();window.addEventListener('resize',function(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function spd(m){SPD=m;['s1','s2','s4'].forEach(function(id){document.getElementById(id).classList.remove('on');});document.getElementById('s'+m).classList.add('on');}
function tgCam(){followCam=!followCam;var b=document.getElementById('bf');b.classList.toggle('on',followCam);b.textContent=followCam?'CAM':'MAP';if(!followCam){camera.position.set(0,1.8,2.2);camera.lookAt(0,0,0);}}
function tgPlay(){playing=!playing;document.getElementById('bp').textContent=playing?'⏸':'▶';}
function rst(){playing=false;frac=0;drawn=-1;cumDP=0;lcp=-1;lastSI=-1;segs.forEach(function(s){scene.remove(s);s.geometry.dispose();s.material.dispose();});segs=[];dot.position.set(COORDS[0][0],COORDS[0][1]+0.022,COORDS[0][2]);halo.position.copy(dot.position);camera.position.set(0,1.2,1.8);camera.lookAt(0,0,0);document.getElementById('progf').style.width='0%';document.getElementById('bp').textContent='⏸';playing=true;}
</script></body></html>"""

    html=html.replace("__TOT__",TOT).replace("__DPLUS__",DPLUS).replace("__EMIN__",EMIN).replace("__EMAX__",EMAX)
    html=html.replace("__COORDS__",COORDS_JS).replace("__DIST__",DIST_JS).replace("__ELEV__",ELEV_JS)
    html=html.replace("__CP__",CP_JS).replace("__GRID__",GRID_JS).replace("__GN__",GN)
    html=html.replace("__TOT2__",TOT).replace("__EMIN2__",EMIN).replace("__EMAX2__",EMAX)
    return html


# ══════════════════════════════════════════════════════════════
# v9.0 — TEMPS DE MAINTIEN & ZONES CARDIAQUES (séance)
#   • Courbe des records : meilleure vitesse moyenne réellement tenue sur
#     chaque durée (1 min, 3 min, 5 min, 10 min, 20 min, 30 min, 1 h…).
#     C'est la contrepartie mesurée de la table de maintien théorique de
#     l'onglet Vitesse Critique.
#   • Temps passé par zone cardiaque, calculé sur le temps en mouvement.
# Ces deux résultats sont enregistrés avec la séance : ils restent comparables
# d'une séance à l'autre, même si le modèle évolue par la suite.
# ══════════════════════════════════════════════════════════════

RECORD_DURATIONS_S = (60, 180, 300, 600, 1200, 1800, 3600, 7200)
HR_ZONE_BOUNDS = (0.60, 0.70, 0.80, 0.90)      # en fraction de FC max
HR_ZONE_LABELS = ("Z1 · récup", "Z2 · endurance", "Z3 · tempo", "Z4 · seuil", "Z5 · VMA")

def _session_grid(df, step_s=1.0, moving_speed_ms=0.3):
    """Grille temporelle régulière (1 s) : distance, vitesse, FC, masque de
    mouvement. Base commune aux records et aux zones cardiaques."""
    if df is None or "elapsed_s" not in df.columns:
        return None
    d = df.dropna(subset=["elapsed_s"]).sort_values("elapsed_s").reset_index(drop=True)
    if len(d) < 10:
        return None
    t = d["elapsed_s"].astype(float).values
    # Garde-fou : certains fichiers donnent un temps absolu (epoch) au lieu d'un
    # temps écoulé. On repart toujours de 0, et on refuse les durées absurdes
    # plutôt que d'allouer une grille de plusieurs milliards de points.
    t = t - float(t[0])
    t_max = float(t[-1])
    if t_max < 60 or t_max > 36 * 3600:
        return None
    dist = d["distance_m"].astype(float).values if "distance_m" in d.columns else np.full(len(d), np.nan)
    if np.isnan(dist).all() or np.nanmax(dist) <= 1.0:
        if "speed_ms" in d.columns and d["speed_ms"].notna().any():
            sp = pd.Series(d["speed_ms"].astype(float)).interpolate().fillna(0.0).values
            dist = np.concatenate([[0.0], np.cumsum(sp[1:] * np.diff(t))])
        else:
            dist = None
    grid = np.arange(0.0, t_max, float(step_s))
    out = {"grid": grid, "step_s": float(step_s)}
    if dist is not None:
        dist = pd.Series(dist).interpolate(limit_direction="both").fillna(0.0).values
        dist = np.maximum.accumulate(dist)
        dist_g = np.interp(grid, t, dist)
        out["dist"] = dist_g
        spd = np.gradient(dist_g, float(step_s))
        out["speed_ms"] = np.clip(spd, 0.0, 12.0)
        out["moving"] = out["speed_ms"] >= float(moving_speed_ms)
    else:
        out["dist"] = None; out["speed_ms"] = None
        out["moving"] = np.ones(len(grid), dtype=bool)
    if "heart_rate" in d.columns and d["heart_rate"].notna().any():
        hr = pd.to_numeric(d["heart_rate"], errors="coerce")
        hr = hr.where((hr >= 40) & (hr <= 230))
        out["hr"] = np.interp(grid, t, hr.interpolate(limit_direction="both").fillna(0.0).values)
    else:
        out["hr"] = None
    # v9.1 — altitude et cadence, utilisées par le découpage en quarts
    if "altitude_m" in d.columns and d["altitude_m"].notna().any():
        alt = pd.Series(d["altitude_m"].astype(float)).interpolate(limit_direction="both").fillna(0.0)
        alt_g = np.interp(grid, t, alt.values)
        w_a = max(3, int(round(30.0 / float(step_s))))
        if w_a % 2 == 0:
            w_a += 1
        out["alt"] = pd.Series(alt_g).rolling(w_a, center=True, min_periods=1).median().values
    else:
        out["alt"] = None
    if "cadence_spm" in d.columns and d["cadence_spm"].notna().any():
        cad = pd.to_numeric(d["cadence_spm"], errors="coerce")
        _valid = cad.dropna(); _valid = _valid[_valid > 20]
        cad_g = np.interp(grid, t, cad.interpolate(limit_direction="both").fillna(0.0).values)
        if len(_valid) and float(_valid.median()) < 110.0:
            cad_g = cad_g * 2.0            # cadence par jambe → pas/min
        out["cadence"] = cad_g
    else:
        out["cadence"] = None
    return out

def compute_session_quarters(df, n_parts=4):
    """Découpe la séance en n parts de TEMPS EN MOUVEMENT égales (les pauses ne
    décalent pas le découpage) et calcule pour chacune : durée, distance, allure,
    D+, FC moyenne et max, cadence. Sert à voir si l'allure tient et comment la
    FC dérive au fil de la séance."""
    g = _session_grid(df)
    if g is None or g.get("dist") is None:
        return []
    step = g["step_s"]; mov = g["moving"]
    idx_mov = np.flatnonzero(mov)
    if len(idx_mov) < 4 * n_parts:
        return []
    chunks = np.array_split(idx_mov, int(n_parts))
    rows = []
    for i, ch in enumerate(chunks):
        if len(ch) < 2:
            continue
        i0, i1 = int(ch[0]), int(ch[-1])
        dur = float(len(ch) * step)
        dist = float(g["dist"][i1] - g["dist"][i0])
        row = {"part": i + 1, "libelle": f"Q{i + 1}",
               "t_debut_s": float(g["grid"][i0]), "t_fin_s": float(g["grid"][i1]),
               "duree_s": round(dur), "distance_m": round(dist),
               "vitesse_kmh": round(dist / max(1.0, dur) * 3.6, 2),
               "allure_s_km": round(dur / max(0.001, dist / 1000.0), 1) if dist > 10 else None}
        if g.get("hr") is not None:
            _h = g["hr"][ch]; _h = _h[(_h >= 40) & (_h <= 230)]
            if len(_h) > 5:
                row["fc_moy"] = round(float(np.mean(_h)), 1)
                row["fc_max"] = round(float(np.percentile(_h, 95)))
        if g.get("alt") is not None:
            _a = g["alt"][i0:i1 + 1]
            if len(_a) > 2:
                _d = np.diff(_a)
                row["d_plus"] = round(float(np.sum(np.clip(_d, 0, None))))
                row["d_moins"] = round(float(-np.sum(np.clip(_d, None, 0))))
        if g.get("cadence") is not None:
            _c = g["cadence"][ch]; _c = _c[_c > 20]
            if len(_c) > 5:
                row["cadence"] = round(float(np.median(_c)))
        if g.get("alt") is not None and g.get("speed_ms") is not None and len(ch) > 5:
            # VAP du quart : vitesse ramenée à son équivalent sur le plat (Minetti),
            # pour comparer des quarts qui n'ont pas le même dénivelé
            _alt_c = g["alt"][ch]; _dist_c = g["dist"][ch]
            _dd = np.diff(_dist_c); _de = np.diff(_alt_c)
            _grade = np.zeros(len(ch))
            _ok = _dd > 0.2
            _grade[1:][_ok] = np.clip(_de[_ok] / _dd[_ok] * 100.0, -45, 45)
            _ratios = np.array([vap_cost_ratio(gg) for gg in _grade])
            _spd_kmh = g["speed_ms"][ch] * 3.6
            row["vap_kmh"] = round(float(np.mean(_spd_kmh * _ratios)), 2)
        rows.append(row)
    # dérives par rapport au 1er quart : lecture immédiate de la tenue de l'effort
    if rows:
        p0 = rows[0].get("allure_s_km"); h0 = rows[0].get("fc_moy"); v0 = rows[0].get("vap_kmh")
        for r in rows:
            if p0 and r.get("allure_s_km"):
                r["derive_allure_pct"] = round((r["allure_s_km"] - p0) / p0 * 100.0, 1)
            if h0 and r.get("fc_moy"):
                r["derive_fc_bpm"] = round(r["fc_moy"] - h0, 1)
            if v0 and r.get("vap_kmh"):
                r["derive_vap_pct"] = round((r["vap_kmh"] - v0) / v0 * 100.0, 1)
    return rows

def plot_session_quarters(quarters):
    """Allure et FC par quart : deux panneaux distincts partageant l'axe des quarts
    (jamais deux échelles sur un même axe)."""
    has_hr = any(q.get("fc_moy") for q in quarters)
    fig, axes = plt.subplots(2 if has_hr else 1, 1, figsize=(11.5, 4.9 if has_hr else 3.1),
                             sharex=True, gridspec_kw={"hspace": 0.16})
    axes = np.atleast_1d(axes)
    x = np.arange(len(quarters))
    labels = [q["libelle"] for q in quarters]
    paces = [q.get("allure_s_km") or 0 for q in quarters]
    cols = [C_WHITE if i == 0 else (C_RED if (q.get("derive_allure_pct") or 0) > 3 else C_GREY)
            for i, q in enumerate(quarters)]
    axes[0].bar(x, paces, color=cols, width=0.6)
    for xi, q in zip(x, quarters):
        if q.get("allure_s_km"):
            _lab = pace_str(q["allure_s_km"]) + "/km"
            if q.get("derive_allure_pct") is not None and q["part"] > 1:
                _lab += f"\n{q['derive_allure_pct']:+.1f} %"
            axes[0].annotate(_lab, xy=(xi, q["allure_s_km"]), xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=8, color=C_TEXT)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Allure (min/km)")
    axes[0].set_yticks(axes[0].get_yticks())
    axes[0].set_yticklabels([pace_str(v) for v in axes[0].get_yticks()])
    axes[0].set_ylim(max(paces) * 1.16, min(p for p in paces if p) * 0.86)
    chart_title(axes[0], "Séance découpée en quarts",
                "Quarts de temps en mouvement · en rouge, un quart ralenti de plus de 3 % vs le premier")
    if has_hr:
        hrs = [q.get("fc_moy") or 0 for q in quarters]
        axes[1].bar(x, hrs, color=[C_WHITE if i == 0 else C_RED_SOFT for i in range(len(quarters))], width=0.6)
        for xi, q in zip(x, quarters):
            if q.get("fc_moy"):
                _lab = f"{q['fc_moy']:.0f} bpm"
                if q.get("derive_fc_bpm") is not None and q["part"] > 1:
                    _lab += f"\n{q['derive_fc_bpm']:+.0f}"
                axes[1].annotate(_lab, xy=(xi, q["fc_moy"]), xytext=(0, 6), textcoords="offset points",
                                 ha="center", fontsize=8, color=C_TEXT)
        _lo = min(h for h in hrs if h); _hi = max(hrs)
        axes[1].set_ylim(_lo * 0.90, _hi * 1.10)
        axes[1].set_ylabel("FC moyenne (bpm)")
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Quart de séance")
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        fig.tight_layout()
    return fig

def compute_session_records(df, durations_s=RECORD_DURATIONS_S):
    """Meilleure vitesse moyenne tenue sur chaque durée (fenêtre glissante sur
    toute la séance, pauses incluses dans la fenêtre : c'est bien la vitesse
    réellement soutenue sur ce laps de temps)."""
    g = _session_grid(df)
    if g is None or g.get("dist") is None:
        return []
    dist = g["dist"]; step = g["step_s"]; total_s = float(g["grid"][-1])
    rows = []
    for D in durations_s:
        n = int(round(D / step))
        if n < 2 or n >= len(dist):
            continue
        gains = dist[n:] - dist[:-n]
        if len(gains) == 0:
            continue
        i_best = int(np.argmax(gains))
        best_m = float(gains[i_best])
        if best_m <= 1.0:
            continue
        v_kmh = best_m / D * 3.6
        rows.append({"duree_s": int(D), "distance_m": round(best_m), "vitesse_kmh": round(v_kmh, 3),
                     "allure_s_km": round(D / (best_m / 1000.0), 1), "debut_s": float(g["grid"][i_best])})
    return rows

def compute_hr_zone_times(df, hr_max, bounds=HR_ZONE_BOUNDS):
    """Temps passé dans chaque zone cardiaque, en secondes, sur le temps en
    mouvement uniquement (les arrêts ne gonflent pas la zone 1)."""
    g = _session_grid(df)
    if g is None or g.get("hr") is None or not hr_max or hr_max <= 0:
        return []
    hr = g["hr"][g["moving"]]
    hr = hr[(hr >= 40) & (hr <= 230)]
    if len(hr) < 10:
        return []
    step = g["step_s"]
    frac = hr / float(hr_max)
    edges = [0.0] + list(bounds) + [10.0]
    rows = []
    for i in range(5):
        n = int(np.sum((frac >= edges[i]) & (frac < edges[i + 1])))
        rows.append({"zone": HR_ZONE_LABELS[i],
                     "bpm_min": round(edges[i] * hr_max) if i > 0 else None,
                     "bpm_max": round(edges[i + 1] * hr_max) if i < 4 else None,
                     "temps_s": round(n * step), "pct": round(n / max(1, len(hr)) * 100.0, 1)})
    return rows

def plot_records_curve(records, vc_ms=None, compare=None, title_suffix=""):
    """Courbe des records : vitesse moyenne maximale tenue en fonction de la durée.
    `compare` = liste de (label, records) pour superposer d'autres séances."""
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    _x = [r["duree_s"] / 60.0 for r in records]
    _y = [r["vitesse_kmh"] for r in records]
    if compare:
        for _i, (_lab, _rec) in enumerate(compare):
            if not _rec:
                continue
            _c = [C_GREY, C_WHITE, C_DIM, C_RED_SOFT][_i % 4]
            ax.plot([r["duree_s"] / 60.0 for r in _rec], [r["vitesse_kmh"] for r in _rec],
                    "-o", color=_c, lw=1.6, ms=5, alpha=0.85, mec=C_SURFACE, mew=1.2, label=_lab)
    ax.plot(_x, _y, "-o", color=C_RED, lw=2.2, ms=7, mec=C_SURFACE, mew=1.6,
            label="Séance analysée" if compare else None)
    for r in records:
        ax.annotate(pace_str(r["allure_s_km"]), xy=(r["duree_s"] / 60.0, r["vitesse_kmh"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color=C_TEXT_MUT)
    if vc_ms:
        ax.axhline(vc_ms * 3.6, color=C_WHITE, lw=1.2, ls=":",
                   label=f"Vitesse critique ({vc_ms*3.6:.2f} km/h)")
    ax.set_xscale("log")
    ax.set_xticks([r["duree_s"] / 60.0 for r in records])
    ax.set_xticklabels([(f"{r['duree_s']//60} min" if r["duree_s"] < 3600 else f"{r['duree_s']//3600} h")
                        for r in records], fontsize=8)
    ax.set_xlabel("Durée tenue"); ax.set_ylabel("Vitesse moyenne (km/h)")
    chart_title(ax, "Temps de maintien — meilleures vitesses tenues",
                "Pour chaque durée, la meilleure moyenne réellement réalisée dans la séance" + title_suffix)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig

def plot_hr_zones(zones, hr_max):
    """Répartition du temps en mouvement par zone cardiaque."""
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    labels = [z["zone"] for z in zones]
    vals = [z["temps_s"] / 60.0 for z in zones]
    cols = [C_DIM, C_GREY, C_WHITE, C_RED_SOFT, C_RED]
    bars = ax.barh(labels, vals, color=cols, height=0.62)
    for b, z in zip(bars, zones):
        if z["temps_s"] > 0:
            ax.annotate(f"{seconds_to_hms(z['temps_s'])}  ·  {z['pct']:.0f} %",
                        xy=(b.get_width(), b.get_y() + b.get_height() / 2), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8, color=C_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Temps (min)")
    ax.set_xlim(0, max(vals) * 1.28 if max(vals) > 0 else 1)
    chart_title(ax, "Temps par zone cardiaque",
                f"Zones calculées sur FC max = {hr_max:.0f} bpm · temps en mouvement uniquement")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# v9.2 — ZONES CALIBRÉES SUR LA VITESSE CRITIQUE
# Les zones d'intensité sont exprimées en % de la VC (vitesse critique) plutôt
# qu'en % de FC max : la VC est un repère métabolique mesuré sur l'athlète, là
# où la FC max est souvent estimée et bouge avec la chaleur, la fatigue ou la
# caféine. En trail, le temps par zone est calculé sur la VAP (vitesse ramenée
# au plat) quand l'altitude est disponible — sinon une montée classerait une
# séance intense en zone basse.
# ══════════════════════════════════════════════════════════════
VC_ZONE_BOUNDS_DEFAULT = (65.0, 75.0, 85.0, 95.0, 110.0)
VC_ZONE_LABELS = ("Sous-Z1 · marche / récup", "Z1 · endurance basse", "Z2 · endurance",
                  "Z3 · tempo", "Z4 · seuil / VC", "Z5 · au-dessus de VC")
VC_ZONE_COLORS_IDX = (0, 1, 2, 3, 4, 5)

def vc_zone_table(vc_ms, bounds=VC_ZONE_BOUNDS_DEFAULT):
    """Bornes de zones en % de VC, converties en vitesse et en allure."""
    if not vc_ms or vc_ms <= 0:
        return []
    edges = [0.0] + list(bounds) + [999.0]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        v_lo = vc_ms * lo / 100.0
        v_hi = vc_ms * hi / 100.0 if hi < 999 else None
        rows.append({
            "zone": VC_ZONE_LABELS[i] if i < len(VC_ZONE_LABELS) else f"Z{i}",
            "pct_lo": lo, "pct_hi": None if hi >= 999 else hi,
            "v_lo_kmh": round(v_lo * 3.6, 2), "v_hi_kmh": round(v_hi * 3.6, 2) if v_hi else None,
            "allure_lo_s_km": (1000.0 / v_hi) if v_hi and v_hi > 0 else None,   # borne rapide
            "allure_hi_s_km": (1000.0 / v_lo) if v_lo > 0 else None,            # borne lente
        })
    return rows

def compute_vc_zone_times(df, vc_ms, bounds=VC_ZONE_BOUNDS_DEFAULT, use_vap=True):
    """Temps passé dans chaque zone de VC, sur le temps en mouvement.
    use_vap=True : la vitesse de chaque instant est ramenée à son équivalent sur
    le plat (Minetti) avant d'être comparée à la VC — indispensable en trail."""
    if not vc_ms or vc_ms <= 0:
        return [], {"error": "Pas de vitesse critique connue pour cet athlète."}
    g = _session_grid(df)
    if g is None or g.get("speed_ms") is None:
        return [], {"error": "Pas de données de vitesse exploitables."}
    step = g["step_s"]
    mov = g["moving"]
    v = np.array(g["speed_ms"], dtype=float)
    used_vap = False
    if use_vap and g.get("alt") is not None and g.get("dist") is not None:
        d_dist = np.gradient(g["dist"], step)
        d_alt = np.gradient(g["alt"], step)
        with np.errstate(divide="ignore", invalid="ignore"):
            grade = np.where(d_dist > 0.2, d_alt / np.maximum(d_dist, 1e-6) * 100.0, 0.0)
        grade = np.clip(np.nan_to_num(grade), -45.0, 45.0)
        ratios = np.array([vap_cost_ratio(x) for x in grade])
        v = v * ratios
        used_vap = True
    v_mov = v[mov]
    if len(v_mov) < 10:
        return [], {"error": "Trop peu de temps en mouvement."}
    pct = v_mov / float(vc_ms) * 100.0
    edges = [0.0] + list(bounds) + [1e9]
    rows = []
    total = len(pct)
    for i in range(len(edges) - 1):
        n = int(np.sum((pct >= edges[i]) & (pct < edges[i + 1])))
        rows.append({"zone": VC_ZONE_LABELS[i] if i < len(VC_ZONE_LABELS) else f"Z{i}",
                     "pct_lo": edges[i], "pct_hi": None if edges[i + 1] > 1e8 else edges[i + 1],
                     "temps_s": round(n * step), "pct": round(n / max(1, total) * 100.0, 1)})
    infos = {"used_vap": used_vap, "vc_ms": float(vc_ms),
             "pct_median": round(float(np.median(pct)), 1),
             "moving_s": round(float(total * step))}
    return rows, infos

def plot_vc_zones(zones, vc_ms, used_vap=True):
    """Temps par zone d'intensité (référence : vitesse critique)."""
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    labels = [z["zone"] for z in zones]
    vals = [z["temps_s"] / 60.0 for z in zones]
    cols = [C_DIM, C_GREY, C_WHITE, "#C9C3B8", C_RED_SOFT, C_RED]
    bars = ax.barh(labels, vals, color=cols[:len(zones)], height=0.62)
    for b, z in zip(bars, zones):
        if z["temps_s"] > 0:
            _rng = (f"{z['pct_lo']:.0f}–{z['pct_hi']:.0f} %" if z.get("pct_hi") else f"> {z['pct_lo']:.0f} %")
            ax.annotate(f"{seconds_to_hms(z['temps_s'])}  ·  {z['pct']:.0f} %   ({_rng} VC)",
                        xy=(b.get_width(), b.get_y() + b.get_height() / 2), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8, color=C_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Temps (min)")
    ax.set_xlim(0, max(vals) * 1.45 if max(vals) > 0 else 1)
    chart_title(ax, "Temps par zone d'intensité — référence vitesse critique",
                f"VC = {vc_ms*3.6:.2f} km/h ({pace_str(1000.0/vc_ms)}/km)" +
                (" · calculé sur la VAP (vitesse ramenée au plat)" if used_vap else " · vitesse brute"))
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# v9.3 — PROFIL MÉTABOLIQUE À PARTIR D'UN MASQUE **CO₂ SEUL**
#
# Le masque mesure la ventilation et le CO₂ expiré. Il ne mesure PAS l'O₂.
# Cela change tout ce qu'on a le droit d'affirmer :
#
#   ✅ MESURÉ            : VE, VCO₂, et donc l'équivalent ventilatoire du CO₂
#                          (VE/VCO₂) → les SEUILS ventilatoires sont solides.
#   🟠 MODÉLISÉ          : VO₂, déduit de la mécanique de course (vitesse, pente,
#                          masse, économie de course). C'est une estimation, pas
#                          une mesure — l'économie varie de ±10 % entre coureurs.
#   🔴 DÉRIVÉ DU MODÈLE  : la partition glucides/lipides, qui repose sur le RER
#                          = VCO₂/VO₂ : sans O₂ mesuré, elle hérite de toute
#                          l'incertitude sur le VO₂ modélisé.
#
# L'app publie donc TROIS confiances distinctes plutôt qu'une seule : seuils,
# dépense énergétique, partition des substrats. C'est la façon honnête de
# présenter ce que ce capteur peut et ne peut pas dire.
#
# Équations :
#   • VO₂ modélisé : VO₂net (mL/kg/min) = économie (mL/kg/km) × vitesse (km/min)
#     × coût relatif de la pente (Minetti) ; VO₂brut = VO₂net + 3.5 (repos).
#   • Dépense : Weir (1949) réécrit à partir du CO₂ mesuré —
#     kcal/min = VCO₂ × (3.941/RER + 1.106). Sensibilité au RER ≈ ±12 % sur
#     toute la plage physiologique : la dépense reste correcte même quand la
#     partition, elle, ne l'est pas.
#   • Substrats : Jeukendrup & Wallis (2005), avec VO₂ modélisé.
#   • Seuils : VT2 = nadir de VE/VCO₂ (méthode standard du point de compensation
#     respiratoire, ne nécessite pas d'O₂) ; VT1 = rupture de pente de VCO₂ vs
#     intensité (le CO₂ s'emballe quand le tampon bicarbonate entre en jeu).
# ══════════════════════════════════════════════════════════════
CHO_A, CHO_B = 4.210, 2.962      # Jeukendrup & Wallis (2005)
FAT_A, FAT_B = 1.695, 1.701
WEIR_A, WEIR_B = 3.941, 1.106    # Weir (1949)
KCAL_PER_G_CHO = 4.07
KCAL_PER_G_FAT = 9.75
GLYCOGEN_G_PER_KG = 7.0          # réserves utilisables (muscle + foie), ~500 g pour 70 kg
ECONOMY_DEFAULT_ML_KG_KM = 200.0  # économie de course : 180 (élite) à 230 (loisir)
VO2_REST_ML_KG_MIN = 3.5

def estimate_vo2_running(speed_kmh, mass_kg, economy_ml_kg_km=ECONOMY_DEFAULT_ML_KG_KM, grade_pct=0.0):
    """VO₂ MODÉLISÉ à partir de la mécanique de course (pas mesuré).
    Le coût de la pente réutilise le modèle de Minetti déjà employé ailleurs
    dans l'app, pour rester cohérent d'un onglet à l'autre."""
    v_kmh = float(speed_kmh or 0.0)
    if v_kmh <= 0.3:
        return None
    ratio = vap_cost_ratio(grade_pct)
    vo2_ml_kg_min = float(economy_ml_kg_km) * (v_kmh / 60.0) * ratio + VO2_REST_ML_KG_MIN
    return vo2_ml_kg_min * float(mass_kg) / 1000.0        # L/min

def energy_from_vco2(vco2_lmin, rer):
    """Dépense énergétique à partir du CO₂ MESURÉ et d'un RER (mesuré ou modélisé).
    kcal/min = 3.941·VO₂ + 1.106·VCO₂ avec VO₂ = VCO₂/RER."""
    vco2 = max(0.0, float(vco2_lmin or 0.0))
    r = float(np.clip(rer or 0.85, 0.65, 1.30))
    return vco2 * (WEIR_A / r + WEIR_B)

def substrate_from_gas(vo2_lmin, vco2_lmin):
    """Oxydation des substrats. Nécessite un VO₂ : mesuré si l'appareil le donne,
    sinon modélisé — l'appelant doit le signaler à l'utilisateur."""
    vo2 = max(0.0, float(vo2_lmin or 0.0))
    vco2 = max(0.0, float(vco2_lmin or 0.0))
    if vo2 <= 0.05:
        return None
    rer = vco2 / vo2
    kcal_min = WEIR_A * vo2 + WEIR_B * vco2
    if rer < 0.70:
        # Sous 0.70 la partition n'a plus de sens physiologique : chez un coureur en
        # mouvement, cela signale un CO₂ encore en retard sur l'effort (début de test,
        # hyperventilation transitoire) ou une vitesse surestimée — pas une oxydation
        # lipidique record. On publie la dépense, pas la partition.
        return {"rer": round(rer, 3), "kcal_min": round(kcal_min, 3),
                "cho_g_min": None, "fat_g_min": None, "pct_cho_kcal": None,
                "mode": "rer_trop_bas"}
    if rer <= 1.0:
        cho = max(0.0, CHO_A * vco2 - CHO_B * vo2)
        fat = max(0.0, FAT_A * vo2 - FAT_B * vco2)
        mode = "partition"
    else:
        cho = kcal_min / KCAL_PER_G_CHO
        fat = 0.0
        mode = "cho_exclusif"
    return {"rer": round(rer, 3), "kcal_min": round(kcal_min, 3),
            "cho_g_min": round(cho, 3), "fat_g_min": round(fat, 3),
            "pct_cho_kcal": round(min(100.0, cho * KCAL_PER_G_CHO / max(1e-6, kcal_min) * 100.0), 1),
            "mode": mode}

def target_rer_at_intensity(pct_vc, target_rer_easy=0.82):
    """RER attendu en fonction de l'intensité relative. À basse intensité on brûle
    un mélange lipides/glucides (RER ~0.80) ; en approchant la vitesse critique la
    part glucidique domine (RER ~0.92). Sans intensité connue, on garde la valeur
    « facile » par défaut. C'est cette courbe qui sert d'ancrage au calage de
    l'économie quand aucun palier vraiment facile n'a été couru."""
    if pct_vc is None:
        return float(target_rer_easy)
    p = float(pct_vc)
    if p <= 70.0:
        return float(target_rer_easy) - 0.02
    if p >= 100.0:
        return float(target_rer_easy) + 0.10
    return (float(target_rer_easy) - 0.02) + (p - 70.0) * 0.12 / 30.0

def calibrate_economy(stage_means, mass_kg, target_rer_easy=0.82, easy_max_pct=None, vc_ms=None,
                      min_stages=3):
    """Cale l'économie de course pour que le RER des paliers les moins intenses
    retombe sur la valeur physiologiquement attendue à leur intensité. C'est ce qui
    remplace la mesure d'O₂ absente : on ancre le modèle là où l'on sait ce que le
    RER doit valoir.

    Deux garde-fous appris sur des tests réels :
      • le tout premier palier d'une rampe est écarté quand d'autres sont
        disponibles — le CO₂ expiré y est encore en train de monter depuis le
        repos, ce qui sous-estime l'économie ;
      • si moins de `min_stages` paliers tombent sous le seuil « facile » (cas d'un
        test qui démarre déjà vite), la fenêtre s'élargit vers le haut plutôt que
        de caler sur un seul palier — avec un RER cible qui suit l'intensité.
    Retourne (économie, n_paliers, écart-type, fenêtre_%VC réellement utilisée)."""
    usable = []
    for i, s in enumerate(stage_means):
        v = s.get("vitesse_kmh"); vco2 = s.get("vco2_lmin")
        if not v or not vco2 or v <= 0.3:
            continue
        pct = (v / 3.6 / float(vc_ms) * 100.0) if vc_ms else None
        usable.append({"i": i, "v": float(v), "vco2": float(vco2), "pct": pct,
                       "grade": s.get("grade_pct", 0.0)})
    if not usable:
        return ECONOMY_DEFAULT_ML_KG_KM, 0, None, None
    if len(usable) >= 4 and usable[0]["i"] == 0:
        usable = usable[1:]                     # rampe : on jette le palier de mise en route

    def _eco_of(c):
        rer_cible = target_rer_at_intensity(c["pct"], target_rer_easy)
        vo2_ml_kg_min = (c["vco2"] / rer_cible) * 1000.0 / max(1.0, float(mass_kg))
        ratio = vap_cost_ratio(c["grade"])
        return (vo2_ml_kg_min - VO2_REST_ML_KG_MIN) / max(1e-6, (c["v"] / 60.0) * ratio)

    if vc_ms and easy_max_pct is not None:
        ordered = sorted(usable, key=lambda c: (c["pct"] if c["pct"] is not None else 1e9))
        window = float(easy_max_pct)
        sel = [c for c in ordered if c["pct"] is not None and c["pct"] <= window]
        while len(sel) < int(min_stages) and window < 100.0:
            window = min(100.0, window + 5.0)
            sel = [c for c in ordered if c["pct"] is not None and c["pct"] <= window]
        if len(sel) < int(min_stages):
            sel = ordered[:max(int(min_stages), 1)]
            window = max([c["pct"] for c in sel if c["pct"] is not None] or [None]) if sel else None
    else:
        sel = usable
        window = None

    cands = [e for e in (_eco_of(c) for c in sel) if 120.0 <= e <= 320.0]
    if not cands:
        return ECONOMY_DEFAULT_ML_KG_KM, 0, None, window
    eco = float(np.clip(np.median(cands), 150.0, 260.0))
    return (eco, len(cands), (float(np.std(cands)) if len(cands) > 1 else None),
            (round(float(window), 0) if window else None))

def _piecewise_breakpoint(x, y):
    """Rupture de pente : ajuste deux droites et retourne le point de cassure, le
    gain de R² par rapport à une droite unique, et les deux pentes."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 6:
        return None
    _, _, r_single, _, _ = sp_stats.linregress(x, y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    best = None
    for i in range(2, len(x) - 3):
        s1, i1, _, _, _ = sp_stats.linregress(x[:i + 1], y[:i + 1])
        s2, i2, _, _, _ = sp_stats.linregress(x[i:], y[i:])
        pred = np.concatenate([s1 * x[:i + 1] + i1, s2 * x[i + 1:] + i2])
        ss = float(np.sum((y - pred) ** 2))
        if best is None or ss < best[0]:
            best = (ss, float(x[i]), float(s1), float(s2))
    if best is None or ss_tot <= 0:
        return None
    r2_2seg = 1.0 - best[0] / ss_tot
    return {"x_break": best[1], "slope_before": best[2], "slope_after": best[3],
            "r2_2seg": round(r2_2seg, 3), "r2_1seg": round(float(r_single ** 2), 3),
            "gain_r2": round(r2_2seg - float(r_single ** 2), 3)}

def detect_thresholds_co2(stages, vc_ms=None):
    """Seuils ventilatoires SANS oxygène :
      • VT2 (point de compensation respiratoire) = nadir de VE/VCO₂, puis remontée ;
      • VT1 = rupture de pente de VCO₂ en fonction de l'intensité.
    L'axe d'intensité est la vitesse si elle est connue, sinon la FC, sinon le
    numéro de palier : les seuils restent situables même sans tapis instrumenté."""
    st_ = [s for s in stages if s.get("vco2_lmin") and s.get("ve_lmin")]
    _key = ("vitesse_kmh" if all(s.get("vitesse_kmh") for s in st_) else
            ("hr" if all(s.get("hr") for s in st_) else "palier"))
    _unit = {"vitesse_kmh": "km/h", "hr": "bpm", "palier": "n° de palier"}[_key]
    st_ = sorted(st_, key=lambda s: s[_key])
    out = {"vt1": None, "vt2": None, "n_paliers": len(st_), "axe": _key, "axe_unite": _unit}
    if len(st_) < 5:
        out["message"] = "Il faut au moins 5 paliers exploitables pour situer les seuils."
        return out
    v = np.array([s[_key] for s in st_], dtype=float)
    ve = np.array([s["ve_lmin"] for s in st_], dtype=float)
    vco2 = np.array([s["vco2_lmin"] for s in st_], dtype=float)
    eqco2 = ve / np.maximum(1e-6, vco2)
    # ── VT2 : minimum de VE/VCO₂ (le nadir), suivi d'une remontée franche ──
    i_min = int(np.argmin(eqco2))
    if 0 < i_min < len(v) - 1:
        rise = float(eqco2[-1] - eqco2[i_min])
        out["vt2"] = {"x": round(float(v[i_min]), 2), "axe": _key,
                      "vitesse_kmh": (round(float(v[i_min]), 2) if _key == "vitesse_kmh"
                                      else st_[i_min].get("vitesse_kmh")),
                      "eqco2_nadir": round(float(eqco2[i_min]), 2),
                      "remontee": round(rise, 2), "palier": st_[i_min]["palier"],
                      "hr": st_[i_min].get("hr"),
                      "pct_vc": (round(float(v[i_min]) / 3.6 / vc_ms * 100.0, 1)
                                 if (vc_ms and _key == "vitesse_kmh") else None),
                      "fiable": bool(rise >= 1.5 and i_min >= 2)}
    # ── confirmation indépendante de VT2 : après le point de compensation, le CO₂
    # expiré CHUTE (l'hyperventilation « lave » le CO₂). Deux marqueurs concordants
    # valent mieux qu'un seul.
    _fe_all = [s.get("feco2") for s in st_]
    if all(f is not None for f in _fe_all) and len(_fe_all) >= 5 and out.get("vt2"):
        _fe_arr = pd.Series(_fe_all, dtype=float).rolling(3, center=True, min_periods=1).mean().values
        _i_peak = int(np.argmax(_fe_arr))
        _chute = float(_fe_arr[_i_peak] - _fe_arr[-1]) / max(1e-6, _fe_arr[_i_peak]) * 100.0
        if _i_peak < len(_fe_arr) - 1 and _chute >= 3.0:
            _x_peak = float(v[_i_peak])
            _ecart = abs(_x_peak - out["vt2"]["x"])
            _pas = float(np.median(np.diff(v))) if len(v) > 2 else 1.0
            out["vt2"]["confirmation_feco2"] = {"x": round(_x_peak, 2), "chute_pct": round(_chute, 1),
                                                "concordant": bool(_ecart <= 1.5 * abs(_pas) + 1e-6)}
            if out["vt2"]["confirmation_feco2"]["concordant"]:
                out["vt2"]["fiable"] = True

    # ── VT1 : le CO₂ EXPIRÉ (FeCO₂ ≈ PetCO₂) monte jusqu'à VT1 puis fait un plateau,
    # avant de chuter après VT2. La fin de la montée est donc un marqueur de VT1 qui ne
    # demande, lui non plus, aucun oxygène. On y recourt en priorité, et l'on retombe sur
    # la rupture de pente du VCO₂ si le CO₂ expiré n'est pas dans le fichier.
    _fe = [s.get("feco2") for s in st_]
    bp = None
    _source_vt1 = None
    if all(f is not None for f in _fe) and len(_fe) >= 6:
        _bpf = _piecewise_breakpoint(v, np.array(_fe, dtype=float))
        # plateau/chute après le point de rupture : la pente doit nettement s'affaisser
        if _bpf and _bpf["slope_before"] > 0 and _bpf["slope_after"] < _bpf["slope_before"] * 0.4:
            bp = {"x_break": _bpf["x_break"], "slope_before": _bpf["slope_after"],
                  "slope_after": _bpf["slope_before"], "gain_r2": _bpf["gain_r2"], "r2_2seg": _bpf["r2_2seg"]}
            _source_vt1 = "plateau du CO₂ expiré (FeCO₂)"
    if bp is None:
        bp = _piecewise_breakpoint(v, vco2)
        _source_vt1 = "rupture de pente du VCO₂"
    if bp and bp["slope_after"] > bp["slope_before"] * 1.05:
        i_bp = int(np.argmin(np.abs(v - bp["x_break"])))
        _vt2_v = (out.get("vt2") or {}).get("x")
        # garde-fou : un VT1 situé au-dessus du VT2 n'a pas de sens physiologique
        _coherent = (_vt2_v is None) or (bp["x_break"] < _vt2_v - 0.05)
        out["vt1"] = {"x": round(bp["x_break"], 2), "axe": _key,
                      "vitesse_kmh": (round(bp["x_break"], 2) if _key == "vitesse_kmh"
                                      else st_[i_bp].get("vitesse_kmh")),
                      "palier": st_[i_bp]["palier"], "hr": st_[i_bp].get("hr"),
                      "pct_vc": (round(bp["x_break"] / 3.6 / vc_ms * 100.0, 1)
                                 if (vc_ms and _key == "vitesse_kmh") else None),
                      "gain_r2": bp["gain_r2"], "r2": bp["r2_2seg"], "coherent_vs_vt2": bool(_coherent),
                      "methode": _source_vt1,
                      "fiable": bool(bp["gain_r2"] >= 0.02 and bp["r2_2seg"] >= 0.95 and _coherent)}
        out["vt1_note"] = (f"VT1 obtenu par {_source_vt1}. Sans oxygène, la méthode de référence (V-slope, "
                           "qui compare VCO₂ et VO₂) est inapplicable : ce repère est indicatif, contrairement "
                           "à VT2 que le nadir de VE/VCO₂ situe directement.")
    if out.get("vt1") is None or not (out.get("vt1") or {}).get("fiable"):
        _vt2x = (out.get("vt2") or {}).get("x")
        if _vt2x and float(v[0]) >= 0.72 * float(_vt2x):
            out["vt1_message"] = (
                f"VT1 non identifiable : le test démarre déjà à {v[0]:.1f} {_unit}, soit "
                f"{float(v[0])/float(_vt2x)*100:.0f} % de l'intensité de VT2. Le premier seuil est "
                f"vraisemblablement SOUS la plage testée — pour le capter, il faudrait commencer le "
                f"protocole nettement plus bas (2 à 3 paliers très faciles).")
        else:
            out["vt1_message"] = ("VT1 non identifiable : aucune rupture nette dans le CO₂ produit ni dans le "
                                  "CO₂ expiré. Des paliers plus longs (3-5 min) rendraient ce signal plus lisible.")
    out["eqco2_curve"] = [{"x": round(float(a), 2), "eqco2": round(float(b), 2)}
                          for a, b in zip(v, eqco2)]
    return out

def _confidences(dur_s, cv_mean_pct, rer, n_prior_tests, n_samples, missing_frac,
                 vo2_measured, eco_calibrated, eco_spread_pct):
    """Trois confiances distinctes : ce que le capteur mesure vraiment (seuils),
    ce qu'il permet d'estimer correctement (dépense), et ce qui dépend d'un VO₂
    modélisé (partition des substrats)."""
    stab = float(np.clip(100.0 - 6.0 * float(cv_mean_pct), 0.0, 100.0))
    d = float(dur_s)
    if d < 90:      dur_score = 55.0 + (d / 90.0) * 10.0
    elif d < 180:   dur_score = 65.0 + (d - 90) / 90.0 * 15.0
    elif d < 300:   dur_score = 80.0 + (d - 180) / 120.0 * 12.0
    else:           dur_score = 92.0 + min(6.0, (d - 300) / 300.0 * 6.0)
    r = float(rer or 0.85)
    if 0.75 <= r <= 0.95:   rer_score = 100.0
    elif 0.95 < r <= 1.00:  rer_score = 100.0 - (r - 0.95) / 0.05 * 15.0
    elif r > 1.00:          rer_score = float(np.clip(85.0 - (r - 1.0) / 0.15 * 55.0, 30.0, 85.0))
    else:                   rer_score = float(np.clip(60.0 + (r - 0.70) / 0.05 * 30.0, 50.0, 100.0))
    calib = float(min(85.0, 55.0 + 6.0 * int(n_prior_tests)))
    signal = float(np.clip(100.0 - 60.0 * float(missing_frac) - (10.0 if n_samples < 5 else 0.0), 0.0, 100.0))
    # qualité du VO₂ : mesuré = 100 ; modélisé = pénalisé, un peu moins si l'économie
    # a pu être calée sur les paliers faciles de CE test
    if vo2_measured:
        vo2_score = 100.0
    else:
        vo2_score = 62.0 if eco_calibrated else 45.0
        if eco_spread_pct is not None:
            vo2_score -= float(np.clip(eco_spread_pct, 0.0, 20.0))    # économies dispersées = moins sûr
        vo2_score = float(np.clip(vo2_score, 30.0, 70.0))
    conf_seuils = float(np.clip(0.45 * stab + 0.25 * dur_score + 0.20 * signal + 0.10 * calib, 35.0, 95.0))
    conf_energie = float(np.clip(0.28 * stab + 0.22 * dur_score + 0.20 * rer_score
                                 + 0.20 * (85.0 if not vo2_measured else 100.0) + 0.10 * calib, 35.0, 95.0))
    conf_substrats = float(np.clip(0.22 * stab + 0.18 * dur_score + 0.20 * rer_score
                                   + 0.30 * vo2_score + 0.10 * calib, 25.0, 95.0))
    return {"conf_seuils": round(conf_seuils, 1), "conf_energie": round(conf_energie, 1),
            "conf_substrats": round(conf_substrats, 1), "confiance": round(conf_energie, 1),
            "q_stabilite": round(stab), "q_duree": round(dur_score), "q_rer": round(rer_score),
            "q_calibration": round(calib), "q_signal": round(signal), "q_vo2": round(vo2_score)}

def confidence_label(conf):
    c = float(conf or 0)
    if c >= 90: return "Très forte"
    if c >= 80: return "Forte"
    if c >= 70: return "Bonne"
    if c >= 60: return "Modérée"
    if c >= 50: return "Faible"
    return "Insuffisante"

def detect_protocol(durations_s):
    """Ramp, paliers longs, ou protocole mixte (ramp + validation)."""
    if not durations_s:
        return "inconnu", ""
    d = np.array(durations_s, dtype=float)
    med = float(np.median(d))
    short = int(np.sum(d < 110)); long_ = int(np.sum(d >= 200))
    if short >= 3 and long_ >= 2:
        return "mixte", ("Ramp + paliers de validation — le meilleur des deux : les paliers courts "
                         "situent les transitions, les longs calibrent les estimations.")
    if med < 110:
        return "ramp", ("Ramp (~1 min/palier) — excellent pour situer les transitions ventilatoires, "
                        "mais l'état stable n'est pas atteint : les substrats sont publiés comme des "
                        "plages, avec une confiance plafonnée.")
    if med < 200:
        return "paliers_courts", "Paliers courts (2-3 min) — compromis correct entre stabilité et durée du test."
    return "paliers_longs", "Paliers longs (≥ 3 min) — réponse ventilatoire stabilisée, estimations les plus fiables."

def _stage_means(df_gz, window_frac=0.5):
    """Moyennes de fin de palier (partie la plus proche de l'état stable)."""
    out = []
    for pal, sub in df_gz[df_gz["palier"] > 0].groupby("palier"):
        sub = sub.sort_values("elapsed_s")
        if len(sub) < 3:
            continue
        t0, t1 = float(sub["elapsed_s"].iloc[0]), float(sub["elapsed_s"].iloc[-1])
        dur = max(1.0, t1 - t0)
        win = sub[sub["elapsed_s"] >= t1 - dur * float(window_frac)]
        if len(win) < 3:
            win = sub
        vco2 = pd.to_numeric(win.get("VCO2"), errors="coerce")
        ve = pd.to_numeric(win.get("VE"), errors="coerce")
        missing = float(vco2.isna().mean())
        vco2 = vco2.dropna(); ve = ve.dropna()
        if len(vco2) < 2 or vco2.mean() <= 0.02:
            continue
        _feco2 = pd.to_numeric(win.get("FeCO2"), errors="coerce").dropna() if "FeCO2" in win.columns else []
        rec = {"palier": int(pal), "t_debut_s": t0, "duree_s": round(dur), "n": int(len(vco2)),
               "missing_frac": missing,
               "feco2": (round(float(np.mean(_feco2)) * 100.0, 3) if len(_feco2) else None),
               "vco2_lmin": float(vco2.mean()), "vco2_sd": float(vco2.std() or 0.0),
               "ve_lmin": float(ve.mean()) if len(ve) else None,
               "hr": (lambda h: round(float(h.mean())) if len(h) and float(h.mean()) > 40 else None)(
                   pd.to_numeric(win.get("HR"), errors="coerce").dropna()),
               "vitesse_kmh": (lambda v: round(float(v.mean()), 2) if len(v) and float(v.mean()) > 0.3 else None)(
                   pd.to_numeric(win.get("Cadence"), errors="coerce").dropna()),
               "grade_pct": 0.0}
        vo2_col = pd.to_numeric(win.get("VO2_Lmin"), errors="coerce").dropna() if "VO2_Lmin" in win.columns else []
        if len(vo2_col) >= 2 and float(vo2_col.mean()) > 0.05:
            rec["vo2_mesure_lmin"] = float(vo2_col.mean())
            rec["vo2_mesure_sd"] = float(vo2_col.std() or 0.0)
        out.append(rec)
    out = sorted(out, key=lambda r: r["palier"])
    # Un dernier « palier » nettement plus court que les autres est presque toujours
    # un retour au calme ou une coupure d'enregistrement : il fausserait les seuils
    # (vitesse qui redescend) et l'ancrage de l'économie. On l'écarte, en le disant.
    if len(out) >= 4:
        _med = float(np.median([r["duree_s"] for r in out]))
        _drop = [r["palier"] for r in out if r["duree_s"] < 0.60 * _med]
        if _drop:
            out = [r for r in out if r["palier"] not in _drop]
            for r in out:
                r.setdefault("_paliers_ecartes", _drop)
    return out

def analyze_metabolic_stages(df_gz, mass_kg=70.0, vc_ms=None, n_prior_tests=0, window_frac=0.5,
                             economy_ml_kg_km=None, auto_calibrate_economy=True,
                             prefer_measured_vo2=True, target_rer_easy=0.82, easy_max_pct=80.0,
                             speed_by_stage=None):
    """Analyse palier par palier avec un masque CO₂ seul.
    Le VO₂ est modélisé (mécanique de course + économie) sauf si le fichier
    contient un VO₂ réellement mesuré. Trois confiances sont produites."""
    if df_gz is None or "palier" not in df_gz.columns:
        return [], {"error": "Fichier sans découpage en paliers."}
    means = _stage_means(df_gz, window_frac)
    if not means:
        return [], {"error": "Aucun palier exploitable (VCO₂ manquant ou trop court)."}
    # vitesses saisies à la main (tapis sans capteur, allure notée sur le carnet…) :
    # elles priment sur ce que contient le fichier
    if speed_by_stage:
        for m in means:
            _v = speed_by_stage.get(m["palier"]) or speed_by_stage.get(str(m["palier"]))
            if _v and float(_v) > 0.3:
                m["vitesse_kmh"] = round(float(_v), 2)
    has_speed = any(m.get("vitesse_kmh") for m in means)
    vo2_measured = prefer_measured_vo2 and all(m.get("vo2_mesure_lmin") for m in means)
    eco_used, eco_n, eco_sd, eco_window = None, 0, None, None
    substrats_possibles = True
    if not vo2_measured:
        if not has_speed:
            # Pas d'O₂ ET pas de vitesse : les seuils ventilatoires et la dépense restent
            # calculables (le CO₂ suffit), la partition des substrats non — on le dit au
            # lieu de refuser tout le fichier.
            substrats_possibles = False
        elif economy_ml_kg_km:
            eco_used = float(economy_ml_kg_km)
        elif auto_calibrate_economy and has_speed:
            eco_used, eco_n, eco_sd, eco_window = calibrate_economy(
                means, mass_kg, target_rer_easy, easy_max_pct if vc_ms else None, vc_ms)
        else:
            eco_used = ECONOMY_DEFAULT_ML_KG_KM
    eco_spread_pct = (eco_sd / eco_used * 100.0) if (eco_sd and eco_used) else None
    eco_alerte = None
    if eco_used and eco_n and (eco_used <= 151.0 or eco_used >= 259.0):
        eco_alerte = (f"L'économie calée bute sur une borne ({eco_used:.0f} mL/kg/km). Concrètement, le CO₂ "
                      f"mesuré ne colle pas aux vitesses fournies : vérifie les allures saisies, la masse, ou "
                      f"le RER de référence des paliers faciles. Les seuils et la dépense restent valables ; "
                      f"la partition glucides/lipides, elle, est à prendre avec des pincettes.")
    elif eco_used and eco_n and not (170.0 <= eco_used <= 240.0):
        eco_alerte = (f"Économie calée à {eco_used:.0f} mL/kg/km, hors de la fourchette habituelle "
                      f"(180-230). Soit l'athlète est réellement très (peu) économe, soit le RER de "
                      f"référence retenu pour les paliers d'ancrage ne lui correspond pas.")
    if eco_used and eco_n and eco_n < 3:
        _msg_n = (f"Économie calée sur {eco_n} palier(s) seulement : l'ancrage est fragile, la partition "
                  f"glucides/lipides plus qu'indicative. Un test qui démarre 2 ou 3 paliers plus bas "
                  f"donnerait un calage beaucoup plus solide.")
        eco_alerte = (eco_alerte + " " + _msg_n) if eco_alerte else _msg_n

    rows = []
    RER_DEFAUT_SANS_VITESSE = 0.90     # hypothèse explicite quand rien ne permet de le modéliser
    for m in means:
        if vo2_measured:
            vo2 = m["vo2_mesure_lmin"]; vo2_sd = m.get("vo2_mesure_sd", 0.0)
        elif substrats_possibles:
            vo2 = estimate_vo2_running(m.get("vitesse_kmh"), mass_kg, eco_used, m.get("grade_pct", 0.0))
            vo2_sd = 0.0
            if vo2 is None:
                continue
        else:
            vo2 = None; vo2_sd = 0.0
        vco2, vco2_sd, n = m["vco2_lmin"], m["vco2_sd"], max(1, m["n"])
        cv = float(vco2_sd / max(1e-6, vco2) * 100.0)
        vco2_sem = vco2_sd / math.sqrt(n)
        vo2_sem = vo2_sd / math.sqrt(n)
        if vo2 is None:
            # dépense seule, avec un RER supposé et une plage large qui l'assume
            sub_m = {"rer": RER_DEFAUT_SANS_VITESSE, "kcal_min": energy_from_vco2(vco2, RER_DEFAUT_SANS_VITESSE),
                     "cho_g_min": None, "fat_g_min": None, "pct_cho_kcal": None, "mode": "sans_vo2"}
        else:
            sub_m = substrate_from_gas(vo2, vco2)
            if sub_m is None:
                continue
        conf = _confidences(m["duree_s"], cv, sub_m["rer"], n_prior_tests, n, m["missing_frac"],
                            vo2_measured, bool(eco_n), eco_spread_pct)
        if vo2 is None:
            conf["conf_substrats"] = 0.0
            cho_lo = cho_hi = fat_lo = fat_hi = None
            _rer_band = 0.12          # RER totalement supposé : plage de dépense élargie
        else:
            # incertitude des substrats : mesure (erreur-type) + incertitude sur le VO₂ modélisé
            vo2_model_unc = 0.0 if vo2_measured else vo2 * 0.08     # ±8 % sur l'économie individuelle
            lo = substrate_from_gas(vo2 + vo2_sem + vo2_model_unc, max(0.01, vco2 - vco2_sem))
            hi = substrate_from_gas(max(0.01, vo2 - vo2_sem - vo2_model_unc), vco2 + vco2_sem)
            widen = (1.0 - conf["conf_substrats"] / 100.0) * 0.25
            if (sub_m.get("cho_g_min") is None or lo.get("cho_g_min") is None
                    or hi.get("cho_g_min") is None):
                # palier hors plage physiologique : pas de partition, donc pas de plage
                cho_lo = cho_hi = fat_lo = fat_hi = None
                conf["conf_substrats"] = 0.0
            else:
                cho_lo = max(0.0, min(lo["cho_g_min"], hi["cho_g_min"]) * (1 - widen))
                cho_hi = max(cho_lo, max(lo["cho_g_min"], hi["cho_g_min"]) * (1 + widen))
                fat_lo = max(0.0, min(lo["fat_g_min"], hi["fat_g_min"]) * (1 - widen))
                fat_hi = max(fat_lo, max(lo["fat_g_min"], hi["fat_g_min"]) * (1 + widen))
            _rer_band = 0.06
        # dépense : bornée par l'incertitude sur le RER (le CO₂, lui, est mesuré)
        kcal_min = energy_from_vco2(vco2, sub_m["rer"])
        kcal_lo = energy_from_vco2(vco2 - vco2_sem, min(1.15, sub_m["rer"] + _rer_band))
        kcal_hi = energy_from_vco2(vco2 + vco2_sem, max(0.70, sub_m["rer"] - _rer_band))
        row = {"palier": m["palier"], "t_debut_s": m["t_debut_s"], "duree_s": m["duree_s"],
               "vco2_lmin": round(vco2, 3), "ve_lmin": round(m["ve_lmin"], 1) if m.get("ve_lmin") else None,
               "eqco2": round(m["ve_lmin"] / max(1e-6, vco2), 1) if m.get("ve_lmin") else None,
               "vo2_lmin": round(vo2, 3) if vo2 else None,
               "vo2_source": "mesuré" if vo2_measured else ("modélisé" if vo2 else "indisponible"),
               "vo2_ml_kg_min": round(vo2 * 1000.0 / max(1.0, mass_kg), 1) if vo2 else None,
               "cv_pct": round(cv, 1), "hr": m.get("hr"), "vitesse_kmh": m.get("vitesse_kmh"),
               "feco2": m.get("feco2"),
               "cho_g_min": sub_m["cho_g_min"],
               "cho_lo": round(cho_lo, 2) if cho_lo is not None else None,
               "cho_hi": round(cho_hi, 2) if cho_hi is not None else None,
               "fat_g_min": sub_m["fat_g_min"],
               "fat_lo": round(fat_lo, 2) if fat_lo is not None else None,
               "fat_hi": round(fat_hi, 2) if fat_hi is not None else None,
               "kcal_h": round(kcal_min * 60.0), "kcal_h_lo": round(kcal_lo * 60.0), "kcal_h_hi": round(kcal_hi * 60.0),
               "rer": sub_m["rer"], "pct_cho_kcal": sub_m["pct_cho_kcal"], "mode": sub_m["mode"]}
        if m.get("vitesse_kmh"):
            km_per_min = m["vitesse_kmh"] / 60.0
            row["kcal_km"] = round(kcal_min / max(1e-6, km_per_min), 1)
            row["kcal_kg_km"] = round(row["kcal_km"] / max(1.0, mass_kg), 3)
            row["cho_g_km"] = (round(sub_m["cho_g_min"] / max(1e-6, km_per_min), 2)
                               if sub_m.get("cho_g_min") is not None else None)
            if vc_ms:
                row["pct_vc"] = round(m["vitesse_kmh"] / 3.6 / float(vc_ms) * 100.0, 1)
        row.update(conf)
        rows.append(row)
    if not rows:
        return [], {"error": "Paliers inexploitables après contrôle des données."}
    proto, proto_msg = detect_protocol([r["duree_s"] for r in rows])
    thresholds = detect_thresholds_co2(rows, vc_ms)
    infos = {"protocole": proto, "protocole_msg": proto_msg, "n_paliers": len(rows),
             "vo2_source": "mesuré" if vo2_measured else "modélisé",
             "economie_ml_kg_km": round(eco_used, 1) if eco_used else None,
             "economie_calibree": bool(eco_n), "economie_n_paliers": eco_n, "economie_alerte": eco_alerte,
             "economie_fenetre_pct": eco_window,
             "paliers_ecartes": (means[0].get("_paliers_ecartes") if means else None),
             "economie_dispersion_pct": round(eco_spread_pct, 1) if eco_spread_pct else None,
             "mass_kg": float(mass_kg), "vc_ms": vc_ms, "n_prior_tests": int(n_prior_tests),
             "seuils": thresholds,
             "conf_seuils": round(float(np.mean([r["conf_seuils"] for r in rows])), 1),
             "conf_energie": round(float(np.mean([r["conf_energie"] for r in rows])), 1),
             "conf_substrats": round(float(np.mean([r["conf_substrats"] for r in rows])), 1)}
    infos["confiance_globale"] = infos["conf_energie"]
    infos["substrats_possibles"] = bool(substrats_possibles or vo2_measured)
    if not infos["substrats_possibles"]:
        infos["substrats_msg"] = (
            "Ni oxygène mesuré, ni vitesse renseignée : la partition glucides/lipides est impossible "
            "(elle exige le RER = VCO₂/VO₂). Les seuils ventilatoires et la dépense énergétique restent "
            "calculés — saisis les vitesses des paliers ci-dessus pour débloquer les substrats.")
    # FatMax : n'a de sens que si la courbe d'oxydation lipidique présente un vrai
    # maximum INTERNE, à une intensité plausible. Un maximum sur le dernier palier
    # (ou au-delà de la vitesse critique) n'est pas un FatMax : c'est du bruit de
    # RER, inévitable avec des paliers d'une minute et un VO₂ modélisé. Mieux vaut
    # dire qu'il n'est pas résolu que d'afficher une valeur fausse.
    _with_fat = [r for r in rows if r.get("fat_g_min") is not None]
    if _with_fat:
        _ordered = sorted(_with_fat, key=lambda r: (r.get("pct_vc") or r.get("vitesse_kmh") or r["palier"]))
        _best = max(_ordered, key=lambda r: r["fat_g_min"])
        _idx = _ordered.index(_best)
        _interne = 0 < _idx < len(_ordered) - 1
        _plausible = (_best.get("pct_vc") is None) or (_best["pct_vc"] <= 92.0)
        if len(_ordered) >= 4 and _interne and _plausible:
            infos["fatmax"] = {"palier": _best["palier"], "fat_g_min": _best["fat_g_min"],
                               "pct_vc": _best.get("pct_vc"), "vitesse_kmh": _best.get("vitesse_kmh"),
                               "hr": _best.get("hr")}
        else:
            infos["fatmax_msg"] = (
                "FatMax non résolu sur ce test : l'oxydation lipidique la plus haute tombe "
                + ("sur un palier extrême du test" if not _interne else
                   f"à {_best['pct_vc']:.0f} % de la VC, trop haut pour être un vrai FatMax")
                + ". Avec des paliers d'une minute et un VO₂ modélisé, le RER est trop bruité pour "
                  "situer un pic ; il faudrait des paliers de 3-5 min démarrant nettement plus bas.")
    _rers = [r["rer"] for r in rows] if infos["substrats_possibles"] else []
    if _rers and (max(_rers) > 1.25 or min(_rers) < 0.68):
        infos["alerte_rer"] = (f"RER modélisé hors plage physiologique ({min(_rers):.2f}–{max(_rers):.2f}) : "
                               "l'économie de course retenue ou la vitesse des paliers est probablement "
                               "inexacte. Ajuste l'économie ou vérifie la colonne vitesse du fichier.")
    return rows, infos

def fueling_plan(stages, vc_ms, bounds=VC_ZONE_BOUNDS_DEFAULT, gut_cap_g_h=90.0, mass_kg=70.0,
                 glycogen_g_per_kg=GLYCOGEN_G_PER_KG):
    """Plan nutritionnel par zone de VC : oxydation glucidique estimée, apport
    conseillé (plafonné par la tolérance intestinale) et autonomie glycogénique."""
    usable = [s for s in stages if s.get("pct_vc") and s.get("cho_g_min") is not None]
    if not usable or not vc_ms:
        return [], {"error": "Il faut la vitesse critique de l'athlète et des paliers avec vitesse "
                             "pour construire un plan nutritionnel."}
    xs = np.array([s["pct_vc"] for s in usable], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    cho = np.array([usable[i]["cho_g_min"] for i in order], dtype=float)
    cho_lo = np.array([usable[i]["cho_lo"] for i in order], dtype=float)
    cho_hi = np.array([usable[i]["cho_hi"] for i in order], dtype=float)
    kcal = np.array([usable[i]["kcal_h"] for i in order], dtype=float)
    conf = np.array([usable[i]["conf_substrats"] for i in order], dtype=float)
    edges = [0.0] + list(bounds) + [130.0]
    stores_g = float(mass_kg) * float(glycogen_g_per_kg)
    rows = []
    for i in range(len(edges) - 1):
        lo_e, hi_e = edges[i], edges[i + 1]
        mid = (lo_e + hi_e) / 2.0
        extrapolated = mid < xs.min() - 5 or mid > xs.max() + 5
        _c = float(np.interp(mid, xs, cho))
        _clo = float(np.interp(mid, xs, cho_lo)); _chi = float(np.interp(mid, xs, cho_hi))
        _k = float(np.interp(mid, xs, kcal))
        _conf = float(np.interp(mid, xs, conf)) * (0.75 if extrapolated else 1.0)
        ox_h = _c * 60.0
        intake = min(float(gut_cap_g_h), ox_h)
        deficit = max(0.0, ox_h - intake)
        autonomy_h = (stores_g / deficit) if deficit > 1 else None
        rows.append({
            "zone": VC_ZONE_LABELS[i] if i < len(VC_ZONE_LABELS) else f"Z{i}",
            "pct_lo": lo_e, "pct_hi": hi_e,
            "allure_s_km": (1000.0 / (vc_ms * mid / 100.0)) if mid > 0 else None,
            "vitesse_kmh": round(vc_ms * mid / 100.0 * 3.6, 2),
            "cho_g_h": round(ox_h), "cho_g_h_lo": round(_clo * 60.0), "cho_g_h_hi": round(_chi * 60.0),
            "kcal_h": round(_k), "apport_g_h": round(intake), "deficit_g_h": round(deficit),
            "autonomie_h": round(autonomy_h, 1) if autonomy_h else None,
            "confiance": round(_conf), "extrapole": bool(extrapolated),
        })
    return rows, {"stores_g": round(stores_g), "gut_cap_g_h": float(gut_cap_g_h)}

def plot_ventilatory_thresholds(stages, thresholds, vc_ms=None):
    """Ce que le masque MESURE vraiment : VE/VCO₂ (nadir = VT2) et VCO₂ vs
    intensité (rupture de pente = VT1). Aucun oxygène n'intervient ici."""
    _key = (thresholds or {}).get("axe") or "vitesse_kmh"
    _unit = (thresholds or {}).get("axe_unite") or "km/h"
    st_ = [s for s in stages if s.get(_key) is not None and s.get("eqco2")]
    st_ = sorted(st_, key=lambda s: s[_key])
    if len(st_) < 4:
        return None
    x = [s[_key] for s in st_]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.14})
    a1.plot(x, [s["eqco2"] for s in st_], "-o", color=C_RED, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    a1.set_ylabel("VE / VCO₂")
    chart_title(a1, "Seuils ventilatoires — mesure directe du masque",
                "Le minimum de VE/VCO₂ marque le point de compensation respiratoire (VT2) ; "
                "la rupture de pente du VCO₂ marque VT1")
    a2.plot(x, [s["vco2_lmin"] for s in st_], "-o", color=C_WHITE, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    a2.set_ylabel("VCO₂ (L/min)")
    a2.set_xlabel({"vitesse_kmh": "Vitesse (km/h)", "hr": "Fréquence cardiaque (bpm)",
                   "palier": "Palier"}[_key])
    for key, color, label, dy in (("vt1", C_WHITE, "VT1", -12), ("vt2", C_RED, "VT2", -32)):
        th = (thresholds or {}).get(key)
        if th and th.get("x") is not None:
            for _a in (a1, a2):
                _a.axvline(th["x"], color=color, lw=1.4,
                           ls="--" if key == "vt1" else ":", alpha=0.85)
            _txt = f"{label} {th['x']:.1f} {_unit}"
            if th.get("hr"):
                _txt += f" · {th['hr']} bpm"
            if th.get("pct_vc"):
                _txt += f" · {th['pct_vc']:.0f} % VC"
            if not th.get("fiable"):
                _txt += " (signal faible)"
            # étiquettes décalées l'une sous l'autre : les deux seuils sont souvent proches
            a1.annotate(_txt, xy=(th["x"], a1.get_ylim()[1]), xytext=(5, dy),
                        textcoords="offset points", fontsize=8, color=color, va="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc=C_SURFACE, ec="none", alpha=0.85))
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        fig.tight_layout()
    return fig

def plot_substrates(stages, vc_ms=None):
    """Oxydation glucides / lipides — deux panneaux distincts. Les paliers dont le
    RER modélisé sort de la plage physiologique (CO₂ en retard en début de test,
    vitesse douteuse) n'ont pas de partition : ils sont simplement absents des
    courbes plutôt que tracés à une valeur inventée."""
    stages = [s for s in stages if s.get("cho_g_min") is not None and s.get("fat_g_min") is not None]
    if len(stages) < 2:
        return None
    use_vc = bool(vc_ms) and all(s.get("pct_vc") for s in stages)
    x = [s["pct_vc"] for s in stages] if use_vc else [s.get("vitesse_kmh") or s["palier"] for s in stages]
    modelled = any(s.get("vo2_source") == "modélisé" for s in stages)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.14})
    a1.fill_between(x, [s["cho_lo"] for s in stages], [s["cho_hi"] for s in stages],
                    color=C_RED, alpha=0.16, linewidth=0)
    a1.plot(x, [s["cho_g_min"] for s in stages], "-o", color=C_RED, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    a1.set_ylabel("Glucides (g/min)")
    chart_title(a1, "Oxydation des substrats selon l'intensité",
                ("VO₂ modélisé (masque CO₂ seul) : lecture qualitative, la plage colorée intègre "
                 "l'incertitude sur l'économie de course" if modelled else
                 "Plage colorée : variabilité mesurée + incertitude du modèle"))
    a2.fill_between(x, [s["fat_lo"] for s in stages], [s["fat_hi"] for s in stages],
                    color=C_WHITE, alpha=0.14, linewidth=0)
    a2.plot(x, [s["fat_g_min"] for s in stages], "-o", color=C_WHITE, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    a2.set_ylabel("Lipides (g/min)")
    a2.set_xlabel("% de la vitesse critique" if use_vc else "Vitesse (km/h)")
    if use_vc:
        for _a in (a1, a2):
            _a.axvline(100, color=C_TEXT_MUT, lw=1.1, ls=":")
    _fm = max(stages, key=lambda s: s["fat_g_min"])
    _i_fm = stages.index(_fm)
    if 0 < _i_fm < len(stages) - 1:     # pic interne seulement : sinon ce n'est pas un FatMax
        _xfm = _fm.get("pct_vc") if use_vc else (_fm.get("vitesse_kmh") or _fm["palier"])
        a2.annotate(f"FatMax ≈ {_fm['fat_g_min']:.2f} g/min", xy=(_xfm, _fm["fat_g_min"]),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color=C_TEXT)
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        fig.tight_layout()
    return fig

def plot_energy(stages, vc_ms=None):
    """Dépense énergétique — la sortie la plus fiable d'un masque CO₂ seul."""
    use_vc = bool(vc_ms) and all(s.get("pct_vc") for s in stages)
    x = [s["pct_vc"] for s in stages] if use_vc else [s.get("vitesse_kmh") or s["palier"] for s in stages]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.fill_between(x, [s["kcal_h_lo"] for s in stages], [s["kcal_h_hi"] for s in stages],
                    color=C_RED, alpha=0.15, linewidth=0)
    ax.plot(x, [s["kcal_h"] for s in stages], "-o", color=C_RED, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    ax.set_ylabel("Dépense (kcal/h)")
    ax.set_xlabel("% de la vitesse critique" if use_vc else "Vitesse (km/h)")
    chart_title(ax, "Dépense énergétique selon l'intensité",
                "Calculée à partir du CO₂ mesuré : la plage reflète l'incertitude sur le RER (±0,06), "
                "qui ne fait bouger la dépense que d'environ ±6 %")
    fig.tight_layout()
    return fig

def plot_economy(stages, mass_kg):
    """Économie de course : coût énergétique par km selon la vitesse."""
    pts = [s for s in stages if s.get("kcal_kg_km") and s.get("vitesse_kmh")]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    x = [s["vitesse_kmh"] for s in pts]
    y = [s["kcal_kg_km"] for s in pts]
    ax.plot(x, y, "-o", color=C_RED, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
    for s in pts:
        ax.annotate(f"{s['kcal_kg_km']:.2f}", xy=(s["vitesse_kmh"], s["kcal_kg_km"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.5, color=C_TEXT_MUT)
    _m = float(np.mean(y))
    ax.axhline(_m, color=C_TEXT_MUT, lw=1, ls=":")
    ax.set_xlabel("Vitesse (km/h)"); ax.set_ylabel("kcal / kg / km")
    chart_title(ax, "Économie de course",
                f"Coût énergétique moyen {_m:.2f} kcal/kg/km (≈ {_m*mass_kg:.0f} kcal/km pour {mass_kg:.0f} kg)")
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# TERRAIN PROFILES — v8.1 : k_up/g0_up/max_cap/fatigue recalibrés
# sur 1245 segments coureurs (top10 réel, 12 ultra-trails). Voir
# calibration_facteurs_v8.md pour la méthodologie et les limites.
# k_down/down_cap/minetti_weight/elev_smooth_window/grade_power/
# base_cap/extra_per_pct NON recalibrés (non identifiables de façon
# fiable à la résolution checkpoint) — valeurs v8 d'origine conservées.
# ══════════════════════════════════════════════════════════════
TERRAIN_PROFILES = {
    "🛣️ Route / Plat": {
        # Inchangé — pas de course route pure dans le jeu de calibration v8.1.
        # v8 : k_up 7→4, grade_power 0.90→0.75, lissage 15→25
        # Moins réactif aux micro-variations GPS sur route plane
        "k_up": 4.0, "k_down": 2.5, "down_cap": -0.03,
        "minetti_weight": 0.80, "elev_smooth_window": 25,
        "grade_power": 0.75, "base_cap": 0.04,
        "extra_per_pct": 0.000, "max_cap": 0.08,
    },
    "🏞️ Trail roulant / peu technique": {
        # v8.1 NOUVEAU — calibré sur Sainte-Lyon, Nantes-Montaigu, EcoTrail
        # (D+/km < 30). R²=0.918, le cluster le mieux identifié des 3.
        "k_up": 19.5, "g0_up": 5.0, "k_down": 4.5, "down_cap": -0.10,
        "minetti_weight": 0.30, "elev_smooth_window": 5,
        "grade_power": 0.70, "base_cap": 0.18,
        "extra_per_pct": 0.012, "max_cap": 0.15,
        "fatigue_threshold": 5, "fatigue_rate": 18,
    },
    "🏔️ Trail modéré": {
        # v8.1 — calibré sur Templiers, Chianti, Saint-Jacques
        # (D+/km 40-50, technique mais pas haute montagne). R²=0.747.
        "k_up": 10.7, "g0_up": 5.0, "k_down": 4.5, "down_cap": -0.10,
        "minetti_weight": 0.30, "elev_smooth_window": 5,
        "grade_power": 0.70, "base_cap": 0.18,
        "extra_per_pct": 0.012, "max_cap": 0.67,
        "fatigue_threshold": 60, "fatigue_rate": 16,
    },
    "⛰️ Ultra-trail montagneux": {
        # v8.1 — calibré sur TDS, UTMB, OCC, Trans Gran Canaria,
        # Transvulcania, MCC (D+/km 53-63). R²=0.604.
        "k_up": 13.2, "g0_up": 5.0, "k_down": 4.0, "down_cap": -0.15,
        "minetti_weight": 0.20, "elev_smooth_window": 5,
        "grade_power": 0.65, "base_cap": 0.25,
        "extra_per_pct": 0.018, "max_cap": 0.79,
        "fatigue_threshold": 5, "fatigue_rate": 18,
    },
}

def get_all_terrain_profiles():
    """Fusionne les 4 profils génériques v8.1 avec les profils calibrés par
    course (sauvegardés depuis l'onglet 👥 Analyse de cohorte, en session ou
    importés via JSON — soit depuis là, soit directement depuis l'onglet
    Prédiction section 4). Les profils calibrés ont la priorité en cas de
    clé identique (ne devrait pas arriver en pratique, noms distincts)."""
    merged = dict(TERRAIN_PROFILES)
    merged.update(st.session_state.get("custom_terrain_profiles", {}))
    return merged

def fit_cohort_profile(athletes_list):
    """Calibre k_up / plafond de pente (max_cap) / seuil et taux de fatigue
    SPÉCIFIQUEMENT sur les splits réels d'une cohorte d'athlètes pour une
    course donnée — même méthode (régression log-allure, effets coureur
    profilés analytiquement) que la calibration v8.1 d'origine sur 1245
    segments/12 courses, appliquée ici à une seule course.
    g0_up/k_down/down_cap/g0_down/max_down sont FIXÉS aux valeurs validées
    globalement : la calibration d'origine a montré qu'ils ne sont pas
    ré-estimables de façon fiable même avec 12 courses et >1000 segments —
    ils le seraient encore moins à partir d'une seule course.
    D+/D- par segment estimé depuis le dénivelé NET entre checkpoints (pas
    le détail GPX) — même limite que le reste de l'onglet cohorte.
    Retourne None si la cohorte est trop petite pour un résultat exploitable."""
    rows = []
    for a in athletes_list:
        d_plus_total = sum(sp["elev"] for sp in a["splits"] if sp["elev"] > 0)
        total_km = a["totalKm"]
        cum_dist = 0.0; cum_dplus = 0.0
        for sp in a["splits"]:
            if sp["dist"] <= 0 or sp["secs"] <= 0:
                continue
            d_up = max(0.0, sp["elev"]); d_down = max(0.0, -sp["elev"])
            cum_dist += sp["dist"]; cum_dplus += d_up
            rows.append({"athlete": a["name"], "d_up": d_up, "d_down": d_down, "dist_m": sp["dist"] * 1000.0,
                         "cum_dplus": cum_dplus, "cum_dist": cum_dist, "d_plus_total": d_plus_total,
                         "total_km": total_km, "pace": sp["secs"] / sp["dist"]})

    n_athletes = len({r["athlete"] for r in rows})
    if len(rows) < 10 or n_athletes < 2:
        return None

    g0_up, g0_down, down_cap, k_down, max_down = 5.0, 2.0, -0.50, 15.0, -0.06
    log_pace = np.array([math.log(r["pace"]) for r in rows])
    _, group_idx = np.unique([r["athlete"] for r in rows], return_inverse=True)
    n_groups = int(group_idx.max()) + 1

    def predict_logmult(params):
        k_up, max_up, fat_thr, fat_rate = params
        out = np.empty(len(rows))
        for i, r in enumerate(rows):
            gm = elev_factor_global(r["d_up"], r["d_down"], r["dist_m"], k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
            fm = fatigue_multiplier_advanced(r["cum_dplus"], r["cum_dist"], r["d_plus_total"], r["total_km"], fat_thr, fat_rate, "mixte")
            out[i] = math.log(gm) + math.log(fm)
        return out

    def residuals(params):
        resid = log_pace - predict_logmult(params)
        sums = np.bincount(group_idx, weights=resid, minlength=n_groups)
        counts = np.bincount(group_idx, minlength=n_groups)
        means = sums / np.maximum(1, counts)
        return resid - means[group_idx]

    x0 = [15.0, 0.55, 40.0, 18.0]
    lb = [1.0, 0.10, 3.0, 0.0]
    ub = [60.0, 1.20, 95.0, 45.0]
    res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=3000)
    resid_final = residuals(res.x)
    ss_res = float(np.sum(resid_final ** 2))
    ss_tot = float(np.sum((log_pace - log_pace.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "k_up": round(float(res.x[0]), 2), "g0_up": g0_up,
        "k_down": 4.5, "down_cap": -0.10, "minetti_weight": 0.30,
        "elev_smooth_window": 5, "grade_power": 0.70, "base_cap": 0.18, "extra_per_pct": 0.012,
        "max_cap": round(float(res.x[1]), 3),
        "fatigue_threshold": round(float(res.x[2])), "fatigue_rate": round(float(res.x[3]), 1),
        "r2": round(r2, 3), "n_segments": len(rows), "n_athletes": n_athletes,
    }

SURFACE_OPTIONS = {
    "🏟️ Route / Piste synthétique":              1.00,
    "🪨 Chemin stabilisé / Gravier":             1.03,
    "🌿 Sentier herbe / Terre sèche":             1.06,
    "🧗 Sentier rocheux / Technique":             1.12,
    "🌧️ Boue / Neige tassée":                    1.18,
    "🎯 Surface calibrée (mix terrain)":         1.11,
    "🤖 Détecté automatiquement (OSM)":          1.06,
}

# ══════════════════════════════════════════════════════════════
# ONGLET 4 — ANALYSE DE COHORTE
# Centralise plusieurs coureurs (splits Strava ou Live-trail) sur une même
# course, les compare entre eux, et les confronte à l'algorithme de prédiction
# (réutilise directement elev_factor_global / fatigue_multiplier_advanced /
# temp_multiplier / TERRAIN_PROFILES déjà définis plus haut — aucune
# duplication de logique de calibration).
# ══════════════════════════════════════════════════════════════

COHORT_PALETTE = list(CHART_CYCLE)

_DIST_RE = re.compile(r"^(\d+[.,]?\d*)\s*km$")
_ELEV_CUM_RE = re.compile(r"^(\d+)\s*m\+$")
_TIME_CUM_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})$")
_CLOCK_RE = re.compile(r"^\w+\.\s+\d{1,2}:\d{2}")
_RANK_RE = re.compile(r"^\d+$")
_BONUS_RE = re.compile(r"^\(\+\d+\)$")
_SPEED_RE = re.compile(r"^\d+[.,]\d*\s*km/h$")
_ALT_LBL_RE = re.compile(r"^Altitude$", re.I)
_ALT_VAL_RE = re.compile(r"^\d+\s*m$")
_REST_RE = re.compile(r"^Temps de repos", re.I)
_SECTION_RE = re.compile(r"^Temps de section", re.I)
_DERNIER_RE = re.compile(r"^Depuis dernier pt", re.I)
_VITESSE_EFFORT_RE = re.compile(r"^Vitesse effort$", re.I)
_MIXED_ELEV_RE = re.compile(r"^\d+\s*m[+-].*\d+\s*m[+-]")
_PARTIAL_ELEV_RE = re.compile(r"^\d+\s*m[+-]")
_HEADER_RE = re.compile(r"^(POINT DE PASSAGE|CLASSEMENT|PASSAGE|TEMPS|VITESSE|DÉNIVELÉ)", re.I)
_PACE_SPLIT_RE = re.compile(r"^(\d+):(\d{2})/km$")

def _itra_parse_dist(l):
    m = _DIST_RE.match(l)
    return float(m.group(1).replace(",", ".")) if m else None

def parse_splits_strava(raw):
    """Parse les splits Strava collés (format avec allure/km OU avec dist+temps)."""
    out = []
    for line in raw.strip().split("\n"):
        parts = [p for p in re.split(r"\t+", line.strip()) if p != ""]
        if len(parts) < 3:
            continue
        try:
            km = int(parts[0])
        except ValueError:
            continue
        m = _PACE_SPLIT_RE.match(parts[1]) if len(parts) > 1 else None
        if m:
            secs = int(m.group(1)) * 60 + int(m.group(2))
            elev_str = parts[3] if len(parts) > 3 else "0"
            elev_digits = re.sub(r"[^\-\d]", "", elev_str)
            elev = int(elev_digits) if elev_digits else 0
            hr_str = parts[4] if len(parts) > 4 else ""
            hr_digits = re.sub(r"[^\d]", "", hr_str)
            hr = int(hr_digits) if hr_digits else None
            out.append({"km": km, "dist": 1.0, "secs": secs, "elev": elev, "hr": hr})
        else:
            if len(parts) < 4:
                continue
            t_parts = (parts[2] if len(parts) > 2 else "0:00").split(":")
            try:
                secs = int(t_parts[0]) * 60 + int(t_parts[1])
            except (ValueError, IndexError):
                secs = 0
            elev_str = parts[4] if len(parts) > 4 else "0"
            elev_digits = re.sub(r"[^\-\d]", "", elev_str)
            elev = int(elev_digits) if elev_digits else 0
            hr_str = parts[5] if len(parts) > 5 else ""
            hr_digits = re.sub(r"[^\d]", "", hr_str)
            hr = int(hr_digits) if hr_digits else None
            dist_str = (parts[1] if len(parts) > 1 else "1,00km").replace(",", ".")
            dist_m = re.match(r"[\d.]+", dist_str)
            dist = float(dist_m.group(0)) if dist_m else 1.0
            out.append({"km": km, "dist": dist, "secs": secs, "elev": elev, "hr": hr})
    return out

def parse_itra(raw):
    """Parse une page de résultats Live-trail/ITRA collée (checkpoints + temps cumulés)."""
    raw_lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    cleaned = []
    n = len(raw_lines)
    i = 0
    while i < n:
        l = raw_lines[i]
        if _REST_RE.match(l) or _SECTION_RE.match(l) or _DERNIER_RE.match(l) or _VITESSE_EFFORT_RE.match(l):
            i += 2
            continue
        if (_HEADER_RE.match(l) or _CLOCK_RE.match(l) or _RANK_RE.match(l) or _BONUS_RE.match(l) or
                _SPEED_RE.match(l) or _ALT_LBL_RE.match(l) or _ALT_VAL_RE.match(l) or
                _MIXED_ELEV_RE.match(l) or _PARTIAL_ELEV_RE.match(l)):
            i += 1
            continue
        cleaned.append(l)
        i += 1

    dist_pos = [k for k in range(len(cleaned)) if _itra_parse_dist(cleaned[k]) is not None]
    if len(dist_pos) < 2:
        return {"cps": [], "splits": []}

    first_val = _itra_parse_dist(cleaned[dist_pos[0]])
    i0 = 1 if first_val > 0.5 else 0

    def _is_text_line(l):
        return _itra_parse_dist(l) is None and not _TIME_CUM_RE.match(l) and not _ELEV_CUM_RE.match(l)

    cp_dist_positions = []
    for k in range(i0, len(dist_pos)):
        pos = dist_pos[k]
        nxt = dist_pos[k + 1] if k + 1 < len(dist_pos) else None
        if nxt is not None and nxt == pos + 1:
            continue
        if pos + 1 < len(cleaned) and _is_text_line(cleaned[pos + 1]):
            cp_dist_positions.append(pos)

    cps = []
    for bi, pos in enumerate(cp_dist_positions):
        cum_dist = _itra_parse_dist(cleaned[pos])
        name = None
        elev = None
        all_times = []
        for k in range(pos + 1, min(len(cleaned), pos + 20)):
            l = cleaned[k]
            if _itra_parse_dist(l) is not None:
                break
            tm = _TIME_CUM_RE.match(l)
            if tm:
                all_times.append(int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + int(tm.group(3)))
                continue
            em = _ELEV_CUM_RE.match(l)
            if em and elev is None:
                elev = int(em.group(1))
                continue
            if not name:
                name = l
        cum_secs = all_times[0] if all_times else None
        cps.append({"name": name or f"CP {bi + 1}", "cumDist": cum_dist, "cumSecs": cum_secs, "elev": elev if elev is not None else 0})

    cps = [cp for cp in cps if cp["cumSecs"] is not None]
    if len(cps) < 2:
        return {"cps": [], "splits": []}

    splits = []
    for i in range(1, len(cps)):
        seg_dist = round((cps[i]["cumDist"] - cps[i - 1]["cumDist"]) * 100) / 100
        seg_secs = cps[i]["cumSecs"] - cps[i - 1]["cumSecs"]
        seg_elev = cps[i]["elev"] - cps[i - 1]["elev"]
        if seg_dist > 0 and seg_secs > 0:
            splits.append({"km": i, "dist": seg_dist, "secs": seg_secs, "elev": seg_elev, "hr": None})

    return {"cps": cps, "splits": splits}

def get_time_at_km(athlete, target_km):
    cum_km = 0.0; cum_secs = 0.0
    for sp in athlete["splits"]:
        nxt = cum_km + sp["dist"]
        if nxt >= target_km:
            return cum_secs + ((target_km - cum_km) / sp["dist"]) * sp["secs"]
        cum_km = nxt; cum_secs += sp["secs"]
    return cum_secs

def plot_cohort_pace_chart(athletes_list, checkpoints_list):
    fig, ax = plt.subplots(figsize=(11, 4))
    for a in athletes_list:
        cum = 0.0; x = []; y = []
        for sp in a["splits"]:
            cum_start = cum; cum += sp["dist"]
            x.append((cum_start + cum) / 2.0)  # position réelle en km (milieu du segment)
            y.append(sp["secs"] / sp["dist"])
        ax.plot(x, y, color=a["color"], lw=2, marker="o" if len(x) < 60 else None, ms=3,
                ls="--" if a.get("dashed") else "-", label=a["name"])
    for cp in checkpoints_list:
        ax.axvline(cp["km"], color=C_RED, lw=1, ls=":", alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Allure")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([pace_str(t) for t in yticks if t > 0])
    ax.set_title("Allure km par km"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plot_cohort_cp_chart(athletes_list, checkpoints_list):
    n_cp = len(checkpoints_list)
    fig_w = max(11, n_cp * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    for a in athletes_list:
        x = [cp["km"] for cp in checkpoints_list]
        y = [get_time_at_km(a, cp["km"]) / 60.0 for cp in checkpoints_list]
        ax.plot(x, y, color=a["color"], lw=2, marker="o", ms=5, ls="--" if a.get("dashed") else "-", label=a["name"])
    ymin, ymax = ax.get_ylim()
    for idx, cp in enumerate(checkpoints_list):
        ax.axvline(cp["km"], color=C_TEXT_MUT, lw=0.5, ls=":", alpha=0.4, zorder=0)
        y_offset = 8 if idx % 2 == 0 else 26  # alterne la hauteur pour limiter le chevauchement
        ax.annotate(f"{cp['name']}\nkm {cp['km']:.1f}", xy=(cp["km"], ymax),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=C_TEXT_MUT)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Temps cumulé")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t // 60)}h{int(round(t % 60)):02d}" for t in yticks])
    ax.set_title("Temps de passage aux checkpoints — positions proportionnelles à la distance réelle")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fetch_daily_weather(lat, lon, date_obj):
    """Météo journalière (tmax/tmin/précip/vent) — endpoint Open-Meteo distinct
    du système météo horaire v7/v8 (get_weather_minutely), adapté au besoin
    d'un résumé jour de course unique pour la cohorte."""
    try:
        today = date.today()
        diff = (date_obj - today).days
        if -1 <= diff <= 15:
            url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
                   f"&timezone=Europe%2FParis&start_date={date_obj}&end_date={date_obj}")
            is_past = False
        else:
            past = date_obj.replace(year=date_obj.year - 1)
            url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                   "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
                   f"&timezone=Europe%2FParis&start_date={past}&end_date={past}")
            is_past = True
        r = requests.get(url, timeout=12)
        d = r.json()
        daily = d.get("daily", {})
        def _first(key):
            v = daily.get(key)
            return v[0] if v else None
        return {"tmax": _first("temperature_2m_max"), "tmin": _first("temperature_2m_min"),
                "precip": _first("precipitation_sum"), "wind": _first("windspeed_10m_max"), "isPast": is_past}
    except Exception:
        return None

def build_prediction_cohort(athlete, profile_key, apply_fatigue, apply_temp, temp_c, mode, manual_pace_sec_km):
    """Prédit l'allure segment par segment pour un athlète de la cohorte, en
    réutilisant directement elev_factor_global/fatigue_multiplier_advanced/
    temp_multiplier et le profil terrain calibré v8.1 sélectionné. D+/D- estimé
    à partir du dénivelé NET par segment (sp['elev']) — pas du détail GPX.
    mode='cale' : l'allure de base est recalée pour que le temps total prédit
    égale le temps réel (compare la RÉPARTITION). mode='manuel' : allure libre."""
    profile = TERRAIN_PROFILES[profile_key]
    g0_up = profile.get("g0_up", 3.0)
    g0_down = 2.5
    max_down = -0.06
    fatigue_threshold = profile.get("fatigue_threshold", 60)
    fatigue_rate = profile.get("fatigue_rate", 18.0)

    total_km = athlete["totalKm"]
    d_plus_total = sum(sp["elev"] for sp in athlete["splits"] if sp["elev"] > 0)

    cum_dist = 0.0; cum_dplus = 0.0
    raw = []
    for sp in athlete["splits"]:
        d_up = max(0.0, sp["elev"]); d_down = max(0.0, -sp["elev"])
        gm = elev_factor_global(d_up, d_down, sp["dist"] * 1000.0, profile["k_up"], profile["k_down"],
                                 profile["down_cap"], g0_up, g0_down, profile["max_cap"], max_down)
        cum_dist += sp["dist"]; cum_dplus += d_up
        fm = fatigue_multiplier_advanced(cum_dplus, cum_dist, d_plus_total, total_km,
                                          fatigue_threshold, fatigue_rate, "mixte") if apply_fatigue else 1.0
        tm = temp_multiplier(temp_c, 10.0, 0.0003, 0.0055, 0.20) if (apply_temp and temp_c is not None) else 1.0
        raw.append({"km": sp["km"], "dist": sp["dist"], "elev": sp["elev"], "gm": gm, "fm": fm, "tm": tm, "mult": gm * fm * tm})

    weighted = sum(r["dist"] * r["mult"] for r in raw)
    if mode == "manuel" and manual_pace_sec_km and manual_pace_sec_km > 0:
        base_pace = manual_pace_sec_km
    else:
        base_pace = athlete["totalSecs"] / max(1e-6, weighted)

    splits = [{**r, "secs": base_pace * r["dist"] * r["mult"], "hr": None} for r in raw]
    total_secs = sum(s["secs"] for s in splits)
    return {"splits": splits, "totalSecs": total_secs, "totalKm": total_km, "basePaceSecKm": base_pace}


# ══════════════════════════════════════════════════════════════
# v8.9 — HISTORIQUE & COMPTES (SQLite local, aucun service externe)
#
#   • Comptes coach : inscription / connexion, mot de passe haché (PBKDF2-
#     SHA256, 240 000 itérations, sel aléatoire par utilisateur). Chaque coach
#     ne voit que SES athlètes et SES données.
#   • Un athlète appartient à un coach ; tests VC, séances et courses/plans
#     sont rattachés à un athlète et horodatés.
#   • Rien n'est écrasé : réenregistrer un test crée une NOUVELLE ligne, ce qui
#     permet de suivre l'évolution dans le temps.
#
# Toutes les données restent dans le fichier coach_data.db, à côté du script.
# Sauvegarde = copier ce fichier. Aucune donnée ne sort de la machine.
# ══════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coach_data.db")
AUTH_ENABLED = True          # passe à False pour un usage strictement personnel sans écran de connexion
PBKDF2_ROUNDS = 240_000

def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def db_init():
    with db_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            pw_hash TEXT NOT NULL,
            pw_salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS athletes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(user_id, name)
        );
        CREATE TABLE IF NOT EXISTS vc_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES athletes(id) ON DELETE CASCADE,
            date TEXT NOT NULL, label TEXT DEFAULT '',
            vc_ms REAL, d_prime REAL, r2 REAL, k_riegel REAL, a_riegel REAL,
            n_refs INTEGER, refs_json TEXT, notes TEXT DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES athletes(id) ON DELETE CASCADE,
            date TEXT NOT NULL, name TEXT DEFAULT '', seance_type TEXT DEFAULT '',
            file_name TEXT DEFAULT '',
            duration_s REAL, distance_m REAL, d_plus REAL,
            hr_avg REAL, hr_max REAL, hr_drift REAL,
            pct_walk REAL, trans_lo REAL, trans_mid REAL, trans_hi REAL,
            vap_slope_montee REAL, vap_last_montee REAL,
            portions_json TEXT, intervals_json TEXT, extra_json TEXT,
            notes TEXT DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES athletes(id) ON DELETE CASCADE,
            date TEXT NOT NULL, name TEXT DEFAULT '', kind TEXT DEFAULT 'plan',
            gpx_name TEXT DEFAULT '', distance_km REAL, d_plus REAL,
            predicted_s REAL, actual_s REAL,
            params_json TEXT, splits_json TEXT, checkpoints_json TEXT, gpx_xml TEXT,
            notes TEXT DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vc_athlete   ON vc_tests(athlete_id, date);
        CREATE INDEX IF NOT EXISTS idx_wk_athlete   ON workouts(athlete_id, date);
        CREATE INDEX IF NOT EXISTS idx_race_athlete ON races(athlete_id, date);
        """)

def _db_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, PBKDF2_ROUNDS)
    return h.hex(), salt.hex()

def create_user(email, name, password):
    email = str(email).strip().lower()
    if not email or "@" not in email:
        return None, "Adresse e-mail invalide."
    if len(str(password)) < 8:
        return None, "Mot de passe trop court (8 caractères minimum)."
    h, s = hash_password(password)
    try:
        with db_conn() as c:
            cur = c.execute("INSERT INTO users(email,name,pw_hash,pw_salt,created_at) VALUES(?,?,?,?,?)",
                            (email, str(name).strip() or email.split("@")[0], h, s, _db_now()))
            return int(cur.lastrowid), None
    except sqlite3.IntegrityError:
        return None, "Un compte existe déjà avec cette adresse."

def verify_user(email, password):
    with db_conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (str(email).strip().lower(),)).fetchone()
    if row is None:
        return None, "Compte inconnu."
    h, _ = hash_password(password, row["pw_salt"])
    if not secrets.compare_digest(h, row["pw_hash"]):
        return None, "Mot de passe incorrect."
    return {"id": int(row["id"]), "email": row["email"], "name": row["name"]}, None

def change_password(user_id, old_pw, new_pw):
    with db_conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()
        if row is None:
            return "Compte introuvable."
        h, _ = hash_password(old_pw, row["pw_salt"])
        if not secrets.compare_digest(h, row["pw_hash"]):
            return "Ancien mot de passe incorrect."
        if len(str(new_pw)) < 8:
            return "Nouveau mot de passe trop court (8 caractères minimum)."
        nh, ns = hash_password(new_pw)
        c.execute("UPDATE users SET pw_hash=?, pw_salt=? WHERE id=?", (nh, ns, int(user_id)))
    return None

def count_users():
    with db_conn() as c:
        return int(c.execute("SELECT COUNT(*) n FROM users").fetchone()["n"])

# ── Athlètes ──────────────────────────────────────────────────────────────
def list_athletes(user_id):
    with db_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM athletes WHERE user_id=? ORDER BY name", (int(user_id),))]

def create_athlete(user_id, name, notes=""):
    name = str(name).strip()
    if not name:
        return None, "Nom d'athlète vide."
    try:
        with db_conn() as c:
            cur = c.execute("INSERT INTO athletes(user_id,name,notes,created_at) VALUES(?,?,?,?)",
                            (int(user_id), name, notes, _db_now()))
            return int(cur.lastrowid), None
    except sqlite3.IntegrityError:
        return None, "Cet athlète existe déjà."

def athlete_belongs_to(athlete_id, user_id):
    with db_conn() as c:
        r = c.execute("SELECT 1 FROM athletes WHERE id=? AND user_id=?",
                      (int(athlete_id), int(user_id))).fetchone()
    return r is not None

def delete_athlete(athlete_id, user_id):
    if not athlete_belongs_to(athlete_id, user_id):
        return False
    with db_conn() as c:
        c.execute("DELETE FROM athletes WHERE id=?", (int(athlete_id),))
    return True

# ── Enregistrements ───────────────────────────────────────────────────────
def save_vc_test(athlete_id, date_str, label, vc_ms, d_prime, r2, k, a, refs, notes=""):
    with db_conn() as c:
        cur = c.execute("""INSERT INTO vc_tests(athlete_id,date,label,vc_ms,d_prime,r2,k_riegel,a_riegel,
                           n_refs,refs_json,notes,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (int(athlete_id), str(date_str), str(label), _db_float(vc_ms), _db_float(d_prime), _db_float(r2),
                         _db_float(k), _db_float(a), len(refs or []), _json.dumps(refs or [], ensure_ascii=False),
                         str(notes), _db_now()))
        return int(cur.lastrowid)

def save_workout(athlete_id, date_str, name, seance_type, file_name, summary,
                 portions=None, intervals=None, extra=None, notes="",
                 category_id=None, tags="", zones=None, records=None, hr_max_used=None, quarters=None,
                 zones_vc=None, vc_ref_ms=None):
    s = summary or {}
    with db_conn() as c:
        cur = c.execute("""INSERT INTO workouts(athlete_id,date,name,seance_type,file_name,duration_s,distance_m,
                           d_plus,hr_avg,hr_max,hr_drift,pct_walk,trans_lo,trans_mid,trans_hi,
                           vap_slope_montee,vap_last_montee,portions_json,intervals_json,extra_json,notes,created_at,
                           category_id,tags,zones_json,records_json,hr_max_used,quarters_json,
                           zones_vc_json,vc_ref_ms)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (int(athlete_id), str(date_str), str(name), str(seance_type), str(file_name),
                         _db_float(s.get("duration_s")), _db_float(s.get("distance_m")), _db_float(s.get("d_plus")),
                         _db_float(s.get("hr_avg")), _db_float(s.get("hr_max")), _db_float(s.get("hr_drift")),
                         _db_float(s.get("pct_walk")), _db_float(s.get("trans_lo")), _db_float(s.get("trans_mid")),
                         _db_float(s.get("trans_hi")), _db_float(s.get("vap_slope_montee")), _db_float(s.get("vap_last_montee")),
                         _json.dumps(portions or [], ensure_ascii=False),
                         _json.dumps(intervals or [], ensure_ascii=False),
                         _json.dumps(extra or {}, ensure_ascii=False), str(notes), _db_now(),
                         int(category_id) if category_id else None, str(tags),
                         _json.dumps(zones or [], ensure_ascii=False),
                         _json.dumps(records or [], ensure_ascii=False), _db_float(hr_max_used),
                         _json.dumps(quarters or [], ensure_ascii=False),
                         _json.dumps(zones_vc or [], ensure_ascii=False), _db_float(vc_ref_ms)))
        return int(cur.lastrowid)

def save_race(athlete_id, date_str, name, kind, gpx_name, distance_km, d_plus,
              predicted_s, actual_s, params, splits, checkpoints=None, notes="", gpx_xml=None,
              stops_s=None, moving_s=None):
    with db_conn() as c:
        cur = c.execute("""INSERT INTO races(athlete_id,date,name,kind,gpx_name,distance_km,d_plus,
                           predicted_s,actual_s,params_json,splits_json,checkpoints_json,gpx_xml,notes,created_at,
                           stops_s,moving_s)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (int(athlete_id), str(date_str), str(name), str(kind), str(gpx_name),
                         _db_float(distance_km), _db_float(d_plus), _db_float(predicted_s), _db_float(actual_s),
                         _json.dumps(params or {}, ensure_ascii=False, default=str),
                         _json.dumps(splits or [], ensure_ascii=False, default=str),
                         _json.dumps(checkpoints or [], ensure_ascii=False, default=str),
                         gpx_xml, str(notes), _db_now(), _db_float(stops_s), _db_float(moving_s)))
        return int(cur.lastrowid)

def _db_float(v):
    """Convertit en float pour SQLite, None si non convertible (jamais d'exception)."""
    try:
        if v is None:
            return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None

def save_metabolic_test(athlete_id, date_str, label, protocole, mass_kg, vc_ms, confiance,
                        stages, fueling, infos, sv1_hr=None, sv2_hr=None, notes=""):
    """Enregistre un test à échanges gazeux : paliers, plan nutritionnel et confiance."""
    fm = (infos or {}).get("fatmax") or {}
    eco = [s.get("kcal_kg_km") for s in (stages or []) if s.get("kcal_kg_km")]
    with db_conn() as c:
        cur = c.execute("""INSERT INTO metabolic_tests(athlete_id,date,label,protocole,mass_kg,vc_ms,
                           confiance,n_paliers,fatmax_pct_vc,fatmax_g_min,eco_kcal_kg_km,sv1_hr,sv2_hr,
                           stages_json,fueling_json,infos_json,notes,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (int(athlete_id), str(date_str), str(label), str(protocole), _db_float(mass_kg),
                         _db_float(vc_ms), _db_float(confiance), len(stages or []),
                         _db_float(fm.get("pct_vc")), _db_float(fm.get("fat_g_min")),
                         _db_float(float(np.mean(eco)) if eco else None), _db_float(sv1_hr), _db_float(sv2_hr),
                         _json.dumps(stages or [], ensure_ascii=False),
                         _json.dumps(fueling or [], ensure_ascii=False),
                         _json.dumps(infos or {}, ensure_ascii=False, default=str),
                         str(notes), _db_now()))
        return int(cur.lastrowid)

def count_metabolic_tests(athlete_id):
    """Nombre de tests déjà enregistrés — sert à la composante « calibration
    individuelle » de la confiance : plus l'athlète a de tests, plus le modèle
    est ancré sur lui."""
    try:
        with db_conn() as c:
            return int(c.execute("SELECT COUNT(*) n FROM metabolic_tests WHERE athlete_id=?",
                                 (int(athlete_id),)).fetchone()["n"])
    except sqlite3.Error:
        return 0

def set_athlete_profile(athlete_id, user_id, mass_kg=None, gut_cap_g_h=None):
    if not athlete_belongs_to(athlete_id, user_id):
        return False
    with db_conn() as c:
        if mass_kg is not None:
            c.execute("UPDATE athletes SET mass_kg=? WHERE id=?", (_db_float(mass_kg), int(athlete_id)))
        if gut_cap_g_h is not None:
            c.execute("UPDATE athletes SET gut_cap_g_h=? WHERE id=?", (_db_float(gut_cap_g_h), int(athlete_id)))
    return True

def list_records(table, athlete_id):
    if table not in ("vc_tests", "workouts", "races", "metabolic_tests"):
        return []
    with db_conn() as c:
        return [dict(r) for r in c.execute(
            f"SELECT * FROM {table} WHERE athlete_id=? ORDER BY date DESC, id DESC", (int(athlete_id),))]

def get_record(table, rec_id):
    if table not in ("vc_tests", "workouts", "races", "metabolic_tests"):
        return None
    with db_conn() as c:
        r = c.execute(f"SELECT * FROM {table} WHERE id=?", (int(rec_id),)).fetchone()
    return dict(r) if r else None

def delete_record(table, rec_id, user_id):
    """Suppression vérifiée : on ne peut effacer qu'un enregistrement rattaché à
    un athlète du coach connecté."""
    if table not in ("vc_tests", "workouts", "races", "metabolic_tests"):
        return False
    with db_conn() as c:
        row = c.execute(f"""SELECT t.id FROM {table} t JOIN athletes a ON a.id=t.athlete_id
                            WHERE t.id=? AND a.user_id=?""", (int(rec_id), int(user_id))).fetchone()
        if row is None:
            return False
        c.execute(f"DELETE FROM {table} WHERE id=?", (int(rec_id),))
    return True

def update_record_notes(table, rec_id, user_id, notes):
    if table not in ("vc_tests", "workouts", "races"):
        return False
    with db_conn() as c:
        row = c.execute(f"""SELECT t.id FROM {table} t JOIN athletes a ON a.id=t.athlete_id
                            WHERE t.id=? AND a.user_id=?""", (int(rec_id), int(user_id))).fetchone()
        if row is None:
            return False
        c.execute(f"UPDATE {table} SET notes=? WHERE id=?", (str(notes), int(rec_id)))
    return True

def update_race_actual(race_id, user_id, actual_hms):
    """Renseigne a posteriori le temps réellement réalisé sur une course."""
    secs = hms_to_seconds(actual_hms)
    with db_conn() as c:
        row = c.execute("""SELECT r.id FROM races r JOIN athletes a ON a.id=r.athlete_id
                           WHERE r.id=? AND a.user_id=?""", (int(race_id), int(user_id))).fetchone()
        if row is None:
            return False
        c.execute("UPDATE races SET actual_s=?, kind='resultat' WHERE id=?", (float(secs), int(race_id)))
    return True

def athlete_counts(athlete_id):
    with db_conn() as c:
        return {t: int(c.execute(f"SELECT COUNT(*) n FROM {t} WHERE athlete_id=?",
                                 (int(athlete_id),)).fetchone()["n"])
                for t in ("vc_tests", "workouts", "races")}

def export_athlete_json(athlete_id):
    """Export complet d'un athlète (sauvegarde ou transfert vers une autre machine)."""
    with db_conn() as c:
        ath = c.execute("SELECT * FROM athletes WHERE id=?", (int(athlete_id),)).fetchone()
        if ath is None:
            return None
        data = {"athlete": dict(ath), "exported_at": _db_now(), "app_version": "v8.9"}
        for t in ("vc_tests", "workouts", "races", "metabolic_tests"):
            data[t] = [dict(r) for r in c.execute(
                f"SELECT * FROM {t} WHERE athlete_id=? ORDER BY date", (int(athlete_id),))]
    return data

def import_athlete_json(user_id, payload, new_name=None):
    """Réimporte un export JSON sous un nouvel athlète du coach connecté."""
    if not isinstance(payload, dict) or "athlete" not in payload:
        return None, "Fichier d'export non reconnu."
    name = new_name or (payload["athlete"].get("name", "Athlète importé") + " (importé)")
    aid, err = create_athlete(user_id, name, payload["athlete"].get("notes", ""))
    if aid is None:
        return None, err
    n = 0
    with db_conn() as c:
        for t in ("vc_tests", "workouts", "races", "metabolic_tests"):
            for rec in payload.get(t, []):
                rec = {k: v for k, v in dict(rec).items() if k not in ("id", "athlete_id")}
                cols = ",".join(rec.keys()); ph = ",".join("?" * len(rec))
                try:
                    c.execute(f"INSERT INTO {t}(athlete_id,{cols}) VALUES(?,{ph})",
                              [aid] + list(rec.values()))
                    n += 1
                except sqlite3.Error:
                    continue
    return {"athlete_id": aid, "n": n}, None

db_init()

# ══════════════════════════════════════════════════════════════
# v9.0 — CATÉGORIES DE SÉANCE & MIGRATIONS NON DESTRUCTIVES
#
# La base de données est un fichier SÉPARÉ du script : modifier l'algorithme,
# ajouter des colonnes ou changer les graphiques ne touche jamais aux données
# déjà enregistrées. À chaque démarrage, db_migrate() ajoute uniquement ce qui
# manque (ALTER TABLE ADD COLUMN) — aucune table n'est jamais recréée ni vidée,
# et une copie de sauvegarde datée est faite avant toute modification de schéma.
# ══════════════════════════════════════════════════════════════
SCHEMA_VERSION = 5
DEFAULT_CATEGORIES = ["Sortie longue", "Endurance fondamentale", "Fractionné court",
                      "Fractionné long", "Seuil", "Côtes / VAM", "Test", "Course", "Récupération"]

def _table_columns(c, table):
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}

def _ensure_column(c, table, col, decl):
    if col not in _table_columns(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        return True
    return False

def db_backup(tag="migration"):
    """Copie datée de la base avant toute évolution de schéma (7 dernières gardées)."""
    if not os.path.exists(DB_PATH):
        return None
    stamp = datetime.now().strftime("%Y%m%d")
    dest = f"{DB_PATH}.bak-{stamp}-{tag}"
    if not os.path.exists(dest):
        try:
            with db_conn() as src, sqlite3.connect(dest) as dst:
                src.backup(dst)                      # copie cohérente, même base ouverte
        except Exception:
            return None
        _baks = sorted(glob.glob(f"{DB_PATH}.bak-*"))
        for _old in _baks[:-7]:
            try:
                os.remove(_old)
            except OSError:
                pass
    return dest

def db_migrate():
    """Ajoute les nouveautés de schéma sans jamais toucher aux données existantes."""
    with db_conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 0
    if current >= SCHEMA_VERSION:
        return current
    # copie de sécurité systématique avant toute évolution de schéma sur une base
    # qui contient déjà quelque chose
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 4096:
        db_backup(f"v{current}-to-v{SCHEMA_VERSION}")
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL, sort_order INTEGER DEFAULT 0, created_at TEXT NOT NULL,
            UNIQUE(user_id, name))""")
        _ensure_column(c, "workouts", "category_id", "INTEGER")
        _ensure_column(c, "workouts", "tags", "TEXT DEFAULT ''")
        _ensure_column(c, "workouts", "zones_json", "TEXT")
        _ensure_column(c, "workouts", "records_json", "TEXT")
        _ensure_column(c, "workouts", "hr_max_used", "REAL")
        _ensure_column(c, "athletes", "hr_max", "REAL")
        _ensure_column(c, "athletes", "birth_year", "INTEGER")
        _ensure_column(c, "races", "gpx_xml", "TEXT")
        _ensure_column(c, "workouts", "quarters_json", "TEXT")      # v4 : découpage en quarts
        _ensure_column(c, "races", "stops_s", "REAL")               # v4 : total des arrêts prévus
        _ensure_column(c, "races", "moving_s", "REAL")              # v4 : temps hors arrêts
        _ensure_column(c, "workouts", "zones_vc_json", "TEXT")      # v5 : zones calibrées sur la VC
        _ensure_column(c, "workouts", "vc_ref_ms", "REAL")
        _ensure_column(c, "athletes", "mass_kg", "REAL")
        _ensure_column(c, "athletes", "gut_cap_g_h", "REAL")
        c.execute("""CREATE TABLE IF NOT EXISTS metabolic_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES athletes(id) ON DELETE CASCADE,
            date TEXT NOT NULL, label TEXT DEFAULT '', protocole TEXT DEFAULT '',
            mass_kg REAL, vc_ms REAL, confiance REAL, n_paliers INTEGER,
            fatmax_pct_vc REAL, fatmax_g_min REAL,
            eco_kcal_kg_km REAL, sv1_hr REAL, sv2_hr REAL,
            stages_json TEXT, fueling_json TEXT, infos_json TEXT,
            notes TEXT DEFAULT '', created_at TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_met_athlete ON metabolic_tests(athlete_id, date)")
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
    return SCHEMA_VERSION

def seed_categories(user_id):
    """Catégories de départ, créées une seule fois par coach. Elles restent
    entièrement modifiables : ce ne sont que des lignes en base."""
    with db_conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM categories WHERE user_id=?", (int(user_id),)).fetchone()["n"]
        if n:
            return
        for i, name in enumerate(DEFAULT_CATEGORIES):
            c.execute("INSERT OR IGNORE INTO categories(user_id,name,sort_order,created_at) VALUES(?,?,?,?)",
                      (int(user_id), name, i, _db_now()))

def list_categories(user_id):
    with db_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM categories WHERE user_id=? ORDER BY sort_order, name", (int(user_id),))]

def create_category(user_id, name):
    name = str(name).strip()
    if not name:
        return None, "Nom de catégorie vide."
    try:
        with db_conn() as c:
            n = c.execute("SELECT COALESCE(MAX(sort_order),0)+1 s FROM categories WHERE user_id=?",
                          (int(user_id),)).fetchone()["s"]
            cur = c.execute("INSERT INTO categories(user_id,name,sort_order,created_at) VALUES(?,?,?,?)",
                            (int(user_id), name, n, _db_now()))
            return int(cur.lastrowid), None
    except sqlite3.IntegrityError:
        return None, "Cette catégorie existe déjà."

def rename_category(cat_id, user_id, new_name):
    new_name = str(new_name).strip()
    if not new_name:
        return "Nom vide."
    try:
        with db_conn() as c:
            r = c.execute("SELECT 1 FROM categories WHERE id=? AND user_id=?",
                          (int(cat_id), int(user_id))).fetchone()
            if r is None:
                return "Catégorie introuvable."
            c.execute("UPDATE categories SET name=? WHERE id=?", (new_name, int(cat_id)))
    except sqlite3.IntegrityError:
        return "Une catégorie porte déjà ce nom."
    return None

def delete_category(cat_id, user_id):
    """Supprime la catégorie mais CONSERVE les séances : elles passent simplement
    en « sans catégorie »."""
    with db_conn() as c:
        r = c.execute("SELECT 1 FROM categories WHERE id=? AND user_id=?",
                      (int(cat_id), int(user_id))).fetchone()
        if r is None:
            return False
        c.execute("UPDATE workouts SET category_id=NULL WHERE category_id=?", (int(cat_id),))
        c.execute("DELETE FROM categories WHERE id=?", (int(cat_id),))
    return True

def set_workout_category(workout_id, user_id, category_id):
    with db_conn() as c:
        r = c.execute("""SELECT w.id FROM workouts w JOIN athletes a ON a.id=w.athlete_id
                         WHERE w.id=? AND a.user_id=?""", (int(workout_id), int(user_id))).fetchone()
        if r is None:
            return False
        c.execute("UPDATE workouts SET category_id=? WHERE id=?",
                  (int(category_id) if category_id else None, int(workout_id)))
    return True

def set_athlete_hr_max(athlete_id, user_id, hr_max):
    if not athlete_belongs_to(athlete_id, user_id):
        return False
    with db_conn() as c:
        c.execute("UPDATE athletes SET hr_max=? WHERE id=?", (_db_float(hr_max), int(athlete_id)))
    return True

def get_athlete(athlete_id):
    with db_conn() as c:
        r = c.execute("SELECT * FROM athletes WHERE id=?", (int(athlete_id),)).fetchone()
    return dict(r) if r else None

def last_vc_ms(athlete_id):
    """Dernière vitesse critique connue de l'athlète (pour situer les records)."""
    with db_conn() as c:
        r = c.execute("""SELECT vc_ms FROM vc_tests WHERE athlete_id=? AND vc_ms IS NOT NULL
                         ORDER BY date DESC, id DESC LIMIT 1""", (int(athlete_id),)).fetchone()
    return float(r["vc_ms"]) if r else None

db_migrate()

# ── Session : utilisateur courant / athlète courant ───────────────────────
def current_user():
    return st.session_state.get("_auth_user")

def current_athlete_id():
    return st.session_state.get("_athlete_id")

def current_athlete_name():
    return st.session_state.get("_athlete_name", "")

def history_ready():
    """Vrai si un coach est connecté ET un athlète sélectionné : les boutons
    d'enregistrement ne s'affichent que dans ce cas."""
    return current_user() is not None and current_athlete_id() is not None

def render_account_sidebar():
    """Panneau Compte + Athlète, en haut de la barre latérale."""
    user = current_user()
    if user is None:
        st.markdown('<div class="sidebar-label">👤 Compte coach</div>', unsafe_allow_html=True)
        if count_users() == 0:
            st.caption("Aucun compte sur cette instance — crée le premier ci-dessous.")
        tab_in, tab_up = st.tabs(["Connexion", "Créer un compte"])
        with tab_in:
            with st.form("login_form"):
                em = st.text_input("E-mail", key="login_email")
                pw = st.text_input("Mot de passe", type="password", key="login_pw")
                if st.form_submit_button("Se connecter"):
                    u, err = verify_user(em, pw)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["_auth_user"] = u
                        st.rerun()
        with tab_up:
            with st.form("signup_form"):
                em2 = st.text_input("E-mail", key="signup_email")
                nm2 = st.text_input("Nom", key="signup_name")
                pw2 = st.text_input("Mot de passe (8 caractères min.)", type="password", key="signup_pw")
                if st.form_submit_button("Créer le compte"):
                    uid, err = create_user(em2, nm2, pw2)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["_auth_user"] = {"id": uid, "email": str(em2).strip().lower(),
                                                          "name": nm2 or em2}
                        st.rerun()
        st.caption("Sans connexion, l'app fonctionne normalement — seuls l'historique et "
                   "l'enregistrement des analyses sont désactivés.")
        return

    seed_categories(user["id"])       # catégories de départ, créées une seule fois
    st.markdown(f'<div class="sidebar-label">👤 {user["name"]}</div>', unsafe_allow_html=True)
    athletes = list_athletes(user["id"])
    names = [a["name"] for a in athletes]
    if athletes:
        idx = names.index(current_athlete_name()) if current_athlete_name() in names else 0
        sel = st.selectbox("Athlète suivi", names, index=idx, key="athlete_select")
        chosen = next(a for a in athletes if a["name"] == sel)
        st.session_state["_athlete_id"] = chosen["id"]
        st.session_state["_athlete_name"] = chosen["name"]
        cnt = athlete_counts(chosen["id"])
        st.caption(f"{cnt['vc_tests']} test(s) VC · {cnt['workouts']} séance(s) · {cnt['races']} course(s)")
    else:
        st.caption("Crée un athlète pour commencer à enregistrer.")
        st.session_state["_athlete_id"] = None
    with st.expander("➕ Nouvel athlète"):
        with st.form("new_athlete_form", clear_on_submit=True):
            nm = st.text_input("Nom de l'athlète")
            if st.form_submit_button("Créer"):
                aid, err = create_athlete(user["id"], nm)
                if err:
                    st.error(err)
                else:
                    st.session_state["_athlete_id"] = aid
                    st.session_state["_athlete_name"] = nm.strip()
                    st.rerun()
    with st.expander("🔐 Compte"):
        with st.form("pw_form", clear_on_submit=True):
            o = st.text_input("Mot de passe actuel", type="password")
            n1 = st.text_input("Nouveau mot de passe", type="password")
            if st.form_submit_button("Changer le mot de passe"):
                err = change_password(user["id"], o, n1)
                st.error(err) if err else st.success("Mot de passe mis à jour.")
        if st.button("Se déconnecter"):
            for k in ("_auth_user", "_athlete_id", "_athlete_name"):
                st.session_state.pop(k, None)
            st.rerun()

def save_gate(context_label):
    """Message affiché à la place d'un bouton d'enregistrement quand il manque
    un compte ou un athlète."""
    if current_user() is None:
        st.caption(f"🔒 Connecte-toi (barre latérale) pour enregistrer {context_label} dans l'historique.")
    else:
        st.caption(f"🔒 Crée ou sélectionne un athlète (barre latérale) pour enregistrer {context_label}.")

# ══════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    render_account_sidebar()
    st.markdown("---")
    st.markdown('<div class="sidebar-label">⚙️ Paramètres — onglets Tests & Entraînement</div>',unsafe_allow_html=True)
    sb_opt_temp=st.slider("Température optimale (°C)",5.0,20.0,10.0,0.5,key="sb_opt_temp")
    sb_k_up=st.number_input("Coefficient montée (k_up)",value=22.0,step=0.5,key="sb_k_up")
    sb_k_down=st.number_input("Coefficient descente (k_down)",value=4.5,step=0.5,key="sb_k_down")
    sb_k_temp_hot=st.number_input("Sensibilité chaleur",value=0.0016,step=0.0002,format="%.4f",key="sb_kth")
    sb_k_temp_cold=st.number_input("Sensibilité froid",value=0.0012,step=0.0002,format="%.4f",key="sb_ktc")
    st.caption("Ces paramètres n'affectent que les onglets 🧪 et ⚙️")

main_tabs=st.tabs(["🏃 Prédiction de course","🧪 Tests d'endurance + VC","⚙️ Analyse entraînement","👥 Analyse de cohorte","📚 Historique"])

# ══════════════════════════════════════════════════════════════
# ONGLET 0 — PRÉDICTION DE COURSE
# ══════════════════════════════════════════════════════════════
with main_tabs[0]:
    # v8.9 — un plan rechargé depuis l'historique réinjecte ses paramètres dans les
    # widgets AVANT leur création (Streamlit interdit de les modifier après).
    _pending_plan = st.session_state.pop("_pending_plan_params", None)
    if _pending_plan:
        for _pk in ("tp_k_up", "tp_k_down", "tp_down_cap", "tp_minetti_weight", "tp_elev_smooth_window",
                    "tp_grade_power", "tp_base_cap", "tp_extra_per_pct", "tp_max_cap",
                    "tp_fatigue_threshold", "tp_fatigue_rate", "tp_fatigue_threshold2", "tp_fatigue_rate2"):
            if _pending_plan.get(_pk) is not None:
                st.session_state[_pk] = _pending_plan[_pk]
        if _pending_plan.get("terrain_profil"):
            st.session_state["terrain_profil_radio"] = _pending_plan["terrain_profil"]
            # neutralise la resynchronisation automatique du profil, qui écraserait
            # les coefficients qu'on vient de restaurer
            st.session_state["_prev_terrain_profil"] = _pending_plan["terrain_profil"]
        if _pending_plan.get("surface_sel"):
            st.session_state["surface_sel"] = _pending_plan["surface_sel"]
        st.session_state["_plan_reloaded_banner"] = _pending_plan.get("_race_name", "plan")
    st.title("🏃 Prédiction de course — Coach & Athlète")
    if st.session_state.get("_plan_reloaded_banner"):
        st.markdown(f'<div class="note-box note-red">♻️ Paramètres rechargés depuis l\'historique : '
                    f'<b>{st.session_state.pop("_plan_reloaded_banner")}</b>. Vérifie la date et l\'heure de '
                    f'départ en section 05, puis relance le calcul en section 06.</div>', unsafe_allow_html=True)
    st.caption("v9.2 — Zones calibrées sur la vitesse critique · profil métabolique & nutrition · quarts de séance · feuille de route ravitos · comptes coach & historique athlète · Analyse trail course/marche · Charte sombre · Filtre GPS · VC FIT/TCX · Prédiction FC · K Riegel relevé · Fatigue à deux phases (parcours + perso)")

    col_mode1,col_mode2=st.columns([2,3])
    with col_mode1:
        mode=st.radio("Mode d'interface",["Simple (recommandé)","Expert (tous les curseurs)"],horizontal=True,key="pred_mode")
    EXPERT="Expert" in mode

    st.markdown("---")
    col_rt1, col_rt2 = st.columns([2, 3])
    with col_rt1:
        mode_activite = st.radio("🏷️ Type d'activité",["🛣️ Route / Piste","🏔️ Trail / Montagne"],
                                  horizontal=True, key="mode_activite")
    IS_TRAIL = "Trail" in mode_activite

    # ── v8.1 : suggestion auto. du profil terrain (étape 1/2 — sans GPX) ──
    # Doit être placé AVANT l'instanciation du widget terrain_profil_radio
    # (section 4) dans ce même run, sinon Streamlit lève une
    # StreamlitAPIException ("cannot be modified after widget is instantiated").
    if st.session_state.get("_last_mode_activite_for_suggestion") != mode_activite:
        st.session_state["_last_mode_activite_for_suggestion"] = mode_activite
        st.session_state["terrain_profil_radio"] = "🛣️ Route / Plat" if not IS_TRAIL else "🏔️ Trail modéré"

    st.markdown("---")
    st.header("01 · Parcours GPX")
    gpx_file=st.file_uploader("📂 Importer le GPX de la course cible",type=["gpx"],key="gpx_main")
    points=None;dem_elevations=None
    # v8.9 — parcours rechargé depuis l'historique : évite de réimporter le GPX
    class _StoredGPX(io.StringIO):
        """Fichier GPX en mémoire, réhydraté depuis la base (même interface qu'un
        fichier importé : .name + .seek/.read)."""
        def __init__(self, name, xml):
            super().__init__(xml); self.name = name
    if gpx_file is None and st.session_state.get("_stored_gpx"):
        _sg = st.session_state["_stored_gpx"]
        gpx_file = _StoredGPX(_sg["name"], _sg["xml"])
        st.caption(f"📂 Parcours rechargé depuis l'historique : **{_sg['name']}** — "
                   "importe un GPX ci-dessus pour le remplacer.")
        if st.button("✖️ Oublier ce parcours", key="forget_stored_gpx"):
            st.session_state.pop("_stored_gpx", None); st.rerun()
    elif gpx_file is not None:
        try:
            _raw_gpx = gpx_file.getvalue()
            st.session_state["_gpx_xml"] = _raw_gpx.decode("utf-8", errors="ignore") if isinstance(_raw_gpx, bytes) else str(_raw_gpx)
        except Exception:
            st.session_state["_gpx_xml"] = None

    if gpx_file:
        _gpx,points=parse_gpx_points(gpx_file)
        if points:
            # ── v8 PATCH 1b : distance filtrée anti-bruit GPS (seuil adaptatif) ──
            tot_tmp=compute_gpx_distance_filtered(points)  # seuil = médiane×10
            dup_tmp,ddn_tmp=compute_dplus_dminus([getattr(p,"elevation",0.0) or 0.0 for p in points])
            avg_alt_tmp=np.mean([getattr(p,"elevation",0.0) or 0.0 for p in points])
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Distance GPX",f"{tot_tmp/1000:.2f} km")
            c2.metric("D+ GPS",f"{dup_tmp:.0f} m")
            c3.metric("D- GPS",f"{ddn_tmp:.0f} m")
            c4.metric("Alt. moy.",f"{avg_alt_tmp:.0f} m")

            # ── v8.1 : suggestion auto. du profil terrain (étape 2/2 — affinée sur D+/km réel) ──
            # Mêmes garanties d'ordre que l'étape 1 : ce bloc s'exécute en
            # section 1, donc toujours avant le widget terrain_profil_radio (section 4).
            _dplus_per_km_tmp = dup_tmp / max(tot_tmp/1000.0, 0.001)
            if IS_TRAIL:
                if _dplus_per_km_tmp < 30: _suggested_profile = "🏞️ Trail roulant / peu technique"
                elif _dplus_per_km_tmp < 50: _suggested_profile = "🏔️ Trail modéré"
                else: _suggested_profile = "⛰️ Ultra-trail montagneux"
            else:
                _suggested_profile = "🛣️ Route / Plat"
            if st.session_state.get("_last_gpx_name_for_suggestion") != (gpx_file.name, mode_activite):
                st.session_state["_last_gpx_name_for_suggestion"] = (gpx_file.name, mode_activite)
                st.session_state["terrain_profil_radio"] = _suggested_profile
            st.info(f"🤖 Profil de terrain suggéré : **{_suggested_profile}** "
                    f"(D+ = {_dplus_per_km_tmp:.0f} m/km sur {tot_tmp/1000:.1f} km) — "
                    f"appliqué automatiquement, modifiable dans la section 4 si besoin.")

            if IS_TRAIL:
                st.markdown("---")
                st.subheader("🔬 Analyse terrain technique automatique")
                with st.spinner("Analyse sinuosité, pentes, variabilité..."):
                    tech_segs, tech_global = detect_technical_terrain(points, dem_elevations, is_trail=IS_TRAIL)
                    st.session_state["tech_segs"]   = tech_segs
                    st.session_state["tech_global"] = tech_global

                if tech_global.get("global_score", 0) > 0:
                    score=tech_global["global_score"]; label=tech_global["label"]
                    pct_tech=tech_global.get("pct_technique",0)
                    badge_css=(("background:rgba(230,57,70,0.18);color:"+C_RED+";border:1px solid rgba(230,57,70,0.45)") if score>0.65 else ("background:"+C_SURFACE_2+";color:"+C_RED_SOFT+";border:1px solid "+C_LINE) if score>0.40 else ("background:"+C_SURFACE_2+";color:"+C_TEXT_MUT+";border:1px solid "+C_LINE))
                    st.markdown(f'<span style="{badge_css};border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:700;">{label} — score {score:.2f}/1.00</span>',unsafe_allow_html=True)
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Score technique global",f"{score:.2f}/1.00"); c2.metric("Segments techniques",f"{pct_tech:.0f}%")
                    c3.metric("Sinuosité moy.",f"{tech_global.get('sinuosity_mean',1.0):.3f}"); c4.metric("Pente max",f"{tech_global.get('grade_max_all',0):.1f}%")
                    st.info(f"💡 Multiplicateurs suggérés : **k_up ×{tech_global['k_up_adj']:.2f}** · **k_down ×{tech_global['k_down_adj']:.2f}** · **surface ×{tech_global['surface_mult_adj']:.2f}**")
                    if tech_segs:
                        with st.expander("📊 Détail par km — score technique"):
                            df_tech=pd.DataFrame(tech_segs)
                            cols_show=["km_start","km_end","label","tech_score","sinuosity","grade_max","grade_std","turns_score_km"]
                            st.dataframe(df_tech[[c for c in cols_show if c in df_tech.columns]],use_container_width=True,hide_index=True)
                            fig_tech,ax_tech=plt.subplots(figsize=(11,3))
                            km_mids=[(s["km_start"]+s["km_end"])/2 for s in tech_segs]; scores=[s["tech_score"] for s in tech_segs]
                            colors_t=[C_DIM if s<0.25 else C_GREY if s<0.5 else C_RED_SOFT if s<0.75 else C_RED for s in scores]
                            ax_tech.bar(km_mids,scores,width=0.85,color=colors_t,alpha=0.85)
                            ax_tech.axhline(0.5,color=C_RED,lw=1,ls="--",label="Seuil technique")
                            ax_tech.axhline(0.75,color=C_RED,lw=1,ls="--",label="Seuil très technique")
                            ax_tech.set_xlabel("Distance (km)"); ax_tech.set_ylabel("Score technique")
                            ax_tech.set_title("Score de technicité par km"); ax_tech.legend(); ax_tech.set_ylim(0,1); ax_tech.grid(alpha=0.3); fig_tech.tight_layout()
                            st.pyplot(fig_tech); plt.close(fig_tech)

    with st.expander("🏔️ Correction altimétrique DEM (optionnel)"):
        st.info("Le GPS vertical a une précision de ±5-15 m. Le DEM donne l'altitude réelle à ±1 m.")
        use_dem=st.checkbox("Activer la correction DEM",value=False,key="use_dem")
        dem_dataset="srtm30m"
        if use_dem:
            dem_dataset=st.selectbox("Dataset",["srtm30m (global, 30m)","eudem25m (Europe, 25m — plus précis)","mapzen (global fusion)"],key="dem_ds").split()[0]
            if gpx_file and points and st.button("🔄 Télécharger et corriger l'altitude"):
                with st.spinner("Correction DEM en cours..."):
                    dem_elevations=list(correct_elevations_dem(points,max_points=100,dataset=dem_dataset))
                    st.session_state["dem_elevations"]=dem_elevations
                    dup_dem,ddn_dem=compute_dplus_dminus([e or 0.0 for e in dem_elevations])
                    st.success(f"DEM OK — D+ DEM: **{dup_dem:.0f} m** | D- DEM: **{ddn_dem:.0f} m**")
        if "dem_elevations" in st.session_state:
            dem_elevations=st.session_state["dem_elevations"]
            tech_segs,tech_global=detect_technical_terrain(points,dem_elevations,is_trail=IS_TRAIL)
            st.session_state["tech_segs"]=tech_segs; st.session_state["tech_global"]=tech_global

    with st.expander("🌿 Détection surface OSM (Overpass API, gratuit)",expanded=False):
        if not IS_TRAIL:
            st.info("ℹ️ La détection OSM de surfaces naturelles est principalement utile en mode Trail.")
        if gpx_file and points and st.button("🔍 Analyser la surface via OSM",key="btn_osm"):
            with st.spinner("Requête Overpass API en cours..."):
                n_pts_osm=len(points); step_osm=max(1,n_pts_osm//100)
                lats_osm=tuple(points[i].latitude  for i in range(0,n_pts_osm,step_osm))
                lons_osm=tuple(points[i].longitude for i in range(0,n_pts_osm,step_osm))
                osm_result=fetch_osm_surface(lats_osm,lons_osm); st.session_state["osm_surface"]=osm_result
        if "osm_surface" in st.session_state:
            osm=st.session_state["osm_surface"]
            if "error" in osm:st.error(f"Erreur OSM : {osm['error']}")
            else:
                c1,c2,c3=st.columns(3)
                c1.metric("Surface dominante",osm.get("dominant_surface","—"))
                c2.metric("Multiplicateur OSM",f"×{osm.get('surface_mult_osm',1.06):.3f}")
                c3.metric("Couverture",f"{osm.get('coverage_pct',0):.0f}%  ({osm.get('ways_found',0)} ways)")
                if osm.get("surface_counts"):
                    df_osm=pd.DataFrame(list(osm["surface_counts"].items()),columns=["Surface","Occurrences"])
                    st.dataframe(df_osm,use_container_width=True,hide_index=True)
                SURFACE_OPTIONS["🤖 Détecté automatiquement (OSM)"]=osm.get("surface_mult_osm",1.06)

    st.markdown("---")
    st.header("02 · Courses de référence")
    st.info("Calibrent le modèle sur l'athlète. Minimum conseillé : **3 références** variées. Chargez des fichiers FIT/TCX pour inclure la FC.")

    if "n_refs" not in st.session_state:st.session_state.n_refs=3
    cc1,cc2=st.columns(2)
    with cc1:
        if st.button("➕ Ajouter une référence") and st.session_state.n_refs<6:st.session_state.n_refs+=1
    with cc2:
        if st.button("➖ Retirer") and st.session_state.n_refs>1:st.session_state.n_refs-=1

    refs_raw=[]
    for i in range(1,st.session_state.n_refs+1):
        with st.expander(f"📌 Référence {i}",expanded=(i<=2)):
            use_file=st.checkbox(f"Importer depuis fichier FIT/TCX",key=f"use_file_{i}")
            c1,c2,c3,c4=st.columns(4)
            dist=c1.number_input("Distance (m)",value=float(st.session_state.get(f"dist_{i}",5000*i)),key=f"dist_{i}")
            temps=hms_input("Temps (hh:mm:ss)",default="0:40:00",key=f"temps_{i}")
            dup=c3.number_input("D+ (m)",value=float(st.session_state.get(f"dup_{i}",0.0)),key=f"dup_{i}")
            ddn=c4.number_input("D- (m)",value=float(st.session_state.get(f"ddn_{i}",0.0)),key=f"ddn_{i}")
            file_in=st.file_uploader(f"Fichier FIT/TCX",type=["fit","tcx"],key=f"fileref_{i}") if use_file else None
            dur_hms_file=avg_temp_ref=avg_wind_ref=avg_hum_ref=hr_ref=None
            fname=file_in.name.lower() if file_in else ""
            fit_data=tcx_data=None
            ref_breakpoint=None
            if file_in:
                if fname.endswith(".fit"):
                    fit_data=parse_fit_ref(file_in)
                    if fit_data:
                        dist,dup,ddn=fit_data["distance"],fit_data["D_up"],fit_data["D_down"]
                        dur_hms_file=fit_data["duration_hms"]
                        avg_temp_ref,avg_wind_ref,avg_hum_ref=fit_data["avg_temp"],fit_data["avg_wind"],fit_data["avg_humidity"]
                        hr_ref=fit_data.get("hr_analysis")
                elif fname.endswith(".tcx"):
                    tcx_data=parse_tcx_ref(file_in)
                    if tcx_data:
                        dist,dup,ddn=tcx_data["distance"],tcx_data["D_up"],tcx_data["D_down"]
                        dur_hms_file=tcx_data["duration_hms"]
                        avg_temp_ref,avg_wind_ref,avg_hum_ref=tcx_data["avg_temp"],tcx_data["avg_wind"],tcx_data["avg_humidity"]
                        hr_ref=tcx_data.get("hr_analysis")
                sh=hms_input("Début segment","0:00:00",key=f"start_{i}",compact=True)
                eh=hms_input("Fin segment","23:59:59",key=f"end_{i}",compact=True)
                start_td,end_td=hms_to_timedelta(sh),hms_to_timedelta(eh)
                pts_src=None
                if fit_data and "points" in fit_data:pts_src=fit_data["points"]
                elif tcx_data and "points" in tcx_data:pts_src=tcx_data["points"]
                _pts_for_breakpoint=pts_src
                if start_td.total_seconds()>0 or end_td.total_seconds()<86399:
                    if pts_src:
                        seg=extract_segment(pts_src,start_td,end_td)
                        _pts_for_breakpoint=seg
                        seg_dist=0.0;seg_elevs=[];seg_times=[]
                        for j in range(1,len(seg)):
                            p1,p2=seg[j-1],seg[j]
                            la1,lo1=(p1["lat"],p1["lon"]) if isinstance(p1,dict) else (p1.latitude,p1.longitude)
                            la2,lo2=(p2["lat"],p2["lon"]) if isinstance(p2,dict) else (p2.latitude,p2.longitude)
                            e2=p2.get("elev",0) if isinstance(p2,dict) else p2.elevation
                            t2=p2.get("time") if isinstance(p2,dict) else p2.time
                            seg_dist+=haversine_m(la1,lo1,la2,lo2);seg_elevs.append(e2)
                            if t2:seg_times.append(t2)
                        dup,ddn=compute_dplus_dminus(seg_elevs)
                        if len(seg_times)>=2:dur_hms_file=seconds_to_hms((seg_times[-1]-seg_times[0]).total_seconds())
                        dist=round(seg_dist)
                # v8.4 — détection automatique du point de rupture d'allure sur CETTE
                # référence (surtout utile pour les références longues/ultra type Saint-
                # Jacques) : donne un signal de fatigue personnel directement depuis tes
                # courses réelles, sans dépendre de l'onglet 🧪 Tests d'endurance + VC —
                # cet onglet reste dédié à l'analyse des tests d'effort/entraînement,
                # indépendant de la prédiction de course.
                if _pts_for_breakpoint:
                    _t_bp,_pace_bp=compute_pace_series_from_points(_pts_for_breakpoint)
                    if _t_bp is not None and len(_t_bp)>=10:
                        ref_breakpoint=detect_pace_breakpoint(_t_bp,_pace_bp)
                        if ref_breakpoint and ref_breakpoint.get("r2",0)>=0.3:
                            st.caption(f"🔍 Rupture d'allure détectée à {ref_breakpoint['pct_break']:.0f}% de cette "
                                       f"course (dégradation {ref_breakpoint['drop_pct']:+.0f}%, R²={ref_breakpoint['r2']:.2f}) "
                                       f"— disponible en option pour la fatigue en section 4 (voir le R² avant de "
                                       f"t'y fier : sous 0.6, le signal reste faible).")
            else:
                if EXPERT:
                    cs2,ce2=st.columns(2)
                    avg_temp_ref=cs2.number_input(f"Temp moy. course (°C)",value=15.0,key=f"avgT_{i}")
                    avg_hum_ref=ce2.number_input(f"Humidité moy. (%)",value=60.0,key=f"avgH_{i}")
                else:avg_temp_ref=avg_hum_ref=None
            # v9.1 — temps d'arrêt inclus dans le chrono de la référence (ravitos, pauses) :
            # on les retire pour calibrer le modèle sur du temps EN MOUVEMENT, puis on
            # rajoutera explicitement les arrêts prévus sur la course cible.
            arret_ref=hms_input("⏱️ Dont temps d'arrêt (ravitos, pauses)",default="0:00:00",key=f"arret_ref_{i}",
                                help="Temps total passé à l'arrêt pendant cette course de référence. "
                                     "Laisse 0:00:00 si le chrono est déjà un temps en mouvement.")
            temps_eff=dur_hms_file if dur_hms_file else temps
            secs_chrono=hms_to_seconds(temps_eff);secs_arret_ref=hms_to_seconds(arret_ref)
            if secs_arret_ref>0 and secs_arret_ref<secs_chrono:
                temps_eff=seconds_to_hms(secs_chrono-secs_arret_ref)
                st.caption(f"→ chrono {seconds_to_hms(secs_chrono)} − {seconds_to_hms(secs_arret_ref)} d'arrêt "
                           f"= **{temps_eff} en mouvement** (c'est cette valeur qui calibre le modèle).")
            elif secs_arret_ref>=secs_chrono and secs_arret_ref>0:
                st.warning("⚠️ Le temps d'arrêt annoncé dépasse le chrono — valeur ignorée.")
            secs_brut=hms_to_seconds(temps_eff);dist_km=safe_float(dist,1.0)/1000.0
            if secs_brut>0 and dist_km>0:
                st.caption(f"📍 {dist:.0f} m · {temps_eff} · **{pace_str(secs_brut/dist_km)}/km**"
                           +(f" · D+ {dup:.0f}m" if dup>0 else "")
                           +(f" · Temp GPS: {avg_temp_ref:.0f}°C" if avg_temp_ref else "")
                           +(f" · FC fiabilité: {hr_ref.get('reliability')}" if hr_ref else ""))
            if hr_ref and hr_ref.get("hr_max"):
                st.caption(f"💓 FC max {hr_ref['hr_max']} bpm · FC moy. {hr_ref.get('hr_avg','—')} bpm · dérive {hr_ref['hr_drift']} bpm · seuil ~{hr_ref['hr_threshold_est']} bpm")
            hr_avg_ref = hr_ref.get("hr_avg") if hr_ref else None
            hr_max_ref = hr_ref.get("hr_max") if hr_ref else None
            refs_raw.append({"distance":float(dist),"temps":str(temps_eff),"arret_s":float(secs_arret_ref),
                              "D_up":float(dup),"D_down":float(ddn),
                              "duration_hms_file":(temps_eff if secs_arret_ref>0 else dur_hms_file),
                              "avg_temp":avg_temp_ref,"avg_humidity":avg_hum_ref,"avg_wind":avg_wind_ref,
                              "hr_analysis":hr_ref,"hr_avg":hr_avg_ref,"hr_max":hr_max_ref,
                              "breakpoint":ref_breakpoint})

    # v8.4 — signature de fatigue personnelle, calculée automatiquement à partir des
    # ruptures d'allure détectées ci-dessus sur tes propres références (aucune action
    # séparée requise, aucune dépendance à l'onglet 🧪 Tests d'endurance + VC).
    # Seuil de détection à 0.3 (affichage), mais le signal reste faible en dessous de
    # 0.6 — la case reste décochée par défaut dans ce cas (voir section 4). Attention
    # en particulier au transfert marathon → ultra-trail : le "mur" du marathon (déplétion
    # glycogénique ~30-35km) et la fatigue d'un ultra de plusieurs heures ne sont pas le
    # même phénomène physiologique, même quand la rupture est nette dans les données.
    _bp_attempted=[r["breakpoint"] for r in refs_raw if r.get("breakpoint")]
    _bp_valid=[b for b in _bp_attempted if b.get("r2",0)>=0.3]
    auto_fatigue_threshold=auto_fatigue_rate=auto_fatigue_r2max=None
    auto_fatigue_n=len(_bp_valid); auto_fatigue_total=len(_bp_attempted); auto_fatigue_std=None
    if _bp_valid:
        _pcts=[b["pct_break"] for b in _bp_valid]; _drops=[abs(b["drop_pct"]) for b in _bp_valid]
        auto_fatigue_threshold=round(float(np.mean(_pcts))); auto_fatigue_rate=round(float(np.mean(_drops)))
        auto_fatigue_std=float(np.std(_pcts)) if len(_pcts)>=2 else None
        auto_fatigue_r2max=float(max(b.get("r2",0) for b in _bp_valid))
    st.session_state["auto_fatigue_threshold"]=auto_fatigue_threshold
    st.session_state["auto_fatigue_rate"]=auto_fatigue_rate
    st.session_state["auto_fatigue_n"]=auto_fatigue_n
    st.session_state["auto_fatigue_total"]=auto_fatigue_total
    st.session_state["auto_fatigue_std"]=auto_fatigue_std
    st.session_state["auto_fatigue_r2max"]=auto_fatigue_r2max


    st.markdown("---")
    st.header("03 · Recalibration des références vers les conditions idéales")
    st.markdown('<div class="highlight-box"><strong>Pourquoi recalibrer ?</strong><br>Une course réalisée par 30°C et 80% d\'humidité vaut <em>physiologiquement mieux</em> qu\'un temps identique par 10°C et temps sec.</div>',unsafe_allow_html=True)
    use_recalibrated=st.checkbox("✅ Recalibrer les références vers les conditions idéales (fortement recommandé)",value=True)
    opt_temp=10.0;use_wbgt=True;elev_ref_power=0.65;temp_ref_power=0.80
    with st.expander("⚙️ Paramètres de recalibration"):
        opt_temp=st.slider("Température optimale de course (°C)",5.0,20.0,10.0,0.5)
        use_wbgt=st.checkbox("Utiliser le WBGT (chaleur+humidité) — recommandé",value=True)
        col_ep1,col_ep2=st.columns(2)
        with col_ep1:elev_ref_power=st.slider("Force correction pente des références",0.0,1.0,0.65,0.05)
        with col_ep2:temp_ref_power=st.slider("Force correction température des références",0.0,1.0,0.80,0.05)

    st.subheader("📋 Résumé de la recalibration")
    _k_up_prev=st.session_state.get("k_up_val",22.0);_k_down_prev=st.session_state.get("k_down_val",4.5)
    _g0u_prev=st.session_state.get("g0_up_val",3.0);_g0d_prev=st.session_state.get("g0_down_val",2.5)
    calib_rows=[];cold_quad=0.0003;hot_quad=0.0055;temp_max_penalty=0.20
    for r in refs_raw:
        t_brut=hms_to_seconds(r.get("duration_hms_file") or r.get("temps",""))
        dist_km_r=safe_float(r.get("distance",1.0))/1000.0
        avg_t=r.get("avg_temp");avg_h=safe_float(r.get("avg_humidity",50.0),50.0)
        wbgt_val=wbgt_simplified(avg_t,avg_h) if avg_t is not None and use_wbgt else None
        t_ideal=(recalibrate_ref_to_ideal(ref={**r,"temps":r.get("duration_hms_file") or r.get("temps","0:00:00")},
            opt_temp=opt_temp,use_wbgt=use_wbgt,cold_quad=cold_quad,hot_quad=hot_quad,
            temp_max_penalty=temp_max_penalty,k_up=_k_up_prev,k_down=_k_down_prev,down_cap=-0.10,
            g0_up=_g0u_prev,g0_down=_g0d_prev,max_up=0.48,max_down=-0.06,
            elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power) if use_recalibrated else float(t_brut))
        gain_s=t_brut-t_ideal
        calib_rows.append({"Distance":f"{safe_float(r['distance'])/1000:.1f} km",
                            "Temps brut":seconds_to_hms(t_brut),"Allure brute":pace_str(t_brut/dist_km_r) if dist_km_r>0 else "-",
                            "D+":f"{r['D_up']:.0f} m","Temp GPS":f"{avg_t:.0f}°C" if avg_t is not None else "?",
                            "WBGT":f"{wbgt_val:.1f}°C" if wbgt_val is not None else "-",
                            "Temps recalibré":seconds_to_hms(t_ideal) if use_recalibrated else "—",
                            "Allure recalibrée":pace_str(t_ideal/dist_km_r) if (use_recalibrated and dist_km_r>0) else "—",
                            "FC moy.":f"{r['hr_avg']} bpm" if r.get('hr_avg') else "—",
                            "Gain correction":f"-{seconds_to_hms(gain_s)}" if gain_s>0 else (f"+{seconds_to_hms(-gain_s)}" if gain_s<0 else "0")})
    st.dataframe(pd.DataFrame(calib_rows),use_container_width=True)

    refs_with_hr = [{"dur_s": hms_to_seconds(r.get("duration_hms_file") or r.get("temps","0")),
                     "hr_avg": r.get("hr_avg"), "hr_max": r.get("hr_max")}
                    for r in refs_raw if r.get("hr_avg")]

    if refs_with_hr:
        st.markdown("#### 💓 Prédiction de zone FC cible — données personnelles de l'athlète")
        _use_hr_regression = st.checkbox(
            "Intégrer ma régression FC personnelle", value=True, key="use_hr_regression_chk",
            help="Désactive cette section si tu ne veux pas t'appuyer sur la régression FC/durée calculée à partir "
                 "de tes références.")
        if _use_hr_regression:
            st.caption("Basé uniquement sur les FC observées lors des courses de référence — pas de valeurs normatives.")
            # Priorité au temps RÉELLEMENT calculé (section 6, si déjà lancé) plutôt qu'à
            # l'estimation grossière (allure moyenne des références × distance GPX) — cette
            # dernière ne sert plus que de repli tant qu'aucun calcul n'a encore été fait.
            # Comme cette section 3 s'exécute avant la section 6 dans le script, le résultat
            # utilisé est celui du dernier calcul lancé (léger décalage d'un run, comme pour
            # k_up_val ailleurs dans l'app) — se met à jour automatiquement après le calcul.
            _res_prev = st.session_state.get("res")
            if _res_prev and _res_prev.get("total_s"):
                _dur_target_s = float(_res_prev["total_s"])
                _dur_source = "temps prédit (dernier calcul lancé)"
            elif st.session_state.get("temps_objectif_target"):
                _dur_target_s = hms_to_seconds(st.session_state.get("temps_objectif_target",""))
                _dur_source = "objectif de temps saisi"
            else:
                _dur_target_s = None; _dur_source = "estimation grossière"
            if not _dur_target_s or _dur_target_s == 0:
                _allures = [hms_to_seconds(r.get("duration_hms_file") or r.get("temps","0")) / max(1, safe_float(r["distance"],1)/1000)
                            for r in refs_raw if safe_float(r.get("distance",0)) > 0]
                _allure_moy = float(np.mean(_allures)) if _allures else 360.0
                _dist_km_est = tot_tmp / 1000.0 if (gpx_file and points) else 21.097
                _dur_target_s = _allure_moy * _dist_km_est
            st.caption(f"⏱️ Durée cible utilisée : **{seconds_to_hms(_dur_target_s)}** ({_dur_source})"
                       + ("" if _res_prev else " — lance le calcul en section 6 pour utiliser le temps réellement prédit."))
            hr_pred = predict_hr_zone(refs_with_hr, _dur_target_s)
            if hr_pred:
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("FC moy. prédite", f"{hr_pred['hr_target_avg']} bpm")
                hc2.metric("Fourchette cible", f"{hr_pred['hr_target_range'][0]}–{hr_pred['hr_target_range'][1]} bpm")
                hc3.metric("FC max estimée",   f"{hr_pred['hr_target_max']} bpm")
                hc4.metric("Références FC",    f"{hr_pred['n_refs']} course(s)",
                           delta=f"R²={hr_pred['r2']}" if hr_pred.get("r2") else "Moyenne brute")
                if hr_pred.get("model") == "regression" and hr_pred.get("r2",0) >= 0.70:
                    st.success(f"✅ Régression FC solide (R²={hr_pred['r2']:.2f}) — vise **{hr_pred['hr_target_range'][0]}–{hr_pred['hr_target_range'][1]} bpm** en moyenne sur la course.")
                elif hr_pred.get("model") == "regression":
                    st.info(f"📊 Régression FC indicative (R²={hr_pred['r2']:.2f}) — ajoute plus de références pour affiner. Cible : **{hr_pred['hr_target_range'][0]}–{hr_pred['hr_target_range'][1]} bpm**.")
                else:
                    st.info(f"💓 FC cible estimée (moyenne de {hr_pred['n_refs']} référence(s)) : **{hr_pred['hr_target_range'][0]}–{hr_pred['hr_target_range'][1]} bpm**.")
                if len(refs_with_hr) >= 2:
                    fig_hr_pred, ax_hr_p = plt.subplots(figsize=(7, 3.5))
                    durs_h = [r["dur_s"]/60.0 for r in refs_with_hr]
                    hrs_h  = [r["hr_avg"] for r in refs_with_hr]
                    ax_hr_p.scatter(durs_h, hrs_h, s=80, color=C_RED, zorder=5, label="FC moy. observée")
                    if hr_pred.get("model") == "regression":
                        d_line = np.linspace(max(1, min(durs_h)*0.8), max(max(durs_h)*1.2, _dur_target_s/60*1.1), 80)
                        hr_line = [hr_pred["slope"]*math.log(max(1,d*60))+hr_pred["intercept"] for d in d_line]
                        ax_hr_p.plot(d_line, hr_line, color=C_WHITE, lw=2, ls="--", label="Régression log(durée)")
                        sigma_viz = (hr_pred['hr_target_range'][1]-hr_pred['hr_target_range'][0])/2
                        ax_hr_p.fill_between(d_line,[h-sigma_viz for h in hr_line],[h+sigma_viz for h in hr_line],alpha=0.12,color=C_WHITE)
                    ax_hr_p.axvline(_dur_target_s/60, color=C_RED, lw=2, ls=":", label=f"Course cible ({_dur_target_s/60:.0f} min)")
                    ax_hr_p.scatter([_dur_target_s/60],[hr_pred["hr_target_avg"]],s=150,color=C_RED,marker="*",zorder=6,label=f"FC cible {hr_pred['hr_target_avg']} bpm")
                    ax_hr_p.set_xlabel("Durée de la course (min)"); ax_hr_p.set_ylabel("FC moyenne (bpm)")
                    ax_hr_p.set_title("Régression FC personnelle — données athlète uniquement")
                    ax_hr_p.legend(fontsize=8); ax_hr_p.grid(alpha=0.3); fig_hr_pred.tight_layout()
                    st.pyplot(fig_hr_pred); plt.close(fig_hr_pred)
    else:
        st.caption("💓 Chargez des fichiers FIT ou TCX dans les références pour obtenir une prédiction de zone cardiaque personnalisée.")


    st.markdown("---")
    st.header("04 · Paramètres du modèle")

    with st.expander("🌡️ Température & Humidité",expanded=False):
        # v8.1 — calibré sur 1245 segments coureurs (top10) de 12 ultra-trails réels :
        # le froid a un effet quasi nul sur l'allure élite (0.0015→0.0003),
        # la chaleur pénalise ~3x plus que l'estimation initiale (0.0020→0.0055)
        cold_quad=0.0003; hot_quad=0.0055; temp_max_penalty=0.20; temp_power=1.0
        if EXPERT:
            c1,c2=st.columns(2)
            cold_quad=c1.number_input("Sensibilité froid",value=0.0003,step=0.0002,format="%.4f")
            hot_quad=c2.number_input("Sensibilité chaleur",value=0.0055,step=0.0002,format="%.4f")
            temp_max_penalty=st.slider("Pénalité max température (%)",0.00,0.30,0.20,0.01)
            temp_power=st.slider("Damping température (puissance)",0.2,1.2,1.0,0.05)

    with st.expander("🏔️ Altitude physiologique (hypoxie)"):
        apply_altitude=st.checkbox("Appliquer la pénalité d'altitude",value=True)
        altitude_ref_m=0.0
        if apply_altitude:altitude_ref_m=st.number_input("Altitude d'entraînement habituelle (m)",value=0.0,step=100.0)

    with st.expander("🎢 Modèle de pente & Terrain",expanded=True):
        apply_grade=st.checkbox("Prendre en compte la pente",value=True)
        use_minetti=st.checkbox("Modèle Minetti",value=True)

        # ── v8.2 : import direct d'un fichier de profils calibrés (JSON) ────────
        # Auparavant, un profil calibré (onglet 👥 Analyse de cohorte) ne pouvait
        # être appliqué ici qu'en repassant par cet autre onglet. Ce bloc permet
        # de déposer directement profils_calibres.json (ou le fichier exporté par
        # l'artefact de calibration externe) et applique aussitôt les coefficients
        # — DOIT rester avant l'instanciation du widget terrain_profil_radio
        # ci-dessous, sinon Streamlit lève une StreamlitAPIException.
        st.markdown("##### 📥 Importer un profil calibré (JSON)")
        st.caption("Fichier exporté depuis l'onglet 👥 Analyse de cohorte, ou depuis l'artefact de calibration "
                   "externe. Les coefficients (k_up, plafond de pente, seuil/taux de fatigue…) s'appliquent "
                   "automatiquement dès le dépôt — plus besoin de repasser par l'onglet Cohorte.")
        _pred_import_file = st.file_uploader(
            "Déposer un fichier .json", type=["json"], key="pred_import_profiles_file")
        if _pred_import_file and st.session_state.get("_last_pred_imported_profile_file") != _pred_import_file.name:
            st.session_state["_last_pred_imported_profile_file"] = _pred_import_file.name
            try:
                _imported_raw = _json.loads(_pred_import_file.read().decode("utf-8"))
                if not isinstance(_imported_raw, dict) or not _imported_raw:
                    raise ValueError("le fichier ne contient aucun profil exploitable.")
                _REQUIRED_KEYS = {"k_up", "max_cap"}
                _valid_imported = {name: params for name, params in _imported_raw.items()
                                    if isinstance(params, dict) and _REQUIRED_KEYS.issubset(params.keys())}
                if not _valid_imported:
                    raise ValueError("aucun profil valide trouvé (clés k_up/max_cap manquantes).")
                if "custom_terrain_profiles" not in st.session_state:
                    st.session_state["custom_terrain_profiles"] = {}
                st.session_state["custom_terrain_profiles"].update(_valid_imported)
                # Sélectionne automatiquement le profil importé (le premier si
                # plusieurs) AVANT l'instanciation du radio ci-dessous, pour que
                # les coefficients s'auto-appliquent sans clic supplémentaire.
                st.session_state["terrain_profil_radio"] = next(iter(_valid_imported))
                st.session_state["_prev_terrain_profil"] = ""  # force la resynchro des sliders tp_*
                _names = ", ".join(_valid_imported.keys())
                st.success(f"✅ {len(_valid_imported)} profil(s) importé(s) et appliqué(s) : {_names}")
            except Exception as e:
                st.error(f"❌ Fichier JSON invalide — {e}")

        st.markdown("##### 🗺️ Profil du parcours")
        _all_terrain_profiles = get_all_terrain_profiles()
        terrain_profil=st.radio("Type de terrain",list(_all_terrain_profiles.keys()),horizontal=True,key="terrain_profil_radio")
        _prev=st.session_state.get("_prev_terrain_profil","")
        if terrain_profil!=_prev:
            st.session_state["_prev_terrain_profil"]=terrain_profil
            _d=_all_terrain_profiles[terrain_profil]
            for k,v in _d.items():
                st.session_state[f"tp_{k}"]=v
        _d=_all_terrain_profiles[terrain_profil]
        _profil_info={
            "🛣️ Route / Plat":"Route, piste, parcours plat. k_up=4 (v8 — moins réactif au bruit GPS), lissage altitude ×25pts. Allure stable sur plat.",
            "🏞️ Trail roulant / peu technique":"v8.1 — Ultra-trail à faible D+/km (<30 m/km), type Sainte-Lyon/Nantes-Montaigu/EcoTrail. k_up=19.5 mais plafond bas (max_cap=0.15) : la pente sature vite car les côtes y sont rares. Fatigue continue dès le début (seuil 5%).",
            "🏔️ Trail modéré":"v8.1 — Profil technique moyen calibré sur Templiers/Chianti/St-Jacques (D+ ~40-50m/km). k_up=10.7, plafond 67%. Seuil de fatigue à 60% confirmé par la donnée réelle (seul profil où l'hypothèse plateau-puis-chute tient).",
            "⛰️ Ultra-trail montagneux":"v8.1 — Calibré sur TDS/UTMB/OCC/TransGC/Transvulcania/MCC (D+ ~55-65m/km). k_up=13.2, plafond le plus élevé (79%) sur les pentes très raides. Fatigue continue dès le début (seuil 5%). DEM recommandé.",
        }
        if terrain_profil in _profil_info:
            st.info(_profil_info[terrain_profil])
        else:
            _eff_fat_thr = st.session_state.get("tp_fatigue_threshold", _d.get("fatigue_threshold","—"))
            _eff_fat_rate = st.session_state.get("tp_fatigue_rate", _d.get("fatigue_rate","—"))
            st.info(f"📌 Profil calibré (importé ou depuis l'onglet 👥 Analyse de cohorte) : k_up={_d.get('k_up')} · "
                    f"plafond={_d.get('max_cap',0)*100:.0f}% · seuil fatigue phase 1={_eff_fat_thr}% · "
                    f"taux fatigue phase 1={_eff_fat_rate}% — paramètres non calibrés (k_down, minetti_weight, "
                    f"lissage…) repris des valeurs standard « Trail modéré ». La phase 2 (fatigue personnelle) se "
                    f"règle plus bas, dans l'expander 🔋 Fatigue en course.")

        tech_global_ui=st.session_state.get("tech_global",{})
        if IS_TRAIL and tech_global_ui.get("global_score",0)>0 and terrain_profil=="🏔️ Trail modéré":
            adj_up=tech_global_ui.get("k_up_adj",1.0);adj_dn=tech_global_ui.get("k_down_adj",1.0)
            sugg_k_up=round(_d["k_up"]*adj_up,1);sugg_k_dn=round(_d["k_down"]*adj_dn,1)
            st.markdown(f'<div class="note-box note-red">🤖 <b>Suggestion auto (terrain score {tech_global_ui["global_score"]:.2f})</b> : k_up → <b>{sugg_k_up}</b> · k_down → <b>{sugg_k_dn}</b></div>',unsafe_allow_html=True)
            if st.button("✅ Appliquer coefficients détectés automatiquement",key="apply_tech_coeffs"):
                st.session_state["tp_k_up"]=sugg_k_up; st.session_state["tp_k_down"]=sugg_k_dn; st.rerun()

        st.markdown("##### 🌿 Surface du terrain")
        if IS_TRAIL: _surf_opts=SURFACE_OPTIONS
        else:
            _surf_opts={k:v for k,v in SURFACE_OPTIONS.items() if any(x in k for x in ["Route","Piste","Gravier","stabilisé","Chemin"])}
            if not _surf_opts:_surf_opts={"🏟️ Route / Piste synthétique":1.00,"🪨 Chemin stabilisé / Gravier":1.03}
        surface_sel=st.selectbox("Type de surface",list(_surf_opts.keys()),key="surface_sel")
        surface_mult=_surf_opts[surface_sel]
        if surface_sel.startswith("🤖") and "osm_surface" in st.session_state:
            surface_mult=st.session_state["osm_surface"].get("surface_mult_osm",1.06)
        st.caption(f"Multiplicateur surface : **×{surface_mult:.3f}** — pénalité **+{(surface_mult-1)*100:.1f}%** sur l'allure de base")

        minetti_weight=0.6
        if use_minetti:
            minetti_weight=st.slider("Part de Minetti",0.0,1.0,float(st.session_state.get("tp_minetti_weight",_d["minetti_weight"])),0.05,key="tp_minetti_weight")

        st.markdown("##### ⚙️ Coefficients détaillés")
        col_r1a,col_r1b,col_r1c=st.columns(3)
        with col_r1a:
            k_up=st.number_input("k_up — coefficient montée",min_value=1.0,max_value=60.0,
                                   value=float(st.session_state.get("tp_k_up",_d["k_up"])),step=0.5,key="tp_k_up")
        with col_r1b:
            k_down=st.number_input("k_down — coefficient descente",min_value=0.5,max_value=20.0,
                                    value=float(st.session_state.get("tp_k_down",_d["k_down"])),step=0.5,key="tp_k_down")
        with col_r1c:
            down_cap=st.number_input("Cap descente (max gain)",min_value=-0.30,max_value=0.0,
                                      value=float(st.session_state.get("tp_down_cap",_d["down_cap"])),
                                      step=0.01,format="%.2f",key="tp_down_cap")
        col_r2a,col_r2b=st.columns(2)
        with col_r2a:
            elev_smooth_window=st.slider("Lissage altitude (fenêtre pts GPS)",1,51,
                                          int(st.session_state.get("tp_elev_smooth_window",_d["elev_smooth_window"])),2,key="tp_elev_smooth_window")
        with col_r2b:
            grade_power=st.slider("Amortissement effet pente",0.2,1.0,
                                   float(st.session_state.get("tp_grade_power",_d["grade_power"])),0.05,key="tp_grade_power")
        st.markdown("**Plafond anti-accumulation**")
        col_r3a,col_r3b,col_r3c=st.columns(3)
        with col_r3a:
            base_cap=st.slider("Plafond de base (%)",0.02,0.40,float(st.session_state.get("tp_base_cap",_d["base_cap"])),0.01,key="tp_base_cap")
        with col_r3b:
            extra_per_pct=st.slider("Extra par % de pente",0.000,0.030,float(st.session_state.get("tp_extra_per_pct",_d["extra_per_pct"])),0.001,format="%.3f",key="tp_extra_per_pct")
        with col_r3c:
            max_cap=st.slider("Plafond absolu (%)",0.05,0.90,float(st.session_state.get("tp_max_cap",_d["max_cap"])),0.01,key="tp_max_cap")
        g0_up=float(_d.get("g0_up",3.0));g0_down=2.5;max_up=float(max_cap);max_down=-0.06
        st.session_state["k_up_val"]=k_up;st.session_state["k_down_val"]=k_down
        st.session_state["g0_up_val"]=g0_up;st.session_state["g0_down_val"]=g0_down
        _terrain_colors={"🛣️ Route / Plat":C_GREY,"🏔️ Trail modéré":C_WHITE,"⛰️ Ultra-trail montagneux":C_RED}
        _col=_terrain_colors.get(terrain_profil,"#444")
        st.markdown(f'<div class="note-box" style="border-left:3px solid {_col};"><b>{terrain_profil}</b> · k_up={k_up:.0f} · k_down={k_down:.0f} · Minetti={minetti_weight:.2f} · Plafond={max_cap:.0%} · Surface <b>{surface_sel.split()[0]} ×{surface_mult:.2f}</b></div>',unsafe_allow_html=True)

    with st.expander("🥾 Marche en forte pente (VAM)", expanded=True):
        st.caption(
            "Au-delà d'un certain % de pente, même les meilleurs grimpeurs marchent — l'allure devient gouvernée "
            "par la VAM (Vitesse d'Ascension Moyenne, m de D+/heure) plutôt que par la vitesse horizontale. Sans "
            "ce correctif, le modèle peut sous-estimer fortement le temps sur les portions très raides (>25-30%), "
            "car le plafond de pénalité calibré (max_cap) reflète des pentes MOYENNES de segments de plusieurs km "
            "dans les données de calibration — pas les pointes instantanées les plus raides du parcours.")
        apply_vam=st.checkbox("Activer le modèle de marche en forte pente",value=True,key="apply_vam")
        vam_threshold_pct=25.0;vam_rate_m_per_h=900.0;vam_blend_width_pct=10.0
        if apply_vam:
            vc1,vc2=st.columns(2)
            vam_threshold_pct=vc1.slider("Pente à partir de laquelle la marche domine (%)",15.0,40.0,25.0,1.0,key="vam_threshold")
            vam_rate_m_per_h=vc2.number_input("VAM soutenable (m de D+ / heure)",min_value=300.0,max_value=2000.0,value=900.0,step=50.0,key="vam_rate",
                                               help="Ordre de grandeur : ~600-800 m/h coureur loisir, ~900-1100 m/h bon traileur entraîné, >1300 m/h niveau élite/skyrunning — dépend aussi de la technicité du terrain.")
            if EXPERT:
                vam_blend_width_pct=st.slider("Largeur de transition (%)",2.0,20.0,10.0,1.0,key="vam_blend",
                                               help="Évite une cassure brutale entre le modèle de pente classique et le modèle de marche.")

    with st.expander("💨 Vent"):
        apply_wind=st.checkbox("Appliquer l'effet du vent",value=True)
        wind_mode="Lissé"
        if IS_TRAIL:
            drag_coeff=0.018;tail_credit=0.40;wind_cap_head=0.12;wind_cap_tail=-0.06;wind_smooth_km=5
        else:
            drag_coeff=0.012;tail_credit=0.35;wind_cap_head=0.06;wind_cap_tail=-0.03;wind_smooth_km=9
        wind_power=1.0;wind_gate_g1=2.0;wind_gate_g2=8.0;wind_gate_min=0.25
        if apply_wind and EXPERT:
            wind_mode=st.selectbox("Mode calcul vent",["Lissé","Global"],key="wmode").split()[0]
            wind_smooth_km=st.slider("Lissage vent (km)",1,15,wind_smooth_km,2)
            c1,c2=st.columns(2)
            drag_coeff=c1.number_input("Coeff. aérodynamique",value=drag_coeff,step=0.002,format="%.3f")
            tail_credit=c2.slider("Crédit vent arrière",0.0,0.8,tail_credit,0.05)
            wind_cap_head=st.slider("Pénalité max vent face (%)",0.00,0.25,wind_cap_head,0.01)
            wind_cap_tail=st.slider("Gain max vent dos (%)",-0.12,0.00,wind_cap_tail,0.01)

    with st.expander("🔋 Fatigue en course — modèle à seuil(s) de dégradation"):
        apply_fatigue=st.checkbox("Activer la fatigue",value=True)
        fatigue_threshold=60.0;fatigue_rate=0.0;fatigue_mode="mixte"
        dual_fatigue=False; fatigue_threshold2=None; fatigue_rate2=None
        if apply_fatigue:
            st.markdown("##### Phase 1 — parcours (issue du profil de terrain / JSON de cohorte)")
            # v8.1 — seuil/taux par défaut dépendants du profil terrain (calibrés sur
            # 1245 segments coureurs réels) : dégradation continue dès le début pour
            # les profils roulant/montagneux, plateau-puis-chute à 60% conservé pour
            # le profil technique moyen (seul cas confirmé par la donnée réelle).
            fatigue_threshold=st.slider(
                "Seuil de dégradation (% de la course)",5,90,
                int(st.session_state.get("tp_fatigue_threshold",_d.get("fatigue_threshold",60))),5,
                key="tp_fatigue_threshold",
                help="% à partir duquel la dégradation s'accélère.")
            fatigue_rate=st.slider(
                "Ralentissement total en fin de course (%)",0.0,35.0,
                float(st.session_state.get("tp_fatigue_rate",_d.get("fatigue_rate",18.0))),0.5,
                key="tp_fatigue_rate",
                help="% de ralentissement cumulé à 100% du parcours.")
            fatigue_mode=st.selectbox("Type de fatigue",["mixte (recommandé)","distance (plat)","d_plus (montagne)"]).split()[0]
            if fatigue_rate>0:
                _tot_km_hint = f"~{fatigue_threshold/100*tot_tmp/1000:.1f} km" if (gpx_file and points) else "distance GPX non chargée"
                st.caption(f"📊 Phase 1 : stable jusqu'à **{fatigue_threshold}%** de la course ({_tot_km_hint}), puis "
                           f"dégradation exponentielle jusqu'à **+{fatigue_rate:.0f}%**.")

            # v8.5 — Phase 2 : fatigue personnelle, EN PLUS de la phase 1 (pas à sa place).
            # Idée : la phase 1 (souvent un seuil bas, ~5-10%) reflète la dynamique propre
            # à CE parcours calibrée sur des coureurs réels (JSON de cohorte) — stabilisation
            # post-départ + effort cumulé du terrain. La phase 2 vient se superposer plus
            # loin dans la course pour représenter TA fatigue profonde personnelle, détectée
            # sur tes propres références (section 2). Une référence courte/route (marathon)
            # n'a pas le même mécanisme physiologique qu'un ultra, mais reste informative sur
            # ta façon de gérer un effort soutenu — à utiliser en connaissance de cause.
            st.markdown("##### Phase 2 — ta fatigue personnelle (s'ajoute à la phase 1)")
            _auto_fat_thr = st.session_state.get("auto_fatigue_threshold")
            _auto_fat_rate = st.session_state.get("auto_fatigue_rate")
            _auto_fat_n = st.session_state.get("auto_fatigue_n", 0)
            _auto_fat_total = st.session_state.get("auto_fatigue_total", 0)
            _auto_fat_std = st.session_state.get("auto_fatigue_std")
            _auto_fat_r2max = st.session_state.get("auto_fatigue_r2max")
            if _auto_fat_thr is not None:
                _consistency = f" · cohérence ±{_auto_fat_std:.0f} pts" if _auto_fat_std is not None else ""
                _r2_txt = f" · R²={_auto_fat_r2max:.2f}" if _auto_fat_r2max is not None else ""
                st.caption(f"{_auto_fat_n}/{_auto_fat_total} référence(s) exploitable(s) (R²≥0.3) parmi celles "
                           f"importées en section 2 — moyenne utilisée si plusieurs.")
                dual_fatigue = st.checkbox(
                    f"🔒 Ajouter ma fatigue personnelle détectée (rupture à {_auto_fat_thr}% · dégradation "
                    f"{_auto_fat_rate}%{_consistency}{_r2_txt})",
                    value=st.session_state.get("dual_fatigue_locked", True), key="dual_fatigue_chk")
                st.session_state["dual_fatigue_locked"] = dual_fatigue
                if dual_fatigue:
                    c_p2a,c_p2b = st.columns(2)
                    fatigue_threshold2 = c_p2a.slider("Seuil phase 2 (% de la course)",
                        min_value=int(fatigue_threshold)+2, max_value=98,
                        value=max(int(fatigue_threshold)+2, int(st.session_state.get("tp_fatigue_threshold2",_auto_fat_thr))),
                        step=1, key="tp_fatigue_threshold2",
                        help="Doit être après le seuil de phase 1. C'est ici que ta fatigue personnelle prend le relais.")
                    fatigue_rate2 = c_p2b.slider("Ralentissement additionnel phase 2 (%)",0.0,40.0,
                        float(st.session_state.get("tp_fatigue_rate2",_auto_fat_rate)),0.5,
                        key="tp_fatigue_rate2",
                        help="S'ajoute par-dessus le ralentissement de la phase 1 (pas à sa place).")
                    if _auto_fat_r2max is not None and _auto_fat_r2max < 0.3:
                        st.warning(f"⚠️ R²={_auto_fat_r2max:.2f} — signal très faible, quasiment du bruit. "
                                   f"Résultat peu fiable si tu gardes cette case cochée.")
                    elif _auto_fat_r2max is not None and _auto_fat_r2max < 0.6:
                        st.caption(f"ℹ️ R²={_auto_fat_r2max:.2f} — signal modéré (typique d'un point de rupture "
                                   f"réel noyé dans du bruit GPS/allure). Traite le résultat comme indicatif, "
                                   f"pas comme une certitude.")
                    st.caption(f"📊 Phase 2 : dégradation stable à +{fatigue_rate:.0f}% (fin de phase 1) jusqu'à "
                               f"**{fatigue_threshold2}%**, puis accélère jusqu'à **+{fatigue_rate+fatigue_rate2:.0f}%** "
                               f"cumulé à l'arrivée.")
            else:
                st.caption("Aucune rupture d'allure exploitable détectée sur tes références (section 2) pour "
                           "l'instant — importe une référence en FIT/TCX pour activer cette option.")

    with st.expander("🎚️ Sensibilité aux conditions selon l'allure",expanded=False):
        st.caption("Plus tu cours vite, moins la météo et le vent te ralentissent en proportion.")
        pace_sens_ref_min=st.slider("Allure de référence (min/km)",min_value=3.0,max_value=10.0,value=6.0,step=0.5,key="pace_sens_ref")
        pace_sensitivity_ref=pace_sens_ref_min*60.0

    with st.expander("⚡ Stratégie de pacing Ultra"):
        apply_ultra=st.checkbox("Activer le pacing ultra",value=False)
        ultra_amp=0.0
        if apply_ultra:ultra_amp=st.slider("Amplitude (%)",0.0,40.0,10.0,0.5)

    show_smooth_pace=True;smooth_window_km=3
    with st.expander("📉 Options d'affichage"):
        show_smooth_pace=st.checkbox("Afficher l'allure lissée",value=True)
        smooth_window_km=st.slider("Fenêtre lissage (km)",1,9,3,2) if show_smooth_pace else 3

    st.markdown("---")
    st.header("05 · Paramètres de la course cible")
    c1,c2=st.columns(2)
    date_course=c1.date_input("📅 Date de course",value=date.today())
    heure_course=c2.time_input("⏰ Heure de départ",value=time(9,0),step=60)  # step=60s → saisie HH:MM:SS possible
    colf1,colf2=st.columns(2)
    with colf1:
        force_dist=st.checkbox("Forcer la distance",value=False)
        dist_forcee=st.number_input("Distance (km)",value=41.0,format="%.3f") if force_dist else None
    with colf2:
        # v8.2 — défaut passé à False : cette case FORCE le temps total sur la
        # valeur saisie (la prédiction ne fait plus que redistribuer ce total
        # entre les km selon la forme du modèle de pente/fatigue) ; laissée
        # cochée par défaut, une valeur périmée d'un test précédent écrasait
        # silencieusement de vraies prédictions (ex : objectif à 10h15 réutilisé
        # tel quel sur un parcours type UTMB → allure moyenne ~3:30/km affichée
        # sans aucun message d'alerte).
        force_temps=st.checkbox("Travailler à partir d'un objectif de temps",value=False,
                                 help="⚠️ Force le temps total sur la valeur saisie ci-dessous — n'est plus une "
                                      "prédiction libre à partir de tes références. Pense à décocher (ou à mettre "
                                      "à jour la valeur) en changeant de course, sous peine d'obtenir un résultat "
                                      "qui n'a plus de rapport avec le parcours chargé.")
        temps_objectif=hms_input("Temps objectif","3:45:00",key="temps_objectif_target") if force_temps else None
        if force_temps:
            st.warning(f"⚠️ Le temps total prédit sera **forcé sur {temps_objectif}** — ce n'est pas un calcul "
                       f"libre à partir de tes références. Décoche la case ci-dessus pour une vraie prédiction.")

    _diff_days_race=(date_course-date.today()).days
    if 0<=_diff_days_race<=15:
        st.info(f"🌡️ **Météo automatique** — prévisions Open-Meteo km par km (J+{_diff_days_race}).")
    elif _diff_days_race<0:
        st.info(f"🌡️ **Météo automatique** — archives Open-Meteo ({date_course}).")
    else:
        st.caption(f"🌡️ Météo : date à J+{_diff_days_race} (hors plage API).")

    meteo_fb={"temp":12.0,"amp":0.0,"wind":0.0,"humidity":60.0,"wind_dir":180.0}

    st.markdown("---")
    with st.expander("🔬 Cross-validation (fiabilité du modèle)"):
        st.info("LOO : prédit chaque référence avec les autres. MAPE < 3% = excellent | < 7% = correct.")
        if st.button("Lancer la cross-validation"):
            refs_cv=prepare_refs(refs_raw,use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,
                                  temp_max_penalty,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                                  elev_ref_power,temp_ref_power)
            cv=crossval_loo(refs_cv)
            if cv is None:st.warning("Au moins 3 références nécessaires.")
            else:
                df_cv,mae,mape=cv;st.dataframe(df_cv,use_container_width=True)
                c1,c2=st.columns(2)
                c1.metric("Erreur absolue moyenne",f"{seconds_to_hms(mae)} ({mae:.0f}s)")
                c2.metric("MAPE",f"{mape:.2f} %")

    st.header("06 · Calcul & Résultats")
    if st.button("▶️ Calculer la prédiction",type="primary"):
        if not gpx_file or points is None:st.error("⚠️ Importe un fichier GPX (section 1).")
        elif not any(safe_float(r.get("distance",0))>0 and hms_to_seconds(r.get("temps","0"))>0 for r in refs_raw):
            st.error("⚠️ Renseigne au moins une référence valide.")
        else:
            st.session_state["_meteo_api_cache"]={}
            # v9.1 — un objectif de temps s'entend chrono TOTAL : on en retire les arrêts
            # prévus pour ne piloter que la partie « en mouvement » du plan.
            _stops_planned_s=sum(max(0.0,float(c.get("arret_s",0) or 0))
                                 for c in st.session_state.get("checkpoints",[]))
            _objectif_mouvement=temps_objectif
            if force_temps and temps_objectif and _stops_planned_s>0:
                _objectif_mouvement=seconds_to_hms(max(60.0,hms_to_seconds(temps_objectif)-_stops_planned_s))
                st.caption(f"Objectif {temps_objectif} − {seconds_to_hms(_stops_planned_s)} d'arrêts prévus "
                           f"= {_objectif_mouvement} de course effective.")
            with st.spinner("Calcul en cours..."):
                try:
                    res=run_prediction(
                        distance_cible_km=dist_forcee if force_dist else None,
                        refs_input=refs_raw,points=points,date_course=date_course,heure_course=heure_course,
                        use_recalibrated=use_recalibrated,opt_temp=opt_temp,use_wbgt=use_wbgt,
                        cold_quad=cold_quad,hot_quad=hot_quad,temp_max_penalty=temp_max_penalty,
                        temp_power=temp_power,elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power,
                        apply_grade=apply_grade,use_minetti=use_minetti,minetti_weight=minetti_weight,
                        k_up=k_up,k_down=k_down,down_cap=down_cap,g0_up=g0_up,g0_down=g0_down,
                        max_up=max_up,max_down=max_down,elev_smooth_window=elev_smooth_window,
                        grade_power=grade_power,apply_altitude=apply_altitude,altitude_ref_m=altitude_ref_m,
                        apply_wind=apply_wind,wind_mode=wind_mode,wind_smooth_km=wind_smooth_km,
                        drag_coeff=drag_coeff,tail_credit=tail_credit,wind_cap_head=wind_cap_head,
                        wind_cap_tail=wind_cap_tail,wind_power=wind_power,
                        wind_gate_g1=wind_gate_g1,wind_gate_g2=wind_gate_g2,wind_gate_min=wind_gate_min,
                        base_cap=base_cap,extra_per_pct=extra_per_pct,max_cap=max_cap,
                        apply_fatigue=apply_fatigue,fatigue_rate=fatigue_rate,
                        fatigue_threshold=float(st.session_state.get("tp_fatigue_threshold",60.0)),
                        fatigue_mode=fatigue_mode,
                        dual_fatigue=dual_fatigue,fatigue_threshold2=fatigue_threshold2,fatigue_rate2=fatigue_rate2,
                        apply_ultra=apply_ultra,ultra_amp=ultra_amp,
                        objective_hms=_objectif_mouvement if force_temps else None,
                        show_smooth_pace=show_smooth_pace,smooth_window_km=smooth_window_km,
                        dem_elevations=dem_elevations,surface_mult=surface_mult,
                        meteo_fallback_temp=meteo_fb["temp"],meteo_fallback_amp=meteo_fb["amp"],
                        meteo_fallback_wind=meteo_fb["wind"],meteo_fallback_humidity=meteo_fb["humidity"],
                        meteo_fallback_wind_dir=meteo_fb["wind_dir"],
                        pace_sensitivity_ref=pace_sensitivity_ref,
                        apply_vam=apply_vam,vam_threshold_pct=vam_threshold_pct,
                        vam_rate_m_per_h=vam_rate_m_per_h,vam_blend_width_pct=vam_blend_width_pct)
                    st.session_state["res"]=res
                    st.session_state["refs_fit_vc"]=res.get("refs_fit",[])
                    st.session_state["K_riegel_vc"]=res.get("K",1.06)
                    # v8.9 — paramètres du calcul, conservés pour l'enregistrement du plan
                    st.session_state["_pred_params"]={
                        "terrain_profil":terrain_profil,"surface_sel":surface_sel,"surface_mult":surface_mult,
                        "tp_k_up":k_up,"tp_k_down":k_down,"tp_down_cap":down_cap,"tp_minetti_weight":minetti_weight,
                        "tp_elev_smooth_window":elev_smooth_window,"tp_grade_power":grade_power,
                        "tp_base_cap":base_cap,"tp_extra_per_pct":extra_per_pct,"tp_max_cap":max_cap,
                        "apply_grade":apply_grade,"use_minetti":use_minetti,
                        "apply_vam":apply_vam,"vam_threshold_pct":vam_threshold_pct,"vam_rate_m_per_h":vam_rate_m_per_h,
                        "apply_fatigue":apply_fatigue,"tp_fatigue_threshold":fatigue_threshold,
                        "tp_fatigue_rate":fatigue_rate,"fatigue_mode":fatigue_mode,"dual_fatigue":dual_fatigue,
                        "tp_fatigue_threshold2":fatigue_threshold2,"tp_fatigue_rate2":fatigue_rate2,
                        "opt_temp":opt_temp,"use_wbgt":use_wbgt,"use_recalibrated":use_recalibrated,
                        "apply_altitude":apply_altitude,"altitude_ref_m":altitude_ref_m,
                        "apply_wind":apply_wind,"apply_ultra":apply_ultra,"ultra_amp":ultra_amp,
                        "date_course":str(date_course),"heure_course":str(heure_course),
                        "force_dist":bool(force_dist),"dist_forcee":dist_forcee,
                        "force_temps":bool(force_temps),"temps_objectif":temps_objectif if force_temps else None,
                        "mode_activite":mode_activite,
                        "references":[{"distance":safe_float(r.get("distance")),
                                       "temps":r.get("duration_hms_file") or r.get("temps")} for r in refs_raw],
                        "K":res.get("K"),"K_raw":res.get("K_raw"),
                    }
                except Exception as e:
                    import traceback;st.error(f"Erreur:{e}");st.code(traceback.format_exc())

    if "res" in st.session_state:
        res=st.session_state["res"]
        st.markdown("---");st.subheader("🎯 Prédiction")
        # ── v8 PATCH 3 : allure moy. basée sur la distance simulée (forcée ou GPX) ──
        _dist_simulated_km = dist_forcee if (force_dist and dist_forcee and dist_forcee>0) else res["dist_gpx_km"]
        avg_pace_s=res["total_s"]/max(_dist_simulated_km,1e-6)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("⏱ Temps prédit",res["total_human"])
        c2.metric("📊 Allure moy.",pace_str(avg_pace_s)+"/km")
        _std_dists={"10 km":10000,"Semi":21097,"Marathon":42195,"50 km":50000,"100 km":100000}
        # ── v8 PATCH 3b : _gpx_m utilise _dist_simulated_km ──
        _gpx_m=_dist_simulated_km*1000.0; _avg_pace_s=res["total_s"]/_gpx_m if _gpx_m>0 else 0
        ci_col1,ci_col2,ci_col3=st.columns([2,1,1])
        with ci_col1:
            _dist_ref_opts={"Distance simulée":_gpx_m}; _dist_ref_opts.update({k:v for k,v in _std_dists.items()})
            _ref_sel=st.selectbox("📏 Distance de référence",list(_dist_ref_opts.keys()),key="ci_dist_ref",index=0)
        with ci_col2: _delta_low=st.slider("Borne basse (%)",-10,0,-2,1,key="ci_delta_low")
        with ci_col3: _delta_high=st.slider("Borne haute (%)",0,10,2,1,key="ci_delta_high")
        _ref_m=_dist_ref_opts[_ref_sel]; _t_ref=_avg_pace_s*_ref_m
        _t_low=_t_ref*(1+_delta_low/100.0); _t_high=_t_ref*(1+_delta_high/100.0)
        c3.metric(f"📏 {_ref_sel} {_delta_low:+d}%",seconds_to_hms(_t_low))
        c4.metric(f"📏 {_ref_sel} {_delta_high:+d}%",seconds_to_hms(_t_high))
        c5.metric("K Riegel",f"{res['K']:.3f}",
                  delta=(f"brut={res['K_raw']:.2f}" if res.get("K_raw") is not None and abs(res["K_raw"]-res["K"])>0.01 else None))

        # v8.3 — signale une extrapolation fragile plutôt que de la laisser silencieuse.
        # Plafonds relevés à [0.85, 1.60] (v8.3) — l'ancien plafond à 1.25 (calibré pour
        # des écarts de distance route) tronquait silencieusement un vrai signal de
        # fatigue ultra quand K brut dépassait 1.25, ce qui pouvait sous-estimer le
        # temps total de plusieurs heures sur une extrapolation vers un 100 miles.
        _K_val = res['K']; _K_raw_val = res.get('K_raw')
        if _K_val >= 1.58 or _K_val <= 0.86:
            _raw_txt = f" (K brut avant plafonnement : {_K_raw_val:.2f})" if _K_raw_val is not None else ""
            st.warning(f"⚠️ K Riegel collé à une borne ({_K_val:.2f}){_raw_txt} — signe que tes références sont trop "
                       f"hétérogènes (distance/terrain/conditions) ou trop peu nombreuses pour une extrapolation "
                       f"fiable vers cette distance. Ajoute une référence de distance intermédiaire si possible.")
        elif _K_raw_val is not None and abs(_K_raw_val - _K_val) > 0.05:
            st.info(f"ℹ️ Le K brut de tes références ({_K_raw_val:.2f}) a été légèrement ajusté à {_K_val:.2f} "
                    f"pour rester dans une plage physiologiquement plausible.")
        _ref_dists_km = [safe_float(r.get("distance",0))/1000.0 for r in refs_raw if safe_float(r.get("distance",0))>0]
        if _ref_dists_km and _dist_simulated_km > max(_ref_dists_km)*1.5:
            st.warning(f"⚠️ La distance cible ({_dist_simulated_km:.0f} km) dépasse largement ta référence la "
                       f"plus longue ({max(_ref_dists_km):.0f} km) — l'extrapolation du modèle Riegel devient "
                       f"moins fiable au-delà de ~1,5× la plus longue référence.")

        df_out=res["df"]
        if not df_out.empty:
            res_t1,res_t2,res_t3=st.tabs(["📈 Allure par km","🔎 Facteurs","📋 Tableau détaillé"])
            with res_t1:
                fig,ax=plt.subplots(figsize=(12,4));pv=[]
                for v in df_out["Allure (min/km)"].values:
                    try:parts=str(v).split(":");pv.append(int(parts[0])+int(parts[1])/60.0)
                    except:pv.append(float("nan"))
                x=list(range(1,len(pv)+1))
                ax.plot(x,pv,lw=1.5,alpha=0.35,color=C_WHITE,label="Allure brute")
                if "Allure lissée (min/km)" in df_out.columns:
                    ps=[]
                    for v in df_out["Allure lissée (min/km)"].values:
                        try:parts=str(v).split(":");ps.append(int(parts[0])+int(parts[1])/60.0)
                        except:ps.append(float("nan"))
                    ax.plot(x,ps,lw=2.5,color=C_RED,label="Allure lissée")
                if apply_fatigue and fatigue_rate>0 and len(x)>0:
                    thresh_km=fatigue_threshold/100.0*len(x)
                    ax.axvline(thresh_km,color=C_RED,lw=1.5,ls="--",alpha=0.7,label=f"Seuil phase 1 ({fatigue_threshold}%)")
                    if dual_fatigue and fatigue_threshold2:
                        thresh2_km=fatigue_threshold2/100.0*len(x)
                        ax.axvline(thresh2_km,color=C_RED_SOFT,lw=1.5,ls="--",alpha=0.7,label=f"Seuil phase 2 perso ({fatigue_threshold2}%)")
                ax.invert_yaxis();ax.set_xlabel("Kilomètre");ax.set_ylabel("Allure (min/km)")
                ax.set_title("Allure prévisionnelle km par km");ax.legend();ax.grid(alpha=0.3)
                st.pyplot(fig);plt.close(fig)
            with res_t2:
                fig2,ax2=plt.subplots(figsize=(12,4));x=list(range(1,len(df_out)+1))
                ax2.plot(x,df_out["Mult Pente"].values,label="Pente",lw=2)
                if "Mult Temp" in df_out.columns:ax2.plot(x,df_out["Mult Temp"].values,label="Température",lw=2)
                if "Mult Vent" in df_out.columns:ax2.plot(x,df_out["Mult Vent"].values,label="Vent",lw=2)
                if "Mult Fatigue" in df_out.columns:ax2.plot(x,df_out["Mult Fatigue"].values,label="Fatigue",lw=2,ls=":")
                if apply_fatigue and fatigue_rate>0:
                    ax2.axvline(fatigue_threshold/100.0*len(x),color=C_RED,lw=1.5,ls="--",alpha=0.7,label=f"Seuil {fatigue_threshold}%")
                ax2.axhline(1.0,color=C_DIM,lw=0.8);ax2.set_xlabel("Kilomètre")
                ax2.set_ylabel("Multiplicateur");ax2.set_title("Décomposition des facteurs")
                ax2.legend();ax2.grid(alpha=0.3);st.pyplot(fig2);plt.close(fig2)
            with res_t3:st.dataframe(df_out,use_container_width=True)

        # ── v9.1 : ravitaillements — arrêts prévus et feuille de route ───
        st.markdown("---")
        st.subheader("🥤 Ravitaillements — temps de passage et arrêts")
        _cps_plan = sorted(st.session_state.get("checkpoints", []), key=lambda c: float(c.get("dist_km", 0)))
        for _cp in _cps_plan:      # identifiant stable : la clé du widget d'arrêt ne bouge plus
            if not _cp.get("cp_id"):
                _cp["cp_id"] = f"{float(_cp.get('dist_km', 0)):.3f}_{str(_cp.get('label', ''))[:14]}"
        if not _cps_plan:
            st.info("Aucun ravitaillement défini. Ajoute-les plus bas dans « 📍 Checkpoints & Ravitaillements » : "
                    "ils apparaîtront ici avec leur heure de passage, et tu pourras y placer des temps d'arrêt.")
        else:
            with st.expander("⏱️ Temps d'arrêt prévus", expanded=True):
                st.caption("Le temps d'arrêt s'ajoute au chrono sans changer l'allure de course : la feuille de "
                           "route ci-dessous décale toutes les heures de passage suivantes.")
                _bs1, _bs2, _bs3 = st.columns([1, 1, 1.4])
                _budget_min = _bs1.number_input("Budget total d'arrêt (min)", 0.0, 240.0, 10.0, 0.5,
                                                key="stop_budget_min")
                _max_stop_min = _bs2.number_input("Maximum par ravito (min)", 0.5, 30.0, 3.5, 0.5,
                                                  key="stop_max_min")
                with _bs3:
                    st.caption(" ")
                    if st.button("↻ Répartir le budget sur les ravitaillements", key="btn_distribute_stops"):
                        _n_srv, _placed, _left = distribute_stop_budget(
                            st.session_state["checkpoints"], _budget_min * 60.0, _max_stop_min * 60.0)
                        # on écrit dans les widgets eux-mêmes : sinon Streamlit réafficherait
                        # l'ancienne valeur du champ tout en utilisant la nouvelle en interne
                        for _cp in st.session_state["checkpoints"]:
                            if _cp.get("cp_id"):
                                st.session_state[f"arret_cp_{_cp['cp_id']}"] = float(_cp.get("arret_s", 0) or 0) / 60.0
                        st.session_state["_stop_flash"] = (
                            (f"⚠️ {_n_srv} ravitaillement(s) × {_max_stop_min:.1f} min = {seconds_to_hms(_placed)} "
                             f"placés ; {seconds_to_hms(_left)} non plaçables avec ce maximum.")
                            if _left > 1 else
                            (f"✅ {seconds_to_hms(_placed)} répartis sur {_n_srv} ravitaillement(s) "
                             f"({_placed/max(1,_n_srv)/60:.1f} min chacun)."))
                        st.rerun()
                if st.session_state.get("_stop_flash"):
                    st.caption(st.session_state.pop("_stop_flash"))
                _cols_stop = st.columns(min(4, len(_cps_plan)))
                for _i_cp, _cp in enumerate(_cps_plan):
                    with _cols_stop[_i_cp % len(_cols_stop)]:
                        _k_cp = f"arret_cp_{_cp['cp_id']}"
                        if _k_cp not in st.session_state:
                            st.session_state[_k_cp] = float(_cp.get("arret_s", 0) or 0) / 60.0
                        st.number_input(f"{_cp['label']} — km {_cp['dist_km']:.1f}",
                                        min_value=0.0, max_value=90.0, step=0.5, key=_k_cp,
                                        help="Temps d'arrêt prévu, en minutes.")
                        _cp["arret_s"] = float(st.session_state[_k_cp]) * 60.0

            _sched, _stops_total = build_checkpoint_schedule(
                df_out, _cps_plan, datetime.combine(date_course, heure_course))
            _moving_s = float(res["total_s"])
            _total_s = _moving_s + _stops_total
            kpi_row([
                ("Temps de course (mouvement)", seconds_to_hms(_moving_s),
                 pace_str(_moving_s / max(_dist_simulated_km, 1e-6)) + "/km"),
                ("Arrêts prévus", seconds_to_hms(_stops_total),
                 f"{len([c for c in _cps_plan if (c.get('arret_s') or 0) > 0])} ravitaillement(s)"),
                ("Temps total (chrono)", seconds_to_hms(_total_s),
                 pace_str(_total_s / max(_dist_simulated_km, 1e-6)) + "/km sur la montre"),
                ("Arrivée prévue",
                 (datetime.combine(date_course, heure_course) + timedelta(seconds=_total_s)).strftime("%d/%m %H:%M"),
                 f"départ {heure_course.strftime('%H:%M')}"),
            ])
            if _sched:
                _rows_sched = []
                for _r in _sched:
                    _rows_sched.append({
                        "Ravitaillement": _r["label"], "Km": round(_r["dist_km"], 1),
                        "Alt (m)": _r.get("alt"),
                        "Arrivée (temps)": seconds_to_hms(_r["arrivee_s"]),
                        "Arrivée (heure)": _r.get("heure_arrivee") or "—",
                        "Arrêt": seconds_to_hms(_r["arret_s"]) if _r["arret_s"] > 0 else "—",
                        "Départ (temps)": seconds_to_hms(_r["depart_s"]),
                        "Départ (heure)": _r.get("heure_depart") or "—",
                        "Segment (km)": round(_r["segment_km"], 1),
                        "Temps segment": seconds_to_hms(_r["segment_s"]),
                        "Allure segment": (pace_str(_r["allure_segment_s_km"]) + "/km")
                                          if _r.get("allure_segment_s_km") else "—",
                        "Allure moy. cumulée": (pace_str(_r["allure_moy_depuis_depart_s_km"]) + "/km")
                                               if _r.get("allure_moy_depuis_depart_s_km") else "—",
                    })
                df_sched = pd.DataFrame(_rows_sched)
                st.dataframe(df_sched, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Feuille de route (CSV)", df_sched.to_csv(index=False).encode("utf-8"),
                                   file_name="feuille_de_route.csv", key="dl_sched")
                st.caption("« Allure segment » est l'allure à tenir entre deux ravitaillements, hors arrêt. "
                           "Les heures tiennent compte des arrêts placés en amont.")

        # ── v8.9 : enregistrement du plan dans l'historique ──────────────
        st.markdown("---")
        st.markdown("#### 💾 Enregistrer ce plan de course")
        if history_ready():
            _gpx_name_save = getattr(gpx_file, "name", "") if gpx_file is not None else \
                             st.session_state.get("_stored_gpx", {}).get("name", "")
            with st.form("save_race_form"):
                _sr1, _sr2 = st.columns(2)
                _race_name = _sr1.text_input("Nom de la course",
                                             value=(_gpx_name_save.rsplit(".", 1)[0] if _gpx_name_save else "Course"))
                _race_date = _sr2.date_input("Date de la course", value=date_course, key="save_race_date")
                _race_notes = st.text_area("Notes (stratégie, objectif, ravitaillements…)", key="save_race_notes")
                _keep_gpx = st.checkbox("Conserver le parcours GPX avec le plan (permet de le recharger sans "
                                        "réimporter le fichier)", value=True, key="save_race_gpx")
                if st.form_submit_button("💾 Enregistrer dans l'historique"):
                    _cols_keep = [c for c in ["Km", "Pente (%)", "D+ seg (m)", "D+ cum (m)", "Temps seg (s)",
                                              "Allure (min/km)", "Allure lissée (min/km)", "Temps cumulé"]
                                  if c in df_out.columns]
                    _params_save = dict(st.session_state.get("_pred_params", {}))
                    _params_save["_race_name"] = _race_name
                    _sched_save, _stops_save = build_checkpoint_schedule(
                        df_out, sorted(st.session_state.get("checkpoints", []),
                                       key=lambda c: float(c.get("dist_km", 0))),
                        datetime.combine(date_course, heure_course))
                    _moving_save = float(res.get("total_s") or 0)
                    _params_save["stops_total_s"] = _stops_save
                    _rid = save_race(current_athlete_id(), _race_date.isoformat(), _race_name, "plan",
                                     _gpx_name_save, _dist_simulated_km, res.get("d_plus_total"),
                                     _moving_save + _stops_save, None, _params_save,
                                     df_out[_cols_keep].to_dict("records"),
                                     _sched_save or st.session_state.get("checkpoints", []), _race_notes,
                                     st.session_state.get("_gpx_xml") if _keep_gpx else None,
                                     stops_s=_stops_save, moving_s=_moving_save)
                    st.success(f"✅ Plan « {_race_name} » enregistré pour {current_athlete_name()} "
                               f"(#{_rid}) — retrouve-le dans l'onglet 📚 Historique.")
        else:
            save_gate("ce plan de course")


    if gpx_file and points:
        # ── v8 PATCH 1d : cum_d_map filtré anti-bruit GPS (seuil adaptatif) ──
        import statistics as _stats_cm
        _steps_cm=[haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude) for i in range(1,len(points))]
        _med_cm=_stats_cm.median(_steps_cm) if _steps_cm else 5.0
        _max_step_cm=max(100.0,min(2000.0,_med_cm*10))
        cum_d_map=[0.0]
        for _ds in _steps_cm:
            cum_d_map.append(cum_d_map[-1]+(_ds if _ds<=_max_step_cm else 0.0))
        total_dist_km=cum_d_map[-1]/1000.0
        lats_m=[p.latitude for p in points];lons_m=[p.longitude for p in points]

        st.markdown("---")
        st.subheader("📍 Checkpoints & Ravitaillements")
        st.caption(f"Distance totale du parcours : **{total_dist_km:.2f} km**")
        if "checkpoints" not in st.session_state:st.session_state["checkpoints"]=[]
        col_cp1,col_cp2,col_cp3=st.columns([2,2,1])
        with col_cp1:cp_dist=st.number_input("Distance du checkpoint (km)",min_value=0.1,max_value=float(total_dist_km),value=min(5.0,float(total_dist_km)),step=0.1,key="cp_dist_input")
        with col_cp2:cp_type=st.selectbox("Type",["🥤 Ravitaillement","⏱ Point de passage","🏔 Sommet","🔻 Col","🏁 Intermédiaire","⚠️ Point clé"],key="cp_type_input")
        with col_cp3:cp_nom=st.text_input("Nom (optionnel)",value="",key="cp_nom_input",placeholder="ex: Ravito km7")
        col_btn1,col_btn2=st.columns(2)
        with col_btn1:
            if st.button("➕ Ajouter ce checkpoint"):
                cp_dist_m=cp_dist*1000.0
                cp_lat=float(np.interp(cp_dist_m,cum_d_map,lats_m));cp_lon=float(np.interp(cp_dist_m,cum_d_map,lons_m))
                y_gps_cp=[getattr(p,"elevation",0.0) or 0.0 for p in points]
                cp_alt=float(np.interp(cp_dist_m,cum_d_map,y_gps_cp))
                label=cp_nom.strip() if cp_nom.strip() else f"{cp_type} km {cp_dist:.1f}"
                st.session_state["checkpoints"].append({"dist_km":cp_dist,"type":cp_type,"label":label,"lat":cp_lat,"lon":cp_lon,"alt":round(cp_alt),"arret_s":0.0})
                st.success(f"✅ Checkpoint ajouté : {label}")
        with col_btn2:
            if st.button("🗑️ Effacer tous les checkpoints"):st.session_state["checkpoints"]=[]
        checkpoints=st.session_state["checkpoints"]
        if checkpoints:
            df_cp=pd.DataFrame([{"Type":c["type"],"Nom":c["label"],"Distance":f"{c['dist_km']:.1f} km","Altitude GPS":f"{c['alt']} m"} for c in sorted(checkpoints,key=lambda x:x["dist_km"])])
            st.dataframe(df_cp,use_container_width=True,hide_index=True)

        st.markdown("---")
        with st.expander("🗺️ Carte satellite & Profil d'altitude",expanded=False):
            map_style_opt=st.radio("Style de carte",["🛰 Satellite ESRI (gratuit)","🗺️ OpenStreetMap","🌄 Topo CartoDB"],horizontal=True,key="map_style_opt")
            n_pts=len(points);step=max(1,n_pts//800)
            coords_line=[[lats_m[i],lons_m[i]] for i in range(0,n_pts,step)]
            center_lat=float(np.mean(lats_m));center_lon=float(np.mean(lons_m))
            if "Satellite" in map_style_opt:tiles_url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";tiles_attr="Tiles &copy; Esri"
            elif "Topo" in map_style_opt:tiles_url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png";tiles_attr="CartoDB"
            else:tiles_url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";tiles_attr="OpenStreetMap"
            y_elev=[getattr(p,"elevation",0.0) or 0.0 for p in points]
            segments_html=[];seg_step=max(1,len(coords_line)//60)
            for si in range(0,len(coords_line)-1,seg_step):
                seg=coords_line[si:si+seg_step+1]
                if len(seg)<2:continue
                prog=si/max(1,len(coords_line));r_c=int(255-prog*40);g_c=int(60+prog*120)
                color=f"rgb({r_c},{g_c},40)";seg_json=_json.dumps(seg)
                segments_html.append(f"L.polyline({seg_json},{{color:'{color}',weight:5,opacity:0.9}}).addTo(map);")
            cp_colors_js={"🥤 Ravitaillement":"#00c864","⏱ Point de passage":"#6496ff","🏔 Sommet":"#ffc800","🔻 Col":"#c864ff","🏁 Intermédiaire":"#aaaaaa","⚠️ Point clé":"#ff5050"}
            cp_markers_html=[]
            for cp in checkpoints:
                col_cp_js=cp_colors_js.get(cp["type"],"#ffcc00");popup=f"{cp['label']} — {cp['dist_km']:.1f} km — {cp['alt']} m"
                cp_markers_html.append(f"L.circleMarker([{cp['lat']},{cp['lon']}],{{radius:10,color:'{col_cp_js}',fillColor:'{col_cp_js}',fillOpacity:0.9,weight:2}}).bindPopup('<b>{popup}</b>').addTo(map);L.marker([{cp['lat']},{cp['lon']}],{{icon:L.divIcon({{className:'',html:'<div style=\"background:{col_cp_js};color:white;padding:2px 5px;border-radius:4px;font-size:11px;font-weight:bold;white-space:nowrap\">{cp['label']}</div>',iconAnchor:[0,20]}})}} ).addTo(map);")
            html_map=f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><style>html,body,#map{{margin:0;padding:0;height:100%;width:100%}}</style></head><body><div id="map" style="height:500px;width:100%"></div><script>var map=L.map('map').setView([{center_lat},{center_lon}],13);L.tileLayer('{tiles_url}',{{maxZoom:19,attribution:'{tiles_attr}'}}).addTo(map);{chr(10).join(segments_html)}{chr(10).join(cp_markers_html)}L.circleMarker([{lats_m[0]},{lons_m[0]}],{{radius:12,color:'#00ff44',fillColor:'#00ff44',fillOpacity:1,weight:3}}).bindPopup('<b>🟢 Départ</b>').addTo(map);L.circleMarker([{lats_m[-1]},{lons_m[-1]}],{{radius:12,color:'#ff2222',fillColor:'#ff2222',fillOpacity:1,weight:3}}).bindPopup('<b>🔴 Arrivée</b>').addTo(map);</script></body></html>"""
            import streamlit.components.v1 as components
            components.html(html_map,height=510,scrolling=False)

            x_km=np.array(cum_d_map)/1000.0;y_gps=np.array(y_elev)
            w_e=int(elev_smooth_window);w_e+=(1 if w_e%2==0 else 0)
            fig3,ax3=plt.subplots(figsize=(11,3.5))
            if w_e>=3 and y_gps.size>=w_e:
                # ── v8 PATCH 2b : lissage altitude padding reflect ──
                pad_e=w_e//2
                y_padded=np.pad(y_gps,pad_e,mode='reflect')
                y_s=np.convolve(y_padded,np.ones(w_e)/w_e,mode='valid')
                if y_s.size>y_gps.size:y_s=y_s[:y_gps.size]
                elif y_s.size<y_gps.size:y_s=np.pad(y_s,(0,y_gps.size-y_s.size),mode='edge')
                ax3.fill_between(x_km,y_s.min()-5,y_s,alpha=0.15,color=C_WHITE)
                ax3.plot(x_km,y_s,lw=2.5,label="Altitude GPS lissée",color=C_WHITE)
                ax3.plot(x_km,y_gps,lw=0.8,alpha=0.2,color=C_DIM,label="GPS brut")
            else:
                ax3.fill_between(x_km,y_gps.min()-5,y_gps,alpha=0.15,color=C_WHITE)
                ax3.plot(x_km,y_gps,lw=2.5,label="Altitude GPS",color=C_WHITE)
            if dem_elevations is not None and len(dem_elevations)==len(points):
                y_dem=np.array([e if e is not None else 0.0 for e in dem_elevations])
                ax3.plot(x_km,y_dem,lw=2,ls="--",label="DEM corrigé",color=C_RED)
            cp_colors_profile={"🥤 Ravitaillement":C_WHITE,"⏱ Point de passage":C_GREY,"🏔 Sommet":C_RED,"🔻 Col":C_RED_SOFT,"🏁 Intermédiaire":C_TEXT_MUT,"⚠️ Point clé":C_RED}
            for cp in sorted(checkpoints,key=lambda x:x["dist_km"]):
                cp_x=cp["dist_km"];cp_y=float(np.interp(cp_x*1000,cum_d_map,y_gps))
                col_p=cp_colors_profile.get(cp["type"],"#ffcc00")
                ax3.axvline(cp_x,color=col_p,lw=1.5,ls="--",alpha=0.7)
                ax3.annotate(cp["label"],xy=(cp_x,cp_y),xytext=(0,12),textcoords="offset points",ha="center",fontsize=7.5,color=col_p,fontweight="bold",arrowprops=dict(arrowstyle="-",color=col_p,lw=1))
                ax3.scatter([cp_x],[cp_y],s=60,color=col_p,zorder=5)
            if IS_TRAIL:
                for seg in st.session_state.get("tech_segs",[]):
                    if seg.get("tech_score",0)>0.45:
                        color_tech=C_RED if seg["tech_score"]>0.70 else C_RED_SOFT
                        ax3.axvspan(seg["km_start"],seg["km_end"],alpha=0.12,color=color_tech,label="_nolegend_")
            ax3.scatter([x_km[0]],[y_gps[0]],s=80,color=C_WHITE,zorder=6,marker="^",label="Départ")
            ax3.scatter([x_km[-1]],[y_gps[-1]],s=80,color=C_RED,zorder=6,marker="s",label="Arrivée")
            ax3.set_xlabel("Distance (km)");ax3.set_ylabel("Altitude (m)")
            ax3.set_title(f"Profil d'altitude — {total_dist_km:.1f} km")
            ax3.legend(fontsize=8);ax3.grid(alpha=0.25);fig3.tight_layout()
            st.pyplot(fig3);plt.close(fig3)

            if checkpoints and "res" in st.session_state:
                res_cp=st.session_state["res"];df_out_cp=res_cp.get("df")
                if df_out_cp is not None and not df_out_cp.empty:
                    st.markdown("#### ⏱ Temps de passage aux checkpoints")
                    km_vals,t_cum_vals=[],[]
                    for _,row_cp in df_out_cp.iterrows():
                        km_str=str(row_cp["Km"])
                        try:km_num=float(km_str.split()[0])
                        except:continue
                        t_s_val=hms_to_seconds(str(row_cp["Temps cumulé"]))
                        km_vals.append(km_num);t_cum_vals.append(t_s_val)
                    if len(km_vals)>=2:
                        passage_rows=[]
                        for cp in sorted(checkpoints,key=lambda x:x["dist_km"]):
                            t_passage=float(np.interp(cp["dist_km"],km_vals,t_cum_vals))
                            passage_rows.append({"Checkpoint":cp["label"],"Distance":f"{cp['dist_km']:.1f} km","Altitude":f"{cp['alt']} m","Temps prévu":seconds_to_hms(t_passage),"Allure moy.":pace_str(t_passage/max(0.001,cp["dist_km"]))+"/km" if cp["dist_km"]>0 else "—"})
                        st.dataframe(pd.DataFrame(passage_rows),use_container_width=True,hide_index=True)

        st.markdown("---")
        st.subheader("🌍 Vue 3D relief réel — style Google Earth")
        col_l1,col_l2,col_l3,col_l4,col_l5=st.columns(5)
        with col_l1: layer_relief=st.checkbox("🏔 Relief 3D",value=True,key="layer_relief")
        with col_l2: layer_allure=st.checkbox("⏱ Allure prédite",value=False,key="layer_allure")
        with col_l3: layer_osm=st.checkbox("🌿 Surfaces OSM",value=False,key="layer_osm")
        with col_l4: layer_tech=st.checkbox("⚠️ Zones tech.",value=IS_TRAIL,key="layer_tech")
        with col_l5: layer_cp=st.checkbox("📍 Checkpoints",value=True,key="layer_cp")
        col_3dh,col_3dp=st.columns([3,1])
        with col_3dh: map_height_3d=st.slider("Hauteur (px)",400,850,600,25,key="map_height_3d")
        with col_3dp: map_pitch_3d=st.slider("Inclinaison (°)",20,75,52,5,key="map_pitch_3d")
        if st.button("🌍 Générer la vue 3D",type="primary",key="btn_3d_pydeck"):
            with st.spinner("Construction vue 3D relief ArcGIS World Elevation..."):
                html_3d=generate_3d_terrain_html(
                    points=points,cum_d_map=cum_d_map,checkpoints=checkpoints if layer_cp else [],
                    df_prediction=st.session_state["res"]["df"] if ("res" in st.session_state and layer_allure) else None,
                    tech_segments=st.session_state.get("tech_segs",[]) if (IS_TRAIL and layer_tech) else [],
                    dem_elevations=dem_elevations,osm_surface_data=st.session_state.get("osm_surface") if layer_osm else None,
                    height=map_height_3d,pitch=map_pitch_3d)
                st.session_state["html_3d_terrain"]=html_3d; st.session_state["html_3d_km"]=round(total_dist_km)
        if "html_3d_terrain" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state["html_3d_terrain"],height=map_height_3d,scrolling=False)
            col_3dl1,col_3dl2=st.columns([1,3])
            with col_3dl1:
                st.download_button("⬇️ Télécharger HTML 3D",data=st.session_state["html_3d_terrain"].encode("utf-8"),
                                   file_name=f"trail_3d_{st.session_state['html_3d_km']}km.html",mime="text/html")
            with col_3dl2: st.caption("💡 Couches interactives · Glisser : rotation · Scroll : zoom")

        st.markdown("---")
        st.subheader("🎬 Animation 3D cinématique — relief néon")
        col_3d1,col_3d2=st.columns(2)
        with col_3d1:anim3d_h=st.slider("Hauteur fenêtre (px)",400,800,600,50,key="anim3d_h")
        with col_3d2:anim3d_cp=st.checkbox("Afficher les checkpoints",value=True,key="anim3d_cp")
        if st.button("🎬 Générer la vue 3D cinématique",type="primary",key="btn_3d_anim"):
            with st.spinner("Construction du terrain 3D..."):
                html_3d=generate_3d_animation(points=points,cum_d_map=cum_d_map,checkpoints=checkpoints if anim3d_cp else [],total_dist_km=total_dist_km,dem_elevations=dem_elevations)
                st.session_state["html_3d"]=html_3d
        if "html_3d" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state["html_3d"],height=int(st.session_state.get("anim3d_h",600)),scrolling=False)
            col_dl1,col_dl2=st.columns([1,3])
            with col_dl1:
                st.download_button(label="⬇️ Télécharger le fichier HTML 3D",data=st.session_state["html_3d"].encode("utf-8"),file_name=f"trail_3d_{round(total_dist_km)}km.html",mime="text/html")
            with col_dl2:st.caption("💡 Fichier autonome — ouvrez dans Chrome/Firefox.")


# ══════════════════════════════════════════════════════════════
# ONGLET 1 — TESTS D'ENDURANCE + VITESSE CRITIQUE
# ══════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.title("🧪 Tests d'endurance + Vitesse Critique")
    st.caption("Évalue ta capacité aérobie, ta VC, ton D' et tes équivalences sur toutes distances.")

    st.header("01 · Performances de référence")
    st.info("Entre ici les chronos de tes courses ou tests (a minima 2 données pour la VC). Format hh:mm:ss.")

    if "n_refs_vc" not in st.session_state:
        st.session_state.n_refs_vc = 3
    cvc1, cvc2 = st.columns(2)
    with cvc1:
        if st.button("➕ Ajouter une référence", key="btn_add_vc") and st.session_state.n_refs_vc < 12:
            st.session_state.n_refs_vc += 1
    with cvc2:
        if st.button("➖ Retirer", key="btn_rm_vc") and st.session_state.n_refs_vc > 1:
            st.session_state.n_refs_vc -= 1

    DIST_PRESETS = {"— Saisie libre —": 0, "400 m": 400, "800 m": 800, "1000 m": 1000,
                    "1500 m": 1500, "1 mile": 1609, "3000 m": 3000, "5 km": 5000,
                    "10 km": 10000, "Semi-marathon": 21097, "Marathon": 42195}
    refs_vc = []
    for i in range(1, st.session_state.n_refs_vc + 1):
        with st.expander(f"📌 Référence {i}", expanded=(i <= 3)):

            use_file_vc = st.checkbox("📂 Importer depuis un fichier FIT/TCX",
                                      key=f"vc_use_file_{i}")
            if not use_file_vc:
                st.session_state.pop(f"vc_points_{i}", None)  # v8.2 — évite une analyse de rupture sur des données obsolètes

            dist_v = 5000.0
            secs_v = 0.0
            d_up_v = 0.0
            hr_ref_vc = None

            if use_file_vc:
                file_vc_i = st.file_uploader(
                    "Fichier FIT ou TCX",
                    type=["fit", "tcx"],
                    key=f"vc_file_{i}"
                )
                if file_vc_i:
                    fname_vc_i = file_vc_i.name.lower()
                    if fname_vc_i.endswith(".fit"):
                        parsed_vc_i = parse_fit_ref(file_vc_i)
                    elif fname_vc_i.endswith(".tcx"):
                        parsed_vc_i = parse_tcx_ref(file_vc_i)
                    else:
                        parsed_vc_i = None

                    if parsed_vc_i:
                        st.success(f"✅ {file_vc_i.name} — {parsed_vc_i['distance']:.0f} m · {parsed_vc_i['duration_hms']} · D+ {parsed_vc_i['D_up']:.0f} m")
                        col_vs, col_ve = st.columns(2)
                        with col_vs:
                            vc_sh = hms_input("Début segment", "0:00:00", key=f"vc_seg_start_{i}")
                        with col_ve:
                            vc_eh = hms_input("Fin segment",   "23:59:59", key=f"vc_seg_end_{i}")
                        start_td_i = hms_to_timedelta(vc_sh)
                        end_td_i   = hms_to_timedelta(vc_eh)
                        pts_i = parsed_vc_i.get("points", [])
                        if pts_i and (start_td_i.total_seconds() > 0 or end_td_i.total_seconds() < 86399):
                            seg_i = extract_segment(pts_i, start_td_i, end_td_i)
                            # ── fix : seuil adaptatif anti-bruit GPS (médiane × 10), même logique que v8 PATCH 1 ──
                            import statistics as _stats_vc
                            _steps_vc_pre = []
                            for _k in range(1, len(seg_i)):
                                _p1k, _p2k = seg_i[_k-1], seg_i[_k]
                                _la1k = _p1k["lat"] if isinstance(_p1k, dict) else _p1k.latitude
                                _lo1k = _p1k["lon"] if isinstance(_p1k, dict) else _p1k.longitude
                                _la2k = _p2k["lat"] if isinstance(_p2k, dict) else _p2k.latitude
                                _lo2k = _p2k["lon"] if isinstance(_p2k, dict) else _p2k.longitude
                                _steps_vc_pre.append(haversine_m(_la1k, _lo1k, _la2k, _lo2k))
                            _med_vc = _stats_vc.median(_steps_vc_pre) if _steps_vc_pre else 5.0
                            _max_step_vc = max(100.0, min(2000.0, _med_vc * 10))
                            seg_dist_i = 0.0; seg_elevs_i = []; seg_times_i = []
                            for j in range(1, len(seg_i)):
                                p1v, p2v = seg_i[j-1], seg_i[j]
                                la1 = p1v["lat"] if isinstance(p1v, dict) else p1v.latitude
                                lo1 = p1v["lon"] if isinstance(p1v, dict) else p1v.longitude
                                la2 = p2v["lat"] if isinstance(p2v, dict) else p2v.latitude
                                lo2 = p2v["lon"] if isinstance(p2v, dict) else p2v.longitude
                                e2  = p2v.get("elev", 0) if isinstance(p2v, dict) else p2v.elevation
                                t2  = p2v.get("time") if isinstance(p2v, dict) else p2v.time
                                _ds = haversine_m(la1, lo1, la2, lo2)
                                if _ds <= _max_step_vc:
                                    seg_dist_i += _ds
                                seg_elevs_i.append(e2)
                                if t2: seg_times_i.append(t2)
                            d_up_v, _ = compute_dplus_dminus(seg_elevs_i)
                            dur_i = seconds_to_hms((seg_times_i[-1]-seg_times_i[0]).total_seconds()) if len(seg_times_i)>=2 else parsed_vc_i["duration_hms"]
                            dist_v = float(round(seg_dist_i))
                        else:
                            dist_v = float(parsed_vc_i["distance"])
                            dur_i  = parsed_vc_i["duration_hms"]
                            d_up_v = float(parsed_vc_i["D_up"])
                        secs_v = float(hms_to_seconds(dur_i))
                        hr_ref_vc = parsed_vc_i.get("hr_analysis")
                        # v8.2 — conserve les points pour la détection de rupture (section 4)
                        st.session_state[f"vc_points_{i}"] = (
                            seg_i if pts_i and (start_td_i.total_seconds() > 0 or end_td_i.total_seconds() < 86399) else pts_i)
                        st.session_state[f"vc_label_{i}"] = f"Réf {i} ({dist_v/1000:.1f} km, {dur_i})"
                        if secs_v > 0 and dist_v > 0:
                            st.caption(f"📍 **{dist_v:.0f} m** · **{dur_i}** · **{pace_str(secs_v/(dist_v/1000))}/km** · D+ {d_up_v:.0f} m")
                        if hr_ref_vc and hr_ref_vc.get("hr_avg"):
                            st.caption(f"💓 FC moy. {hr_ref_vc['hr_avg']} bpm · FC max {hr_ref_vc['hr_max']} bpm · fiabilité {hr_ref_vc['reliability']}")
                    else:
                        st.error("❌ Impossible de lire ce fichier FIT/TCX.")
            else:
                c1, c2, c3 = st.columns([2, 2, 2])
                with c1:
                    preset = st.selectbox("Distance", list(DIST_PRESETS.keys()),
                                           key=f"vc_preset_{i}", index=0)
                    if DIST_PRESETS[preset] > 0:
                        dist_v = float(DIST_PRESETS[preset])
                    else:
                        dist_v = st.number_input("Distance (m)",
                                                  value=float(st.session_state.get(f"vc_dist_{i}", 5000)),
                                                  min_value=100.0, key=f"vc_dist_{i}")
                with c2:
                    t_v = hms_input("Temps", default="0:20:00", key=f"vc_temps_{i}")
                with c3:
                    d_up_v = st.number_input("D+ (m)", value=0.0, step=10.0, key=f"vc_dup_{i}")
                secs_v = float(hms_to_seconds(t_v))
                if dist_v > 0 and secs_v > 0:
                    st.caption(f"Allure : **{pace_str(secs_v/(dist_v/1000))}/km** · Vitesse : **{dist_v/secs_v*3.6:.2f} km/h**")

            refs_vc.append({
                "distance": float(dist_v),
                "temps":    float(secs_v),
                "D_up":     float(d_up_v),
                "hr_analysis": hr_ref_vc,
            })

    refs_vc_valid = [r for r in refs_vc if r["distance"] > 0 and r["temps"] > 0]

    genre_vc = st.selectbox("Genre (pour les standards)", ["H", "F"], key="genre_vc")

    st.markdown("---")
    st.header("02 · Modèle Riegel + Vitesse Critique")

    K_riegel = st.slider("Coefficient Riegel (K)", 0.85, 1.15, 1.06, 0.01,
                          help="K=1.06 est la valeur universelle. Plus K est élevé, plus la fatigue est pénalisante sur les longues distances.")

    if st.button("🔬 Calculer la VC et les équivalences", type="primary", key="btn_calc_vc"):
        if len(refs_vc_valid) < 2:
            st.warning("Au moins 2 références valides nécessaires.")
        else:
            a_r, K_r = fit_loglog(refs_vc_valid)
            st.session_state["vc_a"] = a_r
            st.session_state["vc_K"] = K_r
            distances = [r["distance"] for r in refs_vc_valid]
            durations = [r["temps"]    for r in refs_vc_valid]
            vc_ms, d_prime, r2_vc = compute_vc(distances, durations)
            st.session_state["vc_ms"]     = vc_ms
            st.session_state["d_prime"]   = d_prime
            st.session_state["r2_vc"]     = r2_vc
            st.session_state["refs_fit_vc"] = refs_vc_valid
            st.session_state["K_riegel_vc"] = K_r
            st.success("✅ Calcul terminé !")

    if "vc_ms" in st.session_state and st.session_state["vc_ms"] is not None:
        vc_ms   = st.session_state["vc_ms"]
        d_prime = st.session_state["d_prime"]
        r2_vc   = st.session_state["r2_vc"]
        a_r     = st.session_state.get("vc_a", 240.0)
        K_r     = st.session_state.get("vc_K", K_riegel)
        refs_fit_vc = st.session_state.get("refs_fit_vc", refs_vc_valid)

        st.subheader("📊 Résultats — Modèle linéaire D'")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VC (Vitesse Critique)", f"{vc_ms*3.6:.2f} km/h")
        c2.metric("Allure VC", pace_str(1000.0/vc_ms)+"/km")
        c3.metric("D' (réserve anaérobie)", f"{round(d_prime)} m" if d_prime else "—")
        c4.metric("R² (qualité modèle)", f"{r2_vc:.3f}" if r2_vc is not None else "—")

        if r2_vc is not None and r2_vc < 0.90:
            st.warning("⚠️ R² faible — vérifier la cohérence des références ou exclure des outliers.")

        # ── v8.9 : enregistrement du test dans l'historique ─────────────
        st.markdown("#### 💾 Enregistrer ce test")
        if history_ready():
            with st.form("save_vc_form"):
                _sv1, _sv2 = st.columns(2)
                _vc_date = _sv1.date_input("Date du test", value=date.today(), key="save_vc_date")
                _vc_label = _sv2.text_input("Libellé", value="Test VC", key="save_vc_label")
                _vc_notes = st.text_area("Notes (conditions, forme du jour, protocole…)", key="save_vc_notes")
                if st.form_submit_button("💾 Enregistrer dans l'historique"):
                    _refs_save = [{"distance": r.get("distance"), "temps": r.get("temps"),
                                   "D_up": r.get("D_up", 0)} for r in refs_fit_vc]
                    _tid = save_vc_test(current_athlete_id(), _vc_date.isoformat(), _vc_label,
                                        vc_ms, d_prime, r2_vc, K_r, a_r, _refs_save, _vc_notes)
                    st.success(f"✅ Test enregistré pour {current_athlete_name()} (#{_tid}) — "
                               f"l'évolution de la VC est dans l'onglet 📚 Historique.")
            st.caption("Chaque enregistrement crée une nouvelle ligne : un nouveau test n'efface jamais "
                       "les précédents.")
        else:
            save_gate("ce test de vitesse critique")
        st.markdown("---")

        st.subheader("🏆 Équivalences et prédictions sur distances standard")
        results_pred, best_dist = predict_performances(vc_ms, d_prime, refs_fit_vc, K_r, genre=genre_vc)

        for dist_label, info in results_pred.items():
            with st.expander(f"**{dist_label}** — Prédit : {info['t_pred_hms']} ({info['pace_pred']}/km)", expanded=(dist_label == best_dist)):
                cols_std = st.columns(min(4, len(info["standards"])))
                for si, std_row in enumerate(info["standards"]):
                    col_idx = si % len(cols_std)
                    cols_std[col_idx].metric(
                        f"{std_row['emoji']} {std_row['standard']}",
                        std_row["temps_std"],
                        delta=std_row["diff_str"],
                        delta_color="normal" if std_row["atteint"] else "inverse")

        st.subheader("⏱ Table de maintien (durée maximale par % de VC)")
        df_holding = build_holding_table(vc_ms, d_prime, refs_fit_vc, K_r)
        if not df_holding.empty:
            st.dataframe(df_holding, use_container_width=True, hide_index=True)

        st.subheader("📈 Courbe vitesse-endurance")
        fig_vc, ax_vc = plt.subplots(figsize=(11, 4))
        t_arr = np.linspace(60, 18000, 300)
        v_riegel = [a_r * (t/1000.0)**K_r / t * 1000.0 * 3.6 if a_r > 0 else 0 for t in t_arr]
        ax_vc.plot(t_arr/60, v_riegel, lw=2, color=C_WHITE, label=f"Riegel K={K_r:.3f}")
        if vc_ms > 0:
            v_vc_line = [vc_ms * 3.6 + (d_prime / t * 3.6 if d_prime and t > 0 else 0) for t in t_arr]
            ax_vc.plot(t_arr/60, v_vc_line, lw=2, ls="--", color=C_RED,
                       label=f"Modèle D' (VC={vc_ms*3.6:.2f} km/h, D'={round(d_prime)}m)")
            ax_vc.axhline(vc_ms * 3.6, color=C_RED, lw=1.5, ls=":", label="VC")
        for r in refs_fit_vc:
            if r["distance"] > 0 and r["temps"] > 0:
                v_ref = r["distance"] / r["temps"] * 3.6
                ax_vc.scatter([r["temps"]/60], [v_ref], s=80, zorder=5, color=C_RED)
                ax_vc.annotate(f"{r['distance']/1000:.1f}km", (r["temps"]/60, v_ref),
                               textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax_vc.set_xlabel("Durée (min)"); ax_vc.set_ylabel("Vitesse (km/h)")
        ax_vc.set_title("Courbe vitesse-endurance personnalisée")
        ax_vc.legend(fontsize=8); ax_vc.grid(alpha=0.3); fig_vc.tight_layout()
        st.pyplot(fig_vc); plt.close(fig_vc)

        st.markdown("---")
        st.subheader("🔬 Cross-validation LOO")
        if st.button("Lancer la cross-validation", key="btn_cv_vc"):
            cv_r = crossval_loo(refs_fit_vc)
            if cv_r is None:
                st.warning("Au moins 3 références nécessaires pour la LOO.")
            else:
                df_cv_r, mae_r, mape_r = cv_r
                st.dataframe(df_cv_r, use_container_width=True, hide_index=True)
                c1, c2 = st.columns(2)
                c1.metric("MAE", f"{seconds_to_hms(mae_r)} ({mae_r:.0f}s)")
                c2.metric("MAPE", f"{mape_r:.2f} %")

    st.markdown("---")
    st.header("03 · Test GAZOZ / Masque ventilatoire (analyse SV1/SV2)")
    st.caption("Charge ici le CSV exporté depuis la sonde ZoneX/Cosmed/VO₂Master.")

    gazoz_file = st.file_uploader("📂 CSV ZoneX (Ventilation, VO₂, VCO₂, FC, Cadence…)",
                                   type=["csv"], key="gazoz_file")
    if gazoz_file:
        df_gz = parse_zonex_csv(gazoz_file)
        if df_gz is None:
            st.error("❌ Format CSV non reconnu. Colonnes attendues : timestamp, VE (ou VE L/min), VCO2, Aux1 (eqO2), HR, Event (palier).")
        else:
            st.success(f"✅ {len(df_gz)} lignes chargées · {df_gz['palier'].max()} paliers détectés")
            df_pal = aggregate_by_palier(df_gz)
            _has_o2 = bool(df_gz.attrs.get("has_o2", "RQ" in df_gz.columns))
            if _has_o2 and "RQ" in df_pal.columns and "eqO2" in df_pal.columns:
                thresholds = detect_sv1_sv2(df_pal)
            else:
                thresholds = {"sv1": None, "sv2": None}
                st.markdown('<div class="note-box note-red">🫁 <b>Masque CO₂ seul détecté</b> — ce fichier ne '
                            "contient pas d'oxygène mesuré. La détection SV1/SV2 classique (qui compare VO₂ et "
                            "VCO₂) est donc inapplicable : les seuils ventilatoires sont calculés plus bas, "
                            "dans la section « Profil métabolique », à partir de VE/VCO₂ — une méthode qui ne "
                            "demande que du CO₂.</div>", unsafe_allow_html=True)
            sv1 = thresholds["sv1"]; sv2 = thresholds["sv2"]

            with st.expander("📊 Données brutes par palier"):
                st.dataframe(df_pal.round(2), use_container_width=True, hide_index=True)

            if not _has_o2:
                st.caption("Seuils ventilatoires : voir la section « Profil métabolique » ci-dessous — "
                           "avec un masque CO₂, ils se lisent sur VE/VCO₂.")
            _show_sv = bool(_has_o2)
            if _show_sv:
                st.subheader("🎯 Seuils ventilatoires détectés")
            col_sv1, col_sv2 = st.columns(2) if _show_sv else (st.container(), st.container())
            with col_sv1:
                if sv1 and _show_sv:
                    st.markdown('<div class="test-card">', unsafe_allow_html=True)
                    st.markdown(f"#### 🟢 SV1 — Seuil aérobie")
                    st.metric("FC", f"{sv1['HR']} bpm"); st.metric("Palier", str(sv1["palier"]))
                    st.metric("VO₂", f"{sv1['VO2']:.3f} L/min"); st.metric("RQ", f"{sv1['RQ']:.3f}")
                    st.metric("VE", f"{sv1['VE']:.1f} L/min")
                    if sv1.get("Cadence", 0) > 0: st.metric("Vitesse / Cadence", f"{sv1['Cadence']:.2f} km/h")
                    st.markdown("</div>", unsafe_allow_html=True)
                elif _show_sv: st.info("SV1 non détecté automatiquement.")
            with col_sv2:
                if sv2 and _show_sv:
                    st.markdown('<div class="test-card">', unsafe_allow_html=True)
                    st.markdown(f"#### 🔴 SV2 — Seuil anaérobie")
                    st.metric("FC", f"{sv2['HR']} bpm"); st.metric("Palier", str(sv2["palier"]))
                    st.metric("VO₂", f"{sv2['VO2']:.3f} L/min"); st.metric("RQ", f"{sv2['RQ']:.3f}")
                    st.metric("VE", f"{sv2['VE']:.1f} L/min")
                    if sv2.get("Cadence", 0) > 0: st.metric("Vitesse / Cadence", f"{sv2['Cadence']:.2f} km/h")
                    st.markdown("</div>", unsafe_allow_html=True)
                elif _show_sv: st.info("SV2 non détecté automatiquement.")

            if sv1 and sv2:
                fc_sv1 = sv1["HR"]; fc_sv2 = sv2["HR"]
                st.info(f"💓 Zone 2 : **{fc_sv1}–{fc_sv2} bpm** · Zone 1 : <{fc_sv1} bpm · Zone 3+ : >{fc_sv2} bpm")
                if sv1.get("Cadence", 0) > 0 and sv2.get("Cadence", 0) > 0:
                    st.info(f"🏃 Vitesse SV1 : **{sv1['Cadence']:.2f} km/h** ({pace_str(3600/sv1['Cadence'])}/km) · SV2 : **{sv2['Cadence']:.2f} km/h** ({pace_str(3600/sv2['Cadence'])}/km)")

            st.subheader("📉 Graphiques ventilatoires")
            fig_gz, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.flatten()
            def _ax_palier(ax, y_col, label, color):
                if y_col not in df_pal.columns: return
                x = df_pal["palier"].values; y = df_pal[y_col].values
                ax.plot(x, y, lw=2, color=color, marker="o", ms=5)
                ax.set_xlabel("Palier"); ax.set_ylabel(label); ax.set_title(label); ax.grid(alpha=0.3)
                if sv1: ax.axvline(sv1["palier"], color=C_TEXT_MUT, lw=1.4, ls="--", label="SV1")
                if sv2: ax.axvline(sv2["palier"], color=C_RED,      lw=1.4, ls=":",  label="SV2")
                ax.legend(fontsize=7)
            _ax_palier(axes[0], "VE",   "VE (L/min)",     C_WHITE)
            if _has_o2:
                _ax_palier(axes[1], "RQ",   "Quotient Resp.", C_RED)
                _ax_palier(axes[2], "eqO2", "Éq. O₂",        C_GREY)
            else:
                _ax_palier(axes[1], "VCO2",  "VCO₂ (L/min)",  C_RED)
                _ax_palier(axes[2], "eqCO2", "Éq. CO₂ (VE/VCO₂)", C_GREY)
            _ax_palier(axes[3], "HR",   "FC (bpm)",       C_RED_SOFT)
            plt.tight_layout(); st.pyplot(fig_gz); plt.close(fig_gz)

            # ══════════════════════════════════════════════════════════════
            # v9.3 — PROFIL MÉTABOLIQUE (masque CO₂ seul)
            # ══════════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("🍬 Profil métabolique, économie et nutrition")
            _ath_met = get_athlete(current_athlete_id()) if history_ready() else None
            _vc_last = last_vc_ms(current_athlete_id()) if history_ready() else None
            _n_prior = count_metabolic_tests(current_athlete_id()) if history_ready() else 0
            if _has_o2:
                st.caption("Fichier avec oxygène mesuré : la partition glucides/lipides repose sur des gaz "
                           "réellement mesurés.")
            else:
                st.markdown(
                    '<div class="note-box note-red">🫁 <b>Masque CO₂ seul</b> — ce que l\'app peut affirmer, '
                    'et à quel titre :<br>'
                    '• <b>Mesuré</b> : ventilation et CO₂ → <b>seuils ventilatoires</b> (VT2 = nadir de VE/VCO₂) '
                    'et <b>dépense énergétique</b> (le CO₂ suffit, à ±6 % près sur le RER).<br>'
                    '• <b>Modélisé</b> : le VO₂, déduit de la vitesse, de la masse et de l\'économie de course. '
                    'Ce n\'est pas une mesure.<br>'
                    '• <b>Dérivé du modèle</b> : la partition glucides/lipides, qui dépend du RER = VCO₂/VO₂ et '
                    'hérite donc de toute l\'incertitude du VO₂ estimé.<br>'
                    'C\'est pourquoi trois confiances distinctes sont affichées plutôt qu\'une seule.</div>',
                    unsafe_allow_html=True)

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mass = _mc1.number_input("Masse de l'athlète (kg)", 35.0, 130.0,
                                      float((_ath_met or {}).get("mass_kg") or 70.0), 0.5, key="met_mass")
            _vc_met_kmh = _mc2.number_input("Vitesse critique (km/h)", 5.0, 25.0,
                                            float((_vc_last or 3.9) * 3.6), 0.05, key="met_vc")
            _gut_cap = _mc3.number_input("Tolérance intestinale (g glucides/h)", 30.0, 140.0,
                                         float((_ath_met or {}).get("gut_cap_g_h") or 90.0), 5.0, key="met_gut",
                                         help="60 g/h avec un seul type de glucide, 90-120 g/h avec un mélange "
                                              "glucose+fructose chez un athlète entraîné à s'alimenter.")
            _win_frac = _mc4.slider("Fenêtre d'analyse (fin de palier)", 0.3, 1.0, 0.5, 0.05, key="met_win",
                                    help="Portion finale de chaque palier utilisée pour les moyennes : la plus "
                                         "proche de l'état stable.")
            # ── v9.4 : vitesses des paliers, saisies à la main si besoin ──────
            _diag = []
            _diag.append("O₂ mesuré" if _has_o2 else "CO₂ seul")
            _diag.append("FC exploitable" if df_gz.attrs.get("has_hr") else "FC absente ou figée")
            _diag.append("vitesse dans le fichier" if df_gz.attrs.get("has_speed") else "aucune vitesse dans le fichier")
            _diag.append(f"lissage respiratoire {df_gz.attrs.get('smooth_window_s', 15)} s")
            st.caption("Fichier : " + " · ".join(_diag) + ".")

            _pv = _stage_means(df_gz, _win_frac)
            _skey = f"_met_speeds_{gazoz_file.name}"
            _vkey = f"_met_speeds_ver_{gazoz_file.name}"
            if _skey not in st.session_state:
                st.session_state[_skey] = {int(m["palier"]): float(m.get("vitesse_kmh") or 0.0) for m in _pv}
                st.session_state[_vkey] = 0
            with st.expander("🏃 Vitesse de chaque palier"
                             + ("" if df_gz.attrs.get("has_speed") else " — à renseigner (absente du fichier)"),
                             expanded=not df_gz.attrs.get("has_speed")):
                st.caption("La vitesse sert à modéliser le VO₂, donc à estimer les substrats et l'économie. "
                           "Les seuils ventilatoires et la dépense énergétique, eux, n'en ont pas besoin. "
                           "Si le tapis n'était pas connecté, saisis les vitesses ici.")
                _rc1, _rc2, _rc3 = st.columns([1, 1, 1.3])
                _v0 = _rc1.number_input("Vitesse du 1er palier (km/h)", 3.0, 25.0, 8.0, 0.1, key="met_ramp_v0")
                _dv = _rc2.number_input("Incrément par palier (km/h)", 0.0, 3.0, 0.5, 0.1, key="met_ramp_dv")
                with _rc3:
                    st.caption(" ")
                    if st.button("↻ Remplir en rampe", key="met_ramp_fill"):
                        st.session_state[_skey] = {int(m["palier"]): round(_v0 + _dv * i, 2)
                                                   for i, m in enumerate(_pv)}
                        st.session_state[_vkey] = st.session_state.get(_vkey, 0) + 1
                        st.rerun()
                _hr_col = any(m.get("hr") for m in _pv)      # colonne FC masquée si la FC est figée/absente
                _df_sp = pd.DataFrame([{k: v for k, v in {
                    "Palier": int(m["palier"]),
                    "Durée": seconds_to_hms(m["duree_s"]),
                    "VCO₂ (L/min)": round(m["vco2_lmin"], 2),
                    "VE (L/min)": round(m["ve_lmin"], 1) if m.get("ve_lmin") else None,
                    "FC": (float(m["hr"]) if m.get("hr") else None) if _hr_col else None,
                    "Vitesse (km/h)": float(st.session_state[_skey].get(int(m["palier"]), 0.0) or 0.0),
                }.items() if _hr_col or k != "FC"} for m in _pv])
                _edited = st.data_editor(
                    _df_sp, hide_index=True, use_container_width=True,
                    key=f"met_speed_editor_{gazoz_file.name}_{st.session_state.get(_vkey, 0)}",
                    disabled=[c for c in ["Palier", "Durée", "VCO₂ (L/min)", "VE (L/min)", "FC"]
                              if c in _df_sp.columns],
                    column_config={
                        "Vitesse (km/h)": st.column_config.NumberColumn(
                            "Vitesse (km/h)", min_value=0.0, max_value=25.0, step=0.1, format="%.2f",
                            help="0 = palier ignoré pour les substrats et l'économie."),
                        "FC": st.column_config.NumberColumn("FC", format="%d",
                                                            help="Vide = FC absente ou figée dans le fichier."),
                        "VE (L/min)": st.column_config.NumberColumn("VE (L/min)", format="%.1f")})
                st.session_state[_skey] = {int(r["Palier"]): float(r["Vitesse (km/h)"] or 0.0)
                                           for _, r in _edited.iterrows()}
                _sp_ok = [v for v in st.session_state[_skey].values() if v and v > 0.3]
                if _sp_ok:
                    _allures = " · ".join(f"{v:.1f} km/h = {pace_str(3600.0/v)}/km"
                                          for v in (min(_sp_ok), max(_sp_ok)))
                    st.caption(f"{len(_sp_ok)} palier(s) avec vitesse — {_allures}.")
                else:
                    st.caption("Aucune vitesse renseignée : seuls les seuils et la dépense énergétique seront "
                               "calculés.")
            _speeds_manual = {k: v for k, v in st.session_state[_skey].items() if v and v > 0.3}

            _eco_mode = _eco_manual = None
            if not _has_o2:
                _ec1, _ec2 = st.columns([1.2, 2])
                _eco_mode = _ec1.radio("Économie de course (pour modéliser le VO₂)",
                                       ["Calibrer sur les paliers faciles", "Saisir une valeur"],
                                       key="met_eco_mode")
                if _eco_mode == "Saisir une valeur":
                    _eco_manual = _ec2.number_input("Économie (mL O₂ / kg / km)", 150.0, 260.0, 200.0, 1.0,
                                                    key="met_eco_val",
                                                    help="180-190 chez l'élite, 200-210 chez un bon coureur "
                                                         "entraîné, 220-235 chez un coureur loisir. Si tu as un "
                                                         "test labo avec VO₂ mesuré, mets la valeur exacte.")
                else:
                    _ec2.caption("L'app cale l'économie pour que le RER des paliers faciles retombe sur ~0,82, "
                                 "valeur attendue à basse intensité. C'est ce qui remplace la mesure d'O₂ : on "
                                 "ancre le modèle là où l'on sait ce que le RER doit valoir.")
            if history_ready() and st.button("📌 Mémoriser masse et tolérance dans le profil de l'athlète",
                                             key="met_save_profile"):
                set_athlete_profile(current_athlete_id(), current_user()["id"], _mass, _gut_cap)
                st.success("Profil de l'athlète mis à jour.")

            _stages_met, _infos_met = analyze_metabolic_stages(
                df_gz, mass_kg=_mass, vc_ms=_vc_met_kmh / 3.6, n_prior_tests=_n_prior, window_frac=_win_frac,
                economy_ml_kg_km=_eco_manual, auto_calibrate_economy=(_eco_manual is None),
                speed_by_stage=_speeds_manual)
            if not _stages_met:
                st.warning(f"Analyse métabolique impossible — {_infos_met.get('error', 'paliers inexploitables')}.")
            else:
                _thr = _infos_met.get("seuils") or {}
                _fm = _infos_met.get("fatmax") or {}
                _eco_vals = [s["kcal_kg_km"] for s in _stages_met if s.get("kcal_kg_km")]
                kpi_row([
                    ("Confiance — seuils", f"{_infos_met['conf_seuils']:.0f} %",
                     f"{confidence_label(_infos_met['conf_seuils'])} · VE/VCO₂ mesuré"),
                    ("Confiance — dépense", f"{_infos_met['conf_energie']:.0f} %",
                     f"{confidence_label(_infos_met['conf_energie'])} · CO₂ mesuré, RER estimé"),
                    ("Confiance — substrats", f"{_infos_met['conf_substrats']:.0f} %",
                     f"{confidence_label(_infos_met['conf_substrats'])} · VO₂ {_infos_met['vo2_source']}"),
                    ("Protocole",
                     {"ramp": "Ramp 1 min", "paliers_courts": "Paliers 2-3 min",
                      "paliers_longs": "Paliers ≥ 3 min", "mixte": "Ramp + validation"}.get(
                         _infos_met["protocole"], "—"),
                     f"{_infos_met['n_paliers']} paliers exploitables"),
                ])
                st.markdown(f'<div class="note-box">{_infos_met["protocole_msg"]}</div>', unsafe_allow_html=True)
                if _infos_met.get("economie_ml_kg_km"):
                    _disp = _infos_met.get("economie_dispersion_pct")
                    st.caption(f"Économie retenue pour modéliser le VO₂ : **{_infos_met['economie_ml_kg_km']:.0f} "
                               f"mL/kg/km**"
                               + ((f" — calée sur {_infos_met['economie_n_paliers']} palier(s) d'ancrage"
                                   + (f" (les moins intenses, jusqu'à {_infos_met['economie_fenetre_pct']:.0f} % "
                                      f"de la VC)" if _infos_met.get("economie_fenetre_pct") else "")
                                   + (f", dispersion {_disp:.0f} %" if _disp else ""))
                                  if _infos_met.get("economie_calibree") else " (valeur saisie)")
                               + ". Une erreur de 10 % sur cette valeur déplace la partition glucides/lipides, "
                                 "beaucoup moins la dépense énergétique.")
                if _infos_met.get("paliers_ecartes"):
                    st.caption("Palier(s) " + ", ".join(str(p) for p in _infos_met["paliers_ecartes"])
                               + " écarté(s) : nettement plus courts que les autres (retour au calme ou "
                                 "coupure d'enregistrement).")
                if _infos_met.get("economie_alerte"):
                    st.warning(f"⚠️ {_infos_met['economie_alerte']}")
                if _infos_met.get("alerte_rer"):
                    st.warning(f"⚠️ {_infos_met['alerte_rer']}")
                if not _infos_met.get("substrats_possibles", True):
                    st.markdown(f'<div class="note-box note-red">{_infos_met.get("substrats_msg", "")}</div>',
                                unsafe_allow_html=True)

                # ── 1. ce que le masque mesure : les seuils ──────────────
                st.markdown("##### 🫁 Seuils ventilatoires (mesure directe)")
                _fig_thr = plot_ventilatory_thresholds(_stages_met, _thr, _vc_met_kmh / 3.6)
                if _fig_thr is not None:
                    st.pyplot(_fig_thr); plt.close("all")
                _rows_thr = []
                for _k, _lab in (("vt1", "VT1 — seuil aérobie (indicatif)"),
                                 ("vt2", "VT2 — compensation respiratoire")):
                    _t = _thr.get(_k)
                    if _t:
                        # sans tapis instrumenté, le seuil est situé sur l'axe disponible
                        # (FC ou n° de palier) : la ligne s'adapte au lieu de planter
                        _v_thr = _t.get("vitesse_kmh")
                        _rows_thr.append({
                            "Seuil": _lab,
                            "Situé à": f"{_t.get('x', '—')} {_thr.get('axe_unite', '')}",
                            "Vitesse (km/h)": _v_thr if _v_thr else "—",
                            "Allure": (pace_str(3600.0 / _v_thr) + "/km") if _v_thr else "—",
                            "% VC": _t.get("pct_vc") or "—", "FC": _t.get("hr") or "—",
                            "Palier": _t.get("palier"),
                            "Qualité du signal": ("solide" if _t.get("fiable") else "faible — à confirmer"),
                        })
                if _rows_thr:
                    st.dataframe(pd.DataFrame(_rows_thr), use_container_width=True, hide_index=True)
                if _thr.get("vt1_note") and _thr.get("vt1"):
                    st.caption("ℹ️ " + _thr["vt1_note"])
                if _thr.get("vt1_message"):
                    st.caption("ℹ️ " + _thr["vt1_message"])
                _cf = ((_thr.get("vt2") or {}).get("confirmation_feco2"))
                if _cf:
                    st.caption(("✅ VT2 confirmé par un second marqueur indépendant : le CO₂ expiré chute de "
                                f"{_cf['chute_pct']:.0f} % à partir de {_cf['x']:.1f} {_thr.get('axe_unite','')}."
                                ) if _cf.get("concordant") else
                               ("ℹ️ Le CO₂ expiré chute à partir de "
                                f"{_cf['x']:.1f} {_thr.get('axe_unite','')}, soit un peu après le nadir de "
                                f"VE/VCO₂ : les deux marqueurs encadrent VT2 sans se superposer exactement."))
                if not _rows_thr:
                    st.caption(_thr.get("message", "Seuils non identifiables sur ce test."))

                # ── 2. dépense énergétique ───────────────────────────────
                st.markdown("##### 🔥 Dépense énergétique et économie")
                st.pyplot(plot_energy(_stages_met, _vc_met_kmh / 3.6)); plt.close("all")
                _fig_eco = plot_economy(_stages_met, _mass)
                if _fig_eco is not None:
                    st.pyplot(_fig_eco); plt.close("all")

                # ── 3. substrats ─────────────────────────────────────────
                st.markdown("##### ⚗️ Oxydation des substrats")
                _substrats_ok = bool(_infos_met.get("substrats_possibles", True)) and \
                                any(x.get("cho_g_min") is not None for x in _stages_met)
                if not _has_o2:
                    st.caption("Rappel : sans O₂ mesuré, ces courbes reposent sur un VO₂ modélisé. Elles sont "
                               "utiles pour comparer des intensités entre elles et suivre un athlète dans le "
                               "temps avec le même protocole ; elles ne remplacent pas une mesure de laboratoire.")
                _fig_sub = (plot_substrates([x for x in _stages_met if x.get("cho_g_min") is not None],
                                            _vc_met_kmh / 3.6) if _substrats_ok else None)
                _rer_bas = [x["palier"] for x in _stages_met if x.get("mode") == "rer_trop_bas"]
                if _fig_sub is not None:
                    st.pyplot(_fig_sub); plt.close("all")
                    if _rer_bas:
                        st.caption("Palier(s) " + ", ".join(str(p) for p in _rer_bas)
                                   + " sans partition : RER modélisé sous 0.70, physiologiquement impossible "
                                     "à l'effort. C'est le signe d'un CO₂ encore en retard sur l'effort "
                                     "(typique du tout premier palier) ou d'une vitesse surestimée — la "
                                     "dépense énergétique de ces paliers, elle, reste valable.")
                elif _substrats_ok:
                    st.info("Pas assez de paliers avec une partition exploitable pour tracer les courbes "
                            "de substrats.")
                else:
                    st.info("Substrats non calculables sur ce fichier : renseigne les vitesses des paliers "
                            "ci-dessus (section « Vitesse de chaque palier ») pour les débloquer.")
                if not _fm and _infos_met.get("fatmax_msg") and _substrats_ok:
                    st.markdown(f'<div class="note-box">{_infos_met["fatmax_msg"]}</div>',
                                unsafe_allow_html=True)
                if _fm and _substrats_ok:
                    st.markdown(f'<div class="note-box note-red">FatMax estimé à <b>{_fm.get("fat_g_min", 0):.2f} '
                                f'g/min</b> vers <b>{_fm.get("vitesse_kmh", 0):.1f} km/h</b>'
                                + (f" ({_fm['pct_vc']:.0f} % de la VC)" if _fm.get("pct_vc") else "")
                                + (f", FC ≈ {_fm['hr']} bpm" if _fm.get("hr") else "")
                                + ".</div>", unsafe_allow_html=True)

                st.markdown("##### Détail par palier")
                st.dataframe(pd.DataFrame([{
                    "Palier": s["palier"], "Durée": seconds_to_hms(s["duree_s"]),
                    "Vitesse (km/h)": s.get("vitesse_kmh", "—"), "% VC": s.get("pct_vc", "—"),
                    "FC": s.get("hr", "—"),
                    "VE (L/min)": s.get("ve_lmin", "—"), "VCO₂ (L/min)": s["vco2_lmin"],
                    "VE/VCO₂": s.get("eqco2", "—"),
                    "VO₂ (mL/kg/min)": (f"{s['vo2_ml_kg_min']} ({s['vo2_source']})"
                                        if s.get("vo2_ml_kg_min") else "—"),
                    "RER": (s["rer"] if s.get("mode") != "sans_vo2" else f"{s['rer']} (supposé)"),
                    "Dépense (kcal/h)": f"{s['kcal_h']}  [{s['kcal_h_lo']} – {s['kcal_h_hi']}]",
                    "kcal/kg/km": s.get("kcal_kg_km", "—"),
                    # sans vitesse ni O₂, ces deux colonnes n'ont pas de valeur : on l'affiche
                    "Glucides (g/min)": (f"{s['cho_g_min']:.2f}  [{s['cho_lo']:.2f} – {s['cho_hi']:.2f}]"
                                         if s.get("cho_g_min") is not None else "—"),
                    "Lipides (g/min)": (f"{s['fat_g_min']:.2f}  [{s['fat_lo']:.2f} – {s['fat_hi']:.2f}]"
                                        if s.get("fat_g_min") is not None else "—"),
                    "Stabilité (CV %)": s["cv_pct"],
                    "Conf. seuils / dépense / substrats":
                        f"{s['conf_seuils']:.0f} / {s['conf_energie']:.0f} / {s['conf_substrats']:.0f} %",
                } for s in _stages_met]), use_container_width=True, hide_index=True)
                if any(s["mode"] == "cho_exclusif" for s in _stages_met):
                    st.caption("⚠️ Paliers à RER > 1.00 : une partie du CO₂ provient du tamponnement des ions H⁺ "
                               "et non du métabolisme. L'app y bascule sur « glucides quasi exclusifs » et "
                               "abaisse la confiance.")

                # ── 4. plan nutritionnel ─────────────────────────────────
                st.markdown("##### 🥤 Plan nutritionnel par zone")
                _plan_met, _pinfo = fueling_plan(_stages_met, _vc_met_kmh / 3.6,
                                                 gut_cap_g_h=_gut_cap, mass_kg=_mass)
                if not _plan_met:
                    st.caption(_pinfo.get("error", "Plan indisponible."))
                else:
                    st.dataframe(pd.DataFrame([{
                        "Zone": r["zone"], "% VC": f"{r['pct_lo']:.0f} – {r['pct_hi']:.0f} %",
                        "Allure": (pace_str(r["allure_s_km"]) + "/km") if r.get("allure_s_km") else "—",
                        "Vitesse (km/h)": r["vitesse_kmh"], "Dépense (kcal/h)": r["kcal_h"],
                        "Glucides oxydés (g/h)": f"{r['cho_g_h']}  [{r['cho_g_h_lo']} – {r['cho_g_h_hi']}]",
                        "Apport conseillé (g/h)": r["apport_g_h"], "Déficit horaire (g/h)": r["deficit_g_h"],
                        "Autonomie glycogène": f"{r['autonomie_h']} h" if r.get("autonomie_h") else "≥ 24 h",
                        "Confiance": f"{r['confiance']:.0f} %" + (" (extrapolé)" if r["extrapole"] else ""),
                    } for r in _plan_met]), use_container_width=True, hide_index=True)
                    st.markdown(
                        f'<div class="note-box">Réserves glycogéniques estimées : <b>{_pinfo["stores_g"]} g</b> '
                        f'({GLYCOGEN_G_PER_KG:.0f} g/kg) — ordre de grandeur, pas une mesure. L\'apport est '
                        f'plafonné à <b>{_gut_cap:.0f} g/h</b> : au-delà, ce n\'est plus le métabolisme qui '
                        f'limite mais l\'absorption. La colonne dépense (kcal/h) est la plus fiable de ce '
                        f'tableau ; les grammes de glucides en héritent de l\'incertitude du VO₂ modélisé.</div>',
                        unsafe_allow_html=True)

                st.markdown("##### 🎯 Estimation à une allure donnée")
                _ec1b, _ec2b = st.columns([1, 3])
                _target_pace = _ec1b.text_input("Allure visée (mm:ss/km)", value="5:30", key="met_target_pace")
                _mm = re.match(r"^(\d+):(\d{2})$", _target_pace.strip())
                if _mm:
                    _t_s_km = int(_mm.group(1)) * 60 + int(_mm.group(2))
                    _v_target = 3600.0 / _t_s_km
                    _pct_target = _v_target / _vc_met_kmh * 100.0
                    _pts = [s for s in _stages_met if s.get("pct_vc") and s.get("kcal_h") is not None]
                    if len(_pts) >= 2:
                        _o = np.argsort([s["pct_vc"] for s in _pts])
                        _xs_s = np.array([_pts[i]["pct_vc"] for i in _o], dtype=float)
                        # les paliers sans partition exploitable ne participent qu'à la dépense
                        _pcho = [s for s in _pts if s.get("cho_g_min") is not None]
                        if len(_pcho) >= 2:
                            _oc = np.argsort([s["pct_vc"] for s in _pcho])
                            _cho_i = float(np.interp(
                                _pct_target,
                                np.array([_pcho[i]["pct_vc"] for i in _oc], dtype=float),
                                np.array([_pcho[i]["cho_g_min"] for i in _oc], dtype=float)))
                        else:
                            _cho_i = None
                        _kcal_i = float(np.interp(_pct_target, _xs_s,
                                                  np.array([_pts[i]["kcal_h"] for i in _o], dtype=float)))
                        _conf_i = float(np.interp(_pct_target, _xs_s,
                                                  np.array([_pts[i]["conf_energie"] for i in _o], dtype=float)))
                        _hpts = [s for s in _pts if s.get("hr")]
                        _hr_i = (float(np.interp(_pct_target, [s["pct_vc"] for s in _hpts],
                                                 [s["hr"] for s in _hpts])) if len(_hpts) >= 2 else None)
                        _outside = _pct_target < min(_xs_s) - 3 or _pct_target > max(_xs_s) + 3
                        with _ec2b:
                            kpi_row([
                                ("Intensité", f"{_pct_target:.0f} % VC", f"{_v_target:.2f} km/h"),
                                ("Dépense", f"{_kcal_i:.0f} kcal/h",
                                 f"≈ {_kcal_i/max(0.1,_v_target):.0f} kcal/km · confiance {_conf_i:.0f} %"),
                                ("Glucides oxydés",
                                 f"{_cho_i*60:.0f} g/h" if _cho_i is not None else "—",
                                 (f"apport conseillé {min(_gut_cap, _cho_i*60):.0f} g/h"
                                  + (" — valeur invraisemblable, vérifie les vitesses"
                                     if _cho_i * 60 > 300 else "")
                                  ) if _cho_i is not None else "partition indisponible"),
                                ("FC attendue", f"{_hr_i:.0f} bpm" if _hr_i else "—",
                                 "extrapolé hors plage testée" if _outside else "dans la plage testée"),
                            ])
                else:
                    st.caption("Format attendu : mm:ss (ex. 5:30).")

                st.markdown("##### 💾 Enregistrer ce test métabolique")
                if history_ready():
                    with st.form("save_met_form"):
                        _sm1, _sm2 = st.columns(2)
                        _met_date = _sm1.date_input("Date du test", value=date.today(), key="save_met_date")
                        _met_label = _sm2.text_input("Libellé", value="Test CO₂ — profil métabolique",
                                                     key="save_met_label")
                        _met_notes = st.text_area("Notes (état de forme, alimentation avant test, matériel…)",
                                                  key="save_met_notes")
                        if st.form_submit_button("💾 Enregistrer dans l'historique"):
                            _sv1_hr = (sv1 or {}).get("HR") or ((_thr.get("vt1") or {}).get("hr"))
                            _sv2_hr = (sv2 or {}).get("HR") or ((_thr.get("vt2") or {}).get("hr"))
                            _mid = save_metabolic_test(
                                current_athlete_id(), _met_date.isoformat(), _met_label,
                                _infos_met["protocole"], _mass, _vc_met_kmh / 3.6,
                                _infos_met["conf_energie"], _stages_met, _plan_met if _plan_met else [],
                                _infos_met, sv1_hr=_sv1_hr, sv2_hr=_sv2_hr, notes=_met_notes)
                            st.success(f"✅ Test enregistré (#{_mid}) — évolution dans l'onglet 📚 Historique. "
                                       f"Il compte désormais dans la calibration individuelle de la confiance.")
                    st.caption(f"Calibration individuelle actuelle : {_n_prior} test(s) déjà enregistré(s) "
                               f"pour cet athlète.")
                else:
                    save_gate("ce test métabolique")

    st.markdown("---")
    st.header("04 · Détection du point de rupture (tests à effort maximal)")
    st.caption(
        "Pour chaque référence importée via fichier FIT/TCX (section 1 ci-dessus), détecte le moment où "
        "l'allure décroche le plus nettement — une rupture à un seul changement de régime (allure stable, puis "
        "allure dégradée). Comparer ce point en % de la durée totale entre plusieurs tests de durées différentes "
        "peut révéler une signature de fatigue récurrente propre à l'athlète, utilisable comme point de départ "
        "pour le seuil de fatigue en course longue (onglet 🏃 Prédiction, section 4).")

    _bp_rows = []
    for _i in range(1, st.session_state.n_refs_vc + 1):
        _pts = st.session_state.get(f"vc_points_{_i}")
        if not _pts:
            continue
        _t_bp, _pace_bp = compute_pace_series_from_points(_pts)
        if _t_bp is None or len(_t_bp) < 10:
            continue
        _bp = detect_pace_breakpoint(_t_bp, _pace_bp)
        if _bp is None:
            continue
        _label = st.session_state.get(f"vc_label_{_i}", f"Réf {_i}")
        _bp_rows.append({"Test": _label, "Durée totale": seconds_to_hms(float(_t_bp[-1])),
                         "Rupture à": seconds_to_hms(_bp["t_break_s"]), "% de la durée": round(_bp["pct_break"], 1),
                         "Allure avant": pace_str(_bp["pace_before"]) + "/km", "Allure après": pace_str(_bp["pace_after"]) + "/km",
                         "Dégradation": f"{_bp['drop_pct']:+.1f}%", "Qualité (R²)": round(_bp["r2"], 2)})

    if not _bp_rows:
        st.info("Importe au moins un test via fichier FIT/TCX (section 1 ci-dessus, ≥1 min) pour activer cette "
                "analyse — les références saisies à la main (sans fichier) n'ont pas de courbe d'allure à analyser.")
    else:
        df_bp = pd.DataFrame(_bp_rows)
        st.dataframe(df_bp, use_container_width=True, hide_index=True)

        _valid = [r for r in _bp_rows if r["Qualité (R²)"] >= 0.3]
        if len(_valid) >= 2:
            _pcts = [r["% de la durée"] for r in _valid]
            _drops = [float(r["Dégradation"].replace("%", "")) for r in _valid]
            _avg_pct = float(np.mean(_pcts)); _std_pct = float(np.std(_pcts)); _avg_drop = float(np.mean(_drops))
            st.markdown(f"**Signature moyenne détectée : rupture vers {_avg_pct:.0f}% de l'effort "
                        f"(±{_std_pct:.0f} pts) · dégradation moyenne {_avg_drop:+.1f}%**")
            if _std_pct < 15:
                st.success(f"📍 Cohérent entre tests de durées différentes — signe d'une vraie signature "
                           f"physiologique plutôt qu'un artefact d'un seul test. Tu pourrais tester "
                           f"`fatigue_threshold ≈ {_avg_pct:.0f}` dans l'onglet 🏃 Prédiction comme point de départ "
                           f"pour cet athlète (à confirmer/affiner avec les courses réelles dans l'onglet "
                           f"👥 Analyse de cohorte).")
            else:
                st.warning(f"⚠️ Le point de rupture varie pas mal selon la durée du test (±{_std_pct:.0f} pts) — "
                           f"peut-être pas une signature fixe en %, ou tests pas tous menés au même niveau "
                           f"d'effort relatif (quasi-maximal soutenable sur leur durée respective).")
        elif len(_valid) == 1:
            st.caption("Un seul test exploitable pour l'instant — importe au moins 2-3 tests de durées "
                       "différentes (effort maximal soutenu sur chacun) pour voir si le point de rupture est "
                       "cohérent entre eux.")
        st.caption(
            "⚠️ Le mécanisme de rupture sur un test court (quelques minutes à 1h, effort quasi-maximal) n'est pas "
            "forcément le même phénomène physiologique que la fatigue sur une course de plusieurs heures "
            "(déplétion glycogénique, dommages musculaires, thermorégulation...). À traiter comme une piste à "
            "recouper avec des données de course réelles, pas comme une vérité automatiquement transférable.")


# ══════════════════════════════════════════════════════════════
# ONGLET 2 — ANALYSE ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.title("⚙️ Analyse d'entraînement")
    st.caption("Charge un fichier FIT, GPX, TCX ou CSV pour analyser une séance : FC, allure, dérive, intervalles.")

    train_file = st.file_uploader("📂 Fichier de séance (FIT / GPX / TCX / CSV)",
                                   type=["fit","gpx","tcx","csv"], key="train_file")

    if train_file:
        with st.spinner("Chargement..."):
            df_train = load_activity(train_file)

        if df_train is None:
            st.error("❌ Impossible de lire ce fichier. Vérifiez le format.")
        else:
            n_rows = len(df_train)
            dur_s_train = float(df_train["elapsed_s"].max()) if "elapsed_s" in df_train.columns else 0.0
            dist_m_train = float(df_train["distance_m"].dropna().max()) if "distance_m" in df_train.columns and df_train["distance_m"].notna().any() else None

            c1, c2, c3 = st.columns(3)
            c1.metric("Durée", seconds_to_hms(dur_s_train))
            if dist_m_train:
                c2.metric("Distance", f"{dist_m_train/1000:.2f} km")
                if dur_s_train > 0:
                    c3.metric("Allure moy.", pace_str(dur_s_train/(dist_m_train/1000))+"/km")

            st.markdown("---")
            st.subheader("💓 Analyse fréquence cardiaque")
            hr_stats_train = analyze_heart_rate(df_train)
            if hr_stats_train.get("available"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("FC max (P95)", f"{hr_stats_train['fc_max']} bpm")
                c2.metric("FC moy.", f"{hr_stats_train['fc_avg']} bpm")
                c3.metric("FC min (P5)", f"{hr_stats_train['fc_min']} bpm")
                c4.metric("Dérive cardiaque", f"{hr_stats_train['drift_abs']:+.1f} bpm",
                          delta=f"{hr_stats_train['drift_pct']:+.1f}%",
                          delta_color="inverse" if hr_stats_train["drift_abs"] > 8 else "normal")
                c1b, c2b = st.columns(2)
                c1b.metric("Seuil estimé (~88% FCmax)", f"{hr_stats_train['seuil_estime']} bpm")
                c2b.metric("Fiabilité signal", hr_stats_train["reliability"])

                if "heart_rate" in df_train.columns and "elapsed_s" in df_train.columns:
                    hr_series = df_train["heart_rate"].dropna()
                    hr_series = hr_series[(hr_series >= 40) & (hr_series <= 220)]
                    if len(hr_series) > 10:
                        t_series = df_train.loc[hr_series.index, "elapsed_s"].values / 60.0
                        hr_smooth = smooth_hr(hr_series).values
                        fig_hr_t, ax_hr_t = plt.subplots(figsize=(11, 3))
                        ax_hr_t.plot(t_series, hr_series.values, lw=0.7, alpha=0.25, color=C_DIM, label="FC brute")
                        ax_hr_t.plot(t_series, hr_smooth, lw=2, color=C_RED, label="FC lissée")
                        ax_hr_t.axhline(hr_stats_train["fc_avg"], color=C_WHITE, lw=1, ls="--",
                                        label=f"FC moy. {hr_stats_train['fc_avg']} bpm")
                        ax_hr_t.set_xlabel("Temps (min)"); ax_hr_t.set_ylabel("FC (bpm)")
                        ax_hr_t.set_title("Fréquence cardiaque au fil du temps")
                        ax_hr_t.legend(fontsize=8); ax_hr_t.grid(alpha=0.3); fig_hr_t.tight_layout()
                        st.pyplot(fig_hr_t); plt.close(fig_hr_t)
            else:
                st.info("Pas de données FC dans ce fichier.")

            st.markdown("---")
            st.subheader("🏃 Analyse vitesse & allure")
            spd_stats_train = analyze_speed_kinetics(df_train)
            if spd_stats_train.get("available") and "elapsed_s" in df_train.columns:
                spd = df_train["speed_ms"].dropna()
                spd = spd[spd > 0]
                if len(spd) > 10:
                    t_spd = df_train.loc[spd.index, "elapsed_s"].values / 60.0
                    pace_vals = 1000.0 / spd.values
                    pace_smooth = pd.Series(pace_vals).rolling(15, center=True, min_periods=1).median().values
                    fig_pace, ax_pace = plt.subplots(figsize=(11, 3))
                    ax_pace.plot(t_spd, pace_vals, lw=0.6, alpha=0.2, color=C_DIM)
                    ax_pace.plot(t_spd, pace_smooth, lw=2, color=C_WHITE, label="Allure lissée")
                    ax_pace.invert_yaxis()
                    ax_pace.set_xlabel("Temps (min)"); ax_pace.set_ylabel("Allure (s/km)")
                    ax_pace.set_title("Allure au fil du temps"); ax_pace.grid(alpha=0.3); ax_pace.legend(fontsize=8)
                    yticks = ax_pace.get_yticks()
                    ax_pace.set_yticklabels([pace_str(t) for t in yticks if t > 0], fontsize=8)
                    fig_pace.tight_layout()
                    st.pyplot(fig_pace); plt.close(fig_pace)

            st.markdown("---")
            st.subheader("📐 Séance découpée en quarts")
            st.caption("La séance est coupée en quatre parts de temps en mouvement égales (les pauses ne "
                       "décalent pas le découpage). Pour chaque quart : allure réelle, VAP (allure ramenée "
                       "au plat, comparable même si le dénivelé change), FC et cadence.")
            _n_parts = st.select_slider("Nombre de parts", options=[2, 3, 4, 5, 6], value=4, key="quarters_n")
            _quarters = compute_session_quarters(df_train, n_parts=_n_parts)
            st.session_state["_session_quarters"] = _quarters
            if _quarters:
                _dur_tot_q = sum(q["duree_s"] for q in _quarters)
                _dist_tot_q = sum(q["distance_m"] for q in _quarters)
                _pace_glob = _dur_tot_q / max(0.001, _dist_tot_q / 1000.0)
                _q_last, _q_first = _quarters[-1], _quarters[0]
                kpi_row([
                    ("Allure moyenne (en mouvement)", pace_str(_pace_glob) + "/km",
                     f"{_dist_tot_q/1000:.2f} km en {seconds_to_hms(_dur_tot_q)}"),
                    ("Allure 1er quart", pace_str(_q_first["allure_s_km"]) + "/km" if _q_first.get("allure_s_km") else "—",
                     f"FC {_q_first.get('fc_moy', '—')} bpm"),
                    ("Allure dernier quart", pace_str(_q_last["allure_s_km"]) + "/km" if _q_last.get("allure_s_km") else "—",
                     f"{_q_last.get('derive_allure_pct', 0):+.1f} % vs 1er quart"),
                    ("Dérive cardiaque",
                     f"{_q_last.get('derive_fc_bpm', 0):+.0f} bpm" if _q_last.get("derive_fc_bpm") is not None else "—",
                     f"FC {_q_first.get('fc_moy', '—')} → {_q_last.get('fc_moy', '—')} bpm"),
                ])
                st.pyplot(plot_session_quarters(_quarters)); plt.close("all")
                _rows_q = []
                for q in _quarters:
                    _rows_q.append({
                        "Quart": q["libelle"],
                        "De → à": f"{seconds_to_hms(q['t_debut_s'])} → {seconds_to_hms(q['t_fin_s'])}",
                        "Durée": seconds_to_hms(q["duree_s"]),
                        "Distance (km)": round(q["distance_m"] / 1000.0, 2),
                        "Allure": pace_str(q["allure_s_km"]) + "/km" if q.get("allure_s_km") else "—",
                        "Δ allure": f"{q['derive_allure_pct']:+.1f} %" if q.get("derive_allure_pct") is not None else "—",
                        "VAP (km/h)": q.get("vap_kmh", "—"),
                        "Δ VAP": f"{q['derive_vap_pct']:+.1f} %" if q.get("derive_vap_pct") is not None else "—",
                        "D+ (m)": q.get("d_plus", "—"), "D- (m)": q.get("d_moins", "—"),
                        "FC moy": q.get("fc_moy", "—"), "FC max": q.get("fc_max", "—"),
                        "Δ FC": f"{q['derive_fc_bpm']:+.0f}" if q.get("derive_fc_bpm") is not None else "—",
                        "Cadence": q.get("cadence", "—"),
                    })
                st.dataframe(pd.DataFrame(_rows_q), use_container_width=True, hide_index=True)
                _dv = _q_last.get("derive_vap_pct"); _da = _q_last.get("derive_allure_pct")
                _dh = _q_last.get("derive_fc_bpm")
                if _dv is not None and _dh is not None:
                    _pace_drop = (_da or 0) > 8      # l'allure brute s'est nettement dégradée
                    if _dv <= -6 and _dh >= 8:
                        _msg = ("Effort dégradé : la VAP baisse alors que la FC monte — le coût cardiaque de la "
                                "même vitesse a augmenté (chaleur, déshydratation, fatigue).")
                    elif _dv >= -3 and _pace_drop:
                        _msg = ("L'allure brute chute nettement mais la VAP tient : c'est le <b>terrain</b> qui "
                                "change (plus de dénivelé sur la fin), pas la forme. "
                                + ("La FC monte quand même : effort soutenable mais engagé."
                                   if _dh >= 8 else "La FC reste stable : effort bien géré."))
                    elif _dv >= -3 and _dh >= 8:
                        _msg = ("Allure et VAP tenues, FC en hausse : dérive cardiaque classique sur une sortie "
                                "longue — effort soutenable mais engagé.")
                    elif _dv >= -3 and abs(_dh) < 8:
                        _msg = "Séance très régulière : ni la VAP ni la FC ne dérivent réellement."
                    else:
                        _msg = ("La VAP baisse sans hausse marquée de la FC : gestion volontaire, terrain plus "
                                "exigeant, ou fatigue musculaire plutôt que cardiovasculaire.")
                    st.markdown(f'<div class="note-box note-red">{_msg} — dernier quart : allure '
                                f'<b>{_da:+.1f} %</b>, VAP <b>{_dv:+.1f} %</b>, FC <b>{_dh:+.0f} bpm</b> '
                                f'par rapport au premier quart.</div>', unsafe_allow_html=True)
            else:
                st.caption("Découpage impossible : séance trop courte ou sans données de distance.")

            st.markdown("---")
            st.subheader("⏱️ Temps de maintien & zones cardiaques")
            st.caption("Pour chaque durée, la meilleure vitesse moyenne réellement tenue dans la séance "
                       "(équivalent mesuré de la table de maintien théorique de l'onglet Vitesse Critique), "
                       "puis la répartition du temps en mouvement par zone cardiaque.")
            _hr_obs = (hr_stats_train or {}).get("fc_max") if isinstance(hr_stats_train, dict) else None
            _ath_prof = get_athlete(current_athlete_id()) if history_ready() else None
            _hr_default = float((_ath_prof or {}).get("hr_max") or _hr_obs or 185.0)
            _zc1, _zc2 = st.columns([1, 2])
            _hr_max_used = _zc1.number_input("FC max de référence (bpm)", 120.0, 230.0,
                                             float(_hr_default), 1.0, key="tm_hrmax",
                                             help="Sert uniquement à découper les zones. Par défaut : la FC max "
                                                  "mémorisée pour l'athlète, sinon la plus haute de la séance.")
            with _zc2:
                if history_ready():
                    if st.button("📌 Mémoriser comme FC max de l'athlète", key="tm_hrmax_save"):
                        set_athlete_hr_max(current_athlete_id(), current_user()["id"], _hr_max_used)
                        st.success("FC max enregistrée dans le profil de l'athlète.")
                else:
                    st.caption("Connecte-toi pour mémoriser cette FC max dans le profil de l'athlète.")
            _records_tr = compute_session_records(df_train)
            _zones_tr = compute_hr_zone_times(df_train, _hr_max_used)
            st.session_state["_session_records"] = _records_tr
            st.session_state["_session_zones"] = _zones_tr
            st.session_state["_session_hr_max_used"] = _hr_max_used
            if _records_tr:
                _vc_ref = last_vc_ms(current_athlete_id()) if history_ready() else None
                _by_dur = {r["duree_s"]: r for r in _records_tr}
                kpi_row([(f"Best {d//60} min",
                          f"{_by_dur[d]['vitesse_kmh']:.2f} km/h" if d in _by_dur else "—",
                          (pace_str(_by_dur[d]["allure_s_km"]) + "/km · " +
                           f"{_by_dur[d]['distance_m']:.0f} m") if d in _by_dur else "durée non atteinte")
                         for d in (300, 1200, 1800, 3600)])
                st.pyplot(plot_records_curve(_records_tr, vc_ms=_vc_ref)); plt.close("all")
                if _vc_ref:
                    st.caption(f"Repère : dernière vitesse critique enregistrée pour cet athlète — "
                               f"{_vc_ref*3.6:.2f} km/h ({pace_str(1000.0/_vc_ref)}/km).")
            else:
                st.caption("Pas assez de données de distance pour calculer les temps de maintien.")
            # ── v9.2 : zones d'intensité calibrées sur la VITESSE CRITIQUE ──────
            _vc_athlete = last_vc_ms(current_athlete_id()) if history_ready() else None
            st.markdown("##### 🎯 Zones d'intensité — référence vitesse critique")
            _vz1, _vz2 = st.columns([1, 2])
            _vc_manual = _vz1.number_input("Vitesse critique de référence (km/h)", 5.0, 25.0,
                                           float((_vc_athlete or 3.9) * 3.6), 0.05, key="vc_zone_ref",
                                           help="Reprise du dernier test de VC enregistré pour l'athlète ; "
                                                "modifiable ici pour une simulation.")
            _vc_use = float(_vc_manual) / 3.6
            with _vz2:
                _use_vap_zones = st.checkbox("Calculer sur la VAP (vitesse ramenée au plat) — recommandé en trail",
                                             value=True, key="vc_zone_vap")
                st.caption("Les bornes par défaut (65 / 75 / 85 / 95 / 110 % de VC) sont modifiables ci-dessous.")
            with st.expander("Bornes des zones (% de la vitesse critique)"):
                _b = st.columns(5)
                _bounds_vc = tuple(
                    _b[i].number_input(f"Z{i+1} à partir de (%)", 30.0, 160.0,
                                       float(VC_ZONE_BOUNDS_DEFAULT[i]), 1.0, key=f"vc_bound_{i}")
                    for i in range(5))
            _zones_vc, _zinfo = compute_vc_zone_times(df_train, _vc_use, _bounds_vc, use_vap=_use_vap_zones)
            st.session_state["_session_zones_vc"] = _zones_vc
            st.session_state["_session_vc_ref"] = _vc_use
            if _zones_vc:
                st.pyplot(plot_vc_zones(_zones_vc, _vc_use, _zinfo.get("used_vap", False))); plt.close("all")
                _zt = vc_zone_table(_vc_use, _bounds_vc)
                st.dataframe(pd.DataFrame([{
                    "Zone": z["zone"],
                    "% VC": (f"{z['pct_lo']:.0f} – {z['pct_hi']:.0f} %" if z["pct_hi"] else f"> {z['pct_lo']:.0f} %"),
                    "Vitesse (km/h)": (f"{z['v_lo_kmh']:.2f} – {z['v_hi_kmh']:.2f}" if z["v_hi_kmh"]
                                       else f"> {z['v_lo_kmh']:.2f}"),
                    "Allure": ((pace_str(z["allure_lo_s_km"]) + " – " + pace_str(z["allure_hi_s_km"]))
                               if z["allure_lo_s_km"] else "plus rapide que " + pace_str(z["allure_hi_s_km"])),
                    "Temps passé": seconds_to_hms(zz["temps_s"]), "% du temps": zz["pct"],
                } for z, zz in zip(_zt, _zones_vc)]), use_container_width=True, hide_index=True)
                st.caption(f"Intensité médiane de la séance : **{_zinfo.get('pct_median', 0):.0f} % de la VC**. "
                           "Zones calibrées sur une mesure de terrain (la VC) plutôt que sur une FC max estimée.")
            else:
                st.caption(f"Zones VC non calculables — {_zinfo.get('error', '')}")

            with st.expander("💓 Zones cardiaques (référence FC max) — vue secondaire"):
                if _zones_tr:
                    st.pyplot(plot_hr_zones(_zones_tr, _hr_max_used)); plt.close("all")
                else:
                    st.caption("Pas de données cardiaques exploitables dans ce fichier — zones non calculables.")

            st.markdown("---")
            st.subheader("🏔️ Analyse trail — course / marche & tenue de la performance")
            st.caption("Deux lectures d'une sortie trail : (1) à partir de quelle pente tu bascules de la course "
                       "à la marche, (2) comment ta VAP (Vitesse Ajustée à la Pente) se dégrade au fil de la "
                       "sortie, séparément en montée, sur le plat et en descente. Pauses exclues dans les deux cas.")

            with st.expander("⚙️ Réglages de l'analyse trail", expanded=False):
                tc1, tc2, tc3 = st.columns(3)
                trail_window_s = tc1.slider("Fenêtre vitesse / pente (s)", 10, 60, 20, 5, key="trail_win",
                                            help="Fenêtre glissante de calcul. Plus large = moins de bruit GPS, "
                                                 "mais transitions course/marche moins nettes.")
                trail_alt_smooth = tc2.slider("Lissage altitude (s)", 10, 90, 30, 5, key="trail_alt")
                trail_pause_kmh = tc3.slider("Seuil de pause (km/h)", 0.5, 3.0, 1.0, 0.5, key="trail_pause",
                                             help="En dessous : ravito, arrêt photo… exclu de toute l'analyse.")
                tc4, tc5 = st.columns(2)
                trail_cad_thr = tc4.slider("Seuil cadence marche / course (pas/min)", 100, 165, 135, 1,
                                           key="trail_cad",
                                           help="En dessous de ce seuil de cadence, l'échantillon est classé "
                                                "« marche ». 130-140 convient à la plupart des coureurs.")
                trail_walk_kmh = tc5.slider("Seuil vitesse marche (km/h) — si pas de cadence", 3.0, 10.0, 7.0, 0.5,
                                            key="trail_walkspd")
                tc6, tc7, tc8 = st.columns(3)
                trail_up_thr = tc6.slider("Pente mini d'une montée (%)", 2.0, 12.0, 4.0, 0.5, key="trail_up")
                trail_dn_thr = tc7.slider("Pente maxi d'une descente (%)", -12.0, -2.0, -4.0, 0.5, key="trail_dn")
                trail_min_dur = tc8.slider("Durée mini d'une portion (min)", 1, 15, 2, 1, key="trail_mindur")
                trail_min_dist = st.slider("Distance mini d'une portion (m)", 100, 2000, 300, 50, key="trail_mindist",
                                           help="Une portion doit durer ET mesurer assez pour être comparable aux "
                                                "autres. Monte ces deux seuils si le découpage est trop haché.")

            samples_tr, infos_tr = build_trail_samples(
                df_train, window_s=trail_window_s, alt_smooth_s=trail_alt_smooth,
                pause_speed_kmh=trail_pause_kmh)

            if samples_tr is None:
                st.info(f"Analyse trail indisponible — {infos_tr.get('error', 'données insuffisantes')}")
            else:
                samples_tr, wr_method = classify_walk_run(
                    samples_tr, infos_tr, cadence_threshold=trail_cad_thr, speed_walk_kmh=trail_walk_kmh)
                tz_tr = detect_transition_zone(samples_tr)
                wr_sum = walk_run_summary(samples_tr, infos_tr["grid_step_s"])
                # v8.9 — résumé conservé pour l'enregistrement de la séance
                st.session_state["_trail_summary"] = {
                    "pct_walk": (wr_sum or {}).get("pct_walk"),
                    "walk_speed_med": (wr_sum or {}).get("walk_speed_med"),
                    "run_speed_med": (wr_sum or {}).get("run_speed_med"),
                    "trans_lo": (tz_tr or {}).get("slope_lo"), "trans_mid": (tz_tr or {}).get("slope_mid"),
                    "trans_hi": (tz_tr or {}).get("slope_hi"),
                    "mid_start": (tz_tr or {}).get("mid_start"), "mid_end": (tz_tr or {}).get("mid_end"),
                    "methode_marche": wr_method,
                    "d_plus": float(np.sum(np.clip(np.diff(samples_tr["alt_m"].values), 0, None))),
                }
                st.session_state["_trail_portions"] = []

                # ── A. Biomécanique course / marche ──────────────────────────
                st.markdown("#### 🦵 Biomécanique course / marche")
                if wr_method == "cadence":
                    st.caption("Classement course / marche basé sur la **cadence** (méthode fiable)."
                               + (" Cadence lue par jambe puis doublée en pas/min."
                                  if infos_tr.get("cadence_doubled") else ""))
                else:
                    st.warning("⚠️ Pas de cadence exploitable dans ce fichier — classement de repli basé sur la "
                               "seule **vitesse**. À prendre comme un ordre de grandeur : une marche rapide en "
                               "faux plat et un footing lent en côte ne se distinguent pas.")
                if wr_sum:
                    _has_tz = bool(tz_tr and tz_tr.get("slope_lo") is not None and tz_tr.get("slope_hi") is not None)
                    kpi_row([
                        ("Course", f"{wr_sum['pct_run']:.0f} %", "soit " + seconds_to_hms(wr_sum["run_s"])),
                        ("Marche probable", f"{wr_sum['pct_walk']:.0f} %", "soit " + seconds_to_hms(wr_sum["walk_s"])),
                        ("Zone de transition",
                         f"{tz_tr['slope_lo']:.0f} → {tz_tr['slope_hi']:.0f} %" if _has_tz else "—",
                         "pentes où tu bascules" if _has_tz else "non identifiable"),
                        ("Bascule 50 % marche",
                         f"{tz_tr['slope_mid']:.0f} %" if _has_tz else "—",
                         "1 pas sur 2 en marche" if _has_tz else "—"),
                    ])
                fig_bio = plot_biomecanique(samples_tr, tz_tr, wr_sum, infos_tr, wr_method,
                                            cadence_threshold=trail_cad_thr, speed_walk_kmh=trail_walk_kmh)
                st.pyplot(fig_bio); plt.close(fig_bio)

                if tz_tr and tz_tr.get("slope_lo") is not None and tz_tr.get("slope_hi") is not None:
                    _extra = (" ⚠️ La borne haute est extrapolée : la sortie ne contient pas de pentes assez "
                              "raides pour l'observer directement."
                              if tz_tr.get("extrapolated") else "")
                    if not tz_tr.get("mid_supported", True):
                        _extra += (" ⚠️ Peu de données autour de ce point de bascule (parcours en tout ou rien : "
                                   "du plat couru, des raidillons marchés, rien entre les deux) — l'estimation "
                                   "est peu contrainte sur cette sortie.")
                    st.info(f"🚶 **Zone de transition : {tz_tr['slope_lo']:.0f} % → {tz_tr['slope_hi']:.0f} %.** "
                            f"En dessous de {tz_tr['slope_lo']:.0f} % tu cours l'essentiel du temps ; au-dessus de "
                            f"{tz_tr['slope_hi']:.0f} % tu marches presque toujours ; le basculement se fait vers "
                            f"**{tz_tr['slope_mid']:.0f} %** de pente.{_extra}")
                    if tz_tr.get("mid_start") is not None and tz_tr.get("mid_end") is not None:
                        _drift = tz_tr["mid_end"] - tz_tr["mid_start"]
                        if _drift <= -1.5:
                            st.warning(f"📉 Glissement au fil de la sortie : point de bascule à "
                                       f"**{tz_tr['mid_start']:.0f} %** sur le 1er tiers → **{tz_tr['mid_end']:.0f} %** "
                                       f"sur le dernier ({_drift:+.0f} points). Tu passes en marche sur des pentes de "
                                       f"plus en plus faibles : marqueur de fatigue musculaire/énergétique.")
                        elif _drift >= 1.5:
                            st.success(f"📈 Point de bascule à **{tz_tr['mid_start']:.0f} %** sur le 1er tiers → "
                                       f"**{tz_tr['mid_end']:.0f} %** sur le dernier ({_drift:+.0f} points) : tu cours "
                                       f"des pentes plus raides en fin de sortie (départ prudent, ou terrain plus "
                                       f"roulant en seconde partie).")
                        else:
                            st.caption(f"Point de bascule stable sur la sortie : {tz_tr['mid_start']:.0f} % "
                                       f"(1er tiers) → {tz_tr['mid_end']:.0f} % (dernier tiers).")
                    if wr_sum and wr_sum.get("walk_speed_med") and wr_sum.get("run_speed_med"):
                        st.caption(f"Vitesse médiane en marche : **{wr_sum['walk_speed_med']:.1f} km/h** "
                                   f"(pente médiane {wr_sum['walk_grade_med']:.0f} %) · en course : "
                                   f"**{wr_sum['run_speed_med']:.1f} km/h**.")
                else:
                    st.caption("Pas assez de contraste course / marche sur cette sortie pour situer une zone de "
                               "transition (sortie trop roulante, ou cadence peu exploitable).")

                # ── B. Évolution par terrain (VAP indexée) ───────────────────
                st.markdown("#### 📉 Évolution par terrain — baisse de performance au fil de la sortie")
                st.caption("Chaque portion qualifiée est comparée à la **première portion de la même famille** "
                           "(1re descente, 1er relief roulant, 1re montée = 100 %). La comparaison se fait en VAP "
                           "— vitesse ramenée à son équivalent sur le plat via le coût énergétique de Minetti — "
                           "pour que deux montées de pentes différentes restent comparables.")
                portions_tr = segment_terrain_portions(
                    samples_tr, up_thr=trail_up_thr, down_thr=trail_dn_thr,
                    min_dur_s=trail_min_dur * 60.0, min_dist_m=trail_min_dist)
                if len(portions_tr) < 2:
                    st.info("Pas assez de portions qualifiées pour comparer (sortie trop courte ou trop hachée). "
                            "Baisse la durée/distance minimale d'une portion dans les réglages ci-dessus.")
                else:
                    trends_tr = terrain_trends(portions_tr)
                    # v8.9 — tendances et portions conservées pour l'historique
                    st.session_state["_trail_summary"].update({
                        "vap_slope_montee": (trends_tr.get("Montée") or {}).get("slope_pts_per_h"),
                        "vap_last_montee": (trends_tr.get("Montée") or {}).get("last_pct"),
                        "vap_slope_roulant": (trends_tr.get("Relief roulant") or {}).get("slope_pts_per_h"),
                        "vap_last_roulant": (trends_tr.get("Relief roulant") or {}).get("last_pct"),
                        "vap_slope_descente": (trends_tr.get("Descente") or {}).get("slope_pts_per_h"),
                        "vap_last_descente": (trends_tr.get("Descente") or {}).get("last_pct"),
                    })
                    st.session_state["_trail_portions"] = [
                        {"Portion": p["label"], "Terrain": p["famille"],
                         "Début (effort)": seconds_to_hms(p["t_start_s"]), "Durée": seconds_to_hms(p["dur_s"]),
                         "Dist (km)": round(p["dist_m"] / 1000.0, 2), "D+ (m)": round(p["d_plus"]),
                         "Pente méd. (%)": round(p["grade_med"], 1),
                         "Vitesse (km/h)": round(p["speed_kmh"], 2), "VAP (km/h)": round(p["vap_kmh"], 2),
                         "% réf. famille": round(p["index_pct"], 1)} for p in portions_tr]
                    _kpis = []
                    for _fam, _color, _pref in TERRAIN_FAMILIES:
                        _tr = trends_tr.get(_fam, {})
                        if not _tr or _tr.get("last_pct") is None:
                            _kpis.append((_fam, "—", "aucune portion qualifiée"))
                            continue
                        if _tr.get("slope_pts_per_h") is not None:
                            _sub = (f"{_tr['last_pct'] - 100:+.1f} pts vs référence · "
                                    f"{_tr['slope_pts_per_h']:+.1f} pts/h · {_tr['n']} portions")
                        else:
                            _sub = f"{_tr['n']} portion — tendance non calculable"
                        _kpis.append((f"{_fam} — {_tr['last_label']}", f"{_tr['last_pct']:.0f} %", _sub))
                    kpi_row(_kpis)
                    fig_terr = plot_terrain_evolution(portions_tr, trends_tr)
                    st.pyplot(fig_terr); plt.close(fig_terr)

                    _worst = None
                    for _fam, _c, _p in TERRAIN_FAMILIES:
                        _tr = trends_tr.get(_fam, {})
                        if _tr.get("last_pct") is not None and _tr.get("n", 0) >= 2:
                            if _worst is None or _tr["last_pct"] < _worst[1]["last_pct"]:
                                _worst = (_fam, _tr)
                    if _worst:
                        _worst_fam, _worst_tr = _worst
                        st.info(f"**{_worst_fam} · plus grand écart observé** — dernière portion comparable "
                                f"({_worst_tr['last_label']}) : **{_worst_tr['last_pct']:.1f} %** du niveau de "
                                f"référence, soit **{_worst_tr['last_pct'] - 100:+.1f} points**.")

                    rows_tr = []
                    for p in portions_tr:
                        rows_tr.append({
                            "Portion": p["label"], "Terrain": p["famille"],
                            "Début (effort)": seconds_to_hms(p["t_start_s"]),
                            "Durée": seconds_to_hms(p["dur_s"]),
                            "Dist (km)": round(p["dist_m"] / 1000.0, 2),
                            "D+ (m)": round(p["d_plus"]), "D- (m)": round(p["d_moins"]),
                            "Pente méd. (%)": round(p["grade_med"], 1),
                            "Vitesse réelle (km/h)": round(p["speed_kmh"], 2),
                            "Allure réelle": pace_str(3600.0 / max(0.1, p["speed_kmh"])) + "/km",
                            "VAP (km/h)": round(p["vap_kmh"], 2),
                            "VAP (allure)": pace_str(3600.0 / max(0.1, p["vap_kmh"])) + "/km",
                            "% réf. famille": round(p["index_pct"], 1),
                            "FC moy.": round(p["hr"]) if p.get("hr") else "—",
                            "% marche": round(p["pct_walk"]) if p.get("pct_walk") is not None else "—",
                        })
                    df_tr = pd.DataFrame(rows_tr)
                    st.dataframe(df_tr, use_container_width=True, hide_index=True)
                    st.download_button("⬇️ Portions de terrain (CSV)",
                                       data=df_tr.to_csv(index=False).encode("utf-8"),
                                       file_name=f"portions_terrain_{train_file.name.split('.')[0]}.csv",
                                       mime="text/csv", key="dl_portions_terrain")
                    st.caption("⚠️ La VAP repose sur un modèle de coût énergétique (Minetti) : elle compare des "
                               "efforts, pas des chronos. Une baisse d'index peut aussi venir du terrain "
                               "(technicité, sol gras, nuit) ou d'un choix de gestion, pas seulement de la fatigue.")

            st.markdown("---")
            st.subheader("📐 Analyse d'intervalles")
            st.caption("Découpe la séance en intervalles (efforts + récupérations) pour comparer les répétitions.")

            if "n_intervals" not in st.session_state:
                st.session_state.n_intervals = 2
            ci1, ci2 = st.columns(2)
            with ci1:
                if st.button("➕ Ajouter intervalle", key="btn_add_int") and st.session_state.n_intervals < 10:
                    st.session_state.n_intervals += 1
            with ci2:
                if st.button("➖ Retirer", key="btn_rm_int") and st.session_state.n_intervals > 1:
                    st.session_state.n_intervals -= 1

            interval_defs = []
            for i in range(1, st.session_state.n_intervals + 1):
                with st.expander(f"Intervalle {i}", expanded=(i <= 3)):
                    ci_cols = st.columns(3)
                    int_name  = ci_cols[0].text_input("Nom", value=f"Répét {i}", key=f"int_name_{i}")
                    int_start = hms_input("Début", default="0:00:00", key=f"int_start_{i}")
                    int_end   = hms_input("Fin",   default="0:05:00", key=f"int_end_{i}")
                    interval_defs.append({"name": int_name, "start": int_start, "end": int_end})

            if st.button("📊 Analyser les intervalles", key="btn_analyze_int"):
                int_results = []
                for idef in interval_defs:
                    df_int = extract_interval_df(df_train, idef["start"], idef["end"])
                    res_int = analyze_interval(df_int, idef["name"])
                    int_results.append(res_int)
                st.session_state["int_results"] = int_results

            if "int_results" in st.session_state:
                int_results = st.session_state["int_results"]
                valid_ints = [r for r in int_results if r.get("valid")]
                if valid_ints:
                    rows_int = []
                    for r in valid_ints:
                        rows_int.append({
                            "Intervalle":  r["name"],
                            "Durée":       seconds_to_hms(r["dur_s"]),
                            "Distance":    f"{r['dist_m']:.0f} m" if r["dist_m"] else "—",
                            "Allure moy.": pace_str(r["dur_s"]/(r["dist_m"]/1000)) if r["dist_m"] and r["dist_m"]>0 else "—",
                            "FC max":      f"{r['hr']['fc_max']} bpm" if r["hr"].get("fc_max") else "—",
                            "FC moy.":     f"{r['hr']['fc_avg']} bpm" if r["hr"].get("fc_avg") else "—",
                            "Dérive FC":   f"{r['hr']['drift_abs']:+.1f} bpm" if r["hr"].get("drift_abs") is not None else "—",
                        })
                    st.dataframe(pd.DataFrame(rows_int), use_container_width=True, hide_index=True)

                    fig_int, ax_int = plt.subplots(figsize=(11, 4))
                    colors_int = CHART_CYCLE
                    for ci_int, r in enumerate(valid_ints):
                        if "df" not in r: continue
                        df_i = r["df"]
                        if "heart_rate" not in df_i.columns or "elapsed_s" not in df_i.columns: continue
                        hr_i = df_i["heart_rate"].dropna()
                        hr_i = hr_i[(hr_i >= 40) & (hr_i <= 220)]
                        if len(hr_i) < 5: continue
                        t_i = df_i.loc[hr_i.index, "elapsed_s"].values
                        hr_sm_i = smooth_hr(hr_i).values
                        ax_int.plot(t_i, hr_sm_i, lw=2, color=colors_int[ci_int % len(colors_int)], label=r["name"])
                    ax_int.set_xlabel("Temps dans l'intervalle (s)")
                    ax_int.set_ylabel("FC (bpm)")
                    ax_int.set_title("FC comparée entre intervalles")
                    ax_int.legend(fontsize=8); ax_int.grid(alpha=0.3); fig_int.tight_layout()
                    st.pyplot(fig_int); plt.close(fig_int)

            st.markdown("---")
            st.subheader("📋 Export des données")
            csv_train = df_train.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger les données de séance (CSV)",
                               data=csv_train,
                               file_name=f"seance_{train_file.name.split('.')[0]}.csv",
                               mime="text/csv")

            # ── v8.9 : enregistrement de la séance dans l'historique ─────
            st.markdown("---")
            st.subheader("💾 Enregistrer cette séance")
            if history_ready():
                _ts = st.session_state.get("_trail_summary", {})
                _hr_s = hr_stats_train if isinstance(hr_stats_train, dict) else {}
                with st.form("save_workout_form"):
                    _sw1, _sw2, _sw3 = st.columns(3)
                    _wk_date = _sw1.date_input("Date de la séance", value=date.today(), key="save_wk_date")
                    _wk_name = _sw2.text_input("Nom de la séance",
                                               value=train_file.name.rsplit(".", 1)[0], key="save_wk_name")
                    _cat_list = list_categories(current_user()["id"])
                    _cat_names = [c["name"] for c in _cat_list] or ["Autre"]
                    _wk_type = _sw3.selectbox("Catégorie de séance", _cat_names, key="save_wk_type",
                                              help="Catégories libres, gérées dans 📚 Historique → Séances.")
                    _wk_newcat = st.text_input("…ou nouvelle catégorie (laisse vide pour utiliser celle "
                                               "sélectionnée)", key="save_wk_newcat")
                    _wk_tags = st.text_input("Tags libres (séparés par des virgules)", key="save_wk_tags",
                                             placeholder="ex : chaleur, nuit, sac 5 kg, spécifique UTMB")
                    _wk_notes = st.text_area("Notes / ressenti", key="save_wk_notes")
                    _wk_incl = st.checkbox("Inclure le détail des portions de terrain et des intervalles",
                                           value=True, key="save_wk_incl")
                    if st.form_submit_button("💾 Enregistrer dans l'historique"):
                        _summary_wk = {
                            "duration_s": dur_s_train, "distance_m": dist_m_train,
                            "d_plus": _ts.get("d_plus"),
                            "hr_avg": _hr_s.get("fc_avg"), "hr_max": _hr_s.get("fc_max"),
                            "hr_drift": _hr_s.get("drift_abs"),
                            "pct_walk": _ts.get("pct_walk"), "trans_lo": _ts.get("trans_lo"),
                            "trans_mid": _ts.get("trans_mid"), "trans_hi": _ts.get("trans_hi"),
                            "vap_slope_montee": _ts.get("vap_slope_montee"),
                            "vap_last_montee": _ts.get("vap_last_montee"),
                        }
                        _intervals_save = []
                        for _ri in st.session_state.get("int_results", []):
                            if not _ri.get("valid"):
                                continue
                            _hri = _ri.get("hr", {}) or {}
                            _intervals_save.append({
                                "Intervalle": _ri.get("name"), "Durée": seconds_to_hms(_ri.get("dur_s", 0)),
                                "Distance (m)": round(_ri["dist_m"]) if _ri.get("dist_m") else None,
                                "Allure": (pace_str(_ri["dur_s"] / (_ri["dist_m"] / 1000.0)) + "/km")
                                          if _ri.get("dist_m") else "—",
                                "FC moy": _hri.get("fc_avg"), "FC max": _hri.get("fc_max"),
                                "Dérive FC": _hri.get("drift_abs"),
                            })
                        _cat_final = (_wk_newcat or "").strip() or _wk_type
                        _cat_id = next((c["id"] for c in _cat_list if c["name"] == _cat_final), None)
                        if _cat_id is None and _cat_final:
                            _cat_id, _err_cat = create_category(current_user()["id"], _cat_final)
                        _wid = save_workout(
                            current_athlete_id(), _wk_date.isoformat(), _wk_name, _cat_final,
                            train_file.name, _summary_wk,
                            st.session_state.get("_trail_portions", []) if _wk_incl else [],
                            _intervals_save if _wk_incl else [],
                            {k: v for k, v in _ts.items() if k not in _summary_wk}, _wk_notes,
                            category_id=_cat_id, tags=_wk_tags,
                            zones=st.session_state.get("_session_zones", []),
                            records=st.session_state.get("_session_records", []),
                            hr_max_used=st.session_state.get("_session_hr_max_used"),
                            quarters=st.session_state.get("_session_quarters", []),
                            zones_vc=st.session_state.get("_session_zones_vc", []),
                            vc_ref_ms=st.session_state.get("_session_vc_ref"))
                        st.success(f"✅ Séance « {_wk_name} » enregistrée pour {current_athlete_name()} "
                                   f"(#{_wid}) — comparaison dans l'onglet 📚 Historique.")
                st.caption("Les intervalles sont enregistrés tels que découpés ci-dessus : lance "
                           "« Analyser les intervalles » avant d'enregistrer pour les inclure.")
            else:
                save_gate("cette séance")


# ══════════════════════════════════════════════════════════════
# ONGLET 3 — ANALYSE DE COHORTE
# ══════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.title("👥 Analyse de cohorte")
    st.caption("Centralise les splits de plusieurs coureurs sur une même course (Strava ou Live-trail), compare-les entre eux, et confronte-les à l'algorithme de prédiction calibré v8.1.")

    if "cohort_course" not in st.session_state:
        st.session_state["cohort_course"] = {"name": "", "date": date.today(), "dist": 0.0, "lat": 0.0, "lon": 0.0}
    if "cohort_athletes" not in st.session_state:
        st.session_state["cohort_athletes"] = []
    if "cohort_checkpoints" not in st.session_state:
        st.session_state["cohort_checkpoints"] = []
    if "cohort_weather" not in st.session_state:
        st.session_state["cohort_weather"] = None
    if "cohort_athlete_id_counter" not in st.session_state:
        st.session_state["cohort_athlete_id_counter"] = 0
    if "cohort_cp_id_counter" not in st.session_state:
        st.session_state["cohort_cp_id_counter"] = 0
    if "cohort_add_msg" not in st.session_state:
        st.session_state["cohort_add_msg"] = ""

    cohort_subtabs = st.tabs(["🏁 Course", "🏃 Athlètes", "📍 Checkpoints", "📊 Analyse", "🤖 Analyse de course"])

    # ── Sous-onglet : Course ──
    with cohort_subtabs[0]:
        co1, co2 = st.columns(2)
        st.session_state["cohort_course"]["name"] = co1.text_input(
            "Nom de la course", value=st.session_state["cohort_course"]["name"], key="cohort_course_name", placeholder="ex: UTMB 2025")
        st.session_state["cohort_course"]["date"] = co2.date_input(
            "Date", value=st.session_state["cohort_course"]["date"], key="cohort_course_date")
        co3, co4, co5 = st.columns(3)
        st.session_state["cohort_course"]["dist"] = co3.number_input(
            "Distance (km)", value=float(st.session_state["cohort_course"]["dist"]), key="cohort_course_dist")
        st.session_state["cohort_course"]["lat"] = co4.number_input(
            "Latitude départ", value=float(st.session_state["cohort_course"]["lat"]), format="%.4f", key="cohort_course_lat")
        st.session_state["cohort_course"]["lon"] = co5.number_input(
            "Longitude départ", value=float(st.session_state["cohort_course"]["lon"]), format="%.4f", key="cohort_course_lon")

        if st.button("🌡️ Récupérer la météo", key="cohort_fetch_weather"):
            cc = st.session_state["cohort_course"]
            if cc["lat"] and cc["lon"]:
                with st.spinner("Récupération météo..."):
                    st.session_state["cohort_weather"] = fetch_daily_weather(cc["lat"], cc["lon"], cc["date"])
            else:
                st.warning("Renseigne latitude/longitude.")

        w = st.session_state.get("cohort_weather")
        if w:
            st.markdown("##### Conditions météo")
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("T° max", f"{w['tmax']:.0f}°C" if w.get("tmax") is not None else "—")
            wc2.metric("T° min", f"{w['tmin']:.0f}°C" if w.get("tmin") is not None else "—")
            wc3.metric("Précip.", f"{w['precip']:.1f} mm" if w.get("precip") is not None else "—")
            wc4.metric("Vent max", f"{w['wind']:.0f} km/h" if w.get("wind") is not None else "—")
            st.caption("Source : archive (année précédente, même date)" if w.get("isPast") else "Source : prévisions Open-Meteo")
            if w.get("tmax") is not None and w.get("tmin") is not None:
                tmoy = (w["tmax"] + w["tmin"]) / 2.0
                d_t = tmoy - 10.0
                weather_pen_cohort = min(0.20, d_t * 0.008) if d_t > 0 else max(-0.06, d_t * 0.004)
                bg = C_SURFACE
                st.markdown(f'<div class="note-box note-red" style="background:{bg};">'
                            f'T° moy. estimée <strong>{tmoy:.1f}°C</strong> — référence optimale 10°C — '
                            f'impact estimé <strong>{weather_pen_cohort*100:+.1f}%</strong> sur les temps</div>', unsafe_allow_html=True)

    # ── Sous-onglet : Athlètes ──
    with cohort_subtabs[1]:
        if st.session_state.get("cohort_add_msg"):
            st.success(st.session_state["cohort_add_msg"])
            st.session_state["cohort_add_msg"] = ""

        with st.expander("➕ Ajouter un athlète", expanded=True):
            # Le sélecteur de format reste HORS du formulaire pour que le texte
            # d'exemple ci-dessous se mette à jour immédiatement quand on le change.
            cohort_input_mode = st.radio("Format des données", ["Splits Strava", "Live-trail"], horizontal=True, key="cohort_input_mode")

            # clear_on_submit=True est le mécanisme officiel Streamlit pour vider
            # les champs après validation — affecter directement st.session_state
            # à une clé de widget déjà instancié dans le run lève une
            # StreamlitAPIException, d'où l'usage d'un formulaire ici plutôt que
            # d'une réinitialisation manuelle.
            with st.form("cohort_add_athlete_form", clear_on_submit=True):
                ac1, ac2 = st.columns(2)
                cohort_new_name = ac1.text_input("Nom", key="cohort_new_name", placeholder="ex: Thomas B. ou Athlète A")
                _default_color = COHORT_PALETTE[len(st.session_state["cohort_athletes"]) % len(COHORT_PALETTE)]
                cohort_new_color = ac2.color_picker("Couleur", value=_default_color, key="cohort_new_color")
                cohort_new_splits = st.text_area(
                    "Données (coller le texte brut)", key="cohort_new_splits", height=150,
                    placeholder=("  1\t1,00km\t5:50\t5:50/km\t6 m\t146 bpm\n  2\t1,00km\t6:31\t6:31/km\t91 m\t167 bpm\n  ..."
                                 if cohort_input_mode == "Splits Strava"
                                 else "3.1 km\n20.1 km\nArzon Port Navalo\njeu. 13:22\n1:22:20\n14.6 km/h\n202 m+\n..."))
                cohort_submitted = st.form_submit_button("➕ Ajouter l'athlète")

            if cohort_submitted:
                if not cohort_new_splits.strip():
                    st.warning("Colle les données d'abord.")
                elif cohort_input_mode == "Live-trail":
                    res = parse_itra(cohort_new_splits)
                    if len(res["cps"]) < 2:
                        st.error("Format non reconnu — vérifie les données.")
                    else:
                        st.session_state["cohort_athlete_id_counter"] += 1
                        name = cohort_new_name.strip() or f"Athlète {len(st.session_state['cohort_athletes'])+1}"
                        st.session_state["cohort_athletes"].append({
                            "id": st.session_state["cohort_athlete_id_counter"], "name": name, "color": cohort_new_color,
                            "splits": res["splits"], "totalSecs": res["cps"][-1]["cumSecs"], "totalKm": res["cps"][-1]["cumDist"]})
                        if not st.session_state["cohort_checkpoints"]:
                            for cp in res["cps"]:
                                st.session_state["cohort_cp_id_counter"] += 1
                                st.session_state["cohort_checkpoints"].append(
                                    {"id": st.session_state["cohort_cp_id_counter"], "name": cp["name"], "km": cp["cumDist"]})
                        st.session_state["cohort_add_msg"] = f"✅ {name} importé !"
                        st.rerun()
                else:
                    splits = parse_splits_strava(cohort_new_splits)
                    if len(splits) < 2:
                        st.error("Format non reconnu — vérifie les données.")
                    else:
                        st.session_state["cohort_athlete_id_counter"] += 1
                        name = cohort_new_name.strip() or f"Athlète {len(st.session_state['cohort_athletes'])+1}"
                        st.session_state["cohort_athletes"].append({
                            "id": st.session_state["cohort_athlete_id_counter"], "name": name, "color": cohort_new_color,
                            "splits": splits, "totalSecs": sum(sp["secs"] for sp in splits), "totalKm": sum(sp["dist"] for sp in splits)})
                        st.session_state["cohort_add_msg"] = f"✅ {name} ajouté !"
                        st.rerun()

        if not st.session_state["cohort_athletes"]:
            st.caption("Aucun athlète pour l'instant.")
        else:
            for a in st.session_state["cohort_athletes"]:
                avg_pace_a = a["totalSecs"] / max(a["totalKm"], 0.1)
                col_a, col_b = st.columns([5, 1])
                col_a.markdown(
                    f"<span style='color:{a['color']};font-size:1.1em'>●</span> **{a['name']}** — "
                    f"{a['totalKm']:.1f} km · {seconds_to_hms(a['totalSecs'])} · {pace_str(avg_pace_a)}/km",
                    unsafe_allow_html=True)
                if col_b.button("Supprimer", key=f"cohort_del_athlete_{a['id']}"):
                    st.session_state["cohort_athletes"] = [x for x in st.session_state["cohort_athletes"] if x["id"] != a["id"]]
                    st.rerun()

    # ── Sous-onglet : Checkpoints ──
    with cohort_subtabs[2]:
        with st.expander("➕ Ajouter un checkpoint", expanded=True):
            kc1, kc2 = st.columns([3, 2])
            cohort_cp_name = kc1.text_input("Nom du checkpoint", key="cohort_cp_name", placeholder="ex: Col du Bonhomme")
            cohort_cp_km = kc2.number_input("Kilomètre", min_value=0.0, value=0.0, step=0.5, key="cohort_cp_km")
            if st.button("➕ Ajouter", key="cohort_add_cp"):
                if cohort_cp_name.strip():
                    st.session_state["cohort_cp_id_counter"] += 1
                    st.session_state["cohort_checkpoints"].append(
                        {"id": st.session_state["cohort_cp_id_counter"], "name": cohort_cp_name.strip(), "km": cohort_cp_km})
                    st.session_state["cohort_checkpoints"].sort(key=lambda c: c["km"])
                    st.rerun()
        if not st.session_state["cohort_checkpoints"]:
            st.caption("Aucun checkpoint défini.")
        else:
            for cp in st.session_state["cohort_checkpoints"]:
                kk1, kk2, kk3 = st.columns([4, 2, 1])
                kk1.write(cp["name"]); kk2.write(f"km {cp['km']:.1f}")
                if kk3.button("Retirer", key=f"cohort_del_cp_{cp['id']}"):
                    st.session_state["cohort_checkpoints"] = [x for x in st.session_state["cohort_checkpoints"] if x["id"] != cp["id"]]
                    st.rerun()

    # ── Sous-onglet : Analyse (cohorte) ──
    with cohort_subtabs[3]:
        cohort_athletes_list = st.session_state["cohort_athletes"]
        cohort_checkpoints_list = st.session_state["cohort_checkpoints"]
        if not cohort_athletes_list:
            st.caption("Ajoute au moins un athlète pour lancer l'analyse.")
        else:
            cohort_weather_pen = None
            w = st.session_state.get("cohort_weather")
            if w and w.get("tmax") is not None and w.get("tmin") is not None:
                tmoy = (w["tmax"] + w["tmin"]) / 2.0
                d_t = tmoy - 10.0
                cohort_weather_pen = min(0.20, d_t * 0.008) if d_t > 0 else max(-0.06, d_t * 0.004)

            st.markdown("#### Aperçu global")
            cols_overview = st.columns(len(cohort_athletes_list))
            for col, a in zip(cols_overview, cohort_athletes_list):
                avg_pace_a = a["totalSecs"] / max(a["totalKm"], 0.1)
                with col:
                    st.markdown(f"<span style='color:{a['color']}'>●</span> {a['name']}", unsafe_allow_html=True)
                    st.metric("Temps", seconds_to_hms(a["totalSecs"]))
                    st.caption(f"{pace_str(avg_pace_a)}/km · {a['totalKm']:.1f} km")
                    if cohort_weather_pen is not None:
                        corrected = a["totalSecs"] * (1 + cohort_weather_pen)
                        st.caption(f"corrigé : {seconds_to_hms(corrected)}")
            apercu_rows = [{"Athlète": a["name"], "Distance (km)": round(a["totalKm"],1), "Temps total": seconds_to_hms(a["totalSecs"]),
                            "Allure moy (min/km)": pace_str(a["totalSecs"]/max(a["totalKm"],0.1))} for a in cohort_athletes_list]
            st.download_button("⬇️ Aperçu (CSV)", pd.DataFrame(apercu_rows).to_csv(index=False).encode("utf-8"),
                               file_name=f"apercu_{st.session_state['cohort_course']['name'] or 'course'}.csv", key="cohort_dl_apercu")

            st.markdown("#### Allure km par km")
            st.pyplot(plot_cohort_pace_chart(cohort_athletes_list, cohort_checkpoints_list))

            if cohort_checkpoints_list:
                st.markdown("#### Temps de passage aux checkpoints")
                rows_cp = []
                for cp in cohort_checkpoints_list:
                    times = [get_time_at_km(a, cp["km"]) for a in cohort_athletes_list]
                    min_t = min(times)
                    row = {"Checkpoint": cp["name"], "Km": cp["km"]}
                    for a, t in zip(cohort_athletes_list, times):
                        row[a["name"]] = seconds_to_hms(t) + (" (1er)" if t == min_t else f" (+{seconds_to_hms(t-min_t)})")
                    row["Moyenne"] = seconds_to_hms(sum(times)/len(times))
                    rows_cp.append(row)
                st.dataframe(pd.DataFrame(rows_cp), use_container_width=True, hide_index=True)
                st.pyplot(plot_cohort_cp_chart(cohort_athletes_list, cohort_checkpoints_list))

            st.markdown("#### Détail par athlète")
            for a in cohort_athletes_list:
                avg_pace_a = a["totalSecs"] / max(a["totalKm"], 0.1)
                with st.expander(f"{a['name']} — {pace_str(avg_pace_a)}/km · {seconds_to_hms(a['totalSecs'])}"):
                    rows_d = []
                    cum_km_a = 0.0
                    for sp in a["splits"]:
                        cum_km_a += sp["dist"]
                        pace_sp = sp["secs"] / sp["dist"]
                        diff_sp = pace_sp - avg_pace_a
                        rows_d.append({"Segment": sp["km"], "Km cumulé": round(cum_km_a,1), "Dist (km)": round(sp["dist"],2),
                                       "Allure": pace_str(pace_sp)+"/km",
                                       "D+ seg (m)": sp["elev"], "FC": sp["hr"] if sp["hr"] else "—",
                                       "Écart vs allure moy.": ("+" if diff_sp>0 else "")+pace_str(abs(diff_sp))})
                    df_d = pd.DataFrame(rows_d)
                    st.dataframe(df_d, use_container_width=True, hide_index=True, column_config={
                        "Écart vs allure moy.": st.column_config.TextColumn(
                            help="Différence entre l'allure de ce segment et l'allure MOYENNE de cet athlète sur toute "
                                 "la course (pas l'allure des autres athlètes). Négatif = segment couru plus vite que "
                                 "sa propre moyenne, positif = plus lent.")
                    })
                    st.download_button("⬇️ Splits (CSV)", df_d.to_csv(index=False).encode("utf-8"),
                                       file_name=f"splits_{a['name']}.csv", key=f"cohort_dl_splits_{a['id']}")

    # ── Sous-onglet : Analyse de course (vs algorithme) ──
    with cohort_subtabs[4]:
        cohort_athletes_list = st.session_state["cohort_athletes"]
        cohort_checkpoints_list = st.session_state["cohort_checkpoints"]
        if not cohort_athletes_list:
            st.caption("Ajoute au moins un athlète pour comparer à l'algorithme.")
        else:
            ccm1, ccm2 = st.columns(2)
            cohort_athlete_names = [a["name"] for a in cohort_athletes_list]
            cohort_sel_name = ccm1.selectbox("Coureur à comparer", cohort_athlete_names, key="cohort_comp_athlete")
            cohort_selected = next(a for a in cohort_athletes_list if a["name"] == cohort_sel_name)
            cohort_profile_key = ccm2.selectbox("Profil terrain (calibré v8.1)", list(TERRAIN_PROFILES.keys()), index=2, key="cohort_comp_profile")

            ccm3, ccm4 = st.columns(2)
            cohort_mode_label = ccm3.selectbox(
                "Calage de l'allure",
                ["Recalée sur le temps réel (compare la répartition)", "Allure de base manuelle"],
                key="cohort_comp_mode")
            cohort_mode_key = "cale" if "Recalée" in cohort_mode_label else "manuel"
            cohort_manual_pace = None
            if cohort_mode_key == "manuel":
                cohort_manual_pace_str = ccm4.text_input("Allure de base (mm:ss/km, plat)", value="5:30", key="cohort_comp_pace")
                _mm = re.match(r"^(\d+):(\d{2})$", cohort_manual_pace_str.strip())
                cohort_manual_pace = (int(_mm.group(1)) * 60 + int(_mm.group(2))) if _mm else None

            ccm5, ccm6 = st.columns(2)
            cohort_apply_fatigue = ccm5.checkbox("Appliquer la fatigue", value=True, key="cohort_comp_fatigue")
            w = st.session_state.get("cohort_weather")
            cohort_temp_c = (w["tmax"] + w["tmin"]) / 2.0 if (w and w.get("tmax") is not None and w.get("tmin") is not None) else None
            cohort_apply_temp = ccm6.checkbox("Appliquer la météo", value=True, disabled=(cohort_temp_c is None), key="cohort_comp_temp")
            if cohort_temp_c is None:
                st.caption("Récupère la météo dans le sous-onglet 🏁 Course pour activer cette option.")

            st.caption("⚠️ D+/D- estimé à partir du dénivelé NET entre checkpoints (pas du détail GPX) — sous-estime "
                       "l'effort sur les tronçons très vallonnés. Coefficients calibrés sur 1245 segments coureurs "
                       "réels (12 ultra-trails, top10) — voir l'en-tête du fichier pour la méthodologie v8.1.")

            cohort_predicted = build_prediction_cohort(
                cohort_selected, cohort_profile_key, cohort_apply_fatigue,
                cohort_apply_temp and cohort_temp_c is not None, cohort_temp_c, cohort_mode_key, cohort_manual_pace)
            cohort_predicted_athlete = {"id": "predicted", "name": "Prédiction algo", "color": "#94a3b8",
                                        "splits": cohort_predicted["splits"], "totalSecs": cohort_predicted["totalSecs"],
                                        "totalKm": cohort_predicted["totalKm"], "dashed": True}

            cohort_delta = cohort_selected["totalSecs"] - cohort_predicted["totalSecs"]
            cohort_comparison = []
            _cum_km_comp = 0.0
            for sp, ps in zip(cohort_selected["splits"], cohort_predicted["splits"]):
                _cum_km_comp += sp["dist"]
                real_pace = sp["secs"] / sp["dist"]
                pred_pace = ps["secs"] / ps["dist"]
                cohort_comparison.append({"km": sp["km"], "cum_km": round(_cum_km_comp,1), "dist": sp["dist"], "elev": sp["elev"],
                                          "real_pace": real_pace, "pred_pace": pred_pace, "diff": real_pace - pred_pace,
                                          "gm": ps["gm"], "fm": ps["fm"], "tm": ps["tm"]})

            cohort_mape = (sum(abs(c["diff"]) / max(1.0, c["pred_pace"]) for c in cohort_comparison) / len(cohort_comparison) * 100) if cohort_comparison else 0.0

            _third = max(1, len(cohort_comparison) // 3)
            _first_third = cohort_comparison[:_third]; _last_third = cohort_comparison[-_third:]
            def _avg_diff(lst): return sum(c["diff"] for c in lst) / len(lst) if lst else 0.0
            _diff_start = _avg_diff(_first_third); _diff_end = _avg_diff(_last_third)
            if _diff_start < -8 and _diff_end > 8:
                cohort_trend = (f"🟠 Départ plus rapide que recommandé ({pace_str(abs(_diff_start))}/km plus vite en "
                                f"moyenne sur le 1er tiers), avec une perte de rythme en fin de course "
                                f"({pace_str(_diff_end)}/km plus lent sur le dernier tiers) — signature classique "
                                f"d'un départ trop ambitieux.")
            elif _diff_start > 8 and _diff_end < -8:
                cohort_trend = "🔵 Départ plus prudent que recommandé, avec une accélération en fin de course — marge de progression possible sur la première partie."
            elif abs(_diff_start) > 8 or abs(_diff_end) > 8:
                cohort_trend = "🟡 Écart notable par rapport à la courbe recommandée par l'algorithme sur certaines portions — voir le détail par segment ci-dessous."
            else:
                cohort_trend = "🟢 La répartition d'effort réelle suit d'assez près la courbe recommandée par l'algorithme."

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Temps réel", seconds_to_hms(cohort_selected["totalSecs"]))
            mc2.metric("Temps prédit (algo)", seconds_to_hms(cohort_predicted["totalSecs"]))
            mc3.metric("Écart total", ("+" if cohort_delta > 0 else "-") + seconds_to_hms(abs(cohort_delta)))
            mc4.metric("Écart moy. par segment", f"{cohort_mape:.1f} %")
            st.info(cohort_trend)

            st.markdown("#### Allure réelle vs recommandée par l'algo")
            st.pyplot(plot_cohort_pace_chart([cohort_selected, cohort_predicted_athlete], cohort_checkpoints_list))

            if cohort_checkpoints_list:
                st.markdown("#### Temps aux checkpoints — réel vs prédit")
                st.pyplot(plot_cohort_cp_chart([cohort_selected, cohort_predicted_athlete], cohort_checkpoints_list))
                rows_cpc = []
                for cp in cohort_checkpoints_list:
                    t_real = get_time_at_km(cohort_selected, cp["km"]); t_pred = get_time_at_km(cohort_predicted_athlete, cp["km"])
                    diff_cp = t_real - t_pred
                    rows_cpc.append({"Checkpoint": cp["name"], "Km": cp["km"], "Réel": seconds_to_hms(t_real),
                                     "Prédit": seconds_to_hms(t_pred), "Écart": ("+" if diff_cp>0 else "-")+seconds_to_hms(abs(diff_cp))})
                st.dataframe(pd.DataFrame(rows_cpc), use_container_width=True, hide_index=True)

            st.markdown("#### Détail par segment")
            rows_seg = []
            for c in cohort_comparison:
                rows_seg.append({"Segment": c["km"], "Km cumulé": c["cum_km"], "Dist (km)": round(c["dist"],2), "Allure réelle": pace_str(c["real_pace"])+"/km",
                                 "Allure prédite": pace_str(c["pred_pace"])+"/km",
                                 "Écart": ("+" if c["diff"]>0 else "")+pace_str(abs(c["diff"])),
                                 "Mult pente": round(c["gm"],3), "Mult fatigue": round(c["fm"],3), "Mult météo": round(c["tm"],3)})
            df_seg = pd.DataFrame(rows_seg)
            st.dataframe(df_seg, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Comparaison détaillée (CSV)", df_seg.to_csv(index=False).encode("utf-8"),
                               file_name=f"comparaison_algo_{cohort_selected['name']}.csv", key="cohort_dl_comparison")

            st.markdown("---")
            st.markdown("#### 🔬 Calibrer un profil sur cette course")
            st.caption(
                "Utilise TOUS les athlètes de la cohorte (pas seulement celui sélectionné plus haut) pour ajuster "
                "k_up, le plafond de pente et le modèle de fatigue spécifiquement sur cette course — même méthode "
                "(régression log-allure, effets coureur profilés analytiquement) que la calibration v8.1 d'origine "
                "(1245 segments, 12 courses), appliquée ici à une seule course. Le cap descente, la sensibilité "
                "météo et les paramètres secondaires restent ceux validés globalement — non ré-estimables de façon "
                "fiable à partir d'une seule course, même avec une cohorte complète.")

            if len(cohort_athletes_list) < 2:
                st.info("Ajoute au moins 2 athlètes dans cette cohorte pour pouvoir calibrer un profil sur cette course.")
            else:
                if st.button("🔬 Calibrer un profil sur cette course", key="cohort_calibrate_btn"):
                    with st.spinner("Calibration en cours..."):
                        st.session_state["cohort_fit_result"] = fit_cohort_profile(cohort_athletes_list)

                _fit_result = st.session_state.get("cohort_fit_result")
                if _fit_result is None and "cohort_fit_result" in st.session_state:
                    st.warning("Pas assez de données pour une calibration fiable (minimum recommandé : 2 athlètes "
                               "et ~10 segments cumulés). Ajoute des athlètes ou attends d'avoir plus de checkpoints.")
                elif _fit_result:
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("k_up calibré", f"{_fit_result['k_up']:.1f}")
                    fc2.metric("Plafond pente", f"{_fit_result['max_cap']*100:.0f} %")
                    fc3.metric("Seuil fatigue", f"{_fit_result['fatigue_threshold']:.0f} %")
                    fc4.metric("Taux fatigue", f"{_fit_result['fatigue_rate']:.0f} %")
                    st.caption(f"R² = {_fit_result['r2']:.2f} · {_fit_result['n_athletes']} athlètes · {_fit_result['n_segments']} segments utilisés.")
                    st.caption("ℹ️ k_up et le plafond de pente sont en général bien identifiés par cette méthode. "
                               "Le seuil et le taux de fatigue, eux, peuvent se compenser l'un l'autre (plusieurs "
                               "combinaisons donnent une courbe de dégradation très proche) même quand le R² est "
                               "élevé — fie-toi surtout au temps total prédit pour juger de la qualité du profil, "
                               "pas à ces deux chiffres pris isolément.")
                    if _fit_result["r2"] < 0.5:
                        st.warning("⚠️ R² faible — calibration peu fiable (cohorte trop petite ou trop homogène en "
                                   "allure/dénivelé pour bien séparer pente et fatigue). Utilisable à titre indicatif.")

                    _default_profile_name = f"📌 {st.session_state['cohort_course']['name'] or 'Course'} (calibré)"
                    cohort_profile_name = st.text_input("Nom du profil à sauvegarder", value=_default_profile_name, key="cohort_profile_name_input")
                    if st.button("💾 Sauvegarder comme profil terrain", key="cohort_save_profile_btn"):
                        if "custom_terrain_profiles" not in st.session_state:
                            st.session_state["custom_terrain_profiles"] = {}
                        st.session_state["custom_terrain_profiles"][cohort_profile_name] = {
                            k: v for k, v in _fit_result.items() if k not in ("r2", "n_segments", "n_athletes")
                        }
                        st.success(f"Profil « {cohort_profile_name} » sauvegardé — sélectionnable dans l'onglet "
                                   f"🏃 Prédiction de course, section 4 Paramètres du modèle.")

                _custom_profiles = st.session_state.get("custom_terrain_profiles", {})
                if _custom_profiles:
                    st.markdown("##### 📚 Profils calibrés sauvegardés (cette session)")
                    for _pname in list(_custom_profiles.keys()):
                        _pc1, _pc2 = st.columns([5, 1])
                        _pc1.write(_pname)
                        if _pc2.button("Suppr.", key=f"cohort_del_profile_{_pname}"):
                            del st.session_state["custom_terrain_profiles"][_pname]
                            st.rerun()
                    st.download_button(
                        "⬇️ Exporter ces profils (JSON)",
                        data=_json.dumps(_custom_profiles, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="profils_calibres.json", key="cohort_export_profiles")
                    st.caption("⚠️ Ces profils sont en mémoire de session uniquement — exporte-les en JSON pour "
                               "les retrouver lors d'une prochaine session (à réimporter ci-dessous, ou directement "
                               "dans l'onglet 🏃 Prédiction, section 4).")

                cohort_import_file = st.file_uploader("⬆️ Importer des profils calibrés (JSON)", type=["json"], key="cohort_import_profiles")
                if cohort_import_file and st.session_state.get("_last_imported_profile_file") != cohort_import_file.name:
                    st.session_state["_last_imported_profile_file"] = cohort_import_file.name
                    try:
                        _imported = _json.loads(cohort_import_file.read().decode("utf-8"))
                        if "custom_terrain_profiles" not in st.session_state:
                            st.session_state["custom_terrain_profiles"] = {}
                        st.session_state["custom_terrain_profiles"].update(_imported)
                        st.success(f"{len(_imported)} profil(s) importé(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fichier JSON invalide : {e}")

# ══════════════════════════════════════════════════════════════
# v8.9 — ONGLET HISTORIQUE
# Lecture seule sur la base : aucun calcul n'est refait ici, on relit ce qui a
# été enregistré au moment de l'analyse.
# ══════════════════════════════════════════════════════════════
with main_tabs[4]:
    st.title("📚 Historique")
    _user = current_user()
    if _user is None:
        st.markdown('<div class="highlight-box">Connecte-toi dans la barre latérale pour retrouver '
                    'l\'historique de tes athlètes : tests de vitesse critique, séances et courses '
                    'restent enregistrés d\'une session à l\'autre.</div>', unsafe_allow_html=True)
        st.caption(f"Les données sont stockées localement dans `{os.path.basename(DB_PATH)}`, "
                   "à côté du script. Aucune donnée n'est envoyée ailleurs.")
    elif current_athlete_id() is None:
        st.info("Crée un athlète dans la barre latérale pour commencer à enregistrer des analyses.")
    else:
        _aid = current_athlete_id()
        _aname = current_athlete_name()
        _vc = list_records("vc_tests", _aid)
        _wk = list_records("workouts", _aid)
        _rc = list_records("races", _aid)
        st.caption(f"Athlète : **{_aname}** · {len(_vc)} test(s) VC · {len(_wk)} séance(s) · {len(_rc)} course(s)")

        _last_vc = _vc[0] if _vc else None
        kpi_row([
            ("Dernière VC", f"{_last_vc['vc_ms']*3.6:.2f} km/h" if (_last_vc and _last_vc.get("vc_ms")) else "—",
             f"{_last_vc['date']} · {pace_str(1000.0/_last_vc['vc_ms'])}/km" if (_last_vc and _last_vc.get("vc_ms")) else "aucun test enregistré"),
            ("Tests VC", str(len(_vc)), "évolution suivie dans le temps"),
            ("Séances", str(len(_wk)), "analyses trail & intervalles"),
            ("Courses & plans", str(len(_rc)), "prédictions enregistrées"),
        ])

        _met = list_records("metabolic_tests", _aid)
        h_t1, h_t2, h_t3, h_t5, h_t4 = st.tabs(["📈 Vitesse critique", "🏃 Séances", "🏁 Courses & plans",
                                                "🍬 Métabolisme & nutrition", "🗄️ Données"])

        # ── Historique VC ────────────────────────────────────────────────
        with h_t1:
            if not _vc:
                st.info("Aucun test enregistré. Onglet 🧪 Tests d'endurance + VC → calcule ta VC → "
                        "« Enregistrer ce test ».")
            else:
                df_vc = pd.DataFrame([{
                    "Date": r["date"], "Test": r["label"] or "—",
                    "VC (km/h)": round(r["vc_ms"] * 3.6, 2) if r["vc_ms"] else None,
                    "Allure VC": pace_str(1000.0 / r["vc_ms"]) + "/km" if r["vc_ms"] else "—",
                    "D' (m)": round(r["d_prime"]) if r["d_prime"] else None,
                    "R²": round(r["r2"], 3) if r["r2"] else None,
                    "K Riegel": round(r["k_riegel"], 3) if r["k_riegel"] else None,
                    "Réfs": r["n_refs"], "Notes": r["notes"] or "", "id": r["id"],
                } for r in _vc])
                st.dataframe(df_vc.drop(columns=["id"]), use_container_width=True, hide_index=True)

                _plot = [r for r in _vc if r.get("vc_ms")]
                if len(_plot) >= 2:
                    _plot = sorted(_plot, key=lambda r: r["date"])
                    _x = [datetime.strptime(r["date"], "%Y-%m-%d") for r in _plot]
                    fig_vc_h, (axa, axb) = plt.subplots(2, 1, figsize=(11.5, 5.2), sharex=True,
                                                        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15})
                    axa.plot(_x, [r["vc_ms"] * 3.6 for r in _plot], "-o", color=C_RED, lw=2, ms=7,
                             mec=C_SURFACE, mew=1.6)
                    for _xi, _r in zip(_x, _plot):
                        axa.annotate(f"{_r['vc_ms']*3.6:.2f}", xy=(_xi, _r["vc_ms"] * 3.6), xytext=(0, 9),
                                     textcoords="offset points", ha="center", fontsize=7.5, color=C_TEXT_MUT)
                    axa.set_ylabel("VC (km/h)")
                    chart_title(axa, "Évolution de la vitesse critique",
                                f"{_aname} · {len(_plot)} tests enregistrés")
                    axb.plot(_x, [r["d_prime"] or 0 for r in _plot], "-o", color=C_WHITE, lw=1.8, ms=6,
                             mec=C_SURFACE, mew=1.5)
                    axb.set_ylabel("D' (m)"); axb.set_xlabel("Date du test")
                    fig_vc_h.autofmt_xdate(rotation=0, ha="center")
                    fig_vc_h.tight_layout()
                    st.pyplot(fig_vc_h); plt.close(fig_vc_h)
                    _delta = (_plot[-1]["vc_ms"] - _plot[0]["vc_ms"]) * 3.6
                    _days = max(1, (_x[-1] - _x[0]).days)
                    st.markdown(f'<div class="note-box note-red">Entre le <b>{_plot[0]["date"]}</b> et le '
                                f'<b>{_plot[-1]["date"]}</b> ({_days} jours), la VC évolue de '
                                f'<b>{_delta:+.2f} km/h</b> ({_delta/max(1e-6,_plot[0]["vc_ms"]*3.6)*100:+.1f} %).'
                                f'</div>', unsafe_allow_html=True)
                else:
                    st.caption("Un second test enregistré fera apparaître la courbe d'évolution.")

                with st.expander("🗑️ Supprimer un test"):
                    _opts = {f"{r['date']} — {r['label'] or 'test'} (#{r['id']})": r["id"] for r in _vc}
                    _sel = st.selectbox("Test à supprimer", list(_opts.keys()), key="hist_del_vc")
                    if st.button("Supprimer définitivement", key="hist_del_vc_btn"):
                        if delete_record("vc_tests", _opts[_sel], _user["id"]):
                            st.success("Test supprimé."); st.rerun()

        # ── Historique séances ───────────────────────────────────────────
        with h_t2:
            _cats = list_categories(_user["id"])
            _cat_by_id = {c["id"]: c["name"] for c in _cats}

            with st.expander("🏷️ Gérer mes catégories de séance"):
                st.caption("Les catégories sont libres et propres à ton compte. Supprimer une catégorie "
                           "ne supprime aucune séance : celles qui l'utilisaient repassent simplement "
                           "en « sans catégorie ».")
                _cc1, _cc2 = st.columns([3, 1])
                with _cc1.form("new_cat_form", clear_on_submit=True):
                    _newc = st.text_input("Nouvelle catégorie", placeholder="ex : Sortie longue spécifique UTMB")
                    if st.form_submit_button("➕ Créer"):
                        _cid, _errc = create_category(_user["id"], _newc)
                        st.error(_errc) if _errc else st.rerun()
                if _cats:
                    for _c in _cats:
                        _r1, _r2, _r3 = st.columns([3, 1, 1])
                        _nn = _r1.text_input("Nom", value=_c["name"], key=f"cat_name_{_c['id']}",
                                             label_visibility="collapsed")
                        if _r2.button("Renommer", key=f"cat_ren_{_c['id']}"):
                            _e = rename_category(_c["id"], _user["id"], _nn)
                            st.error(_e) if _e else st.rerun()
                        if _r3.button("Supprimer", key=f"cat_del_{_c['id']}"):
                            delete_category(_c["id"], _user["id"]); st.rerun()

            if not _wk:
                st.info("Aucune séance enregistrée. Onglet ⚙️ Analyse entraînement → charge un fichier → "
                        "« Enregistrer cette séance ».")
            else:
                def _cat_of(r):
                    return _cat_by_id.get(r.get("category_id")) or (r.get("seance_type") or "— sans catégorie —")

                def _rec_speed(r, dur_s):
                    """Meilleure vitesse tenue sur `dur_s` telle qu'enregistrée avec la séance."""
                    try:
                        for _x in _json.loads(r.get("records_json") or "[]"):
                            if int(_x.get("duree_s", 0)) == int(dur_s):
                                return float(_x.get("vitesse_kmh"))
                    except Exception:
                        pass
                    return None

                def _zone_time(r, idx_from=3):
                    """Temps cumulé dans les zones hautes (Z4 + Z5 par défaut), en minutes."""
                    try:
                        _z = _json.loads(r.get("zones_json") or "[]")
                        return sum(float(x.get("temps_s", 0)) for x in _z[idx_from:]) / 60.0 if _z else None
                    except Exception:
                        return None

                def _quart_drift(r, key):
                    """Dérive du dernier quart par rapport au premier, telle qu'enregistrée."""
                    try:
                        _q = _json.loads(r.get("quarters_json") or "[]")
                        return float(_q[-1].get(key)) if _q and _q[-1].get(key) is not None else None
                    except Exception:
                        return None

                for _r in _wk:
                    _r["_cat"] = _cat_of(_r)
                    _r["_v10"] = _rec_speed(_r, 600)
                    _r["_v30"] = _rec_speed(_r, 1800)
                    _r["_z45"] = _zone_time(_r)
                    _r["_dq_pace"] = _quart_drift(_r, "derive_allure_pct")
                    _r["_dq_vap"] = _quart_drift(_r, "derive_vap_pct")
                    _r["_dq_hr"] = _quart_drift(_r, "derive_fc_bpm")
                    try:
                        _q0 = _json.loads(_r.get("quarters_json") or "[]")
                        _r["_pace_moy"] = (sum(x["duree_s"] for x in _q0) /
                                           max(0.001, sum(x["distance_m"] for x in _q0) / 1000.0)) if _q0 else None
                    except Exception:
                        _r["_pace_moy"] = None

                _cat_names = ["(toutes)"] + sorted({r["_cat"] for r in _wk})
                _fc1, _fc2 = st.columns([2, 3])
                _ft = _fc1.selectbox("Filtrer par catégorie", _cat_names, key="hist_wk_cat")
                _wk_f = _wk if _ft == "(toutes)" else [r for r in _wk if r["_cat"] == _ft]
                _fc2.caption(f"{len(_wk_f)} séance(s) dans cette sélection — la comparaison ci-dessous ne "
                             "porte que sur elles.")

                df_wk = pd.DataFrame([{
                    "Date": r["date"], "Séance": r["name"] or r["file_name"] or "—",
                    "Catégorie": r["_cat"], "Tags": r.get("tags") or "",
                    "Durée": seconds_to_hms(r["duration_s"]) if r["duration_s"] else "—",
                    "Dist (km)": round(r["distance_m"] / 1000.0, 2) if r["distance_m"] else None,
                    "D+ (m)": round(r["d_plus"]) if r["d_plus"] else None,
                    "FC moy": round(r["hr_avg"]) if r["hr_avg"] else None,
                    "Allure moy.": (pace_str(r["_pace_moy"]) + "/km") if r.get("_pace_moy") else "—",
                    "Δ allure Q4/Q1": f"{r['_dq_pace']:+.1f} %" if r.get("_dq_pace") is not None else "—",
                    "Δ FC Q4/Q1": f"{r['_dq_hr']:+.0f}" if r.get("_dq_hr") is not None else "—",
                    "Dérive FC": round(r["hr_drift"], 1) if r["hr_drift"] is not None else None,
                    "Z4+Z5 (min)": round(r["_z45"]) if r["_z45"] is not None else None,
                    "Best 10 min (km/h)": round(r["_v10"], 2) if r["_v10"] else None,
                    "Best 30 min (km/h)": round(r["_v30"], 2) if r["_v30"] else None,
                    "% marche": round(r["pct_walk"]) if r["pct_walk"] is not None else None,
                    "Bascule (%)": round(r["trans_mid"], 1) if r["trans_mid"] is not None else None,
                    "VAP montée fin (%)": round(r["vap_last_montee"]) if r["vap_last_montee"] is not None else None,
                } for r in _wk_f])
                st.dataframe(df_wk, use_container_width=True, hide_index=True)

                # ── évolution d'une mesure au choix ──────────────────────
                _num_cols = {
                    "Allure moyenne (s/km — plus bas = plus rapide)": "_pace_moy",
                    "Dérive d'allure dernier quart / premier (%)": "_dq_pace",
                    "Dérive de VAP dernier quart / premier (%)": "_dq_vap",
                    "Dérive FC dernier quart / premier (bpm)": "_dq_hr",
                    "Meilleure vitesse sur 10 min (km/h)": "_v10",
                    "Meilleure vitesse sur 30 min (km/h)": "_v30",
                    "Temps en Z4+Z5 (min)": "_z45",
                    "FC moyenne (bpm)": "hr_avg",
                    "Dérive cardiaque (bpm)": "hr_drift",
                    "Bascule course→marche (%)": "trans_mid",
                    "% du temps en marche": "pct_walk",
                    "VAP montée en fin de séance (% de la 1re)": "vap_last_montee",
                    "Distance (km)": "distance_m", "Dénivelé positif (m)": "d_plus",
                }
                _metric = st.selectbox("Comparer dans le temps", list(_num_cols.keys()), key="hist_wk_metric")
                _key = _num_cols[_metric]
                _pts = [r for r in _wk_f if r.get(_key) is not None]
                if len(_pts) >= 2:
                    _pts = sorted(_pts, key=lambda r: r["date"])
                    _x = [datetime.strptime(r["date"], "%Y-%m-%d") for r in _pts]
                    _scale = 1000.0 if _key == "distance_m" else 1.0
                    _y = [float(r[_key]) / _scale for r in _pts]
                    fig_wk, axw = plt.subplots(figsize=(11.5, 3.9))
                    axw.plot(_x, _y, "-o", color=C_RED, lw=2, ms=7, mec=C_SURFACE, mew=1.6)
                    for _xi, _yi in zip(_x, _y):
                        axw.annotate(f"{_yi:.1f}", xy=(_xi, _yi), xytext=(0, 9), textcoords="offset points",
                                     ha="center", fontsize=7, color=C_TEXT_MUT)
                    if len(_pts) >= 3:
                        _xn = np.array([(d - _x[0]).days for d in _x], dtype=float)
                        _sl, _ic, _rr, _, _ = sp_stats.linregress(_xn, _y)
                        axw.plot(_x, _ic + _sl * _xn, ls="--", lw=1.2, color=C_WHITE, alpha=0.55,
                                 label=f"tendance {_sl*30:+.2f} / mois (R²={_rr**2:.2f})")
                        axw.legend(loc="best")
                    axw.set_ylabel(_metric); axw.set_xlabel("Date de séance")
                    chart_title(axw, _metric,
                                f"{_aname} · {len(_pts)} séances" +
                                (f" · catégorie « {_ft} »" if _ft != "(toutes)" else " · toutes catégories"))
                    fig_wk.autofmt_xdate(rotation=0, ha="center"); fig_wk.tight_layout()
                    st.pyplot(fig_wk); plt.close(fig_wk)
                else:
                    st.caption("Il faut au moins deux séances comportant cette mesure — filtre moins "
                               "restrictif, ou enregistre d'autres séances.")

                # ── courbes de temps de maintien superposées ─────────────
                _with_rec = [r for r in _wk_f if _json.loads(r.get("records_json") or "[]")]
                if _with_rec:
                    st.markdown("##### ⏱️ Temps de maintien — comparaison de séances")
                    _lbl = {f"{r['date']} — {r['name'] or r['file_name'] or 'séance'} (#{r['id']})": r
                            for r in _with_rec}
                    _sel_rec = st.multiselect("Séances à superposer (2 à 4 conseillées)", list(_lbl.keys()),
                                              default=list(_lbl.keys())[:2], key="hist_wk_reccmp")
                    if _sel_rec:
                        _series = [(k.split(" — ")[0] + " · " + (_lbl[k]["name"] or "séance"),
                                    _json.loads(_lbl[k].get("records_json") or "[]")) for k in _sel_rec]
                        _main_lab, _main_rec = _series[-1]
                        fig_rec = plot_records_curve(_main_rec, vc_ms=last_vc_ms(_aid),
                                                     compare=_series[:-1],
                                                     title_suffix=f" · en rouge : {_main_lab}")
                        st.pyplot(fig_rec); plt.close(fig_rec)
                        st.caption("La courbe rouge est la dernière séance sélectionnée. Un décalage vers le "
                                   "haut sur les durées longues = meilleure endurance ; un gain seulement sur "
                                   "les durées courtes = travail de vitesse.")

                # ── temps par zone cardiaque, séance par séance ──────────
                _with_z = [r for r in _wk_f if _json.loads(r.get("zones_json") or "[]")]
                if len(_with_z) >= 1:
                    st.markdown("##### 💓 Répartition du temps par zone cardiaque")
                    _with_z = sorted(_with_z, key=lambda r: r["date"])[-12:]
                    _zlabels = [z["zone"] for z in _json.loads(_with_z[0]["zones_json"])]
                    _mat = []
                    for r in _with_z:
                        _z = _json.loads(r["zones_json"])
                        _mat.append([float(x["temps_s"]) / 60.0 for x in _z])
                    _mat = np.array(_mat)
                    fig_z, axz = plt.subplots(figsize=(11.5, 3.9))
                    _cols_z = [C_DIM, C_GREY, C_WHITE, C_RED_SOFT, C_RED]
                    _bottom = np.zeros(len(_with_z))
                    _xz = np.arange(len(_with_z))
                    for i in range(_mat.shape[1]):
                        axz.bar(_xz, _mat[:, i], bottom=_bottom, color=_cols_z[i % 5], width=0.6,
                                label=_zlabels[i], edgecolor=C_SURFACE, linewidth=1.2)
                        _bottom += _mat[:, i]
                    axz.set_xticks(_xz)
                    axz.set_xlim(-0.75, len(_with_z) - 0.25)   # évite une barre étalée quand il n'y en a qu'une
                    axz.set_xticklabels([f"{r['date'][5:]}\n{(r['name'] or '')[:14]}" for r in _with_z], fontsize=7.5)
                    axz.set_ylabel("Temps (min)")
                    chart_title(axz, "Temps par zone cardiaque, séance par séance",
                                "12 séances les plus récentes de la sélection")
                    axz.legend(loc="upper left", ncol=5, fontsize=7.5)
                    fig_z.tight_layout(); st.pyplot(fig_z); plt.close(fig_z)

                # ── détail d'une séance ─────────────────────────────────
                _opts_w = {f"{r['date']} — {r['name'] or r['file_name'] or 'séance'} (#{r['id']})": r["id"]
                           for r in _wk_f}
                _sw = st.selectbox("Détail d'une séance", list(_opts_w.keys()), key="hist_wk_detail")
                _rec = get_record("workouts", _opts_w[_sw])
                if _rec:
                    _por = _json.loads(_rec["portions_json"] or "[]")
                    _itv = _json.loads(_rec["intervals_json"] or "[]")
                    _zz = _json.loads(_rec["zones_json"] or "[]")
                    _rr_ = _json.loads(_rec["records_json"] or "[]")
                    _qq = _json.loads(_rec.get("quarters_json") or "[]")
                    if _qq:
                        st.markdown("**Découpage en quarts**")
                        st.dataframe(pd.DataFrame([{
                            "Quart": q["libelle"], "Durée": seconds_to_hms(q["duree_s"]),
                            "Distance (km)": round(q["distance_m"] / 1000.0, 2),
                            "Allure": pace_str(q["allure_s_km"]) + "/km" if q.get("allure_s_km") else "—",
                            "Δ allure": f"{q['derive_allure_pct']:+.1f} %" if q.get("derive_allure_pct") is not None else "—",
                            "VAP (km/h)": q.get("vap_kmh", "—"),
                            "Δ VAP": f"{q['derive_vap_pct']:+.1f} %" if q.get("derive_vap_pct") is not None else "—",
                            "D+ (m)": q.get("d_plus", "—"), "FC moy": q.get("fc_moy", "—"),
                            "Δ FC": f"{q['derive_fc_bpm']:+.0f}" if q.get("derive_fc_bpm") is not None else "—",
                            "Cadence": q.get("cadence", "—")} for q in _qq]),
                            use_container_width=True, hide_index=True)
                    if _rr_:
                        st.markdown("**Temps de maintien de cette séance**")
                        st.dataframe(pd.DataFrame([{
                            "Durée": seconds_to_hms(x["duree_s"]), "Distance (m)": x["distance_m"],
                            "Vitesse (km/h)": x["vitesse_kmh"], "Allure": pace_str(x["allure_s_km"]) + "/km",
                            "À partir de": seconds_to_hms(x["debut_s"])} for x in _rr_]),
                            use_container_width=True, hide_index=True)
                    if _zz:
                        st.markdown("**Zones cardiaques**")
                        st.dataframe(pd.DataFrame([{
                            "Zone": x["zone"], "Plage (bpm)": f"{x['bpm_min'] or '—'} – {x['bpm_max'] or '—'}",
                            "Temps": seconds_to_hms(x["temps_s"]), "% du temps": x["pct"]} for x in _zz]),
                            use_container_width=True, hide_index=True)
                    if _por:
                        st.markdown("**Portions de terrain enregistrées**")
                        st.dataframe(pd.DataFrame(_por), use_container_width=True, hide_index=True)
                    if _itv:
                        st.markdown("**Intervalles enregistrés**")
                        st.dataframe(pd.DataFrame(_itv), use_container_width=True, hide_index=True)
                    _dc1, _dc2 = st.columns(2)
                    _cat_opts = ["— sans catégorie —"] + [c["name"] for c in _cats]
                    _cur_cat = _cat_by_id.get(_rec.get("category_id"), "— sans catégorie —")
                    _new_cat = _dc1.selectbox("Catégorie de cette séance", _cat_opts,
                                              index=_cat_opts.index(_cur_cat) if _cur_cat in _cat_opts else 0,
                                              key="hist_wk_setcat")
                    if _dc1.button("Appliquer la catégorie", key="hist_wk_setcat_btn"):
                        _cid2 = next((c["id"] for c in _cats if c["name"] == _new_cat), None)
                        set_workout_category(_rec["id"], _user["id"], _cid2)
                        st.success("Catégorie mise à jour."); st.rerun()
                    _nn2 = _dc2.text_area("Notes", value=_rec["notes"] or "", key="hist_wk_notes")
                    if _dc2.button("💾 Mettre à jour les notes", key="hist_wk_notes_btn"):
                        update_record_notes("workouts", _rec["id"], _user["id"], _nn2)
                        st.success("Notes enregistrées."); st.rerun()
                    if st.button("🗑️ Supprimer cette séance", key="hist_wk_del"):
                        if delete_record("workouts", _rec["id"], _user["id"]):
                            st.success("Séance supprimée."); st.rerun()

        # ── Historique courses & plans ───────────────────────────────────
        with h_t3:
            if not _rc:
                st.info("Aucune course enregistrée. Onglet 🏃 Prédiction de course → lance un calcul → "
                        "« Enregistrer ce plan de course ».")
            else:
                df_rc = pd.DataFrame([{
                    "Date": r["date"], "Course": r["name"] or "—",
                    "Type": "Résultat" if r["kind"] == "resultat" else "Plan",
                    "Dist (km)": round(r["distance_km"], 1) if r["distance_km"] else None,
                    "D+ (m)": round(r["d_plus"]) if r["d_plus"] else None,
                    "Temps prédit": seconds_to_hms(r["predicted_s"]) if r["predicted_s"] else "—",
                    "Temps réel": seconds_to_hms(r["actual_s"]) if r["actual_s"] else "—",
                    "Écart": (("+" if r["actual_s"] - r["predicted_s"] > 0 else "−") +
                              seconds_to_hms(abs(r["actual_s"] - r["predicted_s"])))
                             if (r["actual_s"] and r["predicted_s"]) else "—",
                    "id": r["id"],
                } for r in _rc])
                st.dataframe(df_rc.drop(columns=["id"]), use_container_width=True, hide_index=True)

                _done = [r for r in _rc if r.get("actual_s") and r.get("predicted_s")]
                if len(_done) >= 2:
                    _done = sorted(_done, key=lambda r: r["date"])
                    fig_rc, axr = plt.subplots(figsize=(11.5, 3.8))
                    _xr = np.arange(len(_done))
                    _err = [(r["actual_s"] - r["predicted_s"]) / r["predicted_s"] * 100.0 for r in _done]
                    axr.bar(_xr, _err, color=[C_RED if e > 0 else C_WHITE for e in _err], width=0.55)
                    axr.axhline(0, color=C_TEXT_MUT, lw=1)
                    axr.set_xticks(_xr)
                    axr.set_xticklabels([f"{r['name'][:18]}\n{r['date']}" for r in _done], fontsize=7.5)
                    axr.set_ylabel("Écart réel / prédit (%)")
                    chart_title(axr, "Fiabilité des prédictions",
                                "au-dessus de 0 : couru plus lentement que prédit")
                    fig_rc.tight_layout(); st.pyplot(fig_rc); plt.close(fig_rc)

                _opts_r = {f"{r['date']} — {r['name'] or 'course'} (#{r['id']})": r["id"] for r in _rc}
                _sr = st.selectbox("Détail d'une course / d'un plan", list(_opts_r.keys()), key="hist_rc_detail")
                _rec_r = get_record("races", _opts_r[_sr])
                if _rec_r:
                    _splits = _json.loads(_rec_r["splits_json"] or "[]")
                    _params = _json.loads(_rec_r["params_json"] or "{}")
                    _cps = _json.loads(_rec_r["checkpoints_json"] or "[]")
                    kpi_row([
                        ("Temps total prédit", seconds_to_hms(_rec_r["predicted_s"]) if _rec_r["predicted_s"] else "—",
                         "arrêts inclus"),
                        ("Dont course", seconds_to_hms(_rec_r["moving_s"]) if _rec_r.get("moving_s") else "—",
                         "temps en mouvement"),
                        ("Dont arrêts", seconds_to_hms(_rec_r["stops_s"]) if _rec_r.get("stops_s") else "—",
                         "ravitaillements"),
                        ("Parcours", f"{_rec_r['distance_km']:.1f} km" if _rec_r["distance_km"] else "—",
                         f"{_rec_r['d_plus']:.0f} m D+" if _rec_r["d_plus"] else ""),
                    ])
                    if _cps and isinstance(_cps, list) and _cps and isinstance(_cps[0], dict) and "arrivee_s" in _cps[0]:
                        st.markdown("**Feuille de route enregistrée**")
                        st.dataframe(pd.DataFrame([{
                            "Ravitaillement": c.get("label"), "Km": round(float(c.get("dist_km", 0)), 1),
                            "Arrivée": seconds_to_hms(c.get("arrivee_s", 0)),
                            "Heure": c.get("heure_arrivee") or "—",
                            "Arrêt": seconds_to_hms(c.get("arret_s", 0)) if c.get("arret_s") else "—",
                            "Départ": seconds_to_hms(c.get("depart_s", 0)),
                            "Allure segment": (pace_str(c["allure_segment_s_km"]) + "/km")
                                              if c.get("allure_segment_s_km") else "—"} for c in _cps]),
                            use_container_width=True, hide_index=True)
                    if _splits:
                        st.markdown("**Plan kilomètre par kilomètre (tel qu'enregistré)**")
                        df_sp = pd.DataFrame(_splits)
                        st.dataframe(df_sp, use_container_width=True, hide_index=True, height=320)
                        st.download_button("⬇️ Plan (CSV)", df_sp.to_csv(index=False).encode("utf-8"),
                                           file_name=f"plan_{(_rec_r['name'] or 'course').replace(' ','_')}.csv",
                                           key="hist_dl_plan")
                    if _cps and not (isinstance(_cps[0], dict) and "arrivee_s" in _cps[0]):
                        st.markdown("**Checkpoints (plan enregistré avant la feuille de route)**")
                        st.dataframe(pd.DataFrame(_cps), use_container_width=True, hide_index=True)
                    with st.expander("⚙️ Paramètres du modèle utilisés"):
                        st.json(_params)
                    rr1, rr2, rr3 = st.columns(3)
                    with rr1:
                        _act = hms_input("Temps réellement réalisé", default="0:00:00", key="hist_rc_actual")
                        if st.button("💾 Enregistrer le temps réel", key="hist_rc_actual_btn"):
                            if update_race_actual(_rec_r["id"], _user["id"], _act):
                                st.success("Temps réel enregistré."); st.rerun()
                    with rr2:
                        if st.button("♻️ Recharger ce plan dans l'onglet Prédiction", key="hist_rc_reload"):
                            st.session_state["_pending_plan_params"] = _params
                            if _rec_r.get("gpx_xml"):
                                st.session_state["_stored_gpx"] = {"name": _rec_r["gpx_name"] or "parcours.gpx",
                                                                    "xml": _rec_r["gpx_xml"]}
                            if _cps:
                                st.session_state["checkpoints"] = _cps
                            st.success("Plan rechargé — ouvre l'onglet 🏃 Prédiction de course.")
                    with rr3:
                        if st.button("🗑️ Supprimer", key="hist_rc_del"):
                            if delete_record("races", _rec_r["id"], _user["id"]):
                                st.success("Course supprimée."); st.rerun()

        # ── Historique métabolique ───────────────────────────────────────
        with h_t5:
            if not _met:
                st.info("Aucun test métabolique enregistré. Onglet 🧪 Tests d'endurance + VC → section 03 "
                        "(CSV masque ventilatoire) → « Enregistrer ce test métabolique ».")
            else:
                st.dataframe(pd.DataFrame([{
                    "Date": r["date"], "Test": r["label"] or "—",
                    "Protocole": {"ramp": "Ramp 1 min", "paliers_courts": "Paliers 2-3 min",
                                  "paliers_longs": "Paliers ≥ 3 min", "mixte": "Ramp + validation"}.get(
                                      r["protocole"], r["protocole"] or "—"),
                    "Paliers": r["n_paliers"],
                    "Masse (kg)": round(r["mass_kg"], 1) if r["mass_kg"] else None,
                    "VC (km/h)": round(r["vc_ms"] * 3.6, 2) if r["vc_ms"] else None,
                    "FatMax (g/min)": round(r["fatmax_g_min"], 2) if r["fatmax_g_min"] else None,
                    "FatMax (% VC)": round(r["fatmax_pct_vc"]) if r["fatmax_pct_vc"] else None,
                    "Économie (kcal/kg/km)": round(r["eco_kcal_kg_km"], 3) if r["eco_kcal_kg_km"] else None,
                    "Confiance": f"{r['confiance']:.0f} %" if r["confiance"] else "—",
                } for r in _met]), use_container_width=True, hide_index=True)

                _mplot = [r for r in _met if r.get("eco_kcal_kg_km") or r.get("fatmax_g_min")]
                if len(_mplot) >= 2:
                    _mplot = sorted(_mplot, key=lambda r: r["date"])
                    _xm = [datetime.strptime(r["date"], "%Y-%m-%d") for r in _mplot]
                    fig_m, (axm1, axm2) = plt.subplots(2, 1, figsize=(11.5, 5.2), sharex=True,
                                                       gridspec_kw={"hspace": 0.15})
                    axm1.plot(_xm, [r.get("fatmax_g_min") or np.nan for r in _mplot], "-o",
                              color=C_WHITE, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
                    axm1.set_ylabel("FatMax (g/min)")
                    chart_title(axm1, "Évolution du profil métabolique",
                                f"{_aname} · {len(_mplot)} tests — capacité à oxyder les lipides et coût énergétique")
                    axm2.plot(_xm, [r.get("eco_kcal_kg_km") or np.nan for r in _mplot], "-o",
                              color=C_RED, lw=2, ms=6, mec=C_SURFACE, mew=1.4)
                    axm2.set_ylabel("Économie (kcal/kg/km)"); axm2.set_xlabel("Date du test")
                    axm2.invert_yaxis()   # plus bas = plus économe
                    fig_m.autofmt_xdate(rotation=0, ha="center")
                    try:
                        fig_m.set_layout_engine("constrained")
                    except Exception:
                        fig_m.tight_layout()
                    st.pyplot(fig_m); plt.close(fig_m)
                    st.caption("Axe du bas inversé : une courbe qui monte visuellement = un athlète plus "
                               "économe (moins de kcal pour parcourir un km).")

                _opts_m = {f"{r['date']} — {r['label'] or 'test'} (#{r['id']})": r["id"] for r in _met}
                _sm = st.selectbox("Détail d'un test", list(_opts_m.keys()), key="hist_met_detail")
                _rec_m = get_record("metabolic_tests", _opts_m[_sm])
                if _rec_m:
                    _st_m = _json.loads(_rec_m["stages_json"] or "[]")
                    _fu_m = _json.loads(_rec_m["fueling_json"] or "[]")
                    kpi_row([
                        ("Confiance du modèle", f"{_rec_m['confiance']:.0f} %" if _rec_m["confiance"] else "—",
                         confidence_label(_rec_m["confiance"] or 0)),
                        ("FatMax", f"{_rec_m['fatmax_g_min']:.2f} g/min" if _rec_m["fatmax_g_min"] else "—",
                         f"{_rec_m['fatmax_pct_vc']:.0f} % VC" if _rec_m["fatmax_pct_vc"] else "—"),
                        ("Économie", f"{_rec_m['eco_kcal_kg_km']:.2f} kcal/kg/km" if _rec_m["eco_kcal_kg_km"] else "—",
                         f"≈ {_rec_m['eco_kcal_kg_km']*(_rec_m['mass_kg'] or 70):.0f} kcal/km"
                         if _rec_m["eco_kcal_kg_km"] else "—"),
                        ("Seuils ventilatoires",
                         f"SV1 {_rec_m['sv1_hr']:.0f} / SV2 {_rec_m['sv2_hr']:.0f} bpm"
                         if (_rec_m.get("sv1_hr") and _rec_m.get("sv2_hr")) else "—", "FC aux transitions"),
                    ])
                    if _st_m:
                        st.markdown("**Paliers enregistrés**")
                        st.dataframe(pd.DataFrame([{
                            "Palier": x["palier"], "Vitesse (km/h)": x.get("vitesse_kmh", "—"),
                            "% VC": x.get("pct_vc", "—"), "FC": x.get("hr", "—"), "RER": x.get("rer"),
                            "Glucides (g/min)": f"{x['cho_g_min']:.2f} [{x['cho_lo']:.2f}–{x['cho_hi']:.2f}]",
                            "Lipides (g/min)": f"{x['fat_g_min']:.2f} [{x['fat_lo']:.2f}–{x['fat_hi']:.2f}]",
                            "kcal/h": x.get("kcal_h"), "kcal/kg/km": x.get("kcal_kg_km", "—"),
                            "Confiance": f"{x['confiance']:.0f} %"} for x in _st_m]),
                            use_container_width=True, hide_index=True)
                    if _fu_m:
                        st.markdown("**Plan nutritionnel enregistré**")
                        st.dataframe(pd.DataFrame([{
                            "Zone": x["zone"], "% VC": f"{x['pct_lo']:.0f}–{x['pct_hi']:.0f} %",
                            "Vitesse (km/h)": x.get("vitesse_kmh"),
                            "kcal/h": x.get("kcal_h"),
                            "Glucides oxydés (g/h)": f"{x['cho_g_h']} [{x['cho_g_h_lo']}–{x['cho_g_h_hi']}]",
                            "Apport conseillé (g/h)": x["apport_g_h"], "Déficit (g/h)": x["deficit_g_h"],
                            "Autonomie": f"{x['autonomie_h']} h" if x.get("autonomie_h") else "≥ 24 h",
                            "Confiance": f"{x['confiance']:.0f} %"} for x in _fu_m]),
                            use_container_width=True, hide_index=True)
                        st.download_button("⬇️ Plan nutritionnel (CSV)",
                                           pd.DataFrame(_fu_m).to_csv(index=False).encode("utf-8"),
                                           file_name=f"plan_nutrition_{_aname.replace(' ', '_')}.csv",
                                           key="hist_dl_fueling")
                    if _rec_m.get("notes"):
                        st.caption(f"📝 {_rec_m['notes']}")
                    if st.button("🗑️ Supprimer ce test", key="hist_met_del"):
                        if delete_record("metabolic_tests", _rec_m["id"], _user["id"]):
                            st.success("Test supprimé."); st.rerun()

        # ── Données : export / import / maintenance ──────────────────────
        with h_t4:
            st.markdown("#### Sauvegarde et transfert")
            st.markdown('<div class="note-box note-red">La base de données est un fichier <b>séparé du '
                        'script</b>. Modifier l\'algorithme, ajouter des graphiques ou remplacer le fichier '
                        '.py ne touche jamais aux données : au démarrage, l\'app ajoute seulement les '
                        'colonnes manquantes (aucune table n\'est recréée ni vidée) et fait une copie de '
                        'sauvegarde datée avant toute évolution de schéma.</div>', unsafe_allow_html=True)
            _db_size = os.path.getsize(DB_PATH) / 1024.0 if os.path.exists(DB_PATH) else 0
            st.caption(f"Fichier : `{DB_PATH}` · {_db_size:,.0f} Ko · schéma v{SCHEMA_VERSION}".replace(",", " "))
            _baks = sorted(glob.glob(f"{DB_PATH}.bak-*"), reverse=True)
            if _baks:
                st.caption("Sauvegardes automatiques : " +
                           " · ".join(os.path.basename(b) for b in _baks[:4]))
            with open(DB_PATH, "rb") as _dbf:
                st.download_button("⬇️ Télécharger la base complète (.db)", data=_dbf.read(),
                                   file_name=os.path.basename(DB_PATH), key="hist_dl_db")
            _exp = export_athlete_json(_aid)
            if _exp:
                st.download_button(f"⬇️ Exporter tout l'historique de {_aname} (JSON)",
                                   data=_json.dumps(_exp, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                                   file_name=f"historique_{_aname.replace(' ', '_')}.json",
                                   key="hist_export_json")
            _imp = st.file_uploader("⬆️ Importer un historique athlète (JSON)", type=["json"], key="hist_import_json")
            if _imp and st.session_state.get("_last_hist_import") != _imp.name:
                st.session_state["_last_hist_import"] = _imp.name
                try:
                    _payload = _json.loads(_imp.read().decode("utf-8"))
                    _res_imp, _err_imp = import_athlete_json(_user["id"], _payload)
                    if _err_imp:
                        st.error(_err_imp)
                    else:
                        st.success(f"{_res_imp['n']} enregistrement(s) importé(s).")
                        st.rerun()
                except Exception as _e:
                    st.error(f"Fichier JSON invalide — {_e}")
            st.markdown("---")
            with st.expander("🗑️ Supprimer cet athlète et tout son historique"):
                st.warning("Action irréversible : tests, séances et courses de cet athlète seront effacés.")
                _confirm = st.text_input(f"Tape le nom de l'athlète pour confirmer", key="hist_del_ath_confirm")
                if st.button("Supprimer définitivement l'athlète", key="hist_del_ath_btn"):
                    if _confirm.strip() == _aname:
                        delete_athlete(_aid, _user["id"])
                        st.session_state.pop("_athlete_id", None); st.session_state.pop("_athlete_name", None)
                        st.success("Athlète supprimé."); st.rerun()
                    else:
                        st.error("Le nom saisi ne correspond pas.")
