# analyse_course_v8.py
# Application Streamlit unifiée — 3 onglets
# NOUVEAUTÉS v8.5 :
#   - Modèle de fatigue à DEUX phases empilées (fatigue_multiplier_dual) :
#     Phase 1 (parcours) reste pilotée par le profil terrain / JSON de
#     cohorte comme avant. Phase 2 (optionnelle, décochée par défaut)
#     AJOUTE une fatigue personnelle plus tardive, détectée sur tes
#     propres références (section 2️⃣), par-dessus la phase 1 — au lieu
#     de la remplacer comme le faisait la v8.4. Le R² du point de rupture
#     est affiché à côté du réglage pour juger de sa fiabilité.
#
# NOUVEAUTÉS v8.4 :
#   - Signature de fatigue personnelle (seuil/taux) désormais calculée
#     AUTOMATIQUEMENT depuis tes propres courses de référence (section 2️⃣
#     de l'onglet Prédiction, FIT/TCX importés) — plus besoin de passer par
#     l'onglet 🧪 Tests d'endurance + VC ni de cliquer sur un bouton
#     séparé. La détection tourne dans le même passage de script que le
#     reste de la section 4️⃣, donc pas de décalage d'un run à l'autre.
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
#     section 4️⃣) : pour chaque test importé en FIT/TCX, détecte le moment où
#     l'allure décroche le plus nettement, en % de la durée totale. Comparer
#     ce % entre tests de durées différentes permet de repérer une signature
#     de fatigue propre à l'athlète, utilisable comme point de départ pour le
#     seuil de fatigue en course longue.
#   - Import direct de profils calibrés (JSON) dans l'onglet Prédiction,
#     section 4️⃣ : plus besoin de repasser par l'onglet Cohorte pour
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
import json as _json
from scipy import stats as sp_stats
from scipy.optimize import least_squares

st.set_page_config(page_title="Coach Running — Suite complète", layout="wide", page_icon="🏃")
TZ_NAME_DEFAULT = "Europe/Paris"

st.markdown("""
<style>
.param-box {background:#f8f9fa;border-left:4px solid #1f77b4;border-radius:4px;padding:8px 12px;margin-bottom:8px;font-size:0.88rem;}
.param-up{color:#d62728;font-weight:600;}.param-down{color:#2ca02c;font-weight:600;}
.highlight-box{background:#e8f0fe;border:1px solid #6c8ebf;border-radius:6px;padding:12px 16px;margin:8px 0;color:#1a3a5c;}
.test-card{background:#ffffff;border:1px solid #dee2e6;border-radius:8px;padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.07);}
.test-card h4{margin:0 0 8px 0;color:#1f77b4;font-size:1rem;}
.result-metric{text-align:center;font-size:1.4rem;font-weight:700;}
.sidebar-label{background:#e8f4fd;border-radius:4px;padding:6px 10px;font-size:0.80rem;color:#1f77b4;margin-bottom:10px;}
.interval-card{background:#f0f4ff;border:1px solid #b0c4de;border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.terrain-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;margin-bottom:6px;}
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

def parse_fit_ref(file,tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0);fit=FitFile(file);fit.parse()
        records,times_pts,hr_records=[],[],[]
        start_global=elapsed_global=None
        for msg in fit.get_messages("session"):
            vals={d.name:d.value for d in msg}
            if isinstance(vals.get("start_time"),datetime):start_global=vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"),(int,float)):elapsed_global=float(vals["total_elapsed_time"])
        for msg in fit.get_messages("record"):
            vals={d.name:d.value for d in msg}
            lat_r=vals.get("position_lat");lon_r=vals.get("position_long")
            if lat_r is None or lon_r is None:continue
            lat=lat_r*(180/2**31);lon=lon_r*(180/2**31)
            ts=vals.get("timestamp");dt=ts.replace(tzinfo=None) if isinstance(ts,datetime) else None
            alt=(vals.get("enhanced_altitude") or vals.get("altitude") or 0.0)
            dist=float(vals.get("distance") or 0.0);hr=vals.get("heart_rate")
            hr_records.append(int(hr) if hr is not None else None)
            records.append((lat,lon,float(alt),dist));times_pts.append(dt)
        if not records:return None
        df=pd.DataFrame(records,columns=["lat","lon","elev","dist"])
        valid_t=[t for t in times_pts if t is not None]
        if len(valid_t)>=2:start_dt,end_dt=min(valid_t),max(valid_t)
        elif start_global and elapsed_global:start_dt=start_global;end_dt=start_global+timedelta(seconds=elapsed_global)
        else:
            start_dt=datetime.now().replace(hour=12,minute=0,second=0,microsecond=0)-timedelta(days=1)
            end_dt=start_dt+timedelta(minutes=5)
        avgT,avgW,avgH=get_avg_weather(records[0][0],records[0][1],start_dt,end_dt,tz_name)
        elev_arr=df["elev"].values
        dup=float(np.sum(np.clip(np.diff(elev_arr),0,None))) if elev_arr.size>=2 else 0.0
        ddn=float(-np.sum(np.clip(np.diff(elev_arr),None,0))) if elev_arr.size>=2 else 0.0
        hr_analysis = analyze_hr_v3(hr_records)
        return{"points":[{"lat":r[0],"lon":r[1],"elev":r[2],"dist":r[3],"time":t} for r,t in zip(records,times_pts)],
               "distance":float(df["dist"].max()),"D_up":dup,"D_down":ddn,
               "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
               "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,"hr_analysis":hr_analysis}
    except Exception as e:st.error(f"Erreur FIT:{e}");return None

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
                if ts is None:continue
                rows.append({"timestamp":ts,"heart_rate":hr,"speed_ms":spd,"distance_m":dist,"altitude_m":alt})
        if not rows:return None
        df=pd.DataFrame(rows)
        df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df=df.sort_values("timestamp").reset_index(drop=True)
        t0=df["timestamp"].iloc[0];df["elapsed_s"]=(df["timestamp"]-t0).dt.total_seconds()
    elif fname.endswith(".gpx"):
        file.seek(0);gpx=gpxpy.parse(file);rows=[]
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:rows.append({"timestamp":pt.time,"heart_rate":None,"speed_ms":None,"distance_m":None,"altitude_m":pt.elevation or 0.0})
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
            try:ts=datetime.fromisoformat(tim.text.replace("Z","+00:00"))
            except:continue
            rows.append({"timestamp":ts,"heart_rate":int(hr_el.text) if hr_el is not None else None,
                         "speed_ms":float(spd_el.text) if spd_el is not None else None,
                         "distance_m":float(dist_el.text) if dist_el is not None else None,
                         "altitude_m":float(alt_el.text) if alt_el is not None else None})
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
        df.rename(columns=renames,inplace=True)
        if "elapsed_s" not in df.columns:df["elapsed_s"]=range(len(df))
    if df is None:return None
    for col in["heart_rate","speed_ms","distance_m","altitude_m","elapsed_s"]:
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
            elif cl=="aux1":renames[c]="eqO2_raw"
            elif cl=="aux2":renames[c]="aux2"
        df.rename(columns=renames,inplace=True)
        if "VE" not in df.columns:
            for c in df.columns:
                if "VE" in c and "L/min" in c:df.rename(columns={c:"VE"},inplace=True);break
        required=["timestamp","VE","VCO2","eqO2_raw","HR","palier"]
        missing=[r for r in required if r not in df.columns]
        if missing:return None
        df["timestamp"]=pd.to_numeric(df["timestamp"],errors="coerce")
        df=df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"]=df["timestamp"]-df["timestamp"].iloc[0];df["elapsed_min"]=df["elapsed_s"]/60.0
        for col in["VE","VCO2","eqO2_raw","HR","Cadence","FeCO2","eqCO2"]:
            if col in df.columns:df[col]=pd.to_numeric(df[col],errors="coerce")
        df["VO2_Lmin"]=df["VE"]/df["eqO2_raw"].replace(0,np.nan)
        df["RQ"]=df["VCO2"]/df["VO2_Lmin"].replace(0,np.nan)
        df["eqO2"]=df["eqO2_raw"]
        df["palier"]=pd.to_numeric(df["palier"],errors="coerce").fillna(0).astype(int)
        return df
    except:return None

def aggregate_by_palier(df):
    cols=["elapsed_min","HR","VE","VO2_Lmin","VCO2","RQ","eqO2","eqCO2","Cadence"]
    cols=[c for c in cols if c in df.columns]
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
    Prédiction section 4️⃣). Les profils calibrés ont la priorité en cas de
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

COHORT_PALETTE = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#db2777","#0891b2","#65a30d"]

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
        ax.axvline(cp["km"], color="#f59e0b", lw=1, ls=":", alpha=0.6)
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
        ax.axvline(cp["km"], color="#94a3b8", lw=0.5, ls=":", alpha=0.4, zorder=0)
        y_offset = 8 if idx % 2 == 0 else 26  # alterne la hauteur pour limiter le chevauchement
        ax.annotate(f"{cp['name']}\nkm {cp['km']:.1f}", xy=(cp["km"], ymax),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="#475569")
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
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-label">⚙️ Paramètres — onglets Tests & Entraînement</div>',unsafe_allow_html=True)
    sb_opt_temp=st.slider("Température optimale (°C)",5.0,20.0,10.0,0.5,key="sb_opt_temp")
    sb_k_up=st.number_input("Coefficient montée (k_up)",value=22.0,step=0.5,key="sb_k_up")
    sb_k_down=st.number_input("Coefficient descente (k_down)",value=4.5,step=0.5,key="sb_k_down")
    sb_k_temp_hot=st.number_input("Sensibilité chaleur",value=0.0016,step=0.0002,format="%.4f",key="sb_kth")
    sb_k_temp_cold=st.number_input("Sensibilité froid",value=0.0012,step=0.0002,format="%.4f",key="sb_ktc")
    st.caption("Ces paramètres n'affectent que les onglets 🧪 et ⚙️")

main_tabs=st.tabs(["🏃 Prédiction de course","🧪 Tests d'endurance + VC","⚙️ Analyse entraînement","👥 Analyse de cohorte"])

# ══════════════════════════════════════════════════════════════
# ONGLET 0 — PRÉDICTION DE COURSE
# ══════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.title("🏃 Prédiction de course — Coach & Athlète")
    st.caption("v8.5 — Filtre GPS · Padding lissage · Profil route · VC FIT/TCX · Prédiction FC · Import JSON direct · K Riegel relevé · Fatigue à deux phases (parcours + perso)")

    col_mode1,col_mode2=st.columns([2,3])
    with col_mode1:
        mode=st.radio("Mode d'interface",["🟢 Simple (recommandé)","🔵 Expert (tous les curseurs)"],horizontal=True,key="pred_mode")
    EXPERT="Expert" in mode

    st.markdown("---")
    col_rt1, col_rt2 = st.columns([2, 3])
    with col_rt1:
        mode_activite = st.radio("🏷️ Type d'activité",["🛣️ Route / Piste","🏔️ Trail / Montagne"],
                                  horizontal=True, key="mode_activite")
    IS_TRAIL = "Trail" in mode_activite

    # ── v8.1 : suggestion auto. du profil terrain (étape 1/2 — sans GPX) ──
    # Doit être placé AVANT l'instanciation du widget terrain_profil_radio
    # (section 4️⃣) dans ce même run, sinon Streamlit lève une
    # StreamlitAPIException ("cannot be modified after widget is instantiated").
    if st.session_state.get("_last_mode_activite_for_suggestion") != mode_activite:
        st.session_state["_last_mode_activite_for_suggestion"] = mode_activite
        st.session_state["terrain_profil_radio"] = "🛣️ Route / Plat" if not IS_TRAIL else "🏔️ Trail modéré"

    st.markdown("---")
    st.header("1️⃣  Parcours GPX")
    gpx_file=st.file_uploader("📂 Importer le GPX de la course cible",type=["gpx"],key="gpx_main")
    points=None;dem_elevations=None

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
            # section 1️⃣, donc toujours avant le widget terrain_profil_radio (section 4️⃣).
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
                    f"appliqué automatiquement, modifiable dans la section 4️⃣ si besoin.")

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
                    badge_css=("background:#fee2e2;color:#991b1b" if score>0.65 else "background:#fef3c7;color:#92400e" if score>0.40 else "background:#d1fae5;color:#065f46")
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
                            colors_t=["#22c55e" if s<0.25 else "#eab308" if s<0.5 else "#f97316" if s<0.75 else "#ef4444" for s in scores]
                            ax_tech.bar(km_mids,scores,width=0.85,color=colors_t,alpha=0.85)
                            ax_tech.axhline(0.5,color="orange",lw=1,ls="--",label="Seuil technique")
                            ax_tech.axhline(0.75,color="red",lw=1,ls="--",label="Seuil très technique")
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
    st.header("2️⃣  Courses de référence")
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
                                       f"— disponible en option pour la fatigue en section 4️⃣ (voir le R² avant de "
                                       f"t'y fier : sous 0.6, le signal reste faible).")
            else:
                if EXPERT:
                    cs2,ce2=st.columns(2)
                    avg_temp_ref=cs2.number_input(f"Temp moy. course (°C)",value=15.0,key=f"avgT_{i}")
                    avg_hum_ref=ce2.number_input(f"Humidité moy. (%)",value=60.0,key=f"avgH_{i}")
                else:avg_temp_ref=avg_hum_ref=None
            temps_eff=dur_hms_file if dur_hms_file else temps
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
            refs_raw.append({"distance":float(dist),"temps":str(temps_eff),
                              "D_up":float(dup),"D_down":float(ddn),"duration_hms_file":dur_hms_file,
                              "avg_temp":avg_temp_ref,"avg_humidity":avg_hum_ref,"avg_wind":avg_wind_ref,
                              "hr_analysis":hr_ref,"hr_avg":hr_avg_ref,"hr_max":hr_max_ref,
                              "breakpoint":ref_breakpoint})

    # v8.4 — signature de fatigue personnelle, calculée automatiquement à partir des
    # ruptures d'allure détectées ci-dessus sur tes propres références (aucune action
    # séparée requise, aucune dépendance à l'onglet 🧪 Tests d'endurance + VC).
    # Seuil de détection à 0.3 (affichage), mais le signal reste faible en dessous de
    # 0.6 — la case reste décochée par défaut dans ce cas (voir section 4️⃣). Attention
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
    st.header("3️⃣  Recalibration des références vers les conditions idéales")
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
            # Priorité au temps RÉELLEMENT calculé (section 6️⃣, si déjà lancé) plutôt qu'à
            # l'estimation grossière (allure moyenne des références × distance GPX) — cette
            # dernière ne sert plus que de repli tant qu'aucun calcul n'a encore été fait.
            # Comme cette section 3️⃣ s'exécute avant la section 6️⃣ dans le script, le résultat
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
                       + ("" if _res_prev else " — lance le calcul en section 6️⃣ pour utiliser le temps réellement prédit."))
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
                    ax_hr_p.scatter(durs_h, hrs_h, s=80, color="#d62728", zorder=5, label="FC moy. observée")
                    if hr_pred.get("model") == "regression":
                        d_line = np.linspace(max(1, min(durs_h)*0.8), max(max(durs_h)*1.2, _dur_target_s/60*1.1), 80)
                        hr_line = [hr_pred["slope"]*math.log(max(1,d*60))+hr_pred["intercept"] for d in d_line]
                        ax_hr_p.plot(d_line, hr_line, color="#1f77b4", lw=2, ls="--", label="Régression log(durée)")
                        sigma_viz = (hr_pred['hr_target_range'][1]-hr_pred['hr_target_range'][0])/2
                        ax_hr_p.fill_between(d_line,[h-sigma_viz for h in hr_line],[h+sigma_viz for h in hr_line],alpha=0.12,color="#1f77b4")
                    ax_hr_p.axvline(_dur_target_s/60, color="#f97316", lw=2, ls=":", label=f"Course cible ({_dur_target_s/60:.0f} min)")
                    ax_hr_p.scatter([_dur_target_s/60],[hr_pred["hr_target_avg"]],s=150,color="#f97316",marker="*",zorder=6,label=f"FC cible {hr_pred['hr_target_avg']} bpm")
                    ax_hr_p.set_xlabel("Durée de la course (min)"); ax_hr_p.set_ylabel("FC moyenne (bpm)")
                    ax_hr_p.set_title("Régression FC personnelle — données athlète uniquement")
                    ax_hr_p.legend(fontsize=8); ax_hr_p.grid(alpha=0.3); fig_hr_pred.tight_layout()
                    st.pyplot(fig_hr_pred); plt.close(fig_hr_pred)
    else:
        st.caption("💓 Chargez des fichiers FIT ou TCX dans les références pour obtenir une prédiction de zone cardiaque personnalisée.")


    st.markdown("---")
    st.header("4️⃣  Paramètres du modèle")

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
            st.markdown(f'<div style="background:#fef3c7;border-left:3px solid #f59e0b;border-radius:4px;padding:6px 12px;font-size:0.82rem;">🤖 <b>Suggestion auto (terrain score {tech_global_ui["global_score"]:.2f})</b> : k_up → <b>{sugg_k_up}</b> · k_down → <b>{sugg_k_dn}</b></div>',unsafe_allow_html=True)
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
        _terrain_colors={"🛣️ Route / Plat":"#0C447C","🏔️ Trail modéré":"#3B6D11","⛰️ Ultra-trail montagneux":"#993C1D"}
        _col=_terrain_colors.get(terrain_profil,"#444")
        st.markdown(f'<div style="background:rgba(0,0,0,0.04);border-left:3px solid {_col};border-radius:4px;padding:6px 12px;font-size:0.82rem;margin-top:6px;"><b>{terrain_profil}</b> · k_up={k_up:.0f} · k_down={k_down:.0f} · Minetti={minetti_weight:.2f} · Plafond={max_cap:.0%} · Surface <b>{surface_sel.split()[0]} ×{surface_mult:.2f}</b></div>',unsafe_allow_html=True)

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
            # sur tes propres références (section 2️⃣). Une référence courte/route (marathon)
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
                           f"importées en section 2️⃣ — moyenne utilisée si plusieurs.")
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
                st.caption("Aucune rupture d'allure exploitable détectée sur tes références (section 2️⃣) pour "
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
    st.header("5️⃣  Paramètres de la course cible")
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

    st.header("6️⃣  Calcul & Résultats")
    if st.button("▶️ Calculer la prédiction",type="primary"):
        if not gpx_file or points is None:st.error("⚠️ Importe un fichier GPX (section 1).")
        elif not any(safe_float(r.get("distance",0))>0 and hms_to_seconds(r.get("temps","0"))>0 for r in refs_raw):
            st.error("⚠️ Renseigne au moins une référence valide.")
        else:
            st.session_state["_meteo_api_cache"]={}
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
                        objective_hms=temps_objectif if force_temps else None,
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
                ax.plot(x,pv,lw=1.5,alpha=0.35,color="steelblue",label="Allure brute")
                if "Allure lissée (min/km)" in df_out.columns:
                    ps=[]
                    for v in df_out["Allure lissée (min/km)"].values:
                        try:parts=str(v).split(":");ps.append(int(parts[0])+int(parts[1])/60.0)
                        except:ps.append(float("nan"))
                    ax.plot(x,ps,lw=2.5,color="firebrick",label="Allure lissée")
                if apply_fatigue and fatigue_rate>0 and len(x)>0:
                    thresh_km=fatigue_threshold/100.0*len(x)
                    ax.axvline(thresh_km,color="orange",lw=1.5,ls="--",alpha=0.7,label=f"Seuil phase 1 ({fatigue_threshold}%)")
                    if dual_fatigue and fatigue_threshold2:
                        thresh2_km=fatigue_threshold2/100.0*len(x)
                        ax.axvline(thresh2_km,color="darkred",lw=1.5,ls="--",alpha=0.7,label=f"Seuil phase 2 perso ({fatigue_threshold2}%)")
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
                    ax2.axvline(fatigue_threshold/100.0*len(x),color="orange",lw=1.5,ls="--",alpha=0.7,label=f"Seuil {fatigue_threshold}%")
                ax2.axhline(1.0,color="gray",lw=0.8);ax2.set_xlabel("Kilomètre")
                ax2.set_ylabel("Multiplicateur");ax2.set_title("Décomposition des facteurs")
                ax2.legend();ax2.grid(alpha=0.3);st.pyplot(fig2);plt.close(fig2)
            with res_t3:st.dataframe(df_out,use_container_width=True)


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
                st.session_state["checkpoints"].append({"dist_km":cp_dist,"type":cp_type,"label":label,"lat":cp_lat,"lon":cp_lon,"alt":round(cp_alt)})
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
                ax3.fill_between(x_km,y_s.min()-5,y_s,alpha=0.15,color="steelblue")
                ax3.plot(x_km,y_s,lw=2.5,label="Altitude GPS lissée",color="steelblue")
                ax3.plot(x_km,y_gps,lw=0.8,alpha=0.2,color="gray",label="GPS brut")
            else:
                ax3.fill_between(x_km,y_gps.min()-5,y_gps,alpha=0.15,color="steelblue")
                ax3.plot(x_km,y_gps,lw=2.5,label="Altitude GPS",color="steelblue")
            if dem_elevations is not None and len(dem_elevations)==len(points):
                y_dem=np.array([e if e is not None else 0.0 for e in dem_elevations])
                ax3.plot(x_km,y_dem,lw=2,ls="--",label="DEM corrigé",color="forestgreen")
            cp_colors_profile={"🥤 Ravitaillement":"#00c864","⏱ Point de passage":"#6496ff","🏔 Sommet":"#ffc800","🔻 Col":"#c864ff","🏁 Intermédiaire":"#ffffff","⚠️ Point clé":"#ff5050"}
            for cp in sorted(checkpoints,key=lambda x:x["dist_km"]):
                cp_x=cp["dist_km"];cp_y=float(np.interp(cp_x*1000,cum_d_map,y_gps))
                col_p=cp_colors_profile.get(cp["type"],"#ffcc00")
                ax3.axvline(cp_x,color=col_p,lw=1.5,ls="--",alpha=0.7)
                ax3.annotate(cp["label"],xy=(cp_x,cp_y),xytext=(0,12),textcoords="offset points",ha="center",fontsize=7.5,color=col_p,fontweight="bold",arrowprops=dict(arrowstyle="-",color=col_p,lw=1))
                ax3.scatter([cp_x],[cp_y],s=60,color=col_p,zorder=5)
            if IS_TRAIL:
                for seg in st.session_state.get("tech_segs",[]):
                    if seg.get("tech_score",0)>0.45:
                        color_tech="#ef4444" if seg["tech_score"]>0.70 else "#f97316"
                        ax3.axvspan(seg["km_start"],seg["km_end"],alpha=0.12,color=color_tech,label="_nolegend_")
            ax3.scatter([x_km[0]],[y_gps[0]],s=80,color="lime",zorder=6,marker="^",label="Départ")
            ax3.scatter([x_km[-1]],[y_gps[-1]],s=80,color="red",zorder=6,marker="s",label="Arrivée")
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

    st.header("1️⃣ Performances de référence")
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
                        # v8.2 — conserve les points pour la détection de rupture (section 4️⃣)
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
    st.header("2️⃣ Modèle Riegel + Vitesse Critique")

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
        ax_vc.plot(t_arr/60, v_riegel, lw=2, color="steelblue", label=f"Riegel K={K_r:.3f}")
        if vc_ms > 0:
            v_vc_line = [vc_ms * 3.6 + (d_prime / t * 3.6 if d_prime and t > 0 else 0) for t in t_arr]
            ax_vc.plot(t_arr/60, v_vc_line, lw=2, ls="--", color="firebrick",
                       label=f"Modèle D' (VC={vc_ms*3.6:.2f} km/h, D'={round(d_prime)}m)")
            ax_vc.axhline(vc_ms * 3.6, color="orange", lw=1.5, ls=":", label="VC")
        for r in refs_fit_vc:
            if r["distance"] > 0 and r["temps"] > 0:
                v_ref = r["distance"] / r["temps"] * 3.6
                ax_vc.scatter([r["temps"]/60], [v_ref], s=80, zorder=5, color="forestgreen")
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
    st.header("3️⃣ Test GAZOZ / Masque ventilatoire (analyse SV1/SV2)")
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
            thresholds = detect_sv1_sv2(df_pal)
            sv1 = thresholds["sv1"]; sv2 = thresholds["sv2"]

            with st.expander("📊 Données brutes par palier"):
                st.dataframe(df_pal.round(2), use_container_width=True, hide_index=True)

            st.subheader("🎯 Seuils ventilatoires détectés")
            col_sv1, col_sv2 = st.columns(2)
            with col_sv1:
                if sv1:
                    st.markdown('<div class="test-card">', unsafe_allow_html=True)
                    st.markdown(f"#### 🟢 SV1 — Seuil aérobie")
                    st.metric("FC", f"{sv1['HR']} bpm"); st.metric("Palier", str(sv1["palier"]))
                    st.metric("VO₂", f"{sv1['VO2']:.3f} L/min"); st.metric("RQ", f"{sv1['RQ']:.3f}")
                    st.metric("VE", f"{sv1['VE']:.1f} L/min")
                    if sv1.get("Cadence", 0) > 0: st.metric("Vitesse / Cadence", f"{sv1['Cadence']:.2f} km/h")
                    st.markdown("</div>", unsafe_allow_html=True)
                else: st.info("SV1 non détecté automatiquement.")
            with col_sv2:
                if sv2:
                    st.markdown('<div class="test-card">', unsafe_allow_html=True)
                    st.markdown(f"#### 🔴 SV2 — Seuil anaérobie")
                    st.metric("FC", f"{sv2['HR']} bpm"); st.metric("Palier", str(sv2["palier"]))
                    st.metric("VO₂", f"{sv2['VO2']:.3f} L/min"); st.metric("RQ", f"{sv2['RQ']:.3f}")
                    st.metric("VE", f"{sv2['VE']:.1f} L/min")
                    if sv2.get("Cadence", 0) > 0: st.metric("Vitesse / Cadence", f"{sv2['Cadence']:.2f} km/h")
                    st.markdown("</div>", unsafe_allow_html=True)
                else: st.info("SV2 non détecté automatiquement.")

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
                if sv1: ax.axvline(sv1["palier"], color="green", lw=1.5, ls="--", label="SV1")
                if sv2: ax.axvline(sv2["palier"], color="red",   lw=1.5, ls="--", label="SV2")
                ax.legend(fontsize=7)
            _ax_palier(axes[0], "VE",   "VE (L/min)",     "#1f77b4")
            _ax_palier(axes[1], "RQ",   "Quotient Resp.", "#d62728")
            _ax_palier(axes[2], "eqO2", "Éq. O₂",        "#2ca02c")
            _ax_palier(axes[3], "HR",   "FC (bpm)",       "#e377c2")
            plt.tight_layout(); st.pyplot(fig_gz); plt.close(fig_gz)

    st.markdown("---")
    st.header("4️⃣ Détection du point de rupture (tests à effort maximal)")
    st.caption(
        "Pour chaque référence importée via fichier FIT/TCX (section 1️⃣ ci-dessus), détecte le moment où "
        "l'allure décroche le plus nettement — une rupture à un seul changement de régime (allure stable, puis "
        "allure dégradée). Comparer ce point en % de la durée totale entre plusieurs tests de durées différentes "
        "peut révéler une signature de fatigue récurrente propre à l'athlète, utilisable comme point de départ "
        "pour le seuil de fatigue en course longue (onglet 🏃 Prédiction, section 4️⃣).")

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
        st.info("Importe au moins un test via fichier FIT/TCX (section 1️⃣ ci-dessus, ≥1 min) pour activer cette "
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
                        ax_hr_t.plot(t_series, hr_series.values, lw=0.7, alpha=0.25, color="gray", label="FC brute")
                        ax_hr_t.plot(t_series, hr_smooth, lw=2, color="#e377c2", label="FC lissée")
                        ax_hr_t.axhline(hr_stats_train["fc_avg"], color="steelblue", lw=1, ls="--",
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
                    ax_pace.plot(t_spd, pace_vals, lw=0.6, alpha=0.2, color="gray")
                    ax_pace.plot(t_spd, pace_smooth, lw=2, color="steelblue", label="Allure lissée")
                    ax_pace.invert_yaxis()
                    ax_pace.set_xlabel("Temps (min)"); ax_pace.set_ylabel("Allure (s/km)")
                    ax_pace.set_title("Allure au fil du temps"); ax_pace.grid(alpha=0.3); ax_pace.legend(fontsize=8)
                    yticks = ax_pace.get_yticks()
                    ax_pace.set_yticklabels([pace_str(t) for t in yticks if t > 0], fontsize=8)
                    fig_pace.tight_layout()
                    st.pyplot(fig_pace); plt.close(fig_pace)

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
                    colors_int = plt.cm.tab10.colors
                    for ci_int, r in enumerate(valid_ints):
                        if "df" not in r: continue
                        df_i = r["df"]
                        if "heart_rate" not in df_i.columns or "elapsed_s" not in df_i.columns: continue
                        hr_i = df_i["heart_rate"].dropna()
                        hr_i = hr_i[(hr_i >= 40) & (hr_i <= 220)]
                        if len(hr_i) < 5: continue
                        t_i = df_i.loc[hr_i.index, "elapsed_s"].values
                        hr_sm_i = smooth_hr(hr_i).values
                        ax_int.plot(t_i, hr_sm_i, lw=2, color=colors_int[ci_int % 10], label=r["name"])
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
                bg = "#fef2f2" if weather_pen_cohort > 0.08 else "#fffbeb" if weather_pen_cohort > 0.03 else "#f0fdf4"
                st.markdown(f'<div style="margin-top:8px;padding:10px 14px;border-radius:8px;background:{bg};font-size:13px;">'
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
                                   f"🏃 Prédiction de course, section 4️⃣ Paramètres du modèle.")

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
                               "dans l'onglet 🏃 Prédiction, section 4️⃣).")

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
