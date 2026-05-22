# analyse_course_v6.py
# Application Streamlit unifiée — 3 onglets
# NOUVEAUTÉS v6 :
#   - Trail modéré calibré sur historique course (k_up=22, surface=1.11, fatigue=8.5%)
#   - Vue 3D cinématique Three.js néon sur relief procédural (sans compte, sans API)
#   - Vue relief 3D réel via pydeck TerrainLayer ArcGIS (sans compte, sans clé API)
#   - Détection automatique terrain technique (sinuosité, pente, variabilité, lacets)
#   - Récupération surface OSM via Overpass API (rock/grass/mud/…) sans clé
#   - Sélecteur surface calibrée
#   - opt_temp montagne = 10°C
#   - Fatigue activée par défaut sur profil calibré
#
# pip install streamlit gpxpy fitparse fitdecode pandas numpy pydeck matplotlib requests scipy

import streamlit as st
import math
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
            g_eff=math.tanh((-g)/g0d)*g0d;bonus=min(float(k_down)*g_eff,abs(float(down_cap)));mult=1.0-bonus
        mult=min(mult,1.0+float(max_up));mult=max(mult,1.0+float(max_down))
        return max(0.01,float(mult))
    except:return 1.0

def combined_grade_multiplier(grade_pct,use_minetti,minetti_weight,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down):
    if not use_minetti:return grade_multiplier_heuristic(grade_pct,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    m_min=minetti_multiplier(grade_pct)
    m_heu=grade_multiplier_heuristic(grade_pct,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    w=max(0.0,min(1.0,float(minetti_weight)))
    return w*m_min+(1.0-w)*m_heu

def temp_multiplier(temp_eff, opt_temp, cold_quad, hot_quad, max_penalty):
    """
    Multiplicateur thermique calibré sur données physiologiques trail.
    hot_quad=0.0020 → à +5°C : +5%, à +10°C : +20% (plafonné)
    cold_quad=0.0015 → à -5°C : +3.75% (froid moins pénalisant)
    Température prise en compte même à +2°C de l'optimal.
    """
    if temp_eff is None: return 1.0
    d = float(temp_eff) - float(opt_temp)
    pen = hot_quad * d**2 if d >= 0 else cold_quad * (-d)**2
    return 1.0 + min(float(max_penalty), float(pen))

def wind_components(wind_speed_ms, wind_dir_from_deg, course_bearing_deg):
    """
    Décompose le vent en composante face/dos et latérale.

    Retourne (head_effective, tail_effective) en m/s.

    Physique :
    - Composante longitudinale : ws × cos(δ)
      δ=0° → vent pur face (pénalité max)
      δ=90° → vent côté (composante longitudinale nulle)
      δ=180° → vent pur dos (gain max)
    - Composante latérale : ws × |sin(δ)|
      Un vent de côté crée un effort de stabilisation équivalent à ~20%
      de sa valeur en vent de face (estimation physiologique conservative).
    - 3/4 face (δ=45°) : cos(45°)=0.71 + 0.20×|sin(45°)|=0.14 → 0.85×ws équivalent face

    wind_dir_from_deg : direction D'OÙ vient le vent (convention météo)
    """
    if wind_speed_ms is None or wind_dir_from_deg is None: return 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0: return 0.0, 0.0
    # Angle entre le vent et la direction de course
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians((wind_to - course_bearing_deg + 540.0) % 360.0 - 180.0)
    along = ws * math.cos(delta)              # composante longitudinale
    cross = ws * abs(math.sin(delta)) * 0.20  # composante latérale (×20% = coeff effort stabilisation)

    if along >= 0:
        # Vent de dos : cross réduit légèrement le gain (perturbations)
        tail_eff = max(0.0, along - cross * 0.3)
        head_eff = 0.0
    else:
        # Vent de face : cross s'ajoute à la pénalité
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

def fatigue_multiplier(d_plus_cum,dist_cum,d_plus_total,dist_total,rate_pct,mode):
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


# ══════════════════════════════════════════════════════════════
# SYSTÈME MÉTÉO ROBUSTE v7
# Stratégie par priorité :
#   1. API Open-Meteo Forecast  (J-0 → J+16)
#   2. API Open-Meteo Archive   (données passées)
#   3. API Open-Meteo Historical-Forecast (J-2ans → J+0)
#   4. Modèle thermique diurne  (TOUJOURS disponible, paramètres manuels)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# SYSTÈME MÉTÉO ROBUSTE v7.1
# ──────────────────────────────────────────────────────────────
# Garantie : retourne TOUJOURS une valeur non-None
# Priorité : Forecast API → Archive API → Historical API → Modèle diurne
# Coefficients recalibrés : hot_quad=0.0020, cap_head=0.12, cap_tail=-0.06
# Vent : composante directionnelle cos(δ) + bonus cross-wind
# ══════════════════════════════════════════════════════════════

def _diurnal_weather(hour_float, t_base, t_amp, wind_ms, humidity_pct, wind_dir_deg=180.0):
    """
    Modèle thermique diurne.
    T_min à 6h, T_max à 14h (cycle sinusoïdal demi-journée).
    Vent : +15% en milieu de journée (convection thermique).
    """
    T = t_base + t_amp * math.sin(math.pi * max(0.0, hour_float - 6.0) / 12.0)
    W = wind_ms * (1.0 + 0.15 * math.sin(math.pi * max(0.0, hour_float - 8.0) / 10.0))
    return {
        "temp":     round(float(T), 2),
        "wind":     round(float(max(0.0, W)), 2),
        "humidity": float(humidity_pct),
        "wind_dir": float(wind_dir_deg),
        "source":   "diurnal_model",
    }

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_openmeteo_forecast(lat, lon, tz_name):
    """Open-Meteo Forecast (J+0 → J+16)."""
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
    """Open-Meteo Archive (données historiques)."""
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
    """Open-Meteo Historical-Forecast (J-2ans → J+0)."""
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
    """Interpole les données météo à l'heure exacte de passage."""
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

# Cache session (dict Python) – réinitialisé à chaque lancement mais maintenu pendant le calcul
# Utilise st.session_state pour persister entre reruns Streamlit
def _get_session_meteo_cache():
    if "_meteo_api_cache" not in st.session_state:
        st.session_state["_meteo_api_cache"] = {}
    return st.session_state["_meteo_api_cache"]

def get_weather_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT,
                          fallback_temp=12.0, fallback_temp_amp=4.0,
                          fallback_wind=2.0, fallback_humidity=60.0,
                          fallback_wind_dir=180.0):
    """
    Météo à l'heure exacte de passage — TOUJOURS une valeur non-None.

    Logique :
    1. Tente les APIs selon la plage de dates (avec cache session_state)
    2. Si API indisponible → modèle thermique diurne (paramètres saisis dans l'UI)

    Le modèle diurne fait évoluer T et vent réellement sur la durée de la course.
    """
    hour = dt_local_naive.hour + dt_local_naive.minute / 60.0
    today = datetime.now()
    diff_days = (dt_local_naive.date() - today.date()).days
    date_str = dt_local_naive.strftime("%Y-%m-%d")
    lat_r = round(lat, 2); lon_r = round(lon, 2)
    cache_key = f"{date_str}_{lat_r}_{lon_r}"
    cache = _get_session_meteo_cache()

    # Tentative API (une seule fois par (date, position) grâce au cache)
    if cache_key not in cache:
        md = None
        if 0 <= diff_days <= 15:
            md = _fetch_openmeteo_forecast(lat_r, lon_r, tz_name)
        elif diff_days < 0:
            md = _fetch_openmeteo_archive(lat_r, lon_r, date_str, tz_name)
            if md is None:
                md = _fetch_openmeteo_histforecast(lat_r, lon_r, date_str, tz_name)
        else:
            # Futur lointain : forecast si disponible, sinon archives de l'an passé
            md = _fetch_openmeteo_forecast(lat_r, lon_r, tz_name)
            if md is None:
                past_date = dt_local_naive.replace(year=dt_local_naive.year - 1)
                md = _fetch_openmeteo_archive(lat_r, lon_r,
                                               past_date.strftime("%Y-%m-%d"), tz_name)
        cache[cache_key] = md

    # Interpolation
    result = _interp_meteo(cache.get(cache_key), dt_local_naive)
    if result is not None:
        return result

    # FALLBACK DIURNE — toujours fiable, évolue dans le temps
    return _diurnal_weather(hour, fallback_temp, fallback_temp_amp,
                            fallback_wind, fallback_humidity, fallback_wind_dir)


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
        return{"points":[{"lat":r[0],"lon":r[1],"elev":r[2],"dist":r[3],"time":t} for r,t in zip(records,times_pts)],
               "distance":float(df["dist"].max()),"D_up":dup,"D_down":ddn,
               "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
               "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,"hr_analysis":analyze_hr_v3(hr_records)}
    except Exception as e:st.error(f"Erreur FIT:{e}");return None

def parse_tcx_ref(file,tz_name=TZ_NAME_DEFAULT):
    try:file.seek(0);root=ET.parse(file).getroot()
    except:return None
    ns={"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    pts,times,elevs=[],[],[]
    for tp in root.findall(".//tcx:Trackpoint",ns):
        lat=tp.find("tcx:Position/tcx:LatitudeDegrees",ns);lon=tp.find("tcx:Position/tcx:LongitudeDegrees",ns)
        if lat is None or lon is None:continue
        ele=tp.find("tcx:AltitudeMeters",ns);tim=tp.find("tcx:Time",ns)
        elev=float(ele.text) if ele is not None else 0.0
        try:t=datetime.fromisoformat(tim.text.replace("Z","+00:00")).replace(tzinfo=None)
        except:t=None
        pts.append(SimplePoint(float(lat.text),float(lon.text),elev,t));times.append(t);elevs.append(elev)
    if len(pts)<2:return None
    vt=[t for t in times if t is not None]
    start_dt=vt[0] if vt else datetime.now()-timedelta(days=1)
    end_dt=vt[-1] if len(vt)>1 else start_dt+timedelta(minutes=5)
    avgT,avgW,avgH=get_avg_weather(pts[0].latitude,pts[0].longitude,start_dt,end_dt,tz_name)
    total=sum(pts[i].distance_3d(pts[i-1]) for i in range(1,len(pts)))
    dup,ddn=compute_dplus_dminus(elevs)
    return{"points":pts,"distance":round(total),"D_up":round(dup,1),"D_down":round(ddn,1),
           "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
           "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,"hr_analysis":None}

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

def build_holding_table(vc_ms,d_prime,refs_fit,K_riegel):
    if vc_ms is None or vc_ms<=0:return pd.DataFrame()
    X,Y=[],[]
    for r in refs_fit:
        d_m=float(r.get("distance",0));t_s=float(r.get("temps",0))
        if d_m>0 and t_s>0:X.append(math.log(d_m/1000.0));Y.append(math.log(t_s))
    if len(X)>=2:
        K_fit,loga,_,_,_=sp_stats.linregress(X,Y)
        K_fit=float(max(0.85,min(1.25,K_fit)));a_fit=math.exp(float(loga))
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
        K_fit,loga,_,_,_=sp_stats.linregress(X,Y)
        K_fit=float(max(0.85,min(1.25,K_fit)));a_fit=math.exp(float(loga))
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

def fit_loglog(refs):
    X,Y=[],[]
    for r in refs:
        d_m=safe_float(r.get("distance",0));t=r.get("temps")
        secs=float(t) if isinstance(t,(int,float,np.number)) else hms_to_seconds(str(t))
        if d_m<=0 or secs<=0:continue
        X.append(math.log(d_m/1000.0));Y.append(math.log(secs))
    if len(X)>=2:
        K,loga=np.polyfit(X,Y,1);K=float(max(0.85,min(1.25,K)));a=math.exp(float(loga))
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
                max_up=max_up,max_down=max_down,elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power)
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
    apply_fatigue,fatigue_rate,fatigue_mode,apply_ultra,ultra_amp,
    objective_hms,show_smooth_pace,smooth_window_km,dem_elevations,
    surface_mult=1.0,tz_name=TZ_NAME_DEFAULT,
    # Paramètres météo fallback (utilisés si API indisponible)
    meteo_fallback_temp=12.0, meteo_fallback_amp=4.0,
    meteo_fallback_wind=2.0,  meteo_fallback_humidity=60.0,
    meteo_fallback_wind_dir=180.0,
    # Pace sensitivity : atténuation conditions selon allure (défaut 360=6min/km)
    pace_sensitivity_ref=360.0):

    if not points or len(points)<2:raise ValueError("GPX invalide ou trop court.")
    if dem_elevations is not None and len(dem_elevations)==len(points):
        elev_arr=np.array([e if e is not None else 0.0 for e in dem_elevations],dtype=float)
    else:
        elev_arr=np.array([getattr(p,"elevation",0.0) or 0.0 for p in points],dtype=float)

    total_m=0.0;cum=[0.0]
    for i in range(1,len(points)):
        total_m+=haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude)
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
    elev_s=np.convolve(elev_arr,np.ones(w)/w,mode="same") if w>=3 and elev_arr.size>=w else elev_arr

    diffs_el=np.diff(elev_s);d_plus_total=float(np.sum(np.clip(diffs_el,0,None)))
    avg_alt=float(np.mean(elev_s))

    refs_fit=prepare_refs(refs_input,use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,
                           temp_max_penalty,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                           elev_ref_power,temp_ref_power)
    a,K=fit_loglog(refs_fit)
    # Estimation initiale de la base (sera raffinée si objectif fixé)
    base_total_s=predict_flat(int(distance_cible_km*1000),a,K)
    base_s_per_km=base_total_s/max(distance_cible_km,1e-9)*float(surface_mult)

    # Si objectif de temps : stocker obj_s pour la recherche itérative finale
    obj_s_target = hms_to_seconds(objective_hms) if objective_hms else None
    if obj_s_target and obj_s_target > 0:
        a_init = obj_s_target / (distance_cible_km**K)
        base_s_per_km = (a_init * (distance_cible_km ** K) / max(distance_cible_km,1e-9)) * float(surface_mult)

    # ── Pace sensitivity : facteur d'atténuation des multiplicateurs météo/vent ──
    # Plus l'allure est rapide, moins les conditions ont d'impact relatif.
    # pace_sens_factor = sqrt(base_s_per_km / pace_sens_ref_s)
    # Exemples : 3min/km → ×0.70, 5min/km → ×0.91, 6min/km → ×1.0 (référence)
    pace_sens_ref_s = float(pace_sensitivity_ref)  # paramètre UI (défaut 360 = 6min/km)
    pace_sens_factor = min(1.0, math.sqrt(base_s_per_km / max(1.0, pace_sens_ref_s)))

    alt_mult=altitude_vo2_multiplier(avg_alt,altitude_ref_m) if apply_altitude else 1.0

    km_marks=[i*1000 for i in range(1,int(total_corr//1000)+1)]
    last=total_corr-int(total_corr//1000)*1000
    if last>1e-6:km_marks.append(total_corr)

    lats_arr=np.array([p.latitude for p in points],dtype=float)
    lons_arr=np.array([p.longitude for p in points],dtype=float)
    dt_dep=datetime.combine(date_course,heure_course)

    pre=[];cum_t=cum_dp=cum_dist=0.0
    for i,d in enumerate(km_marks):
        seg_len=1000.0
        if i==len(km_marks)-1 and last>1e-6:seg_len=d-(km_marks[-2] if len(km_marks)>=2 else 0)
        e_cur=float(np.interp(d,dists_corr,elev_s))
        e_prv=float(np.interp(max(d-seg_len,0),dists_corr,elev_s)) if i>0 else e_cur
        grade=(e_cur-e_prv)/max(1e-6,seg_len)*100.0;seg_dp=max(0.0,e_cur-e_prv)
        cum_dp+=seg_dp;cum_dist+=seg_len;t_flat=base_s_per_km*(seg_len/1000.0)
        if apply_grade:
            gm=combined_grade_multiplier(grade,use_minetti,minetti_weight,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
            t1=t_flat*(gm**grade_power)
        else:gm=1.0;t1=t_flat
        t2=t1*alt_mult
        fm=fatigue_multiplier(cum_dp,cum_dist,d_plus_total,total_corr,fatigue_rate,fatigue_mode) if apply_fatigue and fatigue_rate>0 else 1.0
        t3=t2*fm
        passage_dt=dt_dep+timedelta(seconds=cum_t+t3/2.0)
        lat_s=float(np.interp(d,dists_corr,lats_arr));lon_s=float(np.interp(d,dists_corr,lons_arr))
        lat0=float(np.interp(max(d-seg_len,0),dists_corr,lats_arr));lon0=float(np.interp(max(d-seg_len,0),dists_corr,lons_arr))
        cap=bearing_deg(lat0,lon0,lat_s,lon_s)
        meteo=get_weather_minutely(lat_s, lon_s, passage_dt, tz_name,
                                    fallback_temp=meteo_fallback_temp,
                                    fallback_temp_amp=meteo_fallback_amp,
                                    fallback_wind=meteo_fallback_wind,
                                    fallback_humidity=meteo_fallback_humidity,
                                    fallback_wind_dir=meteo_fallback_wind_dir)
        temp_raw=meteo["temp"] if meteo else None;wind_raw=meteo["wind"] if meteo else None
        hum_raw=meteo["humidity"] if meteo else None;wdir_raw=meteo.get("wind_dir") if meteo else None
        temp_eff_val=None
        if temp_raw is not None and hum_raw is not None:temp_eff_val=effective_temp(temp_raw,hum_raw,use_wbgt)
        if temp_eff_val is not None:
            tm_raw=temp_multiplier(temp_eff_val,opt_temp,cold_quad,hot_quad,temp_max_penalty)
            # Atténuation pace-sensitive : coureur rapide = moins sensible aux conditions
            tm=1.0+(tm_raw-1.0)*pace_sens_factor
            t4=t3*(tm**temp_power)
        else:tm=1.0;t4=t3
        pace_local=(t4/seg_len)*1000.0 if seg_len>0 else t4
        # Composantes vent atténuées selon pace_sensitivity
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
        # Appliquer le vent sur t_no_wind (qui contient déjà pente + temp + fatigue)
        t_w=float(row["t_no_wind"])*(wm_gated**wind_power)
        # cap_combined : limiter UNIQUEMENT la composante pente pure
        # (pas le produit total qui inclut météo+fatigue → aplatirait tout)
        gm_capped=cap_combined(float(row["grade_mult"]),g,base_cap,extra_per_pct,max_cap)
        # Reconstruire t_final : remplacer grade_mult par grade_mult_capped
        gm_raw=float(row["grade_mult"])
        if gm_raw > 0:
            t_final=t_w*(gm_capped/gm_raw)
        else:
            t_final=t_w
        t_raw.append(float(t_final));wm_adj_list.append(wm_gated)
    df_pre["wind_mult_adj"]=wm_adj_list;t_raw=np.array(t_raw,dtype=float)

    if apply_ultra and ultra_amp>0:
        t_raw=apply_ultra_pacing(t_raw,df_pre["d"].values,df_pre["seg_len"].values,total_corr,ultra_amp)

    # ── Ajustement objectif de temps — SANS normalisation uniforme ──────────────
    # Principe : on ne multiplie PAS tous les km par le même facteur (ce qui écraserait
    # les variations pente/météo/vent). On recherche par bissection le base_s_per_km
    # tel que sum(t_raw) = obj_s, en PRÉSERVANT les ratios relatifs entre km.
    #
    # t_raw[i] = base_s_per_km * mult_total[i]
    # On extrait mult_total[i] = t_raw[i] / base_s_per_km_initial
    # Puis on cherche base_opt tel que sum(base_opt * mult_total[i]) = obj_s
    # → base_opt = obj_s / sum(mult_total[i])
    # (C'est une équation linéaire, pas besoin de bissection ici !)
    if obj_s_target and obj_s_target > 0 and float(np.sum(t_raw)) > 0:
        # Extraire les multiplicateurs totaux implicites
        # t_raw = base_s_per_km * mult_total * seg_len/1000
        seg_lens_km = df_pre["seg_len"].values / 1000.0
        mult_totaux = t_raw / (base_s_per_km * seg_lens_km + 1e-9)
        # Base optimale : obj_s = sum(base_opt * mult_totaux * seg_lens_km)
        sum_weighted_mults = float(np.sum(mult_totaux * seg_lens_km))
        if sum_weighted_mults > 0:
            base_opt = obj_s_target / sum_weighted_mults
            # Recalculer t_raw avec la nouvelle base, en préservant les multiplicateurs
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
    # Calcul amplitude météo sur la course (info pour l'utilisateur)
    temps_valides = [r["Temp GPS (°C)"] for _,r in pd.DataFrame(rows).iterrows() if r.get("Temp GPS (°C)") is not None]
    meteo_range = None
    if len(temps_valides) >= 2:
        meteo_range = {"t_min": round(min(temps_valides),1), "t_max": round(max(temps_valides),1),
                       "delta": round(max(temps_valides)-min(temps_valides),1)}
    # Fourchette : basée sur distances officielles proches du GPX
    # 10km=10000m, semi=21097m, marathon=42195m, 50k=50000m, 100k=100000m
    _std_dists = {
        "10 km":    10000,   "Semi":     21097,
        "Marathon": 42195,   "50 km":    50000,
        "100 km":   100000,
    }
    _gpx_m = dist_gpx_km * 1000.0
    # Trouver les 2 distances standards encadrantes
    _below = {k:v for k,v in _std_dists.items() if v <= _gpx_m * 1.05}
    _above = {k:v for k,v in _std_dists.items() if v >= _gpx_m * 0.95}
    _ci_ref_low = max(_below.values()) if _below else None
    _ci_ref_high = min(_above.values()) if _above else None
    _label_low = next((k for k,v in _std_dists.items() if v == _ci_ref_low), None)
    _label_high = next((k for k,v in _std_dists.items() if v == _ci_ref_high), None)
    # Temps projeté sur la distance standard (extrapolation linéaire depuis allure moy)
    _avg_pace = total_s / max(_gpx_m, 1.0)  # s/m
    _ci_low_hms  = seconds_to_hms(_avg_pace * _ci_ref_low)  if _ci_ref_low  else seconds_to_hms(total_s*0.97)
    _ci_high_hms = seconds_to_hms(_avg_pace * _ci_ref_high) if _ci_ref_high else seconds_to_hms(total_s*1.03)
    _ci_low_label  = _label_low  if _label_low  else "−3%"
    _ci_high_label = _label_high if _label_high else "+3%"

    return{"df":df_out,"total_s":total_s,"total_human":seconds_to_hms(total_s),
           "ci_low":_ci_low_hms,"ci_high":_ci_high_hms,
           "ci_low_label":_ci_low_label,"ci_high_label":_ci_high_label,
           "meteo_range":meteo_range,
           "dist_gpx_km":dist_gpx_km,"K":K,"avg_alt":avg_alt,"d_plus_total":d_plus_total,
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



# ══════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████
# NOUVEAU v6 — DÉTECTION TERRAIN TECHNIQUE
# ██████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════

def detect_technical_terrain(points, dem_elevations=None, seg_len_m=1000, is_trail=True):
    """
    Analyse le tracé GPX km par km — VERSION CORRIGÉE v6.1

    Corrections :
    - Virages < 45° ignorés (ne contribuent pas — route normale)
    - Virages 45–90° : contribution réduite (×0.3)
    - Virages > 90° (lacets réels) : contribution pleine
    - Bonus ×1.5 si pente > 15% ET virage > 90° simultanément
    - Seuil "technique" relevé à 0.45 (évite faux positifs sur route)
    - En mode Route (is_trail=False) : retourne toujours score 0, pas d'analyse
    - Surface : seules les pentes > 10% et l'irrégularité comptent vraiment

    Retourne segments, global_info
    """
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
        d_start = km_marks[ki]
        d_end   = km_marks[ki + 1]
        idx_seg = [i for i in range(n) if d_start <= cum[i] <= d_end]
        if len(idx_seg) < 3:
            continue

        pts_seg  = [points[i] for i in idx_seg]
        elev_seg = [elevs[i]  for i in idx_seg]
        dist_seg = [cum[i] - d_start for i in idx_seg]

        # ── Sinuosité ──────────────────────────────────────────
        d_gps  = dist_seg[-1]
        d_eucl = haversine_m(pts_seg[0].latitude,  pts_seg[0].longitude,
                             pts_seg[-1].latitude, pts_seg[-1].longitude)
        sinuosity = d_gps / max(1.0, d_eucl)

        # ── Changements de direction — CORRIGÉS ───────────────
        # Seulement les vrais lacets (> 45°), avec pondération par angle
        bearings = []
        for i in range(1, len(pts_seg)):
            dd = dist_seg[i] - dist_seg[i-1]
            if dd > 2.0:  # ignorer points trop proches (bruit GPS)
                b = bearing_deg(pts_seg[i-1].latitude, pts_seg[i-1].longitude,
                                pts_seg[i].latitude,   pts_seg[i].longitude)
                bearings.append(b)

        real_turns_score = 0.0  # score pondéré des vrais virages
        if len(bearings) >= 2:
            for i in range(1, len(bearings)):
                delta = abs((bearings[i] - bearings[i-1] + 180) % 360 - 180)
                if delta < 45:
                    pass  # route normale, ignoré
                elif delta < 90:
                    real_turns_score += delta * 0.30   # virage modéré
                elif delta < 135:
                    real_turns_score += delta * 0.80   # lacet serré
                else:
                    real_turns_score += delta * 1.20   # épingle à cheveux

        real_turns_per_km = real_turns_score / max(0.001, d_gps / 1000.0)

        # ── Pentes ─────────────────────────────────────────────
        grades = []
        for i in range(1, len(pts_seg)):
            dd = dist_seg[i] - dist_seg[i-1]
            if dd > 0.5:
                grades.append((elev_seg[i] - elev_seg[i-1]) / dd * 100.0)
        if not grades:
            grades = [0.0]

        grade_abs  = [abs(g) for g in grades]
        grade_max  = float(np.max(grade_abs))
        grade_std  = float(np.std(grades))
        grade_mean = float(np.mean(grades))

        # Seules les pentes > 10% sont vraiment pénalisantes pour la technicité
        steep_grades = [g for g in grade_abs if g > 10.0]
        steep_ratio  = len(steep_grades) / max(1, len(grade_abs))  # fraction de pentes raides

        # ── Score technique CORRIGÉ ────────────────────────────
        # Normalisation empirique ajustée pour éviter faux positifs route
        norm_sinu  = min(1.0, max(0.0, (sinuosity - 1.05) / 0.35))   # commence à 1.05 (pas 1.0)
        norm_turns = min(1.0, real_turns_per_km / 900.0)              # 900°/km = lacets serrés réels
        norm_grade = min(1.0, grade_max / 35.0)                       # 35% = pente extrême trail
        norm_steep = min(1.0, steep_ratio / 0.40)                     # 40% des pts > 10% = très raide
        norm_std   = min(1.0, grade_std / 12.0)                       # 12% std = très irrégulier

        # Bonus synergique : lacet + forte pente simultanément = vraiment technique
        synergy_bonus = 0.0
        if grade_max > 15.0 and real_turns_per_km > 300.0:
            synergy_bonus = 0.10  # bonus si vrai lacet en côte raide

        tech_score = (
            0.20 * norm_sinu   +
            0.30 * norm_turns  +  # virages corrigés (poids principal)
            0.20 * norm_grade  +
            0.15 * norm_steep  +
            0.15 * norm_std    +
            synergy_bonus
        )
        tech_score = min(1.0, tech_score)

        # Seuil relevé : < 0.30 = non technique (évite faux positifs route)
        if tech_score < 0.25:   label = "🟢 Facile"
        elif tech_score < 0.45: label = "🟡 Modéré"
        elif tech_score < 0.70: label = "🟠 Technique"
        else:                   label = "🔴 Très technique"

        # Multiplicateurs suggestions — conservateurs
        k_up_adj_seg   = 1.0 + 0.30 * max(0, tech_score - 0.25)  # actif seulement si > Facile
        k_down_adj_seg = 1.0 - 0.15 * max(0, tech_score - 0.25)
        surf_adj_seg   = 1.0 + 0.12 * max(0, tech_score - 0.25)

        segments.append({
            "km_start":       round(d_start / 1000.0, 1),
            "km_end":         round(d_end   / 1000.0, 1),
            "sinuosity":      round(sinuosity, 3),
            "turns_score_km": round(real_turns_per_km, 0),
            "grade_max":      round(grade_max, 1),
            "grade_std":      round(grade_std, 1),
            "grade_mean":     round(grade_mean, 1),
            "steep_ratio":    round(steep_ratio, 2),
            "tech_score":     round(tech_score, 3),
            "label":          label,
            "k_up_adj":       round(k_up_adj_seg, 3),
            "k_down_adj":     round(k_down_adj_seg, 3),
            "surface_adj":    round(surf_adj_seg, 3),
        })

    if not segments:
        return [], {"global_score": 0, "label": "—", "k_up_adj": 1.0,
                    "k_down_adj": 1.0, "surface_mult_adj": 1.0}

    global_score   = float(np.mean([s["tech_score"] for s in segments]))
    max_score      = float(np.max([s["tech_score"] for s in segments]))
    weighted_score = 0.65 * global_score + 0.35 * max_score  # passage dur compte moins

    if weighted_score < 0.25:   global_label = "🟢 Terrain facile / Non technique"
    elif weighted_score < 0.42: global_label = "🟡 Trail modéré — quelques passages techniques"
    elif weighted_score < 0.62: global_label = "🟠 Trail technique — rochers, lacets, pentes"
    else:                       global_label = "🔴 Ultra-trail très technique"

    k_up_g   = float(np.median([s["k_up_adj"]   for s in segments]))
    k_down_g = float(np.median([s["k_down_adj"]  for s in segments]))
    surf_g   = float(np.median([s["surface_adj"] for s in segments]))

    global_info = {
        "global_score":      round(weighted_score, 3),
        "label":             global_label,
        "k_up_adj":          round(k_up_g, 3),
        "k_down_adj":        round(k_down_g, 3),
        "surface_mult_adj":  round(surf_g, 3),
        "pct_technique":     round(sum(1 for s in segments if s["tech_score"] > 0.45) / len(segments) * 100, 1),
        "sinuosity_mean":    round(float(np.mean([s["sinuosity"] for s in segments])), 3),
        "grade_max_all":     round(float(np.max([s["grade_max"] for s in segments])), 1),
    }
    return segments, global_info



# ══════════════════════════════════════════════════════════════
# NOUVEAU v6 — SURFACE OSM VIA OVERPASS API
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Analyse surface OSM en cours...")
def fetch_osm_surface(lats_tuple, lons_tuple):
    """
    Interroge l'API Overpass (gratuite, sans clé) pour récupérer
    le type de surface des chemins/routes le long du tracé.
    Retourne dominant_surface, surface_mult_osm pondéré, detail.
    """
    OSM_SURFACE_MULT = {
        "paved": 1.00, "asphalt": 1.00, "concrete": 1.00,
        "cobblestone": 1.08, "sett": 1.08, "unhewn_cobblestone": 1.10,
        "compacted": 1.03, "fine_gravel": 1.04, "gravel": 1.05, "pebblestone": 1.07,
        "unpaved": 1.07, "ground": 1.08, "dirt": 1.08, "earth": 1.08,
        "mud": 1.18, "sand": 1.20,
        "grass": 1.09, "grass_paver": 1.07,
        "rock": 1.12, "rocks": 1.12, "stone": 1.10,
        "wood": 1.06, "woodchips": 1.07,
        "snow": 1.18, "ice": 1.25,
        "unknown": 1.06,
    }
    lats = list(lats_tuple)
    lons = list(lons_tuple)
    lat_min = min(lats) - 0.005; lat_max = max(lats) + 0.005
    lon_min = min(lons) - 0.005; lon_max = max(lons) + 0.005
    query = f"""[out:json][timeout:25];
(
  way["highway"]["surface"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["highway"]["tracktype"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out body geom;"""
    try:
        # Fallback sur plusieurs endpoints Overpass (réseau variable)
        OVERPASS_ENDPOINTS = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.fr/api/interpreter",
        ]
        data = None
        last_err = None
        for ep in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(ep, data={"data": query}, timeout=25)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                continue
        if data is None:
            raise Exception(f"Tous les endpoints Overpass injoignables. Dernier: {last_err}")
    except Exception as e:
        return {"error": str(e), "dominant_surface": "unknown",
                "surface_mult_osm": 1.06, "surface_counts": {}, "detail": []}
    ways = data.get("elements", [])
    if not ways:
        return {"dominant_surface": "unknown", "surface_mult_osm": 1.06,
                "surface_counts": {}, "detail": [], "ways_found": 0}
    step = max(1, len(lats) // 80)
    sample_lats = lats[::step]; sample_lons = lons[::step]
    surface_hits = []; detail = []
    for si, (slat, slon) in enumerate(zip(sample_lats, sample_lons)):
        best_dist = float("inf"); best_surface = None; best_highway = None
        for way in ways:
            geom = way.get("geometry", [])
            surface  = way.get("tags", {}).get("surface")
            tracktype = way.get("tags", {}).get("tracktype")
            highway  = way.get("tags", {}).get("highway", "")
            if not surface and tracktype:
                surface = {"grade1":"compacted","grade2":"fine_gravel","grade3":"gravel",
                           "grade4":"unpaved","grade5":"ground"}.get(tracktype,"unpaved")
            if not surface: continue
            for node in geom[:20]:
                nd = haversine_m(slat, slon, node["lat"], node["lon"])
                if nd < best_dist:
                    best_dist = nd; best_surface = surface; best_highway = highway
        if best_surface and best_dist < 50:
            surface_hits.append(best_surface.lower())
            detail.append({"km": round(si * step / max(1,len(lats)) * (len(lats)/1000.0), 1),
                           "surface": best_surface, "highway": best_highway or "—",
                           "dist_m": round(best_dist, 0)})
    if not surface_hits:
        return {"dominant_surface": "unknown", "surface_mult_osm": 1.06,
                "surface_counts": {}, "detail": [], "ways_found": len(ways)}
    from collections import Counter
    counts = Counter(surface_hits)
    dominant = counts.most_common(1)[0][0]
    total = sum(counts.values())
    weighted_mult = sum(OSM_SURFACE_MULT.get(s, 1.06) * cnt / total for s, cnt in counts.items())
    return {
        "dominant_surface": dominant,
        "surface_mult_osm": round(weighted_mult, 3),
        "surface_counts":   dict(counts.most_common(10)),
        "detail":           detail[:30],
        "ways_found":       len(ways),
        "coverage_pct":     round(len(surface_hits) / len(sample_lats) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════
# NOUVEAU v6 — VUE 3D RELIEF RÉEL PYDECK
# ══════════════════════════════════════════════════════════════

def generate_3d_terrain_html(points, cum_d_map, checkpoints,
                               df_prediction=None, tech_segments=None,
                               dem_elevations=None, osm_surface_data=None,
                               height=600, pitch=52):
    """
    Génère un composant HTML autonome qualité Google Earth Pro :
    - deck.gl 8.9 + MapLibre GL (open source, ZÉRO token, ZÉRO compte)
    - Relief 3D réel ArcGIS World Elevation (tuiles publiques)
    - Fond satellite ESRI World Imagery
    - 5 couches toggle : Relief / Tracé altimétrique / Allure prédite / Surfaces OSM / Zones techniques
    - Interface UI propre, légende dynamique, stats live
    - Tracé qui suit le vrai relief (pas à plat)
    """
    import json as _json
    import math as _math

    n_pts = len(points)
    step  = max(1, n_pts // 700)

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

    # ── D+ total ─────────────────────────────────────────────────
    d_plus = float(np.sum(np.clip(np.diff(elevs_s), 0, None)))
    total_km = cum_d_map[-1] / 1000.0

    # ── Construire les segments de tracé (avec élévation 3D) ─────
    seg_size = max(1, n_sub // 80)
    trace_data = []
    for si in range(0, n_sub - 1, seg_size):
        end_i = min(si + seg_size + 1, n_sub)
        path  = [[lons_s[j], lats_s[j], elevs_s[j] + 4] for j in range(si, end_i)]
        if len(path) < 2:
            continue
        mid_i = (si + end_i) // 2
        mid_i = min(mid_i, n_sub - 1)
        trace_data.append({
            "path":      path,
            "elevation": elevs_s[mid_i],
            "dist":      round(dist_s[mid_i], 2),
        })

    # ── Données allure prédite ──────────────────────────────────
    pace_data = []
    if df_prediction is not None and not df_prediction.empty and "Allure (min/km)" in df_prediction.columns:
        def _p2s(p):
            try: parts = str(p).split(":"); return int(parts[0])*60+int(parts[1])
            except: return 0
        for idx_row, row in df_prediction.iterrows():
            ps = _p2s(row["Allure (min/km)"])
            if ps <= 0: continue
            km_idx = idx_row
            # Interpoler la position GPS pour ce km
            d_m = (km_idx + 0.5) * 1000.0
            if d_m > cum_d_map[-1]: d_m = cum_d_map[-1]
            seg_lat = float(np.interp(d_m, cum_d_map, [points[i].latitude  for i in range(n_pts)]))
            seg_lon = float(np.interp(d_m, cum_d_map, [points[i].longitude for i in range(n_pts)]))
            seg_el  = float(np.interp(d_m, cum_d_map, elevs_s[:n_sub] if n_sub == n_pts else elevs_s))
            # Segment de 1km
            d_start = km_idx * 1000.0
            d_end   = min((km_idx + 1) * 1000.0, cum_d_map[-1])
            idx_s2  = [i for i in range(0, n_sub) if d_start <= (i * (cum_d_map[-1]/(n_sub-1) if n_sub>1 else 1)) <= d_end]
            if len(idx_s2) < 2:
                path_seg = [[seg_lon, seg_lat, seg_el+4]]
            else:
                path_seg = [[lons_s[j], lats_s[j], elevs_s[j]+4] for j in idx_s2]
            pace_data.append({"path": path_seg, "pace_s": ps, "dist": round(d_m/1000,1)})

    # ── Données surfaces OSM ────────────────────────────────────
    osm_segs = []
    if osm_surface_data and osm_surface_data.get("detail"):
        for d_entry in osm_surface_data["detail"]:
            km_k = d_entry.get("km", 0)
            surf = d_entry.get("surface", "unknown")
            d_m  = km_k * 1000.0
            d_end_m = min(d_m + 1000.0, cum_d_map[-1])
            idx_s3 = [i for i in range(n_sub) if d_m <= dist_s[i]*1000 <= d_end_m]
            if len(idx_s3) < 2: continue
            path_seg = [[lons_s[j], lats_s[j], elevs_s[j]+4] for j in idx_s3]
            osm_segs.append({"path": path_seg, "surface": surf, "dist": round(km_k, 1)})

    # ── Zones techniques ────────────────────────────────────────
    tech_data_js = []
    if tech_segments:
        for seg in tech_segments:
            if seg.get("tech_score", 0) > 0.45:
                d_mid = (seg["km_start"] + seg["km_end"]) / 2.0 * 1000.0
                sl = float(np.interp(d_mid, cum_d_map, [points[i].latitude  for i in range(n_pts)]))
                so = float(np.interp(d_mid, cum_d_map, [points[i].longitude for i in range(n_pts)]))
                se = float(np.interp(d_mid, cum_d_map, elevs_s[:n_pts] if len(elevs_s)==n_pts else
                                     [getattr(points[i],"elevation",0.0) or 0.0 for i in range(n_pts)]))
                ts = seg["tech_score"]
                tech_data_js.append({
                    "position": [so, sl, se + 25],
                    "label":    f"{seg['label']} {seg['km_start']:.0f}-{seg['km_end']:.0f}km",
                    "score":    round(ts, 2),
                    "color":    [min(255,int(249*min(1,ts*1.4))), int(80*(1-ts)), 30, 210],
                    "radius":   int(80 + ts * 120),
                })

    # ── Checkpoints ────────────────────────────────────────────
    CP_COL = {
        "🥤 Ravitaillement":[0,229,255,230], "🏔 Sommet":[255,214,0,230],
        "🔻 Col":[224,64,251,230], "⏱ Point de passage":[64,196,255,230],
        "🏁 Intermédiaire":[180,180,180,200], "⚠️ Point clé":[255,23,68,230],
    }
    cp_data_js = []
    for cp in sorted(checkpoints, key=lambda c: c["dist_km"]):
        cp_el = float(np.interp(cp["dist_km"]*1000, cum_d_map,
                                [getattr(points[i],"elevation",0.0) or 0.0 for i in range(n_pts)]))
        cp_data_js.append({
            "position": [cp["lon"], cp["lat"], cp_el + 18],
            "label":    cp["label"],
            "dist":     cp["dist_km"],
            "color":    CP_COL.get(cp.get("type",""), [249,115,22,230]),
        })

    # ── Sérialiser les données ───────────────────────────────────
    def js(obj): return _json.dumps(obj, ensure_ascii=False)

    view_state = {
        "longitude": center_lon, "latitude": center_lat,
        "zoom": round(zoom, 1), "pitch": pitch, "bearing": 0,
        "minPitch": 0, "maxPitch": 85,
    }

    # ── Template HTML ────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trail 3D — {total_km:.1f} km</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:#060c18;font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0;overflow:hidden}}
#c{{width:100%;height:100%;position:relative}}
#ui{{position:absolute;top:10px;left:10px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:12px;padding:11px 15px;min-width:200px;user-select:none}}
#ui h3{{font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;color:#64748b;margin-bottom:9px;font-weight:600}}
.tog{{display:flex;align-items:center;gap:8px;margin:4px 0;cursor:pointer;font-size:.76rem;padding:2px 0}}
.tog input{{accent-color:#f97316;cursor:pointer;flex-shrink:0}}
.tog label{{cursor:pointer;color:#cbd5e1;transition:color .15s}}
.tog:hover label,.tog input:checked+label{{color:#f97316}}
.sep{{border:none;border-top:1px solid rgba(255,255,255,.07);margin:8px 0}}
.btn-grp{{display:flex;gap:5px;margin-top:4px}}
.btn{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#94a3b8;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:.68rem;transition:all .15s}}
.btn:hover,.btn.on{{background:rgba(249,115,22,.25);border-color:#f97316;color:#fff}}
#hud{{position:absolute;bottom:12px;left:12px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:10px;padding:8px 14px;display:flex;gap:18px}}
.hud-item{{display:flex;flex-direction:column;align-items:center}}
.hud-val{{font-size:.85rem;font-weight:700;color:#f97316;font-family:'Courier New',monospace}}
.hud-lbl{{font-size:.58rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-top:1px}}
#leg{{position:absolute;bottom:12px;right:12px;z-index:200;background:rgba(6,12,24,.90);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.11);border-radius:10px;padding:9px 13px;min-width:160px}}
#leg-title{{font-size:.68rem;color:#94a3b8;margin-bottom:5px;font-weight:600}}
#leg-bar{{width:100%;height:10px;border-radius:3px;margin:3px 0}}
#leg-labs{{display:flex;justify-content:space-between;font-size:.60rem;color:#475569}}
#tooltip{{position:absolute;pointer-events:none;z-index:300;background:rgba(6,12,24,.92);border:1px solid rgba(249,115,22,.4);border-radius:8px;padding:7px 12px;font-size:.76rem;color:#e2e8f0;max-width:220px;display:none}}
</style>
</head>
<body>
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
  <div style="font-size:.63rem;color:#374151">Glisser : rotation · Scroll : zoom<br>Ctrl+glisser : déplacer</div>
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
const TRACE  = {js(trace_data)};
const PACE   = {js(pace_data)};
const OSM    = {js(osm_segs)};
const TECH   = {js(tech_data_js)};
const CPS    = {js(cp_data_js)};
const BOUNDS = {js(bounds)};
const EMIN={round(elev_min,1)},EMAX={round(elev_max,1)};
const {{Deck,TerrainLayer,PathLayer,ScatterplotLayer,TextLayer}} = deck;

// ── Couleurs ─────────────────────────────────────────────────────
function altC(t){{
  if(t<.33){{const u=t/.33;return[Math.round(20+u*80),Math.round(100+u*120),Math.round(255-u*80)];}}
  if(t<.66){{const u=(t-.33)/.33;return[Math.round(100+u*140),Math.round(220-u*60),Math.round(175-u*120)];}}
  const u=(t-.66)/.34;return[240,Math.round(160-u*120),Math.round(55-u*40)];
}}
function paceC(t){{return[Math.round(30+t*219),Math.round(200-t*160),Math.round(80-t*60)];}}
const OSMC={{asphalt:[120,120,120],paved:[140,140,140],concrete:[160,160,160],compacted:[200,170,100],gravel:[180,140,80],unpaved:[160,120,60],ground:[139,90,43],dirt:[130,80,30],grass:[60,160,60],rock:[100,100,80],mud:[80,50,20],snow:[220,235,255],ice:[180,220,255],unknown:[150,150,150]}};
function osmC(s){{return OSMC[s]||OSMC.unknown;}}

// ── Légende ──────────────────────────────────────────────────────
function drawLeg(title,cfn,lbl0,lbl1,isText){{
  document.getElementById('leg-title').textContent=title;
  document.getElementById('leg-labs').innerHTML='<span>'+lbl0+'</span><span>'+lbl1+'</span>';
  const cv=document.getElementById('leg-bar');
  const ctx=cv.getContext('2d');
  if(isText){{ctx.fillStyle='rgba(100,116,139,.3)';ctx.fillRect(0,0,cv.width,cv.height);return;}}
  const g=ctx.createLinearGradient(0,0,cv.width,0);
  for(let i=0;i<=12;i++){{const[r,b,c]=cfn(i/12);g.addColorStop(i/12,`rgb(${{r}},${{b}},${{c}})`);}}
  ctx.fillStyle=g;ctx.fillRect(0,0,cv.width,cv.height);
}}
drawLeg('Altitude',altC,Math.round(EMIN)+' m',Math.round(EMAX)+' m',false);

// ── État ─────────────────────────────────────────────────────────
const S={{l0:true,l1:true,l2:false,l3:false,l4:true,l5:true}};
let currentPitch={pitch};

// ── deck.gl ──────────────────────────────────────────────────────
const deckInst = new Deck({{
  container:'c',
  initialViewState:{js(view_state)},
  controller:{{touchRotate:true,touchZoom:true,scrollZoom:true,dragRotate:true}},
  getTooltip:({{object}})=>{{
    if(!object)return null;
    const tt=document.getElementById('tooltip');
    tt.style.display='none';
    return null;
  }},
  onHover:({{object,x,y}})=>{{
    const tt=document.getElementById('tooltip');
    if(object&&(object.label||object.surface||object.dist!==undefined)){{
      let html='';
      if(object.label) html+=`<b>${{object.label}}</b>`;
      if(object.dist!==undefined) html+=`<br>📍 ${{typeof object.dist==='number'?object.dist.toFixed(1):object.dist}} km`;
      if(object.score!==undefined) html+=`<br>⚠️ Score tech: ${{object.score}}`;
      if(object.surface) html+=`<br>🌿 Surface: ${{object.surface}}`;
      if(object.elevation!==undefined) html+=`<br>🏔 Alt: ${{Math.round(object.elevation)}} m`;
      tt.innerHTML=html;
      tt.style.left=(x+14)+'px';tt.style.top=(y-10)+'px';tt.style.display='block';
    }} else {{ tt.style.display='none'; }}
  }},
  layers:[],
}});

function buildLayers(){{
  const L=[];
  // Relief 3D
  if(S.l0) L.push(new TerrainLayer({{
    id:'terrain',minZoom:0,maxZoom:13,
    elevationDecoder:{{rScaler:6553.6,gScaler:25.6,bScaler:0.1,offset:-10000}},
    elevationData:'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer/tile/{{z}}/{{y}}/{{x}}',
    texture:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    bounds:BOUNDS,color:[255,255,255],
  }}));
  // Tracé
  if(S.l1){{
    let data,cfnLeg,l0,l1;
    if(S.l2&&PACE.length>0){{
      const ps=PACE.map(p=>p.pace_s),pmn=Math.min(...ps),pmx=Math.max(...ps);
      data=PACE.map(p=>{{const t=(p.pace_s-pmn)/Math.max(1,pmx-pmn);return{{...p,color:[...paceC(t),220]}}}});
      const fmt=s=>Math.floor(s/60)+':'+(s%60<10?'0':'')+Math.round(s%60)+'/km';
      drawLeg('⏱ Allure',paceC,fmt(pmn),fmt(pmx),false);
    }} else if(S.l3&&OSM.length>0){{
      data=OSM.map(s=>{{return{{...s,color:[...osmC(s.surface),220]}}}});
      drawLeg('🌿 Surfaces',null,'Béton→Terre→Roche','Herbe→Boue',true);
    }} else {{
      data=TRACE.map(s=>{{const t=(s.elevation-EMIN)/Math.max(1,EMAX-EMIN);return{{...s,color:[...altC(t),220]}}}});
      drawLeg('Altitude',altC,Math.round(EMIN)+' m',Math.round(EMAX)+' m',false);
    }}
    L.push(new PathLayer({{id:'trace',data,getPath:d=>d.path,getColor:d=>d.color,getWidth:10,widthMinPixels:3,widthMaxPixels:15,pickable:true}},));
  }}
  // Zones tech
  if(S.l4&&TECH.length>0){{
    L.push(new ScatterplotLayer({{id:'tech',data:TECH,getPosition:d=>d.position,getColor:d=>d.color,getRadius:d=>d.radius,radiusMinPixels:10,radiusMaxPixels:30,pickable:true}}));
    L.push(new TextLayer({{id:'tech_lbl',data:TECH,getPosition:d=>d.position,getText:d=>d.label,getSize:11,getColor:[255,220,50,220],background:true,getBackgroundColor:[0,0,0,150],getPixelOffset:[0,-30],billboard:true}}));
  }}
  // Checkpoints
  if(S.l5&&CPS.length>0){{
    L.push(new ScatterplotLayer({{id:'cp',data:CPS,getPosition:d=>d.position,getColor:d=>d.color,getRadius:100,radiusMinPixels:8,radiusMaxPixels:22,pickable:true}}));
    L.push(new TextLayer({{id:'cp_lbl',data:CPS,getPosition:d=>d.position,getText:d=>d.label,getSize:13,getColor:[255,255,255,230],background:true,getBackgroundColor:[0,0,0,160],getPixelOffset:[0,-24],billboard:true}}));
  }}
  // Départ/Arrivée
  if(TRACE.length>0){{
    const se=[
      {{position:TRACE[0].path[0],color:[0,255,100,255],r:160,label:'🟢 Départ'}},
      {{position:TRACE[TRACE.length-1].path[TRACE[TRACE.length-1].path.length-1],color:[255,50,50,255],r:160,label:'🔴 Arrivée'}},
    ];
    L.push(new ScatterplotLayer({{id:'se',data:se,getPosition:d=>d.position,getColor:d=>d.color,getRadius:d=>d.r,radiusMinPixels:12,radiusMaxPixels:28,pickable:true}}));
  }}
  return L;
}}

function refresh(){{deckInst.setProps({{layers:buildLayers()}});}}
refresh();

// ── Toggles ──────────────────────────────────────────────────────
for(let i=0;i<6;i++){{
  const el=document.getElementById('l'+i);
  el.addEventListener('change',()=>{{S['l'+i]=el.checked;refresh();}});
}}

// ── Boutons de vue ───────────────────────────────────────────────
function setView(p){{
  currentPitch=p;
  deckInst.setProps({{initialViewState:{{...{js(view_state)},pitch:p,transitionDuration:600}}}});
  ['v0','v1','v2'].forEach(id=>document.getElementById(id).classList.remove('on'));
  if(p===52)document.getElementById('v0').classList.add('on');
  else if(p===0)document.getElementById('v1').classList.add('on');
  else document.getElementById('v2').classList.add('on');
}}
</script>
</body>
</html>"""
    return html

# ── Garder generate_pydeck_terrain comme alias pour compatibilité ──
def generate_pydeck_terrain(points, cum_d_map, checkpoints, df_prediction=None,
                             tech_segments=None, dem_elevations=None,
                             osm_surface_data=None, active_layers=None,
                             height=600, pitch=52):
    """Alias de compatibilité -> délègue à generate_3d_terrain_html."""
    html = generate_3d_terrain_html(
        points=points, cum_d_map=cum_d_map, checkpoints=checkpoints,
        df_prediction=df_prediction, tech_segments=tech_segments,
        dem_elevations=dem_elevations, osm_surface_data=osm_surface_data,
        height=height, pitch=pitch,
    )
    return html  # retourne HTML string au lieu d'un deck pydeck object



# ══════════════════════════════════════════════════════════════
# TERRAIN PROFILES — Trail modéré calibré sur historique course
# Jornet 3:38 / Elazzaoui 3:42-3:46 / Merillas 3:42-3:47 / Albon 3:45
# 41 km — D+ ~1500m — départ roulant 4:45/km — double sommet 1328m/1498m
# ══════════════════════════════════════════════════════════════
TERRAIN_PROFILES = {
    "🛣️ Route / Plat": {
        "k_up": 12.0, "k_down": 5.0, "down_cap": -0.08,
        "minetti_weight": 0.60, "elev_smooth_window": 11,
        "grade_power": 0.85, "base_cap": 0.08,
        "extra_per_pct": 0.000, "max_cap": 0.18,
    },
    # ── Trail modéré calibré sur données historiques ─────────────────────────
    # k_up=22 : montées 8-12% réelles, entre trail modéré(18) et ultra(28)
    # k_down=4.5 : descente technique — pas de bonus excessif
    # down_cap=-0.10 : protège les jambes pour la remontée vers 1498m
    # elev_smooth_window=5 : capture les vraies pentes km par km
    # grade_power=0.70 : trail irrégulier
    # max_cap=0.48 : km à +15% peuvent coûter jusqu'à +48%
    # minetti_weight=0.30 : terrain irrégulier, Minetti moins pertinent
    "🏔️ Trail modéré": {
        "k_up": 22.0,
        "k_down": 4.5,
        "down_cap": -0.10,
        "minetti_weight": 0.30,
        "elev_smooth_window": 5,
        "grade_power": 0.70,
        "base_cap": 0.18,
        "extra_per_pct": 0.012,
        "max_cap": 0.48,
    },
    "⛰️ Ultra-trail montagneux": {
        "k_up": 28.0, "k_down": 4.0, "down_cap": -0.15,
        "minetti_weight": 0.20, "elev_smooth_window": 5,
        "grade_power": 0.65, "base_cap": 0.25,
        "extra_per_pct": 0.018, "max_cap": 0.60,
    },
}

SURFACE_OPTIONS = {
    "🏟️ Route / Piste synthétique":              1.00,
    "🪨 Chemin stabilisé / Gravier":             1.03,
    "🌿 Sentier herbe / Terre sèche":             1.06,
    "🧗 Sentier rocheux / Technique":             1.12,
    "🌧️ Boue / Neige tassée":                    1.18,
    # Mix calibré : 30% herbe×1.06 + 50% rocheux×1.12 + 20% neige×1.18 = 1.11
    "🎯 Surface calibrée (historique course)":   1.11,
    "🤖 Détecté automatiquement (OSM)":          1.06,  # valeur remplacée dynamiquement
}


# ══════════════════════════════════════════════════════════════
# VUE 3D CINÉMATIQUE THREE.JS — néon sur relief procédural
# Sans compte, sans clé API — Three.js r128 via jsDelivr CDN
# ══════════════════════════════════════════════════════════════
def generate_3d_animation(points, cum_d_map, checkpoints, total_dist_km, dem_elevations=None):
    n_pts = len(points)
    step = max(1, n_pts // 500)
    lats_a = [points[i].latitude  for i in range(0, n_pts, step)]
    lons_a = [points[i].longitude for i in range(0, n_pts, step)]
    if dem_elevations is not None and len(dem_elevations) == n_pts:
        elevs_raw = [dem_elevations[i] if dem_elevations[i] is not None else 0.0 for i in range(0, n_pts, step)]
    else:
        elevs_raw = [getattr(points[i], "elevation", 0.0) or 0.0 for i in range(0, n_pts, step)]
    dist_a = [cum_d_map[i] / 1000.0 for i in range(0, n_pts, step)]

    lat_min, lat_max = min(lats_a), max(lats_a)
    lon_min, lon_max = min(lons_a), max(lons_a)
    elev_min, elev_max = min(elevs_raw), max(elevs_raw)
    elev_range = max(1.0, elev_max - elev_min)
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    span = max(lat_max - lat_min, lon_max - lon_min) or 1.0

    def norm_lon(v): return (v - lon_center) / span * 2
    def norm_lat(v): return (v - lat_center) / span * 2
    def norm_elev(v): return (v - elev_min) / elev_range

    coords_3d = []
    for la, lo, el in zip(lats_a, lons_a, elevs_raw):
        coords_3d.append([round(norm_lon(lo), 5), round(norm_elev(el) * 0.6, 5), round(-norm_lat(la), 5)])

    cp_3d = []
    color_map_cp = {"🥤 Ravitaillement":"#00e5ff","🏔 Sommet":"#ffd600","🔻 Col":"#e040fb",
                    "⏱ Point de passage":"#40c4ff","🏁 Intermédiaire":"#b0bec5","⚠️ Point clé":"#ff1744"}
    for cp in sorted(checkpoints, key=lambda c: c["dist_km"]):
        cp_elev = float(np.interp(cp["dist_km"]*1000, cum_d_map,
                                   [getattr(points[i], "elevation", 0.0) or 0.0 for i in range(n_pts)]))
        cp_3d.append({"x": round(norm_lon(cp["lon"]), 5),
                      "y": round(norm_elev(cp_elev) * 0.6 + 0.04, 5),
                      "z": round(-norm_lat(cp["lat"]), 5),
                      "label": cp["label"], "dist": cp["dist_km"],
                      "color": color_map_cp.get(cp.get("type", ""), "#f97316")})

    GRID = 40
    grid_x = np.linspace(-1.2, 1.2, GRID)
    grid_z = np.linspace(-1.2, 1.2, GRID)
    grid_pts = []
    for gx in grid_x:
        for gz in grid_z:
            dists_sq = [(gx - c[0])**2 + (gz - c[2])**2 for c in coords_3d]
            nearest = min(range(len(coords_3d)), key=lambda i: dists_sq[i])
            dist_sq = dists_sq[nearest]
            base_y = coords_3d[nearest][1]
            falloff = math.exp(-dist_sq * 8.0)
            noise = (math.sin(gx*7.3+gz*5.1)*0.04 + math.sin(gx*3.1-gz*8.7)*0.03 + math.sin(gx*13.0+gz*2.3)*0.015)
            y_terrain = base_y * falloff + noise * (1.0 - falloff * 0.5)
            grid_pts.append(round(max(-0.05, y_terrain), 4))

    COORDS_JS = _json.dumps(coords_3d)
    DIST_JS   = _json.dumps([round(d, 3) for d in dist_a])
    ELEV_JS   = _json.dumps([round(e, 1) for e in elevs_raw])
    CP_JS     = _json.dumps(cp_3d)
    GRID_JS   = _json.dumps(grid_pts)
    TOT       = str(round(total_dist_km, 1))
    EMIN      = str(round(elev_min))
    EMAX      = str(round(elev_max))
    DPLUS     = str(int(sum(max(0, elevs_raw[i]-elevs_raw[i-1]) for i in range(1, len(elevs_raw)))))
    GN        = str(GRID)
    N         = str(len(coords_3d))

    html = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"/>
<title>Trail 3D</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#050510;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#e2e8f0;}
#c{width:100%;height:100%;display:block;}
#hud{position:fixed;top:0;left:0;right:0;height:44px;
     background:linear-gradient(180deg,rgba(5,5,20,.97) 0%,transparent 100%);
     display:flex;align-items:center;padding:0 16px;gap:6px;z-index:10;}
#hud h1{font-size:.70rem;font-weight:800;letter-spacing:.1em;
        background:linear-gradient(90deg,#a78bfa,#e879f9);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;flex-shrink:0;}
.st{display:flex;flex-direction:column;align-items:center;
    background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);
    border-radius:6px;padding:2px 9px;min-width:50px;}
.sv{font-size:.80rem;font-weight:800;color:#a78bfa;font-variant-numeric:tabular-nums;}
.sl{font-size:.40rem;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;}
#ctrls{margin-left:auto;display:flex;gap:4px;}
.btn{background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.25);
     color:#c4b5fd;border-radius:6px;padding:3px 9px;cursor:pointer;
     font-size:.60rem;font-weight:700;transition:all .15s;}
.btn:hover,.btn.on{background:rgba(167,139,250,.3);border-color:#a78bfa;color:#fff;}
.btn.pl{background:#7c3aed;border-color:#7c3aed;color:#fff;}
#prog{position:fixed;bottom:0;left:0;right:0;height:2px;background:rgba(167,139,250,.1);z-index:10;}
#progf{height:100%;width:0;background:linear-gradient(90deg,#7c3aed,#e879f9);box-shadow:0 0 8px #a78bfa;transition:width .1s;}
#toast{display:none;position:fixed;top:52px;left:50%;transform:translateX(-50%);
       background:rgba(5,5,20,.95);border:1px solid #a78bfa;border-radius:10px;
       padding:5px 14px;z-index:20;text-align:center;pointer-events:none;}
#toast .tn{font-size:.75rem;font-weight:700;color:#a78bfa;}
#toast .tm{font-size:.55rem;color:#6b7280;}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud">
  <h1>⛰ TRAIL __TOT__ KM — D+__DPLUS__M — __EMIN__–__EMAX__M</h1>
  <div class="st"><div class="sv" id="hd">0.0</div><div class="sl">km</div></div>
  <div class="st"><div class="sv" id="he">—</div><div class="sl">m alt</div></div>
  <div class="st"><div class="sv" id="hdp">0</div><div class="sl">d+</div></div>
  <div id="ctrls">
    <button class="btn on" id="s1" onclick="spd(1)">1×</button>
    <button class="btn"    id="s2" onclick="spd(2)">2×</button>
    <button class="btn"    id="s4" onclick="spd(4)">4×</button>
    <button class="btn on" id="bf" onclick="tgCam()">CAM</button>
    <button class="btn pl" id="bp" onclick="tgPlay()">⏸</button>
    <button class="btn"         onclick="rst()">↺</button>
  </div>
</div>
<div id="prog"><div id="progf"></div></div>
<div id="toast"><div class="tn" id="tn"></div><div class="tm" id="tm"></div></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
var COORDS=__COORDS__,DIST=__DIST__,ELEV=__ELEV__,CP=__CP__;
var GRID_Y=__GRID__,GN=__GN__,TOT=__TOT2__,EMIN=__EMIN2__,EMAX=__EMAX2__;
var N=COORDS.length;
var renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.2;
var scene=new THREE.Scene();
scene.background=new THREE.Color(0x050510);
scene.fog=new THREE.FogExp2(0x0a0520,0.32);
var camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.001,20);
camera.position.set(0,1.2,1.8);camera.lookAt(0,0,0);
scene.add(new THREE.AmbientLight(0x1a1040,2.0));
var dlight=new THREE.DirectionalLight(0xff6040,1.8);dlight.position.set(1,2,-1);scene.add(dlight);
var pl1=new THREE.PointLight(0xa78bfa,3.0,4.0);pl1.position.set(0,0.8,0);scene.add(pl1);
var pl2=new THREE.PointLight(0xe879f9,1.5,3.0);pl2.position.set(-0.5,0.6,0.5);scene.add(pl2);
// terrain
(function(){
  var geom=new THREE.PlaneGeometry(2.8,2.8,GN-1,GN-1);geom.rotateX(-Math.PI/2);
  var pos=geom.attributes.position,cols=[];
  for(var i=0;i<pos.count;i++){pos.setY(i,GRID_Y[i]||0);}
  geom.computeVertexNormals();
  for(var i=0;i<pos.count;i++){
    var yv=pos.array[i*3+1],t=Math.max(0,Math.min(1,(yv+0.05)/0.65)),c=new THREE.Color();
    if(t<0.3)c.setRGB(0.10+t*0.3,0.06+t*0.15,0.20+t*0.2);
    else if(t<0.7)c.setRGB(0.28+t*0.25,0.12+t*0.20,0.25);
    else c.setRGB(0.55+t*0.35,0.50+t*0.40,0.55+t*0.35);
    cols.push(c.r,c.g,c.b);
  }
  geom.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  var mat=new THREE.MeshStandardMaterial({roughness:0.85,metalness:0.05,vertexColors:true});
  scene.add(new THREE.Mesh(geom,mat));
})();
// ghost tracé
var trailPts=COORDS.map(function(c){return new THREE.Vector3(c[0],c[1]+0.008,c[2]);});
var curve=new THREE.CatmullRomCurve3(trailPts,false,'catmullrom',0.3);
var gG=new THREE.TubeGeometry(curve,N*2,0.009,6,false);
scene.add(new THREE.Mesh(gG,new THREE.MeshBasicMaterial({color:0x7c3aed,transparent:true,opacity:0.10})));
// départ / arrivée
function mkSph(x,y,z,col,r){
  var g=new THREE.SphereGeometry(r,12,8),m=new THREE.MeshBasicMaterial({color:col});
  var mesh=new THREE.Mesh(g,m);mesh.position.set(x,y,z);scene.add(mesh);
  var gh=new THREE.SphereGeometry(r*2.5,12,8),mh=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.15});
  var mhesh=new THREE.Mesh(gh,mh);mhesh.position.set(x,y,z);scene.add(mhesh);
}
mkSph(COORDS[0][0],COORDS[0][1]+0.015,COORDS[0][2],0x00ff88,0.012);
mkSph(COORDS[N-1][0],COORDS[N-1][1]+0.015,COORDS[N-1][2],0xff4040,0.012);
CP.forEach(function(cp){mkSph(cp.x,cp.y,cp.z,parseInt(cp.color.replace('#','0x')),0.010);});
// dot coureur
var dotG=new THREE.SphereGeometry(0.018,16,12),dotM=new THREE.MeshBasicMaterial({color:0xffffff});
var dot=new THREE.Mesh(dotG,dotM);scene.add(dot);
var haloG=new THREE.SphereGeometry(0.038,16,12),haloM=new THREE.MeshBasicMaterial({color:0xa78bfa,transparent:true,opacity:0.35});
var halo=new THREE.Mesh(haloG,haloM);scene.add(halo);
// étoiles
var sg=new THREE.BufferGeometry(),sv=[];
for(var i=0;i<1200;i++)sv.push((Math.random()-0.5)*12,Math.random()*6+1,(Math.random()-0.5)*12);
sg.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));
scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xffffff,size:0.008,transparent:true,opacity:0.7})));
// couleur néon
function neon(t){
  if(t<0.5)return new THREE.Color().setHSL(0.75+t*0.08,1.0,0.55+t*0.2);
  return new THREE.Color().setHSL(0.83-t*0.05,0.9,0.75+t*0.15);
}
function lerp3(arr,f){
  var i=Math.min(Math.floor(f),arr.length-2),r=f-i;
  return new THREE.Vector3(arr[i][0]*(1-r)+arr[i+1][0]*r,arr[i][1]*(1-r)+arr[i+1][1]*r,arr[i][2]*(1-r)+arr[i+1][2]*r);
}
function lerpA(arr,f){var i=Math.min(Math.floor(f),arr.length-2),r=f-i;return arr[i]*(1-r)+arr[i+1]*r;}
var segs=[],drawn=-1,frac=0,SPD=1,playing=true,followCam=true,lcp=-1,cumDP=0,lastSI=-1;
var BASE=N/(TOT*60);
var clock=new THREE.Clock();
function addSeg(a,b,t){
  var dir=new THREE.Vector3().subVectors(b,a),len=dir.length();
  if(len<0.0001)return;
  var col=neon(t);
  [0.003,0.009].forEach(function(r,ri){
    var g=new THREE.CylinderGeometry(r,r,len,5);g.translate(0,len/2,0);g.rotateX(Math.PI/2);
    var m=new THREE.MeshBasicMaterial({color:col,transparent:ri>0,opacity:ri>0?0.18:1.0});
    var mesh=new THREE.Mesh(g,m);mesh.position.copy(a);mesh.lookAt(b);scene.add(mesh);segs.push(mesh);
  });
}
function animate(){
  requestAnimationFrame(animate);
  var dt=Math.min(clock.getDelta(),0.1);
  if(playing&&frac<N-1){
    frac=Math.min(frac+BASE*SPD*dt*60,N-1);
    var fi=Math.floor(frac);
    for(var si=drawn+1;si<=fi&&si<N-1;si++){
      addSeg(new THREE.Vector3(COORDS[si][0],COORDS[si][1]+0.008,COORDS[si][2]),
             new THREE.Vector3(COORDS[si+1][0],COORDS[si+1][1]+0.008,COORDS[si+1][2]),si/(N-1));
      drawn=si;
    }
    var pos=lerp3(COORDS,frac);pos.y+=0.022;
    dot.position.copy(pos);halo.position.copy(pos);
    var pulse=0.8+0.2*Math.sin(Date.now()*0.006);
    halo.scale.setScalar(pulse);haloM.opacity=0.25+0.15*pulse;
    pl1.position.set(pos.x,pos.y+0.3,pos.z);
    if(fi>0&&fi>lastSI){cumDP+=Math.max(0,(COORDS[fi][1]-COORDS[Math.max(0,fi-1)][1])*(parseFloat(EMAX)-parseFloat(EMIN)));lastSI=fi;}
    var d=lerpA(DIST,frac),e=lerpA(ELEV,frac);
    document.getElementById('hd').textContent=d.toFixed(2);
    document.getElementById('he').textContent=Math.round(e);
    document.getElementById('hdp').textContent=Math.round(cumDP);
    document.getElementById('progf').style.width=(frac/(N-1)*100)+'%';
    CP.forEach(function(cp,ci){
      if(ci!==lcp&&Math.abs(cp.dist-d)<0.3){
        lcp=ci;document.getElementById('tn').textContent=cp.label;
        document.getElementById('tm').textContent=cp.dist.toFixed(1)+' km · '+Math.round(e)+' m';
        var t2=document.getElementById('toast');t2.style.borderColor=cp.color;t2.style.display='block';
        clearTimeout(window._ct);window._ct=setTimeout(function(){t2.style.display='none';},2800);
      }
    });
    if(followCam){
      var ah=Math.min(frac+20,N-1),ap=lerp3(COORDS,ah);
      var ct=new THREE.Vector3(pos.x*0.3+ap.x*0.7,pos.y+0.55,pos.z*0.3+ap.z*0.7);
      camera.position.lerp(ct,0.025);camera.lookAt(pos.x,pos.y+0.05,pos.z);
    }
    if(frac>=N-1){playing=false;document.getElementById('bp').textContent='▶';}
  }
  var t3=Date.now()*0.001;
  pl2.position.x=Math.sin(t3*0.4)*0.3;pl2.position.z=Math.cos(t3*0.3)*0.3;
  renderer.render(scene,camera);
}
animate();
window.addEventListener('resize',function(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function spd(m){SPD=m;['s1','s2','s4'].forEach(function(id){document.getElementById(id).classList.remove('on');});document.getElementById('s'+m).classList.add('on');}
function tgCam(){followCam=!followCam;var b=document.getElementById('bf');b.classList.toggle('on',followCam);b.textContent=followCam?'CAM':'MAP';if(!followCam){camera.position.set(0,1.8,2.2);camera.lookAt(0,0,0);}}
function tgPlay(){playing=!playing;document.getElementById('bp').textContent=playing?'⏸':'▶';}
function rst(){playing=false;frac=0;drawn=-1;cumDP=0;lcp=-1;lastSI=-1;segs.forEach(function(s){scene.remove(s);s.geometry.dispose();s.material.dispose();});segs=[];dot.position.set(COORDS[0][0],COORDS[0][1]+0.022,COORDS[0][2]);halo.position.copy(dot.position);camera.position.set(0,1.2,1.8);camera.lookAt(0,0,0);document.getElementById('progf').style.width='0%';document.getElementById('bp').textContent='⏸';playing=true;}
</script></body></html>"""

    html = html.replace("__TOT__", TOT).replace("__DPLUS__", DPLUS).replace("__EMIN__", EMIN).replace("__EMAX__", EMAX)
    html = html.replace("__COORDS__", COORDS_JS).replace("__DIST__", DIST_JS).replace("__ELEV__", ELEV_JS)
    html = html.replace("__CP__", CP_JS).replace("__GRID__", GRID_JS).replace("__GN__", GN)
    html = html.replace("__TOT2__", TOT).replace("__EMIN2__", EMIN).replace("__EMAX2__", EMAX)
    return html


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

main_tabs=st.tabs(["🏃 Prédiction de course","🧪 Tests d'endurance + VC","⚙️ Analyse entraînement"])

# ══════════════════════════════════════════════════════════════
# ONGLET 0 — PRÉDICTION DE COURSE
# ══════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.title("🏃 Prédiction de course — Coach & Athlète")
    st.caption("v6 — Relief 3D réel pydeck · Terrain technique auto · Surface OSM · Trail calibré · Three.js · WBGT · DEM")

    col_mode1,col_mode2=st.columns([2,3])
    with col_mode1:
        mode=st.radio("Mode d'interface",["🟢 Simple (recommandé)","🔵 Expert (tous les curseurs)"],horizontal=True,key="pred_mode")
    EXPERT="Expert" in mode

    # ── NOUVEAU v6.1 : Sélecteur Route / Trail ──────────────────────────────
    st.markdown("---")
    col_rt1, col_rt2 = st.columns([2, 3])
    with col_rt1:
        mode_activite = st.radio(
            "🏷️ Type d'activité",
            ["🛣️ Route / Piste", "🏔️ Trail / Montagne"],
            horizontal=True, key="mode_activite",
            help="Route : masque la technicité, surfaces limitées au revêtu.\nTrail : affiche technicité, surfaces naturelles, 3D relief."
        )
    IS_TRAIL = "Trail" in mode_activite

    st.markdown("---")
    st.header("1️⃣  Parcours GPX")
    gpx_file=st.file_uploader("📂 Importer le GPX de la course cible",type=["gpx"],key="gpx_main")
    points=None;dem_elevations=None

    if gpx_file:
        _gpx,points=parse_gpx_points(gpx_file)
        if points:
            tot_tmp=sum(haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude) for i in range(1,len(points)))
            dup_tmp,ddn_tmp=compute_dplus_dminus([getattr(p,"elevation",0.0) or 0.0 for p in points])
            avg_alt_tmp=np.mean([getattr(p,"elevation",0.0) or 0.0 for p in points])
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Distance GPX",f"{tot_tmp/1000:.2f} km")
            c2.metric("D+ GPS",f"{dup_tmp:.0f} m")
            c3.metric("D- GPS",f"{ddn_tmp:.0f} m")
            c4.metric("Alt. moy.",f"{avg_alt_tmp:.0f} m")

            # ── NOUVEAU v6.1 : Détection terrain technique (Trail uniquement) ──
            if IS_TRAIL:
              st.markdown("---")
              st.subheader("🔬 Analyse terrain technique automatique")
              with st.spinner("Analyse sinuosité, pentes, variabilité..."):
                tech_segs, tech_global = detect_technical_terrain(points, dem_elevations, is_trail=IS_TRAIL)
                st.session_state["tech_segs"]   = tech_segs
                st.session_state["tech_global"] = tech_global

              if tech_global.get("global_score", 0) > 0:
                score = tech_global["global_score"]; label = tech_global["label"]
                pct_tech = tech_global.get("pct_technique", 0)
                badge_css = ("background:#fee2e2;color:#991b1b" if score > 0.65
                             else "background:#fef3c7;color:#92400e" if score > 0.40
                             else "background:#d1fae5;color:#065f46")
                st.markdown(f'<span style="{badge_css};border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:700;">{label} — score {score:.2f}/1.00</span>',unsafe_allow_html=True)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Score technique global",f"{score:.2f}/1.00")
                c2.metric("Segments techniques",f"{pct_tech:.0f}%")
                c3.metric("Sinuosité moy.",f"{tech_global.get('sinuosity_mean',1.0):.3f}")
                c4.metric("Pente max",f"{tech_global.get('grade_max_all',0):.1f}%")
                st.info(f"💡 Multiplicateurs suggérés : **k_up ×{tech_global['k_up_adj']:.2f}** · **k_down ×{tech_global['k_down_adj']:.2f}** · **surface ×{tech_global['surface_mult_adj']:.2f}**")
                if tech_segs:
                    with st.expander("📊 Détail par km — score technique"):
                        df_tech=pd.DataFrame(tech_segs)
                        cols_show=["km_start","km_end","label","tech_score","sinuosity","grade_max","grade_std","dir_change_km"]
                        st.dataframe(df_tech[[c for c in cols_show if c in df_tech.columns]],use_container_width=True,hide_index=True)
                        fig_tech,ax_tech=plt.subplots(figsize=(11,3))
                        km_mids=[(s["km_start"]+s["km_end"])/2 for s in tech_segs]
                        scores=[s["tech_score"] for s in tech_segs]
                        colors_t=["#22c55e" if s<0.25 else "#eab308" if s<0.5 else "#f97316" if s<0.75 else "#ef4444" for s in scores]
                        ax_tech.bar(km_mids,scores,width=0.85,color=colors_t,alpha=0.85)
                        ax_tech.axhline(0.5,color="orange",lw=1,ls="--",label="Seuil technique")
                        ax_tech.axhline(0.75,color="red",lw=1,ls="--",label="Seuil très technique")
                        ax_tech.set_xlabel("Distance (km)");ax_tech.set_ylabel("Score technique")
                        ax_tech.set_title("Score de technicité par km");ax_tech.legend();ax_tech.set_ylim(0,1);ax_tech.grid(alpha=0.3);fig_tech.tight_layout()
                        st.pyplot(fig_tech);plt.close(fig_tech)

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
            # Relancer détection terrain avec DEM
            tech_segs,tech_global=detect_technical_terrain(points,dem_elevations,is_trail=IS_TRAIL)
            st.session_state["tech_segs"]=tech_segs
            st.session_state["tech_global"]=tech_global

    # ── NOUVEAU v6 : Surface OSM ────────────────────────────────────────────
    with st.expander("🌿 Détection surface OSM (Overpass API, gratuit)", expanded=False):
        if not IS_TRAIL:
            st.info("ℹ️ La détection OSM de surfaces naturelles est principalement utile en mode Trail.")
        st.info("Interroge OpenStreetMap pour détecter automatiquement le type de surface du parcours (rock, grass, mud…).")
        if gpx_file and points and st.button("🔍 Analyser la surface via OSM",key="btn_osm"):
            with st.spinner("Requête Overpass API en cours..."):
                n_pts_osm=len(points);step_osm=max(1,n_pts_osm//100)
                lats_osm=tuple(points[i].latitude  for i in range(0,n_pts_osm,step_osm))
                lons_osm=tuple(points[i].longitude for i in range(0,n_pts_osm,step_osm))
                osm_result=fetch_osm_surface(lats_osm,lons_osm)
                st.session_state["osm_surface"]=osm_result
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
                st.success(f"💡 Sélectionne **🤖 Détecté automatiquement (OSM)** dans le sélecteur de surface pour appliquer ×{osm.get('surface_mult_osm',1.06):.3f}")
                SURFACE_OPTIONS["🤖 Détecté automatiquement (OSM)"]=osm.get("surface_mult_osm",1.06)

    st.markdown("---")
    st.header("2️⃣  Courses de référence")
    st.info("Calibrent le modèle sur l'athlète. Minimum conseillé : **3 références** variées.")

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
                sh=hms_input("Début segment","0:00:00",key=f"start_{i}",compact=True)
                eh=hms_input("Fin segment","23:59:59",key=f"end_{i}",compact=True)
                start_td,end_td=hms_to_timedelta(sh),hms_to_timedelta(eh)
                if start_td.total_seconds()>0 or end_td.total_seconds()<86399:
                    pts_src=None
                    if fit_data and "points" in fit_data:pts_src=fit_data["points"]
                    elif tcx_data and "points" in tcx_data:pts_src=tcx_data["points"]
                    if pts_src:
                        seg=extract_segment(pts_src,start_td,end_td)
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
                st.caption(f"💓 FC max {hr_ref['hr_max']} bpm · dérive {hr_ref['hr_drift']} bpm · seuil ~{hr_ref['hr_threshold_est']} bpm")
            refs_raw.append({"distance":float(dist),"temps":str(temps_eff),
                              "D_up":float(dup),"D_down":float(ddn),"duration_hms_file":dur_hms_file,
                              "avg_temp":avg_temp_ref,"avg_humidity":avg_hum_ref,"avg_wind":avg_wind_ref,"hr_analysis":hr_ref})

    st.markdown("---")
    st.header("3️⃣  Recalibration des références vers les conditions idéales")
    st.markdown('<div class="highlight-box"><strong>Pourquoi recalibrer ?</strong><br>Une course réalisée par 30°C et 80% d\'humidité vaut <em>physiologiquement mieux</em> qu\'un temps identique par 10°C et temps sec.</div>',unsafe_allow_html=True)
    use_recalibrated=st.checkbox("✅ Recalibrer les références vers les conditions idéales (fortement recommandé)",value=True)
    # Valeurs calibrées montagne
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
    calib_rows=[];cold_quad=0.0012;hot_quad=0.0016;temp_max_penalty=0.10
    for r in refs_raw:
        t_brut=hms_to_seconds(r.get("duration_hms_file") or r.get("temps",""))
        dist_km=safe_float(r.get("distance",1.0))/1000.0
        avg_t=r.get("avg_temp");avg_h=safe_float(r.get("avg_humidity",50.0),50.0)
        wbgt_val=wbgt_simplified(avg_t,avg_h) if avg_t is not None and use_wbgt else None
        t_ideal=(recalibrate_ref_to_ideal(ref={**r,"temps":r.get("duration_hms_file") or r.get("temps","0:00:00")},
            opt_temp=opt_temp,use_wbgt=use_wbgt,cold_quad=cold_quad,hot_quad=hot_quad,
            temp_max_penalty=temp_max_penalty,k_up=_k_up_prev,k_down=_k_down_prev,down_cap=-0.10,
            g0_up=_g0u_prev,g0_down=_g0d_prev,max_up=0.48,max_down=-0.06,
            elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power) if use_recalibrated else float(t_brut))
        gain_s=t_brut-t_ideal
        calib_rows.append({"Distance":f"{safe_float(r['distance'])/1000:.1f} km",
                            "Temps brut":seconds_to_hms(t_brut),"Allure brute":pace_str(t_brut/dist_km) if dist_km>0 else "-",
                            "D+":f"{r['D_up']:.0f} m","Temp GPS":f"{avg_t:.0f}°C" if avg_t is not None else "?",
                            "WBGT":f"{wbgt_val:.1f}°C" if wbgt_val is not None else "-",
                            "Temps recalibré":seconds_to_hms(t_ideal) if use_recalibrated else "—",
                            "Allure recalibrée":pace_str(t_ideal/dist_km) if (use_recalibrated and dist_km>0) else "—",
                            "Gain correction":f"-{seconds_to_hms(gain_s)}" if gain_s>0 else (f"+{seconds_to_hms(-gain_s)}" if gain_s<0 else "0")})
    st.dataframe(pd.DataFrame(calib_rows),use_container_width=True)


    st.markdown("---")
    st.header("4️⃣  Paramètres du modèle")

    with st.expander("🌡️ Température & Humidité",expanded=False):
        # Valeurs recalibrées v7 (physiologie trail réelle)
        cold_quad=0.0015; hot_quad=0.0020; temp_max_penalty=0.20; temp_power=1.0
        if EXPERT:
            c1,c2=st.columns(2)
            cold_quad=c1.number_input("Sensibilité froid",value=0.0015,step=0.0002,format="%.4f",
                                       help="+5°C en dessous de l'optimal → +3.75%")
            hot_quad=c2.number_input("Sensibilité chaleur",value=0.0020,step=0.0002,format="%.4f",
                                      help="+5°C au-dessus de l'optimal → +5% | +10°C → +20%")
            temp_max_penalty=st.slider("Pénalité max température (%)",0.00,0.30,0.20,0.01)
            temp_power=st.slider("Damping température (puissance)",0.2,1.2,1.0,0.05)

    with st.expander("🏔️ Altitude physiologique (hypoxie)"):
        apply_altitude=st.checkbox("Appliquer la pénalité d'altitude",value=True)
        altitude_ref_m=0.0
        if apply_altitude:altitude_ref_m=st.number_input("Altitude d'entraînement habituelle (m)",value=0.0,step=100.0)

    with st.expander("🎢 Modèle de pente & Terrain",expanded=True):
        apply_grade=st.checkbox("Prendre en compte la pente",value=True)
        use_minetti=st.checkbox("Modèle Minetti",value=True)
        st.markdown("##### 🗺️ Profil du parcours")
        terrain_profil=st.radio("Type de terrain",list(TERRAIN_PROFILES.keys()),horizontal=True,
                                key="terrain_profil_radio",help="Pré-remplit les coefficients ci-dessous.")
        _prev=st.session_state.get("_prev_terrain_profil","")
        if terrain_profil!=_prev:
            st.session_state["_prev_terrain_profil"]=terrain_profil
            _d=TERRAIN_PROFILES[terrain_profil]
            for k,v in _d.items():st.session_state[f"tp_{k}"]=v
        _d=TERRAIN_PROFILES[terrain_profil]
        _profil_info={
            "🛣️ Route / Plat":"Route, piste, parcours plat. Modèle Minetti bien calibré. k_up=12.",
            "🏔️ Trail modéré":"✅ Calibré sur historique : Jornet 3:38 / Elazzaoui 3:42-3:46 / Merillas 3:42. k_up=22, surface×1.11, fatigue 8.5%.",
            "⛰️ Ultra-trail montagneux":"D+ moyen > 100m/km. Montées techniques. k_up=28. DEM obligatoire.",
        }
        st.info(_profil_info.get(terrain_profil,""))

        # ── Suggestion automatique depuis la détection terrain ───────────
        tech_global_ui=st.session_state.get("tech_global",{})
        if IS_TRAIL and tech_global_ui.get("global_score",0)>0 and terrain_profil=="🏔️ Trail modéré":
            adj_up=tech_global_ui.get("k_up_adj",1.0);adj_dn=tech_global_ui.get("k_down_adj",1.0)
            sugg_k_up=round(_d["k_up"]*adj_up,1);sugg_k_dn=round(_d["k_down"]*adj_dn,1)
            st.markdown(f'<div style="background:#fef3c7;border-left:3px solid #f59e0b;border-radius:4px;padding:6px 12px;font-size:0.82rem;">🤖 <b>Suggestion auto (terrain score {tech_global_ui["global_score"]:.2f})</b> : k_up → <b>{sugg_k_up}</b> · k_down → <b>{sugg_k_dn}</b></div>',unsafe_allow_html=True)
            if st.button("✅ Appliquer coefficients détectés automatiquement",key="apply_tech_coeffs"):
                st.session_state["tp_k_up"]=sugg_k_up
                st.session_state["tp_k_down"]=sugg_k_dn
                st.rerun()

        st.markdown("##### 🌿 Surface du terrain")
        # Filtrer surfaces selon Route/Trail
        if IS_TRAIL:
            _surf_opts = SURFACE_OPTIONS
        else:
            _surf_opts = {k:v for k,v in SURFACE_OPTIONS.items() if any(x in k for x in ["Route","Piste","Gravier","stabilisé","Chemin"])}
            if not _surf_opts: _surf_opts = {"🏟️ Route / Piste synthétique":1.00,"🪨 Chemin stabilisé / Gravier":1.03}
        surface_sel=st.selectbox("Type de surface",list(_surf_opts.keys()),key="surface_sel",
                                  help="Coût énergétique additionnel selon la surface. Route = ×1.00 (référence).")
        surface_mult=_surf_opts[surface_sel]
        # OSM auto
        if surface_sel.startswith("🤖") and "osm_surface" in st.session_state:
            surface_mult=st.session_state["osm_surface"].get("surface_mult_osm",1.06)
        st.caption(f"Multiplicateur surface : **×{surface_mult:.3f}** — pénalité **+{(surface_mult-1)*100:.1f}%** sur l'allure de base")

        minetti_weight=0.6
        if use_minetti:
            minetti_weight=st.slider("Part de Minetti (0=heuristique pur, 1=Minetti pur)",0.0,1.0,
                                      float(st.session_state.get("tp_minetti_weight",_d["minetti_weight"])),0.05,key="tp_minetti_weight")

        st.markdown("##### ⚙️ Coefficients détaillés")
        col_r1a,col_r1b,col_r1c=st.columns(3)
        with col_r1a:
            k_up=st.number_input("k_up — coefficient montée",min_value=1.0,max_value=60.0,
                                   value=float(st.session_state.get("tp_k_up",_d["k_up"])),step=0.5,key="tp_k_up",
                                   help="Route≈12 · Trail calibré≈22 · Ultra≈28")
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
                                          int(st.session_state.get("tp_elev_smooth_window",_d["elev_smooth_window"])),
                                          2,key="tp_elev_smooth_window")
        with col_r2b:
            grade_power=st.slider("Amortissement effet pente",0.2,1.0,
                                   float(st.session_state.get("tp_grade_power",_d["grade_power"])),
                                   0.05,key="tp_grade_power")
        st.markdown("**Plafond anti-accumulation**")
        col_r3a,col_r3b,col_r3c=st.columns(3)
        with col_r3a:
            base_cap=st.slider("Plafond de base (%)",0.02,0.40,
                                float(st.session_state.get("tp_base_cap",_d["base_cap"])),0.01,key="tp_base_cap")
        with col_r3b:
            extra_per_pct=st.slider("Extra par % de pente",0.000,0.030,
                                     float(st.session_state.get("tp_extra_per_pct",_d["extra_per_pct"])),
                                     0.001,format="%.3f",key="tp_extra_per_pct")
        with col_r3c:
            max_cap=st.slider("Plafond absolu (%)",0.05,0.80,
                               float(st.session_state.get("tp_max_cap",_d["max_cap"])),0.01,key="tp_max_cap")
        g0_up=3.0;g0_down=2.5;max_up=float(max_cap);max_down=-0.06
        st.session_state["k_up_val"]=k_up;st.session_state["k_down_val"]=k_down
        st.session_state["g0_up_val"]=g0_up;st.session_state["g0_down_val"]=g0_down
        _terrain_colors={"🛣️ Route / Plat":"#0C447C","🏔️ Trail modéré":"#3B6D11","⛰️ Ultra-trail montagneux":"#993C1D"}
        _col=_terrain_colors.get(terrain_profil,"#444")
        st.markdown(f'<div style="background:rgba(0,0,0,0.04);border-left:3px solid {_col};border-radius:4px;padding:6px 12px;font-size:0.82rem;margin-top:6px;"><b>{terrain_profil}</b> · k_up={k_up:.0f} · k_down={k_down:.0f} · Minetti={minetti_weight:.2f} · Plafond={max_cap:.0%} · Surface <b>{surface_sel.split()[0]} ×{surface_mult:.2f}</b></div>',unsafe_allow_html=True)

    with st.expander("💨 Vent"):
        apply_wind=st.checkbox("Appliquer l'effet du vent",value=True)
        wind_mode="Lissé";wind_smooth_km=5
        # Recalibrés v7 : cap_head=0.12 (vent face 10m/s = +12%), cap_tail=-0.06 (vent dos = -6%)
        drag_coeff=0.018;tail_credit=0.40;wind_cap_head=0.12;wind_cap_tail=-0.06;wind_power=1.0
        wind_gate_g1=2.0;wind_gate_g2=8.0;wind_gate_min=0.25
        if apply_wind and EXPERT:
            wind_mode=st.selectbox("Mode calcul vent",["Lissé","Global"],key="wmode").split()[0]
            wind_smooth_km=st.slider("Lissage vent (km)",1,11,5,2)
            c1,c2=st.columns(2)
            drag_coeff=c1.number_input("Coeff. aérodynamique",value=0.018,step=0.002,format="%.3f",
                                        help="0.018 = trail (posture penchée) | 0.012 = route")
            tail_credit=c2.slider("Crédit vent arrière",0.0,0.8,0.40,0.05)
            wind_cap_head=st.slider("Pénalité max vent face (%)",0.00,0.25,0.12,0.01,
                                     help="10 m/s en face ≈ +12-15% sur trail")
            wind_cap_tail=st.slider("Gain max vent dos (%)",-0.12,0.00,-0.06,0.01,
                                     help="10 m/s dans le dos ≈ -6%")

    with st.expander("🔋 Fatigue en course"):
        # Fatigue activée par défaut pour le profil trail calibré
        apply_fatigue=st.checkbox("Activer la fatigue",value=True)
        fatigue_rate=0.0;fatigue_mode="mixte"
        if apply_fatigue:
            fatigue_rate=st.slider("Ralentissement total fin de course (%)",0.0,30.0,8.5,0.5)
            fatigue_mode=st.selectbox("Type de fatigue",["mixte (recommandé)","distance (plat)","d_plus (montagne)"]).split()[0]

    with st.expander("🎚️ Sensibilité aux conditions selon l'allure", expanded=False):
        st.caption("Plus tu cours vite, moins la météo et le vent te ralentissent en proportion (temps d'exposition réduit).")
        pace_sens_ref_min=st.slider(
            "Allure de référence (min/km) — au-dessus : plein effet · en dessous : effet réduit",
            min_value=3.0,max_value=10.0,value=6.0,step=0.5,key="pace_sens_ref",
            help="6 min/km = défaut neutre. Baisser si tu cours < 4min/km pour éviter les sur-corrections météo.")
        pace_sensitivity_ref=pace_sens_ref_min*60.0
        # Aperçu à l'allure cible (utiliser les valeurs de session si dispo)
        try:
            _obj_hms = st.session_state.get("temps_objectif_target","")
            _dist_km = st.session_state.get("dist_km_input", 21.0)
            if _obj_hms and hms_to_seconds(_obj_hms)>0 and _dist_km>0:
                _allure_s=hms_to_seconds(_obj_hms)/float(_dist_km)
                _pf=min(1.0,(_allure_s/pace_sensitivity_ref)**0.5)
                st.info(f"À ~{_allure_s/60:.1f}min/km : impact météo/vent réduit à **{_pf*100:.0f}%** (pace factor={_pf:.2f})")
        except Exception: pass

    pace_sensitivity_ref=st.session_state.get("pace_sens_ref",6.0)*60.0  # défaut si pas encore défini
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
    heure_course=c2.time_input("⏰ Heure de départ",value=time(9,0))
    colf1,colf2=st.columns(2)
    with colf1:
        force_dist=st.checkbox("Forcer la distance",value=False)
        dist_forcee=st.number_input("Distance (km)",value=41.0,format="%.3f") if force_dist else None
    with colf2:
        force_temps=st.checkbox("Travailler à partir d'un objectif de temps",value=True)
        temps_objectif=hms_input("Temps objectif","3:45:00",key="temps_objectif_target") if force_temps else None

    # ── Info météo (automatique uniquement) ────────────────────────
    _diff_days_race = (date_course - date.today()).days
    if 0 <= _diff_days_race <= 15:
        st.info(f"🌡️ **Météo automatique** — prévisions Open-Meteo km par km (J+{_diff_days_race}).")
    elif _diff_days_race < 0:
        st.info(f"🌡️ **Météo automatique** — archives Open-Meteo ({date_course}).")
    else:
        st.caption(f"🌡️ Météo : date à J+{_diff_days_race} (hors plage API) — si disponible, archives de l'an passé.")

    # Paramètres météo fallback neutres (pas de saisie manuelle)
    meteo_fb = {"temp": 12.0, "amp": 0.0, "wind": 0.0, "humidity": 60.0, "wind_dir": 180.0}
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
            st.session_state["_meteo_api_cache"] = {}  # forcer re-fetch météo
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
                        apply_fatigue=apply_fatigue,fatigue_rate=fatigue_rate,fatigue_mode=fatigue_mode,
                        apply_ultra=apply_ultra,ultra_amp=ultra_amp,
                        objective_hms=temps_objectif if force_temps else None,
                        show_smooth_pace=show_smooth_pace,smooth_window_km=smooth_window_km,
                        dem_elevations=dem_elevations,surface_mult=surface_mult,
                        # Paramètres météo fallback
                        meteo_fallback_temp=meteo_fb["temp"],
                        meteo_fallback_amp=meteo_fb["amp"],
                        meteo_fallback_wind=meteo_fb["wind"],
                        meteo_fallback_humidity=meteo_fb["humidity"],
                        meteo_fallback_wind_dir=meteo_fb["wind_dir"],
                        pace_sensitivity_ref=pace_sensitivity_ref)
                    st.session_state["res"]=res
                    st.session_state["refs_fit_vc"]=res.get("refs_fit",[])
                    st.session_state["K_riegel_vc"]=res.get("K",1.06)
                except Exception as e:
                    import traceback;st.error(f"Erreur:{e}");st.code(traceback.format_exc())

    if "res" in st.session_state:
        res=st.session_state["res"]
        st.markdown("---");st.subheader("🎯 Prédiction")
        avg_pace_s=res["total_s"]/max(res["dist_gpx_km"],1e-6)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("⏱ Temps prédit",res["total_human"])
        c2.metric("📊 Allure moy.",pace_str(avg_pace_s)+"/km")
        # Fourchette avec curseur décalage manuel
        _std_dists={"10 km":10000,"Semi":21097,"Marathon":42195,"50 km":50000,"100 km":100000}
        _gpx_m=res["dist_gpx_km"]*1000.0
        _avg_pace_s=res["total_s"]/_gpx_m if _gpx_m>0 else 0
        # Sélection distance de référence
        ci_col1,ci_col2,ci_col3=st.columns([2,1,1])
        with ci_col1:
            _dist_ref_opts={"Distance GPX":_gpx_m}
            _dist_ref_opts.update({k:v for k,v in _std_dists.items()})
            _ref_sel=st.selectbox("📏 Distance de référence",list(_dist_ref_opts.keys()),
                                   key="ci_dist_ref",index=0,
                                   help="Projeter le temps sur cette distance à allure identique")
        with ci_col2:
            _delta_low=st.slider("Borne basse (%)",-10,0,-2,1,key="ci_delta_low",
                                  help="Décalage négatif : objectif ambitieux")
        with ci_col3:
            _delta_high=st.slider("Borne haute (%)",0,10,2,1,key="ci_delta_high",
                                   help="Décalage positif : objectif conservateur")
        _ref_m=_dist_ref_opts[_ref_sel]
        _t_ref=_avg_pace_s*_ref_m
        _t_low=_t_ref*(1+_delta_low/100.0)
        _t_high=_t_ref*(1+_delta_high/100.0)
        _lbl_low=f"{_ref_sel} {_delta_low:+d}%"
        _lbl_high=f"{_ref_sel} {_delta_high:+d}%"
        c3.metric(f"📏 {_lbl_low}",seconds_to_hms(_t_low))
        c4.metric(f"📏 {_lbl_high}",seconds_to_hms(_t_high))
        c5.metric("K Riegel",f"{res['K']:.3f}")
        # Amplitude météo évolutive
        mr=res.get("meteo_range")
        if mr and mr.get("delta",0)>1.5:
            st.info(f"🌡️ **Météo évolutive** : {mr['t_min']}°C → {mr['t_max']}°C (Δ {mr['delta']}°C) — chaque km intègre la météo horaire réelle de passage.")
        elif mr:
            st.caption(f"🌡️ Météo stable : {mr['t_min']}°C → {mr['t_max']}°C")
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
                ax.invert_yaxis();ax.set_xlabel("Kilomètre");ax.set_ylabel("Allure (min/km)")
                ax.set_title("Allure prévisionnelle km par km");ax.legend();ax.grid(alpha=0.3)
                st.pyplot(fig);plt.close(fig)
            with res_t2:
                fig2,ax2=plt.subplots(figsize=(12,4));x=list(range(1,len(df_out)+1))
                ax2.plot(x,df_out["Mult Pente"].values,label="Pente",lw=2)
                if "Mult Temp" in df_out.columns:ax2.plot(x,df_out["Mult Temp"].values,label="Température",lw=2)
                if "Mult Vent" in df_out.columns:ax2.plot(x,df_out["Mult Vent"].values,label="Vent",lw=2)
                if "Mult Fatigue" in df_out.columns:ax2.plot(x,df_out["Mult Fatigue"].values,label="Fatigue",lw=2,ls=":")
                ax2.axhline(1.0,color="gray",lw=0.8);ax2.set_xlabel("Kilomètre")
                ax2.set_ylabel("Multiplicateur");ax2.set_title("Décomposition des facteurs")
                ax2.legend();ax2.grid(alpha=0.3);st.pyplot(fig2);plt.close(fig2)
            with res_t3:st.dataframe(df_out,use_container_width=True)


    if gpx_file and points:
        cum_d_map=[0.0]
        for i in range(1,len(points)):
            cum_d_map.append(cum_d_map[-1]+haversine_m(points[i-1].latitude,points[i-1].longitude,points[i].latitude,points[i].longitude))
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
                cp_lat=float(np.interp(cp_dist_m,cum_d_map,lats_m))
                cp_lon=float(np.interp(cp_dist_m,cum_d_map,lons_m))
                y_gps_cp=[getattr(p,"elevation",0.0) or 0.0 for p in points]
                cp_alt=float(np.interp(cp_dist_m,cum_d_map,y_gps_cp))
                label=cp_nom.strip() if cp_nom.strip() else f"{cp_type} km {cp_dist:.1f}"
                st.session_state["checkpoints"].append({"dist_km":cp_dist,"type":cp_type,"label":label,"lat":cp_lat,"lon":cp_lon,"alt":round(cp_alt)})
                st.success(f"✅ Checkpoint ajouté : {label}")
        with col_btn2:
            if st.button("🗑️ Effacer tous les checkpoints"):st.session_state["checkpoints"]=[]
        checkpoints=st.session_state["checkpoints"]
        if checkpoints:
            df_cp=pd.DataFrame([{"Type":c["type"],"Nom":c["label"],"Distance":f"{c['dist_km']:.1f} km","Altitude GPS":f"{c['alt']} m","Lat":round(c["lat"],5),"Lon":round(c["lon"],5)} for c in sorted(checkpoints,key=lambda x:x["dist_km"])])
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
                y_s=np.convolve(y_gps,np.ones(w_e)/w_e,mode="same")
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
            # Overlay zones techniques sur le profil (Trail only)
            if IS_TRAIL:
              for seg in st.session_state.get("tech_segs",[]):
                if seg.get("tech_score",0)>0.45:
                    color_tech="#ef4444" if seg["tech_score"]>0.70 else "#f97316"
                    ax3.axvspan(seg["km_start"],seg["km_end"],alpha=0.12,color=color_tech,label="_nolegend_")
            ax3.scatter([x_km[0]],[y_gps[0]],s=80,color="lime",zorder=6,marker="^",label="Départ")
            ax3.scatter([x_km[-1]],[y_gps[-1]],s=80,color="red",zorder=6,marker="s",label="Arrivée")
            ax3.set_xlabel("Distance (km)");ax3.set_ylabel("Altitude (m)")
            ax3.set_title(f"Profil d'altitude — {total_dist_km:.1f} km  (zones orange/rouge = terrain technique)")
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
                            passage_rows.append({"Checkpoint":cp["label"],"Distance":f"{cp['dist_km']:.1f} km","Altitude":f"{cp['alt']} m","Temps prévu":seconds_to_hms(t_passage),"Allure moy. jusque-là":pace_str(t_passage/max(0.001,cp["dist_km"]))+"/km" if cp["dist_km"]>0 else "—"})
                        st.dataframe(pd.DataFrame(passage_rows),use_container_width=True,hide_index=True)

        # ══════════════════════════════════════════════════════
        # 🌍 NOUVEAU v6.1 — VUE 3D RELIEF RÉEL PYDECK AVEC COUCHES
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🌍 Vue 3D relief réel — style Google Earth")
        st.caption("ArcGIS World Elevation 3D · Gratuit · Sans compte · Sans clé API · Tracé sur vrai relief 3D")

        # ── Toggles de couches ──────────────────────────────────
        st.markdown("**🎛️ Couches actives**")
        col_l1,col_l2,col_l3,col_l4,col_l5=st.columns(5)
        with col_l1: layer_relief     = st.checkbox("🏔 Relief 3D",     value=True,  key="layer_relief")
        with col_l2: layer_allure     = st.checkbox("⏱ Allure prédite", value=False, key="layer_allure",
                                                     help="Colorie le tracé par allure km/km — nécessite une prédiction calculée")
        with col_l3: layer_osm        = st.checkbox("🌿 Surfaces OSM",  value=False, key="layer_osm",
                                                     help="Colorie par type de surface détecté (nécessite l'analyse OSM)")
        with col_l4: layer_tech       = st.checkbox("⚠️ Zones tech.",   value=IS_TRAIL, key="layer_tech",
                                                     help="Surligne les zones techniques (Trail uniquement)")
        with col_l5: layer_cp         = st.checkbox("📍 Checkpoints",   value=True,  key="layer_cp")

        col_3dh, col_3dp = st.columns([3,1])
        with col_3dh: map_height_3d = st.slider("Hauteur (px)",400,850,600,25,key="map_height_3d")
        with col_3dp: map_pitch_3d  = st.slider("Inclinaison (°)",20,75,52,5,key="map_pitch_3d")

        if st.button("🌍 Générer la vue 3D",type="primary",key="btn_3d_pydeck"):
            with st.spinner("Construction vue 3D relief ArcGIS World Elevation..."):
                tech_segs_3d = st.session_state.get("tech_segs",[]) if (IS_TRAIL and layer_tech) else []
                df_pred_3d   = st.session_state["res"]["df"] if ("res" in st.session_state and layer_allure) else None
                osm_data_3d  = st.session_state.get("osm_surface") if layer_osm else None

                html_3d = generate_3d_terrain_html(
                    points=points, cum_d_map=cum_d_map,
                    checkpoints=checkpoints if layer_cp else [],
                    df_prediction=df_pred_3d,
                    tech_segments=tech_segs_3d,
                    dem_elevations=dem_elevations,
                    osm_surface_data=osm_data_3d,
                    height=map_height_3d,
                    pitch=map_pitch_3d,
                )
                st.session_state["html_3d_terrain"] = html_3d
                st.session_state["html_3d_km"] = round(total_dist_km)

        if "html_3d_terrain" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state["html_3d_terrain"],
                            height=map_height_3d, scrolling=False)
            col_3dl1, col_3dl2 = st.columns([1, 3])
            with col_3dl1:
                st.download_button(
                    "⬇️ Télécharger HTML 3D",
                    data=st.session_state["html_3d_terrain"].encode("utf-8"),
                    file_name=f"trail_3d_{st.session_state['html_3d_km']}km.html",
                    mime="text/html",
                    help="Fichier autonome — ouvre directement dans Chrome/Firefox, sans serveur"
                )
            with col_3dl2:
                st.caption("💡 Couches interactives dans la carte · Glisser : rotation · Scroll : zoom · Vue oblique 52° par défaut")

        # ══════════════════════════════════════════════════════
        # 🎬 ANIMATION 3D CINÉMATIQUE THREE.JS
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🎬 Animation 3D cinématique — relief néon")
        st.caption("Three.js r128 — tracé néon violet sur relief 3D procédural. Sans compte, sans clé API.")

        col_3d1,col_3d2=st.columns(2)
        with col_3d1:anim3d_h=st.slider("Hauteur fenêtre (px)",400,800,600,50,key="anim3d_h")
        with col_3d2:anim3d_cp=st.checkbox("Afficher les checkpoints",value=True,key="anim3d_cp")

        if st.button("🎬 Générer la vue 3D cinématique",type="primary",key="btn_3d_anim"):
            with st.spinner("Construction du terrain 3D..."):
                html_3d=generate_3d_animation(points=points,cum_d_map=cum_d_map,
                    checkpoints=checkpoints if anim3d_cp else [],
                    total_dist_km=total_dist_km,dem_elevations=dem_elevations)
                st.session_state["html_3d"]=html_3d

        if "html_3d" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state["html_3d"],height=int(st.session_state.get("anim3d_h",600)),scrolling=False)
            col_dl1,col_dl2=st.columns([1,3])
            with col_dl1:
                st.download_button(label="⬇️ Télécharger le fichier HTML 3D",
                                   data=st.session_state["html_3d"].encode("utf-8"),
                                   file_name=f"trail_3d_{round(total_dist_km)}km.html",mime="text/html")
            with col_dl2:st.caption("💡 Fichier autonome — ouvrez dans Chrome/Firefox, partageable sans connexion.")

        # ══════════════════════════════════════════════════════
        # 🎬 ANIMATION 2D LEAFLET (version originale conservée)
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🗺️ Animation 2D Leaflet — style présentation d'étape")
        with st.expander("⚙️ Paramètres de l'animation 2D",expanded=False):
            col_an1,col_an2,col_an3=st.columns(3)
            with col_an1:anim_frames=st.slider("Nombre de frames",60,300,120,10);anim_duration=st.slider("Durée totale (secondes)",5,60,20,5)
            with col_an2:anim_color=st.selectbox("Colorier le tracé par",["Altitude","Pente","Distance"],key="anim_color_by");anim_style=st.selectbox("Style de carte fond",["Sombre (trail)","Satellite ESRI","Topo","Blanc (épuré)"],key="anim_map_style")
            with col_an3:anim_width=st.slider("Épaisseur du tracé",2,8,4);anim_dot_size=st.slider("Taille du point",8,30,16);anim_show_elev=st.checkbox("Afficher profil d'altitude animé",value=True);anim_show_cp=st.checkbox("Afficher les checkpoints 2D",value=True)
        if st.button("🎬 Générer l'animation 2D",type="primary",key="btn_generate_anim"):
            with st.spinner("Génération de l'animation en cours..."):
                n_pts_a=len(points);step_a=max(1,n_pts_a//600)
                lats_a=lats_m[::step_a];lons_a=lons_m[::step_a]
                elev_a=[getattr(p,"elevation",0.0) or 0.0 for p in points][::step_a]
                dist_a=[cum_d_map[i]/1000.0 for i in range(0,n_pts_a,step_a)]
                n_sub=len(lats_a);slopes_a=[0.0]
                for i in range(1,n_sub):
                    dd=(dist_a[i]-dist_a[i-1])*1000.0;de=elev_a[i]-elev_a[i-1]
                    slopes_a.append((de/dd*100.0) if dd>0.5 else 0.0)
                if anim_color=="Altitude":color_vals=elev_a
                elif anim_color=="Pente":color_vals=slopes_a
                else:color_vals=dist_a
                cv_min=min(color_vals);cv_max=max(color_vals) if max(color_vals)!=cv_min else cv_min+1
                cv_norm=[(v-cv_min)/(cv_max-cv_min) for v in color_vals]
                _n=len(lons_a)
                LO_js=_json.dumps([round(x,6) for x in lons_a]);LA_js=_json.dumps([round(x,6) for x in lats_a])
                DI_js=_json.dumps([round(x,3) for x in dist_a]);EL_js=_json.dumps([round(e,1) for e in elev_a])
                SL_js=_json.dumps([round(s,1) for s in slopes_a]);CV_js=_json.dumps([round(v,4) for v in cv_norm])
                CP_js=_json.dumps([{"lat":c["lat"],"lon":c["lon"],"label":c["label"],"dist":c["dist_km"],"elev":c["alt"],"color":{"🥤 Ravitaillement":"#22d3ee","🏔 Sommet":"#f59e0b","🔻 Col":"#a855f7","⚠️ Point clé":"#ef4444","⏱ Point de passage":"#60a5fa","🏁 Intermédiaire":"#9ca3af"}.get(c["type"],"#f97316")} for c in sorted(checkpoints,key=lambda x:x["dist_km"])] if anim_show_cp and checkpoints else [])
                _dplus_a=int(sum(max(0,elev_a[i]-elev_a[i-1]) for i in range(1,_n)))
                _emin=round(min(elev_a));_emax=round(max(elev_a))
                _clat=round(float(np.mean(lats_a)),6);_clon=round(float(np.mean(lons_a)),6)
                _total_km=round(total_dist_km,1)
                _head=f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Trail {_total_km} km</title><link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&family=Space+Mono&display=swap" rel="stylesheet"/><link href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" rel="stylesheet"/><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><style>*{{margin:0;padding:0;box-sizing:border-box;}}html,body{{height:100%;overflow:hidden;background:#060a14;font-family:'Space Grotesk',sans-serif;color:#e2e8f0;}}#app{{display:grid;grid-template-rows:48px 1fr 130px;height:100vh;}}#hdr{{display:flex;align-items:center;justify-content:space-between;padding:0 16px;background:rgba(6,10,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.08);z-index:200;gap:10px;}}#ti h1{{font-size:.78rem;font-weight:800;letter-spacing:.06em;}}#ti p{{font-size:.55rem;color:#64748b;font-family:'Space Mono';}}#stats{{display:flex;gap:4px;flex:1;justify-content:center;}}.sc{{display:flex;flex-direction:column;align-items:center;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:6px;padding:2px 9px;min-width:52px;}}.sv{{font-size:.85rem;font-weight:800;font-family:'Space Mono';color:#f97316;}}.sl{{font-size:.45rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;}}#ctrls{{display:flex;gap:4px;}}.btn{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#e2e8f0;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:.65rem;font-weight:700;font-family:'Space Grotesk';transition:all .15s;}}.btn:hover,.btn.on{{background:rgba(249,115,22,.3);border-color:#f97316;color:#fff;}}.btn.pl{{background:#f97316;border-color:#f97316;color:#fff;}}#mwrap{{position:relative;min-height:0;}}#map{{width:100%;height:100%;}}.leaflet-container{{background:#060a14!important;}}.leaflet-control-attribution{{display:none!important;}}#prog{{position:absolute;bottom:0;left:0;right:0;height:3px;background:rgba(255,255,255,.06);z-index:500;}}#progf{{height:100%;width:0%;background:linear-gradient(90deg,#f97316,#ef4444);box-shadow:0 0 6px #f97316;transition:width .08s;}}#badge{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);background:rgba(6,10,20,.9);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:3px 12px;font-size:.7rem;font-weight:700;font-family:'Space Mono';color:#f97316;z-index:500;white-space:nowrap;}}#toast{{display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(6,10,20,.93);backdrop-filter:blur(12px);border-radius:10px;padding:7px 16px;z-index:999;pointer-events:none;text-align:center;min-width:180px;border:1px solid #f97316;}}.tn{{font-size:.82rem;font-weight:700;}}.tm{{font-size:.6rem;color:#64748b;font-family:'Space Mono';}}#bottom{{display:grid;background:rgba(6,10,20,.97);border-top:1px solid rgba(255,255,255,.07);}}#prof{{padding:6px 16px 4px;}}#prof canvas{{width:100%;height:100%;display:block;}}</style></head><body><div id="app"><div id="hdr"><div id="ti"><h1>🏔 TRAIL — {_total_km} KM</h1><p>D+{_dplus_a}M · {_emin}–{_emax}M</p></div><div id="stats"><div class="sc"><div class="sv" id="sd">0.0</div><div class="sl">km</div></div><div class="sc"><div class="sv" id="se">—</div><div class="sl">m alt</div></div><div class="sc"><div class="sv" id="ss">—</div><div class="sl">pente</div></div><div class="sc"><div class="sv" id="sdp">0</div><div class="sl">d+</div></div></div><div id="ctrls"><button class="btn on" id="s1" onclick="spd(1)">1x</button><button class="btn" id="s2" onclick="spd(2)">2x</button><button class="btn" id="s4" onclick="spd(4)">4x</button><button class="btn on" id="bf" onclick="tgF()">CAM</button><button class="btn pl" id="bp" onclick="tgP()">&#9646;&#9646;</button><button class="btn" onclick="rst()">&#8635;</button></div></div><div id="mwrap"><div id="map"></div><div id="prog"><div id="progf"></div></div><div id="badge">0.00 / {_total_km} km</div><div id="toast"><div class="tn" id="tn"></div><div class="tm" id="tm"></div></div></div><div id="bottom"><div id="prof"><canvas id="pc"></canvas></div></div></div>"""
                _js_raw="""<script>
var LO=__LO__,LA=__LA__,DI=__DI__,EL=__EL__,SL=__SL__,CV=__CV__,CP=__CP__;
var N=LO.length,EMIN=__EMIN__,EMAX=__EMAX__,TOT=__TOT__,CLT=__CLT__,CLO=__CLO__;
var map=L.map('map',{zoomControl:false,attributionControl:false}).setView([CLT,CLO],14);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,opacity:.88}).addTo(map);
L.tileLayer('https://cartodb-basemaps-a.global.ssl.fastly.net/dark_only_labels/{z}/{x}/{y}.png',{maxZoom:18,opacity:.45}).addTo(map);
var ghostCoords=LA.map(function(la,i){return [la,LO[i]];});
L.polyline(ghostCoords,{color:'rgba(255,255,255,.07)',weight:2}).addTo(map);
function mkDot(c){return L.divIcon({className:'',iconAnchor:[7,7],html:'<div style="width:14px;height:14px;background:'+c+';border-radius:50%;border:2px solid white;box-shadow:0 0 10px '+c+'"></div>'});}
L.marker([LA[0],LO[0]],{icon:mkDot('#4ade80')}).addTo(map);
L.marker([LA[N-1],LO[N-1]],{icon:mkDot('#f87171')}).addTo(map);
CP.forEach(function(cp){var el=document.createElement('div');el.style.cssText='background:rgba(6,10,20,.92);color:#fff;padding:2px 8px;border-radius:9px;font-size:10px;font-weight:700;white-space:nowrap;border:1.5px solid '+cp.color+';font-family:Space Grotesk,sans-serif';el.textContent=cp.label;L.marker([cp.lat,cp.lon],{icon:L.divIcon({className:'',iconAnchor:[0,22],html:el.outerHTML})}).addTo(map);L.circleMarker([cp.lat,cp.lon],{radius:8,color:cp.color,fillColor:cp.color,fillOpacity:.9,weight:2}).addTo(map);});
function cs(t){if(t>.85){var v=Math.round(210+(t-.85)/.15*45);return'rgb('+v+','+v+','+v+')';}if(t>.6)return'rgb(249,'+Math.round(115+(t-.6)/.25*80)+',22)';if(t>.35)return'rgb('+Math.round(74+(t-.35)/.25*175)+','+Math.round(222-(t-.35)/.25*110)+',128)';return'rgb(38,'+Math.round(180+t*60)+',248)';}
function lerp(a,f){var i=Math.min(Math.floor(f),a.length-2),r=f-i;return a[i]*(1-r)+a[i+1]*r;}
var segs=[],dotMk=null,cur=0,frac=0,playing=true,follow=true,lastTs=null,lcp=-1,dp=0,SPD=1,BASE=N/(TOT*48);
function frame(ts){
if(!playing)return;if(!lastTs)lastTs=ts;var dt=Math.min((ts-lastTs)/1000,0.1);lastTs=ts;frac=Math.min(frac+BASE*SPD*dt*60,N-1);
var fi=Math.min(Math.floor(frac),N-2),ff=frac-fi;var la=LA[fi]*(1-ff)+LA[fi+1]*ff,lo=LO[fi]*(1-ff)+LO[fi+1]*ff,el=EL[fi]*(1-ff)+EL[fi+1]*ff;
var nc=Math.floor(frac);while(cur<nc&&cur<N-1){cur++;segs.push(L.polyline([[LA[cur-1],LO[cur-1]],[LA[cur],LO[cur]]],{color:cs(CV[cur]),weight:5,opacity:.95,smoothFactor:.5,lineCap:'round',lineJoin:'round'}).addTo(map));}
if(dotMk)map.removeLayer(dotMk);dotMk=L.marker([la,lo],{icon:L.divIcon({className:'',iconAnchor:[11,11],html:'<div style="width:22px;height:22px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#fff 0%,#f97316 45%,#dc2626 100%);border:3px solid rgba(255,255,255,.95);box-shadow:0 0 0 5px rgba(249,115,22,.25),0 0 18px rgba(249,115,22,.8)"></div>'}),zIndexOffset:2000}).addTo(map);
if(follow){var ah=Math.min(frac+30,N-1),ai=Math.min(Math.floor(ah),N-2),af=ah-ai;var ala=LA[ai]*(1-af)+LA[ai+1]*af,alo=LO[ai]*(1-af)+LO[ai+1]*af;map.setView([(la+ala)/2,(lo+alo)/2],map.getZoom(),{animate:true,duration:.3,noMoveStart:true});}
var d=lerp(DI,frac),s=lerp(SL,frac);if(frac>BASE*SPD)dp+=Math.max(0,el-lerp(EL,frac-BASE*SPD));
document.getElementById('sd').textContent=d.toFixed(2);document.getElementById('se').textContent=Math.round(el);
var ss=document.getElementById('ss');ss.textContent=(s>=0?'+':'')+s.toFixed(1)+'%';ss.style.color=s>12?'#f87171':s>5?'#fb923c':s<-8?'#38bdf8':'#f97316';
document.getElementById('sdp').textContent=Math.round(dp);document.getElementById('badge').textContent=d.toFixed(2)+' / '+TOT+' km';document.getElementById('progf').style.width=(frac/(N-1)*100)+'%';drawProf(frac);
CP.forEach(function(cp,ci){if(ci!==lcp&&Math.abs(cp.dist-d)<0.4){lcp=ci;document.getElementById('tn').textContent=cp.label;document.getElementById('tn').style.color=cp.color;document.getElementById('tm').textContent=cp.dist.toFixed(1)+' km · '+Math.round(el)+' m';var t=document.getElementById('toast');t.style.borderColor=cp.color;t.style.display='block';clearTimeout(window._ct);window._ct=setTimeout(function(){t.style.display='none';},2500);}});
if(frac>=N-1){playing=false;document.getElementById('bp').innerHTML='&#9654;';return;}requestAnimationFrame(frame);}
var pc=document.getElementById('pc'),pctx=pc.getContext('2d');
function drawProf(c){pc.width=pc.offsetWidth||800;pc.height=pc.offsetHeight||130;var W=pc.width,H=pc.height;pctx.clearRect(0,0,W,H);var H3=Math.round(H*.62),fi=Math.max(0,Math.min(Math.floor(c||0),N-1));var WIN=Math.min(80,Math.floor(N/2)),startI=Math.max(0,fi-WIN),endI=Math.min(N-1,fi+WIN);function proj(i){var prog=(i-startI)/(endI-startI||1),elN=(EL[i]-EMIN)/(EMAX-EMIN+1),skew=(i-fi)/(WIN||1)*.06*H3;return{x:prog*W,y:H3-elN*(H3*.78)-8+skew};}pctx.beginPath();var p0=proj(startI);pctx.moveTo(p0.x,p0.y);for(var i=startI+1;i<=endI;i++){var p=proj(i);pctx.lineTo(p.x,p.y);}var pe=proj(endI);pctx.lineTo(pe.x,H3);pctx.lineTo(p0.x,H3);pctx.closePath();var gM=pctx.createLinearGradient(0,0,0,H3);gM.addColorStop(0,'rgba(80,160,80,.55)');gM.addColorStop(.4,'rgba(130,100,60,.45)');gM.addColorStop(.8,'rgba(160,140,120,.35)');gM.addColorStop(1,'rgba(40,40,40,.2)');pctx.fillStyle=gM;pctx.fill();for(var i=startI+1;i<=endI;i++){var pa=proj(i-1),pb=proj(i);pctx.beginPath();pctx.moveTo(pa.x,pa.y);pctx.lineTo(pb.x,pb.y);pctx.strokeStyle=i<=fi?cs(CV[i]):'rgba(255,255,255,.18)';pctx.lineWidth=2;pctx.stroke();}if(fi>=startI&&fi<=endI){var pc2=proj(fi);pctx.beginPath();pctx.arc(pc2.x,pc2.y,12,0,Math.PI*2);pctx.fillStyle='rgba(249,115,22,.2)';pctx.fill();pctx.beginPath();pctx.arc(pc2.x,pc2.y,6,0,Math.PI*2);pctx.fillStyle='#f97316';pctx.fill();pctx.strokeStyle='white';pctx.lineWidth=2.5;pctx.stroke();pctx.fillStyle='#f97316';pctx.font='bold 11px Space Mono,monospace';pctx.fillText(Math.round(EL[fi])+' m',pc2.x+10,pc2.y-4);}pctx.fillStyle='rgba(255,255,255,.06)';pctx.fillRect(0,H3,W,1);var HB=H-H3-1,offY=H3+1,g=pctx.createLinearGradient(0,offY,0,H);g.addColorStop(0,'rgba(249,115,22,.18)');g.addColorStop(1,'rgba(0,0,0,0)');pctx.beginPath();EL.forEach(function(e,i){var x=i/N*W,y=offY+HB-((e-EMIN)/(EMAX-EMIN+1))*(HB-4)-2;i?pctx.lineTo(x,y):pctx.moveTo(x,y);});pctx.lineTo(W,H);pctx.lineTo(0,H);pctx.closePath();pctx.fillStyle=g;pctx.fill();for(var i=1;i<N;i++){var x0=(i-1)/N*W,x1=i/N*W,y0=offY+HB-((EL[i-1]-EMIN)/(EMAX-EMIN+1))*(HB-4)-2,y1=offY+HB-((EL[i]-EMIN)/(EMAX-EMIN+1))*(HB-4)-2;pctx.beginPath();pctx.moveTo(x0,y0);pctx.lineTo(x1,y1);pctx.strokeStyle=i<=(c||0)?cs(CV[i]):'rgba(255,255,255,.08)';pctx.lineWidth=1.4;pctx.stroke();}if(c>=0&&c<N){var cx2=c/N*W,cy2=offY+HB-((EL[Math.min(Math.floor(c),N-1)]-EMIN)/(EMAX-EMIN+1))*(HB-4)-2;pctx.beginPath();pctx.moveTo(cx2,offY);pctx.lineTo(cx2,H);pctx.strokeStyle='rgba(249,115,22,.5)';pctx.lineWidth=1;pctx.stroke();}}
function tgF(){follow=!follow;var b=document.getElementById('bf');b.classList.toggle('on',follow);b.textContent=follow?'CAM':'MAP';}
function tgP(){playing=!playing;document.getElementById('bp').innerHTML=playing?'&#9646;&#9646;':'&#9654;';if(playing){lastTs=null;requestAnimationFrame(frame);}}
function spd(m){SPD=m;['s1','s2','s4'].forEach(function(id){document.getElementById(id).classList.remove('on');});document.getElementById('s'+m).classList.add('on');}
function rst(){playing=false;frac=0;cur=0;dp=0;lcp=-1;lastTs=null;segs.forEach(function(s){map.removeLayer(s);});segs=[];if(dotMk){map.removeLayer(dotMk);dotMk=null;}document.getElementById('bp').innerHTML='&#9646;&#9646;';document.getElementById('progf').style.width='0%';map.setView([CLT,CLO],14);drawProf(-1);playing=true;lastTs=null;requestAnimationFrame(frame);}
window.addEventListener('load',function(){drawProf(-1);requestAnimationFrame(frame);});
window.addEventListener('resize',function(){drawProf(frac>0?frac:-1);});
</script></body></html>"""
                _js=_js_raw.replace('__LO__',LO_js).replace('__LA__',LA_js).replace('__DI__',DI_js).replace('__EL__',EL_js)
                _js=_js.replace('__SL__',SL_js).replace('__CV__',CV_js).replace('__CP__',CP_js)
                _js=_js.replace('__EMIN__',str(_emin)).replace('__EMAX__',str(_emax)).replace('__TOT__',str(_total_km))
                _js=_js.replace('__CLT__',str(_clat)).replace('__CLO__',str(_clon))
                html_anim=_head+_js;st.session_state["html_anim"]=html_anim;st.session_state["html_anim_km"]=int(total_dist_km)
        if "html_anim" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state["html_anim"],height=620,scrolling=False)
            col_dl1,col_dl2=st.columns([1,3])
            with col_dl1:st.download_button(label="⬇️ Télécharger HTML 2D",data=st.session_state["html_anim"].encode("utf-8"),file_name=f"trail_{st.session_state['html_anim_km']}km.html",mime="text/html")
            with col_dl2:st.caption("💡 Fichier autonome — ouvrez dans Chrome, partageable sans connexion.")


# ══════════════════════════════════════════════════════════════
# ONGLET 1 — TESTS D'ENDURANCE + VC
# ══════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.title("🧪 Tests d'endurance & Vitesse Critique (VC)")
    st.markdown('<div class="highlight-box"><strong>Principe :</strong> réalise 3 à 6 efforts à intensités variées (ex : 6 min, 12 min, 20 min, 30 min). La <em>Vitesse Critique</em> est la vitesse maximale que l\'athlète peut maintenir indéfiniment.</div>',unsafe_allow_html=True)
    if not HAS_FITDECODE:st.warning("⚠️ `fitdecode` non installé — `pip install fitdecode`")
    n_tests=st.number_input("Nombre de tests",min_value=2,max_value=6,value=3,step=1,key="n_tests_vc")
    st.info(f"**{n_tests} tests** à charger. Conseil : mélanger des efforts courts (6 min) et longs (20-30 min).")
    test_files=[];test_names=[];test_ranges=[]
    for i in range(int(n_tests)):
        with st.expander(f"📁 Test {i+1}",expanded=(i<2)):
            c_name,c_file=st.columns([2,3])
            with c_name:t_name=st.text_input("Nom du test",value=f"Test {i+1}",key=f"tname_{i}")
            with c_file:t_file=st.file_uploader("Fichier activité",type=["fit","gpx","tcx","csv"],key=f"tfile_{i}")
            use_range=st.checkbox("Délimiter une plage de temps",key=f"frange_{i}")
            if use_range:
                cr1,cr2=st.columns(2)
                with cr1:t_start_hms=hms_input("Début de l'effort","0:00:00",key=f"tstart_hms_{i}")
                with cr2:t_end_hms=hms_input("Fin de l'effort","0:20:00",key=f"tend_hms_{i}")
                t_start_s=float(hms_to_seconds(t_start_hms)) if validate_hms(t_start_hms) else 0.0
                t_end_s=float(hms_to_seconds(t_end_hms)) if validate_hms(t_end_hms) else None
                if t_end_s is not None and t_end_s<=t_start_s:st.warning("⚠️ La fin doit être postérieure au début.");t_start_s,t_end_s=None,None
            else:t_start_s,t_end_s=None,None
            test_files.append(t_file);test_names.append(t_name);test_ranges.append((t_start_s,t_end_s))
    st.markdown("---")
    if st.button("🔍 Analyser tous les tests",type="primary",key="btn_analyze_vc"):
        loaded=[]
        for i,(f,name,(rng_start,rng_end)) in enumerate(zip(test_files,test_names,test_ranges)):
            if f is None:st.warning(f"Test {i+1} ({name}) : aucun fichier — ignoré.");continue
            df_act=load_activity(f)
            if df_act is None or df_act.empty:st.warning(f"Test {i+1} ({name}) : impossible de lire.");continue
            if rng_start is not None or rng_end is not None:
                t0_act=float(df_act["elapsed_s"].min());t_lo=rng_start if rng_start is not None else t0_act
                t_hi=rng_end if rng_end is not None else float(df_act["elapsed_s"].max())
                df_act=df_act[(df_act["elapsed_s"]>=t_lo)&(df_act["elapsed_s"]<=t_hi)].copy()
                df_act["elapsed_s"]=df_act["elapsed_s"]-t_lo
            if df_act.empty:st.warning(f"Test {i+1} ({name}) : données vides après troncature.");continue
            dur_s=float(df_act["elapsed_s"].max())
            if df_act["distance_m"].notna().any():
                d_valid=df_act["distance_m"].dropna();dist_m=float(d_valid.iloc[-1]-d_valid.iloc[0])
                if dist_m<=0:dist_m=float(d_valid.max()-d_valid.min())
            else:dist_m=None
            hr_stats=analyze_heart_rate(df_act);spd_stats=analyze_speed_kinetics(df_act)
            loaded.append({"name":name,"idx":i,"df":df_act,"dur_s":dur_s,"dist_m":dist_m,"hr":hr_stats,"spd":spd_stats})
        st.session_state["vc_loaded"]=loaded

    if "vc_loaded" in st.session_state:
        loaded=st.session_state["vc_loaded"]
        if not loaded:st.error("Aucun test valide chargé.")
        else:
            st.subheader(f"📊 Résultats des {len(loaded)} tests")
            n_cols=2
            for row_start in range(0,len(loaded),n_cols):
                row_items=loaded[row_start:row_start+n_cols];cols=st.columns(n_cols)
                for col_idx,item in enumerate(row_items):
                    with cols[col_idx]:
                        hr=item["hr"];spd=item["spd"]
                        dur_str=seconds_to_hms(item["dur_s"]);dist_str=f"{item['dist_m']/1000:.2f} km" if item["dist_m"] else "—"
                        avg_v=(item["dist_m"]/item["dur_s"]) if item["dist_m"] and item["dur_s"]>0 else None
                        st.markdown(f'<div class="test-card"><h4>🔵 {item["name"]}</h4>',unsafe_allow_html=True)
                        m1,m2,m3=st.columns(3);m1.metric("Durée",dur_str);m2.metric("Distance",dist_str)
                        m3.metric("Vitesse moy.",f"{avg_v:.2f} m/s" if avg_v else "—")
                        if hr.get("available"):
                            st.markdown("**Fréquence cardiaque**");h1,h2,h3=st.columns(3)
                            h1.metric("FC max (P95)",f"{hr['fc_max']} bpm");h2.metric("FC moy.",f"{hr['fc_avg']:.0f} bpm")
                            h3.metric("Dérive",f"+{hr['drift_abs']:.1f} bpm")
                            st.caption(f"Seuil estimé ~{hr['seuil_estime']} bpm · Fiabilité : {hr['reliability']}")
                            df_act=item["df"]
                            if "heart_rate" in df_act.columns and df_act["heart_rate"].notna().any():
                                hr_s=smooth_hr(df_act["heart_rate"].ffill(),window=15)
                                fig_hr,ax_hr=plt.subplots(figsize=(5,2))
                                ax_hr.plot(df_act["elapsed_s"]/60.0,hr_s,color="#d62728",lw=1.5)
                                ax_hr.set_xlabel("Temps (min)");ax_hr.set_ylabel("FC (bpm)")
                                ax_hr.set_title(f"FC — {item['name']}",fontsize=9);ax_hr.grid(alpha=0.3);fig_hr.tight_layout()
                                st.pyplot(fig_hr);plt.close(fig_hr)
                        else:st.caption("FC non disponible dans ce fichier.")
                        if spd.get("available"):
                            pace_avg=pace_str(1000.0/spd["speed_avg_ms"]) if spd["speed_avg_ms"]>0 else "—"
                            st.caption(f"Vitesse moy. : {spd['speed_avg_ms']:.2f} m/s ({pace_avg}/km) | r={spd['r_value']:.2f}")
                        st.markdown('</div>',unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📐 Vitesse Critique (D = VC × T + D')")
            vc_points=[(item["dist_m"],item["dur_s"]) for item in loaded if item["dist_m"] is not None and item["dur_s"]>0]
            if len(vc_points)>=2:
                dists_vc=[p[0] for p in vc_points];durs_vc=[p[1] for p in vc_points]
                vc,d_prime,r2=compute_vc(dists_vc,durs_vc)
                cv1,cv2,cv3,cv4=st.columns(4)
                if vc and vc>0:
                    cv1.metric("Vitesse Critique (VC)",f"{vc:.2f} m/s");cv2.metric("Allure VC",pace_str(1000.0/vc)+"/km")
                    cv3.metric("D' (réserve anaérobie)",f"{d_prime:.0f} m" if d_prime else "—");cv4.metric("R² régression",f"{r2:.3f}")
                    if r2<0.90:st.warning(f"⚠️ R²={r2:.3f} — régression faible.")
                    else:st.success(f"✅ R²={r2:.3f} — bonne qualité.")
                    fig_vc,ax_vc=plt.subplots(figsize=(7,4))
                    T_arr=np.array(durs_vc);D_arr=np.array(dists_vc)
                    T_line=np.linspace(T_arr.min()*0.8,T_arr.max()*1.2,100);D_line=vc*T_line+d_prime
                    ax_vc.scatter(T_arr/60,D_arr/1000,s=80,color="#1f77b4",zorder=5,label="Tests réels")
                    ax_vc.plot(T_line/60,D_line/1000,color="#d62728",lw=2,label=f"VC={vc:.2f} m/s | D'={d_prime:.0f} m")
                    ax_vc.set_xlabel("Durée (min)");ax_vc.set_ylabel("Distance (km)")
                    ax_vc.set_title("Modèle D = VC × T + D'");ax_vc.legend();ax_vc.grid(alpha=0.3)
                    st.pyplot(fig_vc);plt.close(fig_vc)
                    st.markdown("---")
                    st.subheader("📊 Seuils physiologiques (SV1 & SV2)")
                    col_sv1,col_sv2=st.columns(2)
                    with col_sv1:sv1_input=hms_input("🟢 SV1 — allure (mm:ss/km)",default="0:05:00",key="sv1_pace_input")
                    with col_sv2:sv2_input=hms_input("🔴 SV2 — allure (mm:ss/km)",default="0:04:00",key="sv2_pace_input")
                    sv1_pace_s=hms_to_seconds(sv1_input);sv2_pace_s=hms_to_seconds(sv2_input)
                    sv1_ms=(1000.0/sv1_pace_s) if sv1_pace_s>0 else None
                    sv2_ms=(1000.0/sv2_pace_s) if sv2_pace_s>0 else None
                    if sv1_ms and sv2_ms and sv1_ms>0 and sv2_ms>0 and sv1_ms<sv2_ms:
                        m1,m2,m3,m4=st.columns(4)
                        m1.metric("🟢 SV1",pace_str(1000/sv1_ms)+"/km",delta=f"{sv1_ms/vc*100:.1f}% de VC")
                        m2.metric("🔴 SV2",pace_str(1000/sv2_ms)+"/km",delta=f"{sv2_ms/vc*100:.1f}% de VC")
                        m3.metric("🔵 VC",pace_str(1000/vc)+"/km",delta="100% VC — référence")
                        ecart_sv1_sv2_pct=(sv2_ms-sv1_ms)/sv1_ms*100;ecart_sv2_vc_pct=(vc-sv2_ms)/sv2_ms*100
                        m4.metric("Écart SV1→SV2",f"+{ecart_sv1_sv2_pct:.1f}%",delta=f"SV2→VC : +{ecart_sv2_vc_pct:.1f}%")
                        sv1_pct_vc=sv1_ms/vc*100;sv2_pct_vc=sv2_ms/vc*100;zone_aerob_pct=sv2_pct_vc-sv1_pct_vc
                        col_a,col_b=st.columns(2)
                        with col_a:
                            st.markdown(f"""| Indicateur | Valeur |\n|---|---|\n| SV1 / VC | **{sv1_pct_vc:.1f} %** |\n| SV2 / VC | **{sv2_pct_vc:.1f} %** |\n| Zone aérobie (SV1→SV2) | **{zone_aerob_pct:.1f} pts %VC** |\n| Écart SV1→SV2 | **+{ecart_sv1_sv2_pct:.1f} %** |\n| Écart SV2→VC | **+{ecart_sv2_vc_pct:.1f} %** |""")
                        with col_b:
                            if sv1_pct_vc<65:st.success("🟢 **SV1 bas** (< 65% VC) — priorité à l'**endurance fondamentale** : volume Z2.")
                            elif sv2_pct_vc<85:st.warning("🔴 **SV2 bas** (< 85% VC) — priorité au **seuil** : tempo, allure marathon.")
                            elif ecart_sv2_vc_pct>15:st.info("🔵 **Bon profil aérobie** — priorité à la **VC** : intervalles 8-15 min.")
                            else:st.success("✅ **Profil équilibré** — travail polyvalent Z2 + Z4.")
                        fig_sv,ax_sv=plt.subplots(figsize=(9,2.5))
                        zones_def=[("Z1 Récup",0.0,sv1_ms*0.85,"#81c784"),("Z2 EF",sv1_ms*0.85,sv1_ms,"#aed581"),
                                   ("Z3 Tempo",sv1_ms,sv2_ms,"#ffb74d"),("Z4 Seuil",sv2_ms,vc,"#ef5350"),("Z5 >VC",vc,vc*1.15,"#ab47bc")]
                        for(lbl_z,v_lo,v_hi,col_z) in zones_def:
                            if v_hi<=v_lo:continue
                            ax_sv.barh(0,v_hi-v_lo,left=v_lo,height=0.5,color=col_z,alpha=0.85)
                            ax_sv.text((v_lo+v_hi)/2,0,lbl_z,ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
                        for v_mark,lbl_m,col_m in[(sv1_ms,f"SV1 {pace_str(1000/sv1_ms)}/km {sv1_pct_vc:.0f}%VC","#2e7d32"),(sv2_ms,f"SV2 {pace_str(1000/sv2_ms)}/km {sv2_pct_vc:.0f}%VC","#b71c1c"),(vc,f"VC {pace_str(1000/vc)}/km 100%","#4a148c")]:
                            ax_sv.axvline(v_mark,color=col_m,lw=2);ax_sv.text(v_mark,0.32,lbl_m,ha="center",fontsize=7,color=col_m,fontweight="bold",va="bottom")
                        ax_sv.set_xlabel("Vitesse (m/s)");ax_sv.set_yticks([]);ax_sv.set_xlim(sv1_ms*0.75,vc*1.20)
                        ax_sv.set_title("Positionnement SV1 / SV2 / VC");ax_sv.grid(axis="x",alpha=0.3);fig_sv.tight_layout()
                        st.pyplot(fig_sv);plt.close(fig_sv)
                    elif sv1_ms and sv2_ms and sv1_ms>=sv2_ms:st.warning("⚠️ SV1 doit être inférieur à SV2.")
                    else:st.info("Entrez vos allures SV1 et SV2 pour voir l'analyse.")
                    st.markdown("---")
                    st.subheader("📋 Table des temps de maintien")
                    refs_for_table=st.session_state.get("refs_fit_vc",[])
                    if not refs_for_table:refs_for_table=[{"distance":item["dist_m"],"temps":item["dur_s"]} for item in loaded if item["dist_m"] and item["dur_s"]>0]
                    K_for_table=st.session_state.get("K_riegel_vc",1.06)
                    df_hold=build_holding_table(vc,d_prime,refs_for_table,K_for_table)
                    if not df_hold.empty:
                        def style_vc_row(row):
                            if row["% VC"]=="100 %":return["background-color:#2c5282;color:white;font-weight:bold"]*len(row)
                            return[""]*len(row)
                        st.dataframe(df_hold.style.apply(style_vc_row,axis=1),use_container_width=True)
                        fig_hold,ax_hold=plt.subplots(figsize=(9,4))
                        pcts=[float(p.replace(" %","")) for p in df_hold["% VC"]]
                        mask_ri=df_hold["Modèle"]=="Riegel";mask_dp=df_hold["Modèle"]=="Modèle D'"
                        if mask_ri.any():ax_hold.plot([pcts[i] for i in df_hold.index[mask_ri]],df_hold.loc[mask_ri,"Durée (min)"],"o-",color="#1f77b4",lw=2.5,ms=6,label="Riegel")
                        if mask_dp.any():ax_hold.plot([pcts[i] for i in df_hold.index[mask_dp]],df_hold.loc[mask_dp,"Durée (min)"],"o--",color="#d62728",lw=2.5,ms=6,label="Modèle D'")
                        ax_hold.axvline(100,color="gray",lw=1.5,ls=":",label=f"VC ({pace_str(1000/vc)}/km)")
                        ax_hold.set_xlabel("% de la Vitesse Critique");ax_hold.set_ylabel("Durée (min)")
                        ax_hold.set_title("Temps de maintien par % VC");ax_hold.legend();ax_hold.grid(alpha=0.3);ax_hold.set_ylim(0)
                        fig_hold.tight_layout();st.pyplot(fig_hold);plt.close(fig_hold)
                    st.markdown("---")
                    st.subheader("🏆 Prédictions & Standards de performance")
                    genre_sel=st.radio("Catégorie",["H","F"],horizontal=True,key="genre_standards")
                    refs_pred=st.session_state.get("refs_fit_vc",[])
                    if not refs_pred:refs_pred=[{"distance":item["dist_m"],"temps":item["dur_s"]} for item in loaded if item["dist_m"] and item["dur_s"]>0]
                    K_pred=st.session_state.get("K_riegel_vc",1.06)
                    perf_results,best_dist=predict_performances(vc,d_prime,refs_pred,K_pred,genre=genre_sel)
                    if perf_results:
                        cols_dist=st.columns(4)
                        for col_d,(dist_label,info) in zip(cols_dist,perf_results.items()):
                            with col_d:
                                delta_txt=None
                                if info["closest"]:
                                    c=info["closest"];sign="✅" if c["atteint"] else "⏳"
                                    delta_txt=f"{sign} {c['diff_str']} de {c['standard']}"
                                star=" ⭐" if dist_label==best_dist else ""
                                st.metric(label=f"{dist_label}{star}",value=info["t_pred_hms"],delta=delta_txt,delta_color="inverse")
                        if best_dist:st.success(f"⭐ Distance de prédilection : **{best_dist}**")
                        for dist_label,info in perf_results.items():
                            with st.expander(f"📏 {dist_label} — {info['t_pred_hms']} ({info['pace_pred']}/km)"+("  ⭐" if dist_label==best_dist else ""),expanded=(dist_label==best_dist)):
                                if not info["standards"]:st.caption("Aucun standard disponible.");continue
                                rows_std=[]
                                for s in info["standards"]:rows_std.append({"Standard":f"{s['emoji']} {s['standard']}","Temps standard":s["temps_std"],"Votre prévu":info["t_pred_hms"],"Écart":s["diff_str"],"Statut":"✅ Atteint" if s["atteint"] else "⏳ À réaliser"})
                                df_std=pd.DataFrame(rows_std)
                                def style_std_row(row):
                                    if "✅" in row["Statut"]:return["background-color:#d4edda"]*len(row)
                                    return[""]*len(row)
                                st.dataframe(df_std.style.apply(style_std_row,axis=1),use_container_width=True,hide_index=True)
                                c=info["closest"]
                                if c:
                                    if c["atteint"]:st.success(f"{c['emoji']} Vous atteignez **{c['standard']}** avec **{c['diff_str']}** d'avance.")
                                    else:st.info(f"{c['emoji']} Standard le plus proche : **{c['standard']}** — il manque **{c['diff_str'].lstrip('+')}** (objectif : {c['temps_std']}).")
                    st.markdown("---")
                    st.subheader("📄 Export PDF")
                    if st.button("Générer le PDF",key="btn_pdf_vc"):
                        buf=io.BytesIO()
                        with PdfPages(buf) as pdf:
                            fig_p1,axes=plt.subplots(2,1,figsize=(8.27,11.69))
                            axes[0].scatter(T_arr/60,D_arr/1000,s=80,color="#1f77b4",label="Tests réels",zorder=5)
                            axes[0].plot(T_line/60,D_line/1000,color="#d62728",lw=2,label=f"VC={vc:.2f} m/s | D'={d_prime:.0f} m")
                            axes[0].set_xlabel("Durée (min)");axes[0].set_ylabel("Distance (km)")
                            axes[0].set_title("Modèle Vitesse Critique");axes[0].legend();axes[0].grid(alpha=0.3)
                            if not df_hold.empty:
                                pcts_pdf=[float(p.replace(" %","")) for p in df_hold["% VC"]]
                                if mask_ri.any():axes[1].plot([pcts_pdf[i] for i in df_hold.index[mask_ri]],df_hold.loc[mask_ri,"Durée (min)"],"o-",color="#1f77b4",lw=2,label="Riegel")
                                if mask_dp.any():axes[1].plot([pcts_pdf[i] for i in df_hold.index[mask_dp]],df_hold.loc[mask_dp,"Durée (min)"],"o--",color="#d62728",lw=2,label="Modèle D'")
                                axes[1].axvline(100,color="gray",lw=1.5,ls=":")
                                axes[1].set_xlabel("% VC");axes[1].set_ylabel("Durée (min)")
                                axes[1].set_title("Temps de maintien");axes[1].legend();axes[1].grid(alpha=0.3);axes[1].set_ylim(0)
                            fig_p1.tight_layout();pdf.savefig(fig_p1);plt.close(fig_p1)
                            for item in loaded:
                                df_act=item["df"]
                                if "heart_rate" not in df_act.columns or not df_act["heart_rate"].notna().any():continue
                                fig_fc,ax_fc=plt.subplots(figsize=(8.27,4))
                                hr_s=smooth_hr(df_act["heart_rate"].ffill(),window=15)
                                ax_fc.plot(df_act["elapsed_s"]/60.0,hr_s,color="#d62728",lw=1.5)
                                ax_fc.set_xlabel("Temps (min)");ax_fc.set_ylabel("FC (bpm)")
                                ax_fc.set_title(f"FC — {item['name']}");ax_fc.grid(alpha=0.3);fig_fc.tight_layout()
                                pdf.savefig(fig_fc);plt.close(fig_fc)
                        buf.seek(0);st.download_button("⬇️ Télécharger le rapport PDF",data=buf,file_name="rapport_vc.pdf",mime="application/pdf")
            else:st.info("Chargez au moins 2 tests pour calculer la Vitesse Critique.")
