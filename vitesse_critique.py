# analyse_course_v4.py
# Application Streamlit unifiée — 3 onglets
#   [0] 🏃 Prédiction de course (v3)
#   [1] 🧪 Tests d'endurance + Vitesse Critique
#   [2] ⚙️  Analyse entraînement (avec analyse fractionnés)
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
from scipy import stats as sp_stats

# ══════════════════════════════════════════════════════════════
# CONFIG — un seul set_page_config pour toute l'app
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Coach Running — Suite complète", layout="wide", page_icon="🏃")
TZ_NAME_DEFAULT = "Europe/Paris"

# ══════════════════════════════════════════════════════════════
# CSS UNIFIÉ
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
.param-box {
    background: #f8f9fa;
    border-left: 4px solid #1f77b4;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.88rem;
}
.param-up   { color: #d62728; font-weight: 600; }
.param-down { color: #2ca02c; font-weight: 600; }
.highlight-box {
    background: #e8f0fe;
    border: 1px solid #6c8ebf;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #1a3a5c;
}
.test-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.test-card h4 { margin: 0 0 8px 0; color: #1f77b4; font-size: 1rem; }
.result-metric { text-align: center; font-size: 1.4rem; font-weight: 700; }

.sidebar-label {
    background: #e8f4fd;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 0.80rem;
    color: #1f77b4;
    margin-bottom: 10px;
}
.interval-card {
    background: #f0f4ff;
    border: 1px solid #b0c4de;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════

def param_help(text_up: str, text_down: str, note: str = ""):
    note_html = f"<br><em>{note}</em>" if note else ""
    st.markdown(
        f'<div class="param-box">'
        f'<span class="param-up">⬆️ Augmenter</span> : {text_up}<br>'
        f'<span class="param-down">⬇️ Diminuer</span> : {text_down}'
        f'{note_html}</div>',
        unsafe_allow_html=True
    )





# ══════════════════════════════════════════════════════════════
# UTILITAIRES PARTAGÉS
# ══════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    try:
        if val is None: return float(default)
        if isinstance(val, str):
            s = val.strip()
            if s in ("", "nan", "none"): return float(default)
            return float(s.replace(",", "."))
        if isinstance(val, (float, int, np.number)):
            if np.isnan(val) or np.isinf(val): return float(default)
            return float(val)
        return float(val)
    except Exception:
        return float(default)


def hms_to_seconds(hms: str) -> int:
    """Convertit hh:mm:ss, mm:ss ou ss en secondes entières."""
    if hms is None: return 0
    try:
        parts = [int(p) for p in str(hms).strip().split(":")]
        if len(parts) == 3:   h, m, s = parts
        elif len(parts) == 2: h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 1: h, m, s = 0, 0, parts[0]
        else: return 0
        if not (0 <= m <= 59 and 0 <= s <= 59): return 0
        return max(0, h*3600 + m*60 + s)
    except Exception:
        return 0


def seconds_to_hms(s: float) -> str:
    """Convertit des secondes en hh:mm:ss."""
    s = int(round(s))
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"


def hms_to_timedelta(hms: str) -> timedelta:
    return timedelta(seconds=hms_to_seconds(hms))


def validate_hms(val: str) -> bool:
    """Retourne True si le format hh:mm:ss / mm:ss est valide."""
    try:
        parts = [int(p) for p in val.strip().split(":")]
        if len(parts) == 3:
            return 0 <= parts[1] <= 59 and 0 <= parts[2] <= 59
        if len(parts) == 2:
            return 0 <= parts[0] <= 59 and 0 <= parts[1] <= 59
        return False
    except Exception:
        return False


def hms_input(label: str, default: str = "0:00:00", key: str = None,
              help: str = None, compact: bool = False) -> str:
    """
    Champ de saisie d'un temps au format hh:mm:ss.
    Retourne la valeur saisie (chaîne). Affiche un avertissement si format invalide.
    """
    val = st.text_input(
        label,
        value=str(st.session_state.get(key, default)) if key else default,
        key=key,
        help=help or "Format : hh:mm:ss  (ex: 1:23:45)",
        placeholder="hh:mm:ss",
    )
    if val and not validate_hms(val):
        st.warning(f"⚠️ Format invalide : **{val}** — attendu hh:mm:ss (ex: 0:45:00)")
    return val


def pace_str(secs_per_km: float) -> str:
    if secs_per_km is None or secs_per_km <= 0 or not math.isfinite(secs_per_km):
        return "0:00"
    t = int(round(float(secs_per_km)))
    return f"{t//60}:{t%60:02d}"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compute_dplus_dminus(elevs):
    arr = np.array([safe_float(e, np.nan) for e in elevs], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2: return 0.0, 0.0
    diffs = np.diff(arr)
    return float(np.sum(np.clip(diffs, 0, None))), float(-np.sum(np.clip(diffs, None, 0)))


# ══════════════════════════════════════════════════════════════
# MODÈLES PHYSIQUES (v3)
# ══════════════════════════════════════════════════════════════

def wbgt_simplified(T_c: float, RH: float) -> float:
    try:
        RH_c = max(0.0, min(100.0, float(RH)))
        T = float(T_c)
        Tw = (T * math.atan(0.151977 * (RH_c + 8.313659)**0.5)
              + math.atan(T + RH_c)
              - math.atan(RH_c - 1.676331)
              + 0.00391838 * RH_c**1.5 * math.atan(0.023101 * RH_c)
              - 4.686035)
        Tg = T + 2.0
        return 0.7*Tw + 0.2*Tg + 0.1*T
    except Exception:
        return float(T_c)


def effective_temp(T_c, RH, use_wbgt):
    return wbgt_simplified(T_c, RH) if use_wbgt else float(T_c)


def altitude_vo2_multiplier(altitude_m, altitude_ref_m=0.0):
    alt = max(0.0, float(altitude_m))
    alt_ref = max(0.0, float(altitude_ref_m))
    effective_alt = max(0.0, alt - max(1500.0, alt_ref))
    penalty = min(0.25, 0.01 * (effective_alt / 100.0))
    return 1.0 + penalty


def minetti_cost(grade_fraction):
    g = max(-0.45, min(0.45, float(grade_fraction)))
    c = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    return max(0.1, float(c))


def minetti_multiplier(grade_pct):
    return float(max(0.92, min(1.35, minetti_cost(float(grade_pct)/100.0) / 3.6)))


def grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    try:
        g = float(grade_pct)/100.0
        g0u = max(1e-6, float(g0_up)/100.0)
        g0d = max(1e-6, float(g0_down)/100.0)
        if g >= 0:
            g_eff = math.tanh(g/g0u)*g0u
            mult = 1.0 + float(k_up)*g_eff
        else:
            g_eff = math.tanh((-g)/g0d)*g0d
            bonus = min(float(k_down)*g_eff, abs(float(down_cap)))
            mult = 1.0 - bonus
        mult = min(mult, 1.0+float(max_up))
        mult = max(mult, 1.0+float(max_down))
        return max(0.01, float(mult))
    except Exception:
        return 1.0


def combined_grade_multiplier(grade_pct, use_minetti, minetti_weight,
                               k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    if not use_minetti:
        return grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    m_min = minetti_multiplier(grade_pct)
    m_heu = grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    w = max(0.0, min(1.0, float(minetti_weight)))
    return w*m_min + (1.0-w)*m_heu


def temp_multiplier(temp_eff, opt_temp, cold_quad, hot_quad, max_penalty):
    if temp_eff is None: return 1.0
    d = float(temp_eff) - float(opt_temp)
    pen = hot_quad*d**2 if d >= 0 else cold_quad*(-d)**2
    return 1.0 + min(float(max_penalty), float(pen))


def wind_components(wind_speed_ms, wind_dir_from_deg, course_bearing_deg):
    if wind_speed_ms is None or wind_dir_from_deg is None: return 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0: return 0.0, 0.0
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians((wind_to - course_bearing_deg + 540.0) % 360.0 - 180.0)
    along = ws * math.cos(delta)
    return float(max(0.0, -along)), float(max(0.0, along))


def wind_multiplier(head_ms, tail_ms, pace_s_per_km, drag_coeff, tail_credit, cap_head, cap_tail):
    pace = max(150.0, float(pace_s_per_km))
    v_run = 1000.0/pace
    w_along = float(head_ms) - float(tail_ms)
    v_rel = max(0.0, v_run + w_along)
    base = max(1e-9, v_run**2)
    extra = (v_rel**2 - v_run**2)/base
    if extra < 0: extra = float(tail_credit)*extra
    mult = 1.0 + float(drag_coeff)*extra
    return float(max(1.0+cap_tail, min(1.0+cap_head, mult)))


def wind_gate(grade_pct, g1=2.0, g2=8.0, min_gate=0.25):
    g = max(0.0, float(grade_pct))
    if g <= g1: return 1.0
    if g >= g2: return float(min_gate)
    return float(1.0 - (g-g1)/(g2-g1)*(1.0-min_gate))


def cap_combined(mult_total, grade_pct, base_cap, extra_per_pct, max_cap):
    g = max(0.0, float(grade_pct))
    cap = min(float(max_cap), float(base_cap) + float(extra_per_pct)*g)
    return min(float(mult_total), 1.0+cap)


def fatigue_multiplier(d_plus_cum, dist_cum, d_plus_total, dist_total, rate_pct, mode):
    if rate_pct <= 0: return 1.0
    rate = rate_pct/100.0
    prog_dist  = min(1.0, dist_cum/max(1.0, dist_total))
    prog_dplus = min(1.0, d_plus_cum/max(1.0, d_plus_total))
    dplus_ratio = d_plus_total/max(1.0, dist_total)
    w_dplus = min(0.8, dplus_ratio*10.0)
    if mode == "distance":    prog = prog_dist
    elif mode == "d_plus":    prog = prog_dplus
    else:                     prog = w_dplus*prog_dplus + (1.0-w_dplus)*prog_dist
    k = 2.0
    factor = (math.exp(k*prog) - 1.0)/(math.exp(k) - 1.0)
    return 1.0 + rate*factor


# ══════════════════════════════════════════════════════════════
# MÉTÉO
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_weather_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data: return None
        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums  = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]
        dt = dt_local_naive
        for i in range(len(times)-1):
            if times[i] <= dt <= times[i+1]:
                r = (dt-times[i]).total_seconds()/max(1.0,(times[i+1]-times[i]).total_seconds())
                a1, a2 = float(wdirs[i])%360, float(wdirs[i+1])%360
                da = (a2-a1+540.0)%360.0-180.0
                return {"temp": temps[i]+r*(temps[i+1]-temps[i]),
                        "wind": winds[i]+r*(winds[i+1]-winds[i]),
                        "humidity": hums[i]+r*(hums[i+1]-hums[i]),
                        "wind_dir": (a1+r*da)%360.0}
        idx = min(range(len(times)), key=lambda i: abs(times[i]-dt))
        return {"temp": float(temps[idx]), "wind": float(winds[idx]),
                "humidity": float(hums[idx]), "wind_dir": float(wdirs[idx])}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_weather_archive_day(lat, lon, date_obj, tz_name=TZ_NAME_DEFAULT):
    try:
        ds = date_obj.strftime("%Y-%m-%d")
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={ds}&end_date={ds}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data: return None
        return ([datetime.fromisoformat(t) for t in data["hourly"]["time"]],
                data["hourly"]["temperature_2m"], data["hourly"]["wind_speed_10m"],
                data["hourly"]["relativehumidity_2m"], data["hourly"]["wind_direction_10m"])
    except Exception:
        return None


def get_avg_weather(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    if start_dt is None or end_dt is None: return None, None, None
    if (end_dt-start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2); end_dt += timedelta(minutes=2)
    res = get_weather_archive_day(lat, lon, start_dt.date(), tz_name=tz_name)
    if not res: return None, None, None
    times, temps, winds, hums, _ = res
    selT = [T for t,T in zip(times,temps) if start_dt<=t<=end_dt]
    selW = [W for t,W in zip(times,winds) if start_dt<=t<=end_dt]
    selH = [H for t,H in zip(times,hums)  if start_dt<=t<=end_dt]
    if not selT:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-start_dt))
        return float(temps[idx]), float(winds[idx]), float(hums[idx])
    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# ══════════════════════════════════════════════════════════════
# DEM
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Correction altimétrique DEM...")
def fetch_dem_elevations(lats: tuple, lons: tuple, dataset: str = "srtm30m") -> list:
    try:
        locs = "|".join(f"{la},{lo}" for la,lo in zip(lats,lons))
        data = requests.get(f"https://api.opentopodata.org/v1/{dataset}?locations={locs}", timeout=30).json()
        if data.get("status") != "OK": return [None]*len(lats)
        return [r.get("elevation") for r in data["results"]]
    except Exception:
        return [None]*len(lats)


def correct_elevations_dem(points, max_points=100, dataset="srtm30m"):
    n = len(points)
    if n < 2: return np.array([getattr(p,"elevation",0.0) or 0.0 for p in points])
    step = max(1, n//max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n-1: indices.append(n-1)
    lats = tuple(points[i].latitude for i in indices)
    lons = tuple(points[i].longitude for i in indices)
    dem  = fetch_dem_elevations(lats, lons, dataset=dataset)
    cum_all = [0.0]
    for i in range(1,n):
        cum_all.append(cum_all[-1] + haversine_m(
            points[i-1].latitude, points[i-1].longitude,
            points[i].latitude,   points[i].longitude))
    cum_sub = [cum_all[i] for i in indices]
    valid = [(d,e) for d,e in zip(cum_sub,dem) if e is not None]
    if len(valid) < 2:
        return np.array([getattr(p,"elevation",0.0) or 0.0 for p in points])
    return np.interp(cum_all, [v[0] for v in valid], [v[1] for v in valid])


# ══════════════════════════════════════════════════════════════
# FC ANALYSE (v3)
# ══════════════════════════════════════════════════════════════

def analyze_hr_v3(hr_records: list) -> dict:
    hrs = [h for h in hr_records if h is not None and 50<=h<=220]
    if len(hrs) < 10:
        return {"hr_max":None,"hr_avg":None,"hr_drift":None,"reliability":"inconnue"}
    arr = np.array(hrs, dtype=float)
    n = len(arr)
    hr_max = float(np.percentile(arr, 95))
    hr_avg = float(np.mean(arr))
    q1, q3 = int(n*0.25), int(n*0.75)
    drift = float(np.mean(arr[q3:])) - float(np.mean(arr[:q1]))
    reliability = "haute" if drift<5 else ("moyenne" if drift<12 else "basse (dérive cardiaque forte)")
    return {"hr_max":round(hr_max),"hr_avg":round(hr_avg),
            "hr_drift":round(drift,1),"hr_threshold_est":round(hr_max*0.88),
            "reliability":reliability}


# ══════════════════════════════════════════════════════════════
# PARSING FICHIERS (fitparse — pour onglet Prédiction)
# ══════════════════════════════════════════════════════════════

class SimplePoint:
    def __init__(self, lat, lon, elev=0.0, time=None):
        self.latitude  = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.time      = time
    def distance_3d(self, other):
        h = haversine_m(self.latitude,self.longitude,other.latitude,other.longitude)
        v = self.elevation - other.elevation
        return math.sqrt(h*h+v*v)


def parse_gpx_points(file):
    try:
        file.seek(0)
        gpx = gpxpy.parse(file)
        pts = [p for track in gpx.tracks for seg in track.segments for p in seg.points]
        return gpx, pts
    except Exception as e:
        st.error(f"Erreur GPX : {e}")
        return None, []


def parse_fit_ref(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        fit = FitFile(file); fit.parse()
        records, times_pts, hr_records = [], [], []
        start_global = elapsed_global = None
        for msg in fit.get_messages("session"):
            vals = {d.name:d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int,float)):
                elapsed_global = float(vals["total_elapsed_time"])
        for msg in fit.get_messages("record"):
            vals = {d.name:d.value for d in msg}
            lat_r = vals.get("position_lat"); lon_r = vals.get("position_long")
            if lat_r is None or lon_r is None: continue
            lat = lat_r*(180/2**31); lon = lon_r*(180/2**31)
            ts  = vals.get("timestamp")
            dt  = ts.replace(tzinfo=None) if isinstance(ts,datetime) else None
            alt = (vals.get("enhanced_altitude") or vals.get("altitude") or 0.0)
            dist = float(vals.get("distance") or 0.0)
            hr   = vals.get("heart_rate")
            hr_records.append(int(hr) if hr is not None else None)
            records.append((lat,lon,float(alt),dist)); times_pts.append(dt)
        if not records: return None
        df = pd.DataFrame(records, columns=["lat","lon","elev","dist"])
        valid_t = [t for t in times_pts if t is not None]
        if len(valid_t) >= 2:
            start_dt, end_dt = min(valid_t), max(valid_t)
        elif start_global and elapsed_global:
            start_dt = start_global; end_dt = start_global+timedelta(seconds=elapsed_global)
        else:
            start_dt = datetime.now().replace(hour=12,minute=0,second=0,microsecond=0)-timedelta(days=1)
            end_dt   = start_dt+timedelta(minutes=5)
        avgT,avgW,avgH = get_avg_weather(records[0][0],records[0][1],start_dt,end_dt,tz_name)
        elev_arr = df["elev"].values
        dup = float(np.sum(np.clip(np.diff(elev_arr),0,None))) if elev_arr.size>=2 else 0.0
        ddn = float(-np.sum(np.clip(np.diff(elev_arr),None,0))) if elev_arr.size>=2 else 0.0
        return {"points":[{"lat":r[0],"lon":r[1],"elev":r[2],"dist":r[3],"time":t}
                           for r,t in zip(records,times_pts)],
                "distance":float(df["dist"].max()),
                "D_up":dup,"D_down":ddn,
                "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
                "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,
                "hr_analysis":analyze_hr_v3(hr_records)}
    except Exception as e:
        st.error(f"Erreur FIT : {e}"); return None


def parse_tcx_ref(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0); root = ET.parse(file).getroot()
    except Exception:
        return None
    ns = {"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    pts, times, elevs = [], [], []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        lat = tp.find("tcx:Position/tcx:LatitudeDegrees",  ns)
        lon = tp.find("tcx:Position/tcx:LongitudeDegrees", ns)
        if lat is None or lon is None: continue
        ele = tp.find("tcx:AltitudeMeters", ns)
        tim = tp.find("tcx:Time", ns)
        elev = float(ele.text) if ele is not None else 0.0
        try:   t = datetime.fromisoformat(tim.text.replace("Z","+00:00")).replace(tzinfo=None)
        except: t = None
        pts.append(SimplePoint(float(lat.text),float(lon.text),elev,t))
        times.append(t); elevs.append(elev)
    if len(pts)<2: return None
    vt = [t for t in times if t is not None]
    start_dt = vt[0] if vt else datetime.now()-timedelta(days=1)
    end_dt   = vt[-1] if len(vt)>1 else start_dt+timedelta(minutes=5)
    avgT,avgW,avgH = get_avg_weather(pts[0].latitude,pts[0].longitude,start_dt,end_dt,tz_name)
    total = sum(pts[i].distance_3d(pts[i-1]) for i in range(1,len(pts)))
    dup,ddn = compute_dplus_dminus(elevs)
    return {"points":pts,"distance":round(total),"D_up":round(dup,1),"D_down":round(ddn,1),
            "duration_hms":seconds_to_hms((end_dt-start_dt).total_seconds()),
            "avg_temp":avgT,"avg_wind":avgW,"avg_humidity":avgH,"hr_analysis":None}


def extract_segment(points, start_td, end_td):
    def get_t(p): return p.get("time") if isinstance(p,dict) else getattr(p,"time",None)
    ts = [get_t(p) for p in points if get_t(p) is not None]
    if len(ts)<2: return points
    t0 = min(ts)
    seg = [p for p in points if get_t(p) is not None
           and t0+start_td <= get_t(p) <= t0+end_td+timedelta(seconds=1)]
    return seg if len(seg)>=2 else points


# ══════════════════════════════════════════════════════════════
# CHARGEMENT ACTIVITÉ (fitdecode — pour onglets Tests + Entraînement)
# ══════════════════════════════════════════════════════════════

def load_activity(file):
    """
    Charge FIT (fitdecode), GPX, TCX ou CSV.
    Retourne un DataFrame avec : elapsed_s, heart_rate, speed_ms, distance_m, altitude_m
    """
    if file is None: return None
    fname = file.name.lower()
    df = None

    if fname.endswith(".fit"):
        if not HAS_FITDECODE:
            st.error("Installez 'fitdecode' : pip install fitdecode"); return None
        rows = []
        file.seek(0)
        with fitdecode.FitReader(file) as fit:
            for frame in fit:
                if not isinstance(frame, fitdecode.FitDataMessage): continue
                if frame.name != "record": continue
                def fv(field, default=None):
                    try:    return frame.get_value(field)
                    except: return default
                ts   = fv("timestamp")
                hr   = fv("heart_rate")
                spd  = fv("speed")
                dist = fv("distance")
                alt  = fv("enhanced_altitude") or fv("altitude")
                if ts is None: continue
                rows.append({"timestamp": ts, "heart_rate": hr,
                             "speed_ms": spd, "distance_m": dist, "altitude_m": alt})
        if not rows: return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        t0 = df["timestamp"].iloc[0]
        df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()

    elif fname.endswith(".gpx"):
        file.seek(0)
        gpx = gpxpy.parse(file)
        rows = []
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    rows.append({"timestamp": pt.time,
                                 "heart_rate": None, "speed_ms": None,
                                 "distance_m": None,
                                 "altitude_m": pt.elevation or 0.0})
        if not rows: return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        t0 = df["timestamp"].iloc[0]
        df["elapsed_s"] = (df["timestamp"]-t0).dt.total_seconds()
        lats = []
        file.seek(0)
        gpx2 = gpxpy.parse(file)
        for track in gpx2.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    lats.append((pt.latitude, pt.longitude))
        cumd = [0.0]
        for i in range(1,len(lats)):
            cumd.append(cumd[-1]+haversine_m(lats[i-1][0],lats[i-1][1],lats[i][0],lats[i][1]))
        df["distance_m"] = cumd[:len(df)]

    elif fname.endswith(".tcx"):
        file.seek(0); root = ET.parse(file).getroot()
        ns = {"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
        rows = []
        for tp in root.findall(".//tcx:Trackpoint",ns):
            tim = tp.find("tcx:Time",ns)
            hr_el = tp.find(".//tcx:Value",ns)
            spd_el= tp.find("tcx:Extensions/tcx:TPX/tcx:Speed",ns) or tp.find(".//tcx:Speed",ns)
            dist_el= tp.find("tcx:DistanceMeters",ns)
            alt_el = tp.find("tcx:AltitudeMeters",ns)
            try:   ts = datetime.fromisoformat(tim.text.replace("Z","+00:00"))
            except: continue
            rows.append({"timestamp":ts,
                         "heart_rate": int(hr_el.text) if hr_el is not None else None,
                         "speed_ms": float(spd_el.text) if spd_el is not None else None,
                         "distance_m": float(dist_el.text) if dist_el is not None else None,
                         "altitude_m": float(alt_el.text) if alt_el is not None else None})
        if not rows: return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        t0 = df["timestamp"].iloc[0]
        df["elapsed_s"] = (df["timestamp"]-t0).dt.total_seconds()

    elif fname.endswith(".csv"):
        file.seek(0); df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        renames = {}
        for c in df.columns:
            if "heart" in c or c in ("hr","fc"): renames[c]="heart_rate"
            if "speed" in c and "ms" not in c:    renames[c]="speed_ms"
            if "dist" in c:                        renames[c]="distance_m"
            if "alt" in c or "elev" in c:          renames[c]="altitude_m"
            if "time" in c or "elapsed" in c:      renames[c]="elapsed_s"
        df.rename(columns=renames, inplace=True)
        if "elapsed_s" not in df.columns: df["elapsed_s"] = range(len(df))

    if df is None: return None

    for col in ["heart_rate","speed_ms","distance_m","altitude_m","elapsed_s"]:
        if col not in df.columns: df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["elapsed_s"]).reset_index(drop=True)
    return df if len(df) >= 5 else None


# ══════════════════════════════════════════════════════════════
# ANALYSE FC + CINÉTIQUE
# ══════════════════════════════════════════════════════════════

def smooth_hr(series: pd.Series, window: int = None) -> pd.Series:
    n = len(series)
    if n < 5: return series
    if window is None: window = max(3, n//20)
    if window % 2 == 0: window += 1
    return series.rolling(window, center=True, min_periods=1).mean()


def analyze_heart_rate(df: pd.DataFrame) -> dict:
    col = "heart_rate"
    if col not in df.columns: return {}
    hr = df[col].dropna()
    hr = hr[(hr>=40)&(hr<=220)]
    if len(hr) < 10: return {"available":False}
    arr = hr.values
    n = len(arr)
    fc_max = float(np.percentile(arr, 95))
    fc_avg = float(np.mean(arr))
    fc_min = float(np.percentile(arr, 5))
    q1, q3 = int(n*0.25), int(n*0.75)
    drift_abs = float(np.mean(arr[q3:])) - float(np.mean(arr[:q1]))
    drift_pct = drift_abs/max(1.0,float(np.mean(arr[:q1])))*100.0
    x = np.arange(n, dtype=float)
    slope, intercept, r, p, _ = sp_stats.linregress(x, arr)
    trend_bpm_per_min = slope * 60.0
    reliability = ("haute" if abs(drift_abs)<5 else
                   "moyenne" if abs(drift_abs)<12 else "basse (dérive élevée)")
    return {"available":True,"fc_max":round(fc_max),"fc_avg":round(fc_avg,1),
            "fc_min":round(fc_min),"drift_abs":round(drift_abs,1),
            "drift_pct":round(drift_pct,2),"trend_bpm_per_min":round(trend_bpm_per_min,2),
            "r_value":round(r,3),"reliability":reliability,
            "seuil_estime":round(fc_max*0.88)}


def analyze_speed_kinetics(df: pd.DataFrame) -> dict:
    col = "speed_ms"
    if col not in df.columns: return {}
    spd = df[col].dropna()
    spd = spd[spd>0]
    if len(spd) < 10: return {"available":False}
    arr = spd.values
    x = np.arange(len(arr), dtype=float)
    slope, intercept, r, p, _ = sp_stats.linregress(x, arr)
    return {"available":True,"speed_avg_ms":round(float(np.mean(arr)),3),
            "speed_max_ms":round(float(np.percentile(arr,95)),3),
            "slope":round(slope,5),"r_value":round(r,3)}


# ══════════════════════════════════════════════════════════════
# VITESSE CRITIQUE (VC)
# ══════════════════════════════════════════════════════════════

def compute_vc(distances_m: list, durations_s: list):
    T = np.array(durations_s, dtype=float)
    D = np.array(distances_m, dtype=float)
    if len(T) < 2: return None, None, None
    slope, intercept, r, p, se = sp_stats.linregress(T, D)
    return float(slope), float(intercept), float(r**2)


def build_holding_table(vc_ms: float, d_prime: float,
                        refs_fit: list, K_riegel: float) -> pd.DataFrame:
    """
    Table des temps de maintien de 80 % à 120 % de VC, par pas de 4 %.
    - 80 % VC ≤ v < 100 % VC : Riegel log-log calé sur les références
                                T = [a × (v/1000)^K]^(1/(1−K))
    - 100 % VC ≤ v ≤ 120 % VC : Modèle D'  →  T = D' / (v − VC)
    Colonnes : % VC, Vitesse, Allure, Temps de maintien, Durée (min), Modèle.
    """
    if vc_ms is None or vc_ms <= 0:
        return pd.DataFrame()

    # Recalculer a et K depuis les refs
    X, Y = [], []
    for r in refs_fit:
        d_m = float(r.get("distance", 0))
        t_s = float(r.get("temps", 0))
        if d_m > 0 and t_s > 0:
            X.append(math.log(d_m / 1000.0))
            Y.append(math.log(t_s))
    if len(X) >= 2:
        K_fit, loga, _, _, _ = sp_stats.linregress(X, Y)
        K_fit = float(max(0.85, min(1.25, K_fit)))
        a_fit = math.exp(float(loga))
    elif len(X) == 1:
        K_fit = float(K_riegel)
        a_fit = math.exp(Y[0]) / math.exp(X[0])
    else:
        K_fit = float(K_riegel)
        a_fit = 240.0

    # Paliers fixes de 80 % à 120 % par pas de 4 %
    pct_steps = [pct / 100.0 for pct in range(80, 121, 4)]

    rows = []
    for pct in pct_steps:
        v = vc_ms * pct
        if v <= 0:
            continue
        pct_label = f"{round(pct * 100):.0f} %"
        pace = pace_str(1000.0 / v)

        if pct < 1.0:
            # Riegel
            if abs(1.0 - K_fit) < 1e-6:
                continue
            try:
                t = (a_fit * (v / 1000.0) ** K_fit) ** (1.0 / (1.0 - K_fit))
            except Exception:
                continue
            modele = "Riegel"
        else:
            # Modèle D'
            delta_v = v - vc_ms
            if d_prime is None or delta_v < 0.01:
                continue
            t = d_prime / delta_v
            modele = "Modèle D'"

        t = max(0.0, min(t, 360000.0))   # plafond 100 h
        if t > 0:
            rows.append({
                "% VC":               pct_label,
                "Vitesse (m/s)":      round(v, 2),
                "Allure (/km)":       pace,
                "Temps de maintien":  seconds_to_hms(t),
                "Durée (min)":        round(t / 60.0, 1),
                "Modèle":             modele,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# MODÈLE RIEGEL + LOO-CV + RECALIBRATION (v3)
# ══════════════════════════════════════════════════════════════

def fit_loglog(refs):
    X, Y = [], []
    for r in refs:
        d_m = safe_float(r.get("distance",0))
        t   = r.get("temps")
        secs = float(t) if isinstance(t,(int,float,np.number)) else hms_to_seconds(str(t))
        if d_m<=0 or secs<=0: continue
        X.append(math.log(d_m/1000.0)); Y.append(math.log(secs))
    if len(X)>=2:
        K, loga = np.polyfit(X,Y,1)
        K = float(max(0.85,min(1.25,K)))
        a = math.exp(float(loga))
        return (a if 0<a<1e7 else 240.0), K
    elif len(X)==1:
        return math.exp(Y[0])/(math.exp(X[0])), 1.0
    return 240.0, 1.0


def predict_flat(dist_m, a, K):
    return float(a)*((dist_m/1000.0)**float(K))


def crossval_loo(refs_prepared):
    n = len(refs_prepared)
    if n<3: return None
    rows = []
    for i in range(n):
        train = [r for j,r in enumerate(refs_prepared) if j!=i]
        test  = refs_prepared[i]
        a_cv,K_cv = fit_loglog(train)
        pred_s   = predict_flat(test["distance"],a_cv,K_cv)
        actual_s = float(test["temps"])
        rows.append({"Réf":i+1,"Distance (km)":round(test["distance"]/1000.0,2),
                     "Temps réel":seconds_to_hms(actual_s),"Temps prédit":seconds_to_hms(pred_s),
                     "Erreur (s)":round(pred_s-actual_s,0),
                     "Erreur (%)":round((pred_s-actual_s)/actual_s*100.0,2) if actual_s>0 else 0})
    df_cv = pd.DataFrame(rows)
    mae  = float(np.mean(np.abs(df_cv["Erreur (s)"].values)))
    mape = float(np.mean(np.abs(df_cv["Erreur (%)"].values)))
    return df_cv, mae, mape


def elev_factor_global(D_up_m,D_down_m,dist_m,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down):
    dist = max(1e-6,float(dist_m))
    g_up = float(D_up_m)/dist; g_dn = float(D_down_m)/dist
    g0u  = max(1e-6,float(g0_up)/100.0); g0d = max(1e-6,float(g0_down)/100.0)
    up_term    = float(k_up)*math.tanh(g_up/g0u)*g0u
    down_bonus = min(float(k_down)*math.tanh(g_dn/g0d)*g0d, abs(float(down_cap)))
    mult = 1.0+up_term-down_bonus
    mult = min(mult,1.0+float(max_up)); mult = max(mult,1.0+float(max_down))
    return max(0.01,float(mult))


def recalibrate_ref_to_ideal(ref,opt_temp,use_wbgt,cold_quad,hot_quad,temp_max_penalty,
                              k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                              elev_ref_power,temp_ref_power):
    secs  = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up  = safe_float(ref.get("D_up",0.0))
    D_down= safe_float(ref.get("D_down",0.0))
    dist  = max(1.0,safe_float(ref.get("distance",1000.0)))
    f_elev = elev_factor_global(D_up,D_down,dist,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
    secs_no_elev = secs/(f_elev**float(elev_ref_power))
    temp_real = ref.get("avg_temp"); hum_real = safe_float(ref.get("avg_humidity",50.0),50.0)
    if temp_real is not None:
        temp_eff = effective_temp(temp_real,hum_real,use_wbgt)
        f_temp   = temp_multiplier(temp_eff,opt_temp,cold_quad,hot_quad,temp_max_penalty)
        secs_no_temp = secs_no_elev/(max(0.01,f_temp)**float(temp_ref_power))
    else:
        secs_no_temp = secs_no_elev
    return max(0.0,float(secs_no_temp))


def prepare_refs(refs_input,use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,
                 temp_max_penalty,k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                 elev_ref_power,temp_ref_power):
    out = []
    for r in refs_input:
        d    = safe_float(r.get("distance",0.0))
        raw_t= r.get("duration_hms_file") or r.get("temps","0:00:00")
        if use_recalibrated:
            secs = recalibrate_ref_to_ideal(
                ref={**r,"temps":raw_t}, opt_temp=opt_temp,use_wbgt=use_wbgt,
                cold_quad=cold_quad,hot_quad=hot_quad,temp_max_penalty=temp_max_penalty,
                k_up=k_up,k_down=k_down,down_cap=down_cap,
                g0_up=g0_up,g0_down=g0_down,max_up=max_up,max_down=max_down,
                elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power)
        else:
            secs = float(hms_to_seconds(raw_t))
        out.append({"distance":float(d),"temps":float(secs)})
    return out


# ══════════════════════════════════════════════════════════════
# PACING ULTRA + PRÉDICTION PRINCIPALE (v3)
# ══════════════════════════════════════════════════════════════

def apply_ultra_pacing(t_raw,d_end_m,seg_len_m,total_corr_m,amp_pct):
    if len(t_raw)==0 or amp_pct<=0: return t_raw
    total_corr_m = max(1e-9,float(total_corr_m))
    d_mid = np.asarray(d_end_m)-0.5*np.asarray(seg_len_m)
    prog  = np.clip(d_mid/total_corr_m,0.0,1.0)
    A = amp_pct/100.0; mult = 1.0+A*(2.0*prog-1.0)
    t_adj = np.asarray(t_raw)*mult
    s_raw = np.sum(t_raw); s_adj = np.sum(t_adj)
    if s_raw>0 and s_adj>0: t_adj *= s_raw/s_adj
    return t_adj


def run_prediction(
    distance_cible_km,refs_input,points,date_course,heure_course,
    use_recalibrated,opt_temp,use_wbgt,cold_quad,hot_quad,temp_max_penalty,temp_power,
    elev_ref_power,temp_ref_power,
    apply_grade,use_minetti,minetti_weight,
    k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
    elev_smooth_window,grade_power,
    apply_altitude,altitude_ref_m,
    apply_wind,wind_mode,wind_smooth_km,
    drag_coeff,tail_credit,wind_cap_head,wind_cap_tail,wind_power,
    wind_gate_g1,wind_gate_g2,wind_gate_min,
    base_cap,extra_per_pct,max_cap,
    apply_fatigue,fatigue_rate,fatigue_mode,
    apply_ultra,ultra_amp,
    objective_hms,show_smooth_pace,smooth_window_km,
    dem_elevations,tz_name=TZ_NAME_DEFAULT):

    if not points or len(points)<2: raise ValueError("GPX invalide ou trop court.")
    if dem_elevations is not None and len(dem_elevations)==len(points):
        elev_arr = np.array([e if e is not None else 0.0 for e in dem_elevations],dtype=float)
    else:
        elev_arr = np.array([getattr(p,"elevation",0.0) or 0.0 for p in points],dtype=float)

    total_m=0.0; cum=[0.0]
    for i in range(1,len(points)):
        total_m+=haversine_m(points[i-1].latitude,points[i-1].longitude,
                              points[i].latitude,   points[i].longitude)
        cum.append(total_m)
    dist_gpx_km = total_m/1000.0
    if not distance_cible_km: distance_cible_km=dist_gpx_km
    fac = distance_cible_km/max(dist_gpx_km,1e-9)
    total_corr  = total_m*fac
    dists_corr  = np.array(cum,dtype=float)*fac

    if elev_arr.size!=dists_corr.size:
        xs = np.linspace(0,total_m,elev_arr.size)
        elev_arr = np.interp(np.linspace(0,total_m,dists_corr.size),xs,elev_arr)

    w = int(elev_smooth_window)
    if w%2==0: w+=1
    elev_s = np.convolve(elev_arr,np.ones(w)/w,mode="same") if w>=3 and elev_arr.size>=w else elev_arr

    diffs_el  = np.diff(elev_s)
    d_plus_total = float(np.sum(np.clip(diffs_el,0,None)))
    avg_alt = float(np.mean(elev_s))

    refs_fit = prepare_refs(refs_input,use_recalibrated,opt_temp,use_wbgt,
                             cold_quad,hot_quad,temp_max_penalty,
                             k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                             elev_ref_power,temp_ref_power)
    a,K = fit_loglog(refs_fit)
    if objective_hms:
        obj_s = hms_to_seconds(objective_hms); d_km = distance_cible_km
        a = obj_s/(d_km**K) if d_km>0 else a
    base_total_s  = predict_flat(int(distance_cible_km*1000),a,K)
    base_s_per_km = base_total_s/max(distance_cible_km,1e-9)
    alt_mult = altitude_vo2_multiplier(avg_alt,altitude_ref_m) if apply_altitude else 1.0

    km_marks = [i*1000 for i in range(1,int(total_corr//1000)+1)]
    last = total_corr - int(total_corr//1000)*1000
    if last>1e-6: km_marks.append(total_corr)

    lats_arr = np.array([p.latitude  for p in points],dtype=float)
    lons_arr = np.array([p.longitude for p in points],dtype=float)
    dt_dep = datetime.combine(date_course,heure_course)

    pre=[]; cum_t=cum_dp=cum_dist=0.0
    for i,d in enumerate(km_marks):
        seg_len = 1000.0
        if i==len(km_marks)-1 and last>1e-6:
            seg_len = d-(km_marks[-2] if len(km_marks)>=2 else 0)
        e_cur = float(np.interp(d,dists_corr,elev_s))
        e_prv = float(np.interp(max(d-seg_len,0),dists_corr,elev_s)) if i>0 else e_cur
        grade = (e_cur-e_prv)/max(1e-6,seg_len)*100.0
        seg_dp = max(0.0,e_cur-e_prv)
        cum_dp+=seg_dp; cum_dist+=seg_len
        t_flat = base_s_per_km*(seg_len/1000.0)
        if apply_grade:
            gm = combined_grade_multiplier(grade,use_minetti,minetti_weight,
                                           k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down)
            t1 = t_flat*(gm**grade_power)
        else:
            gm=1.0; t1=t_flat
        t2 = t1*alt_mult
        fm = fatigue_multiplier(cum_dp,cum_dist,d_plus_total,total_corr,fatigue_rate,fatigue_mode) \
             if apply_fatigue and fatigue_rate>0 else 1.0
        t3 = t2*fm
        passage_dt = dt_dep+timedelta(seconds=cum_t+t3/2.0)
        lat_s = float(np.interp(d,dists_corr,lats_arr))
        lon_s = float(np.interp(d,dists_corr,lons_arr))
        lat0  = float(np.interp(max(d-seg_len,0),dists_corr,lats_arr))
        lon0  = float(np.interp(max(d-seg_len,0),dists_corr,lons_arr))
        cap   = bearing_deg(lat0,lon0,lat_s,lon_s)
        meteo = get_weather_minutely(lat_s,lon_s,passage_dt,tz_name)
        temp_raw=meteo["temp"] if meteo else None
        wind_raw=meteo["wind"] if meteo else None
        hum_raw =meteo["humidity"] if meteo else None
        wdir_raw=meteo.get("wind_dir") if meteo else None
        temp_eff_val=None
        if temp_raw is not None and hum_raw is not None:
            temp_eff_val=effective_temp(temp_raw,hum_raw,use_wbgt)
        if temp_eff_val is not None:
            tm=temp_multiplier(temp_eff_val,opt_temp,cold_quad,hot_quad,temp_max_penalty)
            t4=t3*(tm**temp_power)
        else:
            tm=1.0; t4=t3
        pace_local = (t4/seg_len)*1000.0 if seg_len>0 else t4
        head,tail  = wind_components(wind_raw,wdir_raw,cap)
        pre.append({"idx":i,"d":d,"seg_len":seg_len,"grade":grade,"grade_mult":gm,
                    "seg_dp":seg_dp,"cum_dp":cum_dp,"fat_mult":fm,"alt_mult":alt_mult,
                    "temp_raw":temp_raw,"temp_eff":temp_eff_val,"hum":hum_raw,
                    "wind":wind_raw,"wdir":wdir_raw,"cap":cap,"head":head,"tail":tail,
                    "temp_mult":tm,"t_flat":t_flat,"t_no_wind":t4,"pace_no_wind":pace_local})
        cum_t+=t4

    df_pre = pd.DataFrame(pre)
    if apply_wind and not df_pre.empty:
        if wind_mode=="Global":
            hg=float(np.median(df_pre["head"])); tg=float(np.median(df_pre["tail"]))
            pg=float(np.median(df_pre["pace_no_wind"]))
            wm_raw=wind_multiplier(hg,tg,pg,drag_coeff,tail_credit,wind_cap_head,wind_cap_tail)
            df_pre["wind_mult_raw"]=wm_raw
        else:
            w_s=int(max(1,wind_smooth_km)); w_s+=(1 if w_s%2==0 else 0)
            hs=pd.Series(df_pre["head"]).rolling(w_s,center=True,min_periods=1).median()
            ts_=pd.Series(df_pre["tail"]).rolling(w_s,center=True,min_periods=1).median()
            wms=[wind_multiplier(h,t,p,drag_coeff,tail_credit,wind_cap_head,wind_cap_tail)
                 for h,t,p in zip(hs,ts_,df_pre["pace_no_wind"])]
            df_pre["wind_mult_raw"]=wms
            df_pre["head_s"]=hs.values; df_pre["tail_s"]=ts_.values
    else:
        df_pre["wind_mult_raw"]=1.0

    t_raw=[]; wm_adj_list=[]
    for _,row in df_pre.iterrows():
        wm=float(row["wind_mult_raw"]); g=float(row["grade"])
        gate=wind_gate(g,wind_gate_g1,wind_gate_g2,wind_gate_min)
        wm_gated=1.0+gate*(wm-1.0)
        t_w=float(row["t_no_wind"])*(wm_gated**wind_power)
        mt=t_w/max(1e-9,float(row["t_flat"]))
        mt=cap_combined(mt,g,base_cap,extra_per_pct,max_cap)
        t_raw.append(float(row["t_flat"])*mt); wm_adj_list.append(wm_gated)
    df_pre["wind_mult_adj"]=wm_adj_list; t_raw=np.array(t_raw,dtype=float)

    if apply_ultra and ultra_amp>0:
        t_raw=apply_ultra_pacing(t_raw,df_pre["d"].values,df_pre["seg_len"].values,total_corr,ultra_amp)
    if objective_hms:
        s_obj=hms_to_seconds(objective_hms); s_sum=float(np.sum(t_raw))
        t_raw=t_raw*(s_obj/s_sum) if s_sum>0 else t_raw

    rows=[]; cum_t2=0.0
    for i in range(len(df_pre)):
        seg=df_pre.iloc[i]; ts=float(t_raw[i]); cum_t2+=ts
        pace_val=(ts/float(seg["seg_len"]))*1000.0 if seg["seg_len"]>0 else ts
        rows.append({"Km":(int(seg["idx"])+1) if seg["seg_len"]>=999 else f"{int(seg['idx'])+1} ({seg['seg_len']:.0f}m)",
                     "Pente (%)":round(float(seg["grade"]),2),"Mult Pente":round(float(seg["grade_mult"]),4),
                     "D+ seg (m)":round(float(seg["seg_dp"]),1),"D+ cum (m)":round(float(seg["cum_dp"]),1),
                     "Mult Fatigue":round(float(seg["fat_mult"]),4),"Mult Altitude":round(float(seg["alt_mult"]),4),
                     "Temp GPS (°C)":round(float(seg["temp_raw"]),1) if seg["temp_raw"] is not None else None,
                     "Temp eff/WBGT (°C)":round(float(seg["temp_eff"]),1) if seg["temp_eff"] is not None else None,
                     "Mult Temp":round(float(seg["temp_mult"]),4),
                     "Vent (m/s)":round(float(seg["wind"]),1) if seg["wind"] is not None else None,
                     "Headwind (m/s)":round(float(seg.get("head_s",seg["head"])),2),
                     "Tailwind (m/s)":round(float(seg.get("tail_s",seg["tail"])),2),
                     "Mult Vent":round(float(seg["wind_mult_adj"]),4),
                     "Humidité (%)":round(float(seg["hum"]),1) if seg["hum"] is not None else None,
                     "Temps seg (s)":round(ts,1),"Allure (min/km)":pace_str(pace_val),
                     "Temps cumulé":seconds_to_hms(cum_t2)})
    df_out=pd.DataFrame(rows)
    if show_smooth_pace and not df_out.empty:
        w_p=int(max(1,smooth_window_km)); w_p+=(1 if w_p%2==0 else 0)
        s_p=pd.Series(df_out["Temps seg (s)"].astype(float)).rolling(w_p,center=True,min_periods=1).median()
        df_out["Allure lissée (min/km)"]=s_p.apply(pace_str)
    total_s=float(np.sum(t_raw))
    return {"df":df_out,"total_s":total_s,"total_human":seconds_to_hms(total_s),
            "ci_low":seconds_to_hms(total_s*0.95),"ci_high":seconds_to_hms(total_s*1.05),
            "dist_gpx_km":dist_gpx_km,"K":K,"avg_alt":avg_alt,"d_plus_total":d_plus_total,
            "refs_fit":refs_fit,"pre_df":df_pre}


# ══════════════════════════════════════════════════════════════
# ANALYSE FRACTIONNÉ
# ══════════════════════════════════════════════════════════════

def extract_interval_df(df: pd.DataFrame, start_hms: str, end_hms: str) -> pd.DataFrame:
    """Extrait un sous-DataFrame entre start_hms et end_hms (relatif au début de l'activité)."""
    t_start = float(hms_to_seconds(start_hms))
    t_end   = float(hms_to_seconds(end_hms))
    if t_end <= t_start:
        return pd.DataFrame()
    sub = df[(df["elapsed_s"] >= t_start) & (df["elapsed_s"] <= t_end)].copy()
    sub["elapsed_s"] = sub["elapsed_s"] - t_start
    return sub.reset_index(drop=True)


def analyze_interval(df_int: pd.DataFrame, name: str) -> dict:
    """Calcule les stats clés d'un intervalle."""
    if df_int.empty:
        return {"name": name, "valid": False}
    dur_s = float(df_int["elapsed_s"].max())
    dist_m = None
    if "distance_m" in df_int.columns and df_int["distance_m"].notna().any():
        d0 = df_int["distance_m"].dropna().iloc[0]
        d1 = df_int["distance_m"].dropna().iloc[-1]
        dist_m = float(d1 - d0) if d1 > d0 else float(df_int["distance_m"].dropna().max() - df_int["distance_m"].dropna().min())
    hr_stats  = analyze_heart_rate(df_int)
    spd_stats = analyze_speed_kinetics(df_int)
    avg_speed = None
    if dist_m and dur_s > 0:
        avg_speed = dist_m / dur_s
    return {
        "name": name,
        "valid": True,
        "dur_s": dur_s,
        "dist_m": dist_m,
        "avg_speed": avg_speed,
        "hr": hr_stats,
        "spd": spd_stats,
        "df": df_int,
    }


# ══════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-label">⚙️ Paramètres — onglets Tests & Entraînement</div>',
                unsafe_allow_html=True)
    sb_opt_temp = st.slider("Température optimale (°C)", 5.0, 20.0, 12.0, 0.5,
                             key="sb_opt_temp")
    sb_k_up   = st.number_input("Coefficient montée (k_up)",   value=12.0, step=0.5, key="sb_k_up")
    sb_k_down = st.number_input("Coefficient descente (k_down)",value=5.0,  step=0.5, key="sb_k_down")
    sb_k_temp_hot  = st.number_input("Sensibilité chaleur",  value=0.0016, step=0.0002, format="%.4f", key="sb_kth")
    sb_k_temp_cold = st.number_input("Sensibilité froid",    value=0.0012, step=0.0002, format="%.4f", key="sb_ktc")
    st.caption("Ces paramètres n'affectent que les onglets 🧪 et ⚙️")

main_tabs = st.tabs(["🏃 Prédiction de course", "🧪 Tests d'endurance + VC", "⚙️ Analyse entraînement"])


# ══════════════════════════════════════════════════════════════
# ONGLET 0 — PRÉDICTION DE COURSE (v3)
# ══════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.title("🏃 Prédiction de course — Coach & Athlète")
    st.caption("v4 — WBGT · Minetti · DEM · Recalibration · Saisie hh:mm:ss")

    col_mode1, col_mode2 = st.columns([2,3])
    with col_mode1:
        mode = st.radio("Mode d'interface",
                        ["🟢 Simple (recommandé)", "🔵 Expert (tous les curseurs)"],
                        horizontal=True, key="pred_mode")
    EXPERT = "Expert" in mode

    st.markdown("---")
    st.header("1️⃣  Parcours GPX")
    gpx_file = st.file_uploader("📂 Importer le GPX de la course cible", type=["gpx"], key="gpx_main")
    points = None; dem_elevations = None

    if gpx_file:
        _gpx, points = parse_gpx_points(gpx_file)
        if points:
            tot_tmp = sum(haversine_m(points[i-1].latitude,points[i-1].longitude,
                                       points[i].latitude,  points[i].longitude)
                          for i in range(1,len(points)))
            dup_tmp,ddn_tmp = compute_dplus_dminus([getattr(p,"elevation",0.0) or 0.0 for p in points])
            avg_alt_tmp = np.mean([getattr(p,"elevation",0.0) or 0.0 for p in points])
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Distance GPX",f"{tot_tmp/1000:.2f} km")
            c2.metric("D+ GPS",f"{dup_tmp:.0f} m")
            c3.metric("D- GPS",f"{ddn_tmp:.0f} m")
            c4.metric("Alt. moy.",f"{avg_alt_tmp:.0f} m")

    with st.expander("🏔️ Correction altimétrique DEM (optionnel)"):
        st.info("Le GPS vertical a une précision de ±5-15 m. Le DEM donne l'altitude réelle à ±1 m.")
        use_dem = st.checkbox("Activer la correction DEM", value=False, key="use_dem")
        dem_dataset = "srtm30m"
        if use_dem:
            dem_dataset = st.selectbox("Dataset",
                ["srtm30m (global, 30m)","eudem25m (Europe, 25m — plus précis)","mapzen (global fusion)"],
                key="dem_ds").split()[0]
            if gpx_file and points and st.button("🔄 Télécharger et corriger l'altitude"):
                with st.spinner("Correction DEM en cours..."):
                    dem_elevations = list(correct_elevations_dem(points,max_points=100,dataset=dem_dataset))
                    st.session_state["dem_elevations"] = dem_elevations
                    dup_dem,ddn_dem = compute_dplus_dminus([e or 0.0 for e in dem_elevations])
                    st.success(f"DEM OK — D+ DEM: **{dup_dem:.0f} m** | D- DEM: **{ddn_dem:.0f} m**")
        if "dem_elevations" in st.session_state:
            dem_elevations = st.session_state["dem_elevations"]

    st.markdown("---")
    st.header("2️⃣  Courses de référence")
    st.info("Calibrent le modèle sur l'athlète. Minimum conseillé : **3 références** variées (5 km, semi, marathon…).")

    if "n_refs" not in st.session_state: st.session_state.n_refs = 3
    cc1,cc2 = st.columns(2)
    with cc1:
        if st.button("➕ Ajouter une référence") and st.session_state.n_refs<6:
            st.session_state.n_refs+=1
    with cc2:
        if st.button("➖ Retirer") and st.session_state.n_refs>1:
            st.session_state.n_refs-=1

    refs_raw = []
    for i in range(1, st.session_state.n_refs+1):
        with st.expander(f"📌 Référence {i}", expanded=(i<=2)):
            use_file = st.checkbox(f"Importer depuis fichier FIT/TCX", key=f"use_file_{i}")
            c1,c2,c3,c4 = st.columns(4)
            dist  = c1.number_input("Distance (m)", value=float(st.session_state.get(f"dist_{i}",5000*i)), key=f"dist_{i}")
            # ── Temps au format hh:mm:ss ──
            temps = hms_input("Temps (hh:mm:ss)", default="0:40:00", key=f"temps_{i}")
            dup   = c3.number_input("D+ (m)", value=float(st.session_state.get(f"dup_{i}",0.0)), key=f"dup_{i}")
            ddn   = c4.number_input("D- (m)", value=float(st.session_state.get(f"ddn_{i}",0.0)), key=f"ddn_{i}")
            file_in = st.file_uploader(f"Fichier FIT/TCX", type=["fit","tcx"], key=f"fileref_{i}") if use_file else None

            dur_hms_file=avg_temp_ref=avg_wind_ref=avg_hum_ref=hr_ref=None
            fname = file_in.name.lower() if file_in else ""
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
                # ── Segment hh:mm:ss ──
                cs,ce=st.columns(2)
                sh = hms_input("Début segment", "0:00:00", key=f"start_{i}", compact=True)
                eh = hms_input("Fin segment",   "23:59:59", key=f"end_{i}",   compact=True)
                start_td,end_td=hms_to_timedelta(sh),hms_to_timedelta(eh)
                if start_td.total_seconds()>0 or end_td.total_seconds()<86399:
                    pts_src=None
                    if fit_data and "points" in fit_data: pts_src=fit_data["points"]
                    elif tcx_data and "points" in tcx_data: pts_src=tcx_data["points"]
                    if pts_src:
                        seg=extract_segment(pts_src,start_td,end_td)
                        seg_dist=0.0; seg_elevs=[]; seg_times=[]
                        for j in range(1,len(seg)):
                            p1,p2=seg[j-1],seg[j]
                            la1,lo1=(p1["lat"],p1["lon"]) if isinstance(p1,dict) else (p1.latitude,p1.longitude)
                            la2,lo2=(p2["lat"],p2["lon"]) if isinstance(p2,dict) else (p2.latitude,p2.longitude)
                            e2=p2.get("elev",0) if isinstance(p2,dict) else p2.elevation
                            t2=p2.get("time") if isinstance(p2,dict) else p2.time
                            seg_dist+=haversine_m(la1,lo1,la2,lo2); seg_elevs.append(e2)
                            if t2: seg_times.append(t2)
                        dup,ddn=compute_dplus_dminus(seg_elevs)
                        if len(seg_times)>=2: dur_hms_file=seconds_to_hms((seg_times[-1]-seg_times[0]).total_seconds())
                        dist=round(seg_dist)
            else:
                if EXPERT:
                    cs2,ce2=st.columns(2)
                    avg_temp_ref=cs2.number_input(f"Temp moy. course (°C)",value=15.0,key=f"avgT_{i}")
                    avg_hum_ref =ce2.number_input(f"Humidité moy. (%)",   value=60.0,key=f"avgH_{i}")
                else:
                    avg_temp_ref=avg_hum_ref=None

            temps_eff = dur_hms_file if dur_hms_file else temps
            secs_brut = hms_to_seconds(temps_eff)
            dist_km   = safe_float(dist,1.0)/1000.0
            if secs_brut>0 and dist_km>0:
                st.caption(f"📍 {dist:.0f} m · {temps_eff} · **{pace_str(secs_brut/dist_km)}/km**"
                           +(f" · D+ {dup:.0f}m" if dup>0 else "")
                           +(f" · Temp GPS: {avg_temp_ref:.0f}°C" if avg_temp_ref else "")
                           +(f" · FC fiabilité: {hr_ref.get('reliability')}" if hr_ref else ""))
            if hr_ref and hr_ref.get("hr_max"):
                st.caption(f"💓 FC max {hr_ref['hr_max']} bpm · dérive {hr_ref['hr_drift']} bpm · seuil ~{hr_ref['hr_threshold_est']} bpm")
            refs_raw.append({"distance":float(dist),"temps":str(temps_eff),
                              "D_up":float(dup),"D_down":float(ddn),
                              "duration_hms_file":dur_hms_file,
                              "avg_temp":avg_temp_ref,"avg_humidity":avg_hum_ref,"avg_wind":avg_wind_ref,
                              "hr_analysis":hr_ref})

    st.markdown("---")
    st.header("3️⃣  Recalibration des références vers les conditions idéales")
    st.markdown("""
<div class="highlight-box">
<strong>Pourquoi recalibrer ?</strong><br>
Une course réalisée par 30°C et 80% d'humidité vaut <em>physiologiquement mieux</em>
qu'un temps identique par 12°C et temps sec. Sans correction, le modèle sous-estime la performance.
</div>""", unsafe_allow_html=True)

    use_recalibrated = st.checkbox(
        "✅ Recalibrer les références vers les conditions idéales (fortement recommandé)", value=True)

    opt_temp=12.0; use_wbgt=True; elev_ref_power=0.60; temp_ref_power=0.85
    with st.expander("⚙️ Paramètres de recalibration"):
        opt_temp = st.slider("Température optimale de course (°C)", 5.0, 20.0, 12.0, 0.5)
        use_wbgt = st.checkbox("Utiliser le WBGT (chaleur+humidité) — recommandé", value=True)
        col_ep1,col_ep2 = st.columns(2)
        with col_ep1:
            elev_ref_power = st.slider("Force correction pente des références", 0.0, 1.0, 0.60, 0.05)
        with col_ep2:
            temp_ref_power = st.slider("Force correction température des références", 0.0, 1.0, 0.85, 0.05)

    st.subheader("📋 Résumé de la recalibration")
    _k_up_prev=st.session_state.get("k_up_val",12.0); _k_down_prev=st.session_state.get("k_down_val",5.0)
    _g0u_prev=st.session_state.get("g0_up_val",3.0); _g0d_prev=st.session_state.get("g0_down_val",2.5)
    calib_rows=[]; cold_quad=0.0012; hot_quad=0.0016; temp_max_penalty=0.10
    for r in refs_raw:
        t_brut=hms_to_seconds(r.get("duration_hms_file") or r.get("temps",""))
        dist_km=safe_float(r.get("distance",1.0))/1000.0
        avg_t=r.get("avg_temp"); avg_h=safe_float(r.get("avg_humidity",50.0),50.0)
        wbgt_val=wbgt_simplified(avg_t,avg_h) if avg_t is not None and use_wbgt else None
        t_ideal=(recalibrate_ref_to_ideal(
            ref={**r,"temps":r.get("duration_hms_file") or r.get("temps","0:00:00")},
            opt_temp=opt_temp,use_wbgt=use_wbgt,cold_quad=cold_quad,hot_quad=hot_quad,
            temp_max_penalty=temp_max_penalty,k_up=_k_up_prev,k_down=_k_down_prev,down_cap=-0.08,
            g0_up=_g0u_prev,g0_down=_g0d_prev,max_up=0.30,max_down=-0.06,
            elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power)
         if use_recalibrated else float(t_brut))
        gain_s=t_brut-t_ideal
        calib_rows.append({"Distance":f"{safe_float(r['distance'])/1000:.1f} km",
                            "Temps brut":seconds_to_hms(t_brut),
                            "Allure brute":pace_str(t_brut/dist_km) if dist_km>0 else "-",
                            "D+":f"{r['D_up']:.0f} m",
                            "Temp GPS":f"{avg_t:.0f}°C" if avg_t is not None else "?",
                            "WBGT":f"{wbgt_val:.1f}°C" if wbgt_val is not None else "-",
                            "Temps recalibré":seconds_to_hms(t_ideal) if use_recalibrated else "—",
                            "Allure recalibrée":pace_str(t_ideal/dist_km) if (use_recalibrated and dist_km>0) else "—",
                            "Gain correction":f"-{seconds_to_hms(gain_s)}" if gain_s>0 else (f"+{seconds_to_hms(-gain_s)}" if gain_s<0 else "0")})
    st.dataframe(pd.DataFrame(calib_rows), use_container_width=True)

    st.markdown("---")
    st.header("4️⃣  Paramètres du modèle")

    with st.expander("🌡️ Température & Humidité", expanded=False):
        temp_power=1.0
        if EXPERT:
            c1,c2=st.columns(2)
            cold_quad=c1.number_input("Sensibilité froid",value=0.0012,step=0.0002,format="%.4f")
            hot_quad =c2.number_input("Sensibilité chaleur",value=0.0016,step=0.0002,format="%.4f")
            temp_max_penalty=st.slider("Pénalité max température (%)",0.00,0.20,0.10,0.01)
            temp_power=st.slider("Damping température (puissance)",0.2,1.2,1.0,0.05)

    with st.expander("🏔️ Altitude physiologique (hypoxie)"):
        apply_altitude=st.checkbox("Appliquer la pénalité d'altitude",value=True)
        altitude_ref_m=0.0
        if apply_altitude:
            altitude_ref_m=st.number_input("Altitude d'entraînement habituelle (m)",value=0.0,step=100.0)

    with st.expander("🎢 Modèle de pente"):
        apply_grade=st.checkbox("Prendre en compte la pente",value=True)
        use_minetti=st.checkbox("Modèle Minetti",value=True)
        minetti_weight=0.6; elev_smooth_window=11; grade_power=0.85
        k_up=12.0; k_down=5.0; down_cap=-0.08; g0_up=3.0; g0_down=2.5; max_up=0.30; max_down=-0.06
        if use_minetti:
            minetti_weight=st.slider("Part de Minetti",0.0,1.0,0.6,0.1)
        if EXPERT:
            elev_smooth_window=st.slider("Lissage altitude (fenêtre pts GPS)",1,51,11,2)
            grade_power=st.slider("Amortissement effet pente",0.2,1.0,0.85,0.05)
            c1,c2,c3=st.columns(3)
            k_up  =c1.number_input("k_up",  value=12.0,step=0.5)
            k_down=c2.number_input("k_down",value=5.0,step=0.5)
            down_cap=c3.number_input("Cap descente",value=-0.08,step=0.01,format="%.2f")
            st.session_state["k_up_val"]=k_up; st.session_state["k_down_val"]=k_down

    with st.expander("💨 Vent"):
        apply_wind=st.checkbox("Appliquer l'effet du vent",value=True)
        wind_mode="Lissé"; wind_smooth_km=5
        drag_coeff=0.012; tail_credit=0.35; wind_cap_head=0.10; wind_cap_tail=-0.04; wind_power=1.0
        wind_gate_g1=2.0; wind_gate_g2=8.0; wind_gate_min=0.25
        if apply_wind and EXPERT:
            wind_mode=st.selectbox("Mode calcul vent",["Lissé","Global"],key="wmode").split()[0]
            wind_smooth_km=st.slider("Lissage vent (km)",1,11,5,2)
            c1,c2=st.columns(2)
            drag_coeff =c1.number_input("Coeff. aérodynamique",value=0.012,step=0.002,format="%.3f")
            tail_credit=c2.slider("Crédit vent arrière",0.0,0.8,0.35,0.05)
            wind_cap_head=st.slider("Pénalité max vent face (%)",0.00,0.20,0.10,0.01)
            wind_cap_tail=st.slider("Gain max vent dos (%)",-0.10,0.00,-0.04,0.01)

    base_cap=0.08; extra_per_pct=0.004; max_cap=0.18
    if EXPERT:
        with st.expander("🧱 Plafond anti-accumulation"):
            c1,c2,c3=st.columns(3)
            base_cap=c1.slider("Plafond de base (%)",0.02,0.20,0.08,0.01)
            extra_per_pct=c2.slider("Extra par % pente",0.000,0.020,0.004,0.001)
            max_cap=c3.slider("Plafond absolu (%)",0.05,0.40,0.18,0.01)

    with st.expander("🔋 Fatigue en course"):
        apply_fatigue=st.checkbox("Activer la fatigue",value=False)
        fatigue_rate=0.0; fatigue_mode="mixte"
        if apply_fatigue:
            fatigue_rate=st.slider("Ralentissement total fin de course (%)",0.0,30.0,8.0,0.5)
            fatigue_mode=st.selectbox("Type de fatigue",["mixte (recommandé)","distance (plat)","d_plus (montagne)"]).split()[0]

    with st.expander("⚡ Stratégie de pacing Ultra"):
        apply_ultra=st.checkbox("Activer le pacing ultra",value=False)
        ultra_amp=0.0
        if apply_ultra:
            ultra_amp=st.slider("Amplitude (%)",0.0,40.0,10.0,0.5)

    show_smooth_pace=True; smooth_window_km=3
    with st.expander("📉 Options d'affichage"):
        show_smooth_pace=st.checkbox("Afficher l'allure lissée",value=True)
        smooth_window_km=st.slider("Fenêtre lissage (km)",1,9,3,2) if show_smooth_pace else 3

    st.markdown("---")
    st.header("5️⃣  Paramètres de la course cible")
    c1,c2=st.columns(2)
    date_course  = c1.date_input("📅 Date de course",value=date.today())
    heure_course = c2.time_input("⏰ Heure de départ",value=time(9,0))
    colf1,colf2=st.columns(2)
    with colf1:
        force_dist=st.checkbox("Forcer la distance",value=False)
        dist_forcee=st.number_input("Distance (km)",value=42.195,format="%.3f") if force_dist else None
    with colf2:
        force_temps=st.checkbox("Travailler à partir d'un objectif de temps",value=False)
        # ── Objectif au format hh:mm:ss ──
        temps_objectif = hms_input("Temps objectif", "3:30:00", key="temps_objectif_target") if force_temps else None
    st.markdown("---")

    with st.expander("🔬 Cross-validation (fiabilité du modèle)"):
        st.info("LOO : prédit chaque référence avec les autres. MAPE < 3% = excellent | < 7% = correct.")
        if st.button("Lancer la cross-validation"):
            refs_cv=prepare_refs(refs_raw,use_recalibrated,opt_temp,use_wbgt,
                                  cold_quad,hot_quad,temp_max_penalty,
                                  k_up,k_down,down_cap,g0_up,g0_down,max_up,max_down,
                                  elev_ref_power,temp_ref_power)
            cv=crossval_loo(refs_cv)
            if cv is None:
                st.warning("Au moins 3 références nécessaires.")
            else:
                df_cv,mae,mape=cv
                st.dataframe(df_cv,use_container_width=True)
                c1,c2=st.columns(2)
                c1.metric("Erreur absolue moyenne",f"{seconds_to_hms(mae)} ({mae:.0f}s)")
                c2.metric("MAPE",f"{mape:.2f} %")

    st.header("6️⃣  Calcul & Résultats")
    if st.button("▶️ Calculer la prédiction", type="primary"):
        if not gpx_file or points is None:
            st.error("⚠️ Importe un fichier GPX (section 1).")
        elif not any(safe_float(r.get("distance",0))>0 and hms_to_seconds(r.get("temps","0"))>0 for r in refs_raw):
            st.error("⚠️ Renseigne au moins une référence valide.")
        else:
            with st.spinner("Calcul en cours..."):
                try:
                    res=run_prediction(
                        distance_cible_km=dist_forcee if force_dist else None,
                        refs_input=refs_raw,points=points,
                        date_course=date_course,heure_course=heure_course,
                        use_recalibrated=use_recalibrated,opt_temp=opt_temp,use_wbgt=use_wbgt,
                        cold_quad=cold_quad,hot_quad=hot_quad,temp_max_penalty=temp_max_penalty,
                        temp_power=temp_power,elev_ref_power=elev_ref_power,temp_ref_power=temp_ref_power,
                        apply_grade=apply_grade,use_minetti=use_minetti,minetti_weight=minetti_weight,
                        k_up=k_up,k_down=k_down,down_cap=down_cap,
                        g0_up=g0_up,g0_down=g0_down,max_up=max_up,max_down=max_down,
                        elev_smooth_window=elev_smooth_window,grade_power=grade_power,
                        apply_altitude=apply_altitude,altitude_ref_m=altitude_ref_m,
                        apply_wind=apply_wind,wind_mode=wind_mode,wind_smooth_km=wind_smooth_km,
                        drag_coeff=drag_coeff,tail_credit=tail_credit,
                        wind_cap_head=wind_cap_head,wind_cap_tail=wind_cap_tail,wind_power=wind_power,
                        wind_gate_g1=wind_gate_g1,wind_gate_g2=wind_gate_g2,wind_gate_min=wind_gate_min,
                        base_cap=base_cap,extra_per_pct=extra_per_pct,max_cap=max_cap,
                        apply_fatigue=apply_fatigue,fatigue_rate=fatigue_rate,fatigue_mode=fatigue_mode,
                        apply_ultra=apply_ultra,ultra_amp=ultra_amp,
                        objective_hms=temps_objectif if force_temps else None,
                        show_smooth_pace=show_smooth_pace,smooth_window_km=smooth_window_km,
                        dem_elevations=dem_elevations)
                    st.session_state["res"] = res
                    # Partager les refs calibrées et K avec l'onglet VC
                    st.session_state["refs_fit_vc"]  = res.get("refs_fit", [])
                    st.session_state["K_riegel_vc"]  = res.get("K", 1.06)
                except Exception as e:
                    import traceback; st.error(f"Erreur : {e}"); st.code(traceback.format_exc())

    if "res" in st.session_state:
        res=st.session_state["res"]
        st.markdown("---"); st.subheader("🎯 Prédiction")
        avg_pace_s=res["total_s"]/max(res["dist_gpx_km"],1e-6)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("⏱ Temps prédit",res["total_human"])
        c2.metric("📊 Allure moy.",pace_str(avg_pace_s)+"/km")
        c3.metric("Fourchette −5%",res["ci_low"])
        c4.metric("Fourchette +5%",res["ci_high"])
        c5.metric("K Riegel",f"{res['K']:.3f}")

        df_out=res["df"]
        if not df_out.empty:
            res_t1,res_t2,res_t3=st.tabs(["📈 Allure par km","🔎 Facteurs","📋 Tableau détaillé"])
            with res_t1:
                fig,ax=plt.subplots(figsize=(12,4))
                pv=[]
                for v in df_out["Allure (min/km)"].values:
                    try: parts=str(v).split(":"); pv.append(int(parts[0])+int(parts[1])/60.0)
                    except: pv.append(float("nan"))
                x=list(range(1,len(pv)+1))
                ax.plot(x,pv,lw=1.5,alpha=0.35,color="steelblue",label="Allure brute")
                if "Allure lissée (min/km)" in df_out.columns:
                    ps=[]
                    for v in df_out["Allure lissée (min/km)"].values:
                        try: parts=str(v).split(":"); ps.append(int(parts[0])+int(parts[1])/60.0)
                        except: ps.append(float("nan"))
                    ax.plot(x,ps,lw=2.5,color="firebrick",label="Allure lissée")
                ax.invert_yaxis(); ax.set_xlabel("Kilomètre"); ax.set_ylabel("Allure (min/km)")
                ax.set_title("Allure prévisionnelle km par km"); ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)
            with res_t2:
                fig2,ax2=plt.subplots(figsize=(12,4))
                x=list(range(1,len(df_out)+1))
                ax2.plot(x,df_out["Mult Pente"].values,label="Pente",lw=2)
                if "Mult Temp" in df_out.columns: ax2.plot(x,df_out["Mult Temp"].values,label="Température",lw=2)
                if "Mult Vent" in df_out.columns: ax2.plot(x,df_out["Mult Vent"].values,label="Vent",lw=2)
                if "Mult Fatigue" in df_out.columns: ax2.plot(x,df_out["Mult Fatigue"].values,label="Fatigue",lw=2,ls=":")
                ax2.axhline(1.0,color="gray",lw=0.8); ax2.set_xlabel("Kilomètre")
                ax2.set_ylabel("Multiplicateur"); ax2.set_title("Décomposition des facteurs")
                ax2.legend(); ax2.grid(alpha=0.3); st.pyplot(fig2); plt.close(fig2)
            with res_t3:
                st.dataframe(df_out,use_container_width=True)

    if gpx_file and points:
        with st.expander("🗺️ Carte & Profil d'altitude",expanded=False):
            try:
                lats_m=[p.latitude for p in points]; lons_m=[p.longitude for p in points]
                view=pdk.ViewState(latitude=float(np.mean(lats_m)),longitude=float(np.mean(lons_m)),zoom=13,pitch=0)
                deck=pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=view,
                    layers=[pdk.Layer("PathLayer",
                                      data=[{"path":[[lon,lat] for lat,lon in zip(lats_m,lons_m)]}],
                                      get_path="path",get_color=[220,50,50],width_min_pixels=4)])
                st.pydeck_chart(deck,use_container_width=True)
                cum_d=[0.0]
                for i in range(1,len(points)):
                    cum_d.append(cum_d[-1]+haversine_m(points[i-1].latitude,points[i-1].longitude,
                                                        points[i].latitude,points[i].longitude))
                x_km=np.array(cum_d)/1000.0
                y_gps=np.array([getattr(p,"elevation",0.0) or 0.0 for p in points])
                w=int(elev_smooth_window); w+=(1 if w%2==0 else 0)
                fig3,ax3=plt.subplots(figsize=(10,3))
                if w>=3 and y_gps.size>=w:
                    y_s=np.convolve(y_gps,np.ones(w)/w,mode="same")
                    ax3.plot(x_km,y_s,lw=2,label="GPS lissé",color="steelblue")
                    ax3.plot(x_km,y_gps,lw=1,alpha=0.2,color="gray",label="GPS brut")
                else:
                    ax3.plot(x_km,y_gps,lw=2,label="GPS",color="steelblue")
                if dem_elevations is not None and len(dem_elevations)==len(points):
                    y_dem=np.array([e if e is not None else 0.0 for e in dem_elevations])
                    ax3.plot(x_km,y_dem,lw=2,ls="--",label="DEM corrigé",color="forestgreen")
                ax3.set_xlabel("Distance (km)"); ax3.set_ylabel("Altitude (m)")
                ax3.set_title("Profil d'altitude"); ax3.legend(); ax3.grid(alpha=0.3)
                st.pyplot(fig3); plt.close(fig3)
            except Exception as e:
                st.error(f"Impossible d'afficher la carte : {e}")


# ══════════════════════════════════════════════════════════════
# ONGLET 1 — TESTS D'ENDURANCE + VITESSE CRITIQUE
# ══════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.title("🧪 Tests d'endurance & Vitesse Critique (VC)")
    st.markdown("""
<div class="highlight-box">
<strong>Principe :</strong> réalise 3 à 6 efforts à intensités variées (ex : 6 min, 12 min, 20 min, 30 min).
La <em>Vitesse Critique</em> est la vitesse maximale que l'athlète peut maintenir indéfiniment.
</div>""", unsafe_allow_html=True)

    if not HAS_FITDECODE:
        st.warning("⚠️ `fitdecode` non installé — fichiers FIT non disponibles. `pip install fitdecode`")

    col_nt1, col_nt2 = st.columns([1,3])
    with col_nt1:
        n_tests = st.number_input("Nombre de tests", min_value=2, max_value=6, value=3, step=1, key="n_tests_vc")

    st.info(f"**{n_tests} tests** à charger. Conseil : mélanger des efforts courts (6 min) et longs (20-30 min).")

    # test_ranges : liste de (start_s, end_s) ou (None, None) si pas de plage
    test_files = []; test_names = []; test_ranges = []

    for i in range(int(n_tests)):
        with st.expander(f"📁 Test {i+1}", expanded=(i<2)):
            c_name, c_file = st.columns([2, 3])
            with c_name:
                t_name = st.text_input("Nom du test", value=f"Test {i+1}", key=f"tname_{i}")
            with c_file:
                t_file = st.file_uploader("Fichier activité", type=["fit","gpx","tcx","csv"], key=f"tfile_{i}")

            # ── Plage de temps hh:mm:ss (optionnelle) ──
            use_range = st.checkbox("Délimiter une plage de temps dans le fichier",
                                     key=f"frange_{i}",
                                     help="Utile si le fichier contient l'échauffement ou la récupération")
            if use_range:
                cr1, cr2 = st.columns(2)
                with cr1:
                    t_start_hms = hms_input("Début de l'effort", "0:00:00",
                                             key=f"tstart_hms_{i}",
                                             help="Temps relatif depuis le début du fichier (hh:mm:ss)")
                with cr2:
                    t_end_hms = hms_input("Fin de l'effort", "0:20:00",
                                           key=f"tend_hms_{i}",
                                           help="Temps relatif depuis le début du fichier (hh:mm:ss)")
                t_start_s = float(hms_to_seconds(t_start_hms)) if validate_hms(t_start_hms) else 0.0
                t_end_s   = float(hms_to_seconds(t_end_hms))   if validate_hms(t_end_hms)   else None
                if t_end_s is not None and t_end_s <= t_start_s:
                    st.warning("⚠️ La fin doit être postérieure au début.")
                    t_start_s, t_end_s = None, None
                st.caption(f"→ Plage sélectionnée : **{t_start_hms}** → **{t_end_hms}** "
                           f"({seconds_to_hms(t_end_s - t_start_s) if t_end_s else '—'} d'effort)")
            else:
                t_start_s, t_end_s = None, None

            test_files.append(t_file)
            test_names.append(t_name)
            test_ranges.append((t_start_s, t_end_s))

    st.markdown("---")

    if st.button("🔍 Analyser tous les tests", type="primary", key="btn_analyze_vc"):
        loaded = []
        for i, (f, name, (rng_start, rng_end)) in enumerate(zip(test_files, test_names, test_ranges)):
            if f is None:
                st.warning(f"Test {i+1} ({name}) : aucun fichier chargé — ignoré.")
                continue
            df_act = load_activity(f)
            if df_act is None or df_act.empty:
                st.warning(f"Test {i+1} ({name}) : impossible de lire le fichier.")
                continue
            # Appliquer la plage de temps si définie
            if rng_start is not None or rng_end is not None:
                t0_act = float(df_act["elapsed_s"].min())
                t_lo = (rng_start if rng_start is not None else t0_act)
                t_hi = (rng_end   if rng_end   is not None else float(df_act["elapsed_s"].max()))
                df_act = df_act[(df_act["elapsed_s"] >= t_lo) & (df_act["elapsed_s"] <= t_hi)].copy()
                df_act["elapsed_s"] = df_act["elapsed_s"] - t_lo
            if df_act.empty:
                st.warning(f"Test {i+1} ({name}) : données vides après troncature de la plage.")
                continue
            dur_s = float(df_act["elapsed_s"].max())
            if df_act["distance_m"].notna().any():
                d_valid = df_act["distance_m"].dropna()
                dist_m = float(d_valid.iloc[-1] - d_valid.iloc[0])
                if dist_m <= 0:
                    dist_m = float(d_valid.max() - d_valid.min())
            else:
                dist_m = None
            hr_stats  = analyze_heart_rate(df_act)
            spd_stats = analyze_speed_kinetics(df_act)
            loaded.append({"name":name,"idx":i,"df":df_act,"dur_s":dur_s,
                           "dist_m":dist_m,"hr":hr_stats,"spd":spd_stats})
        st.session_state["vc_loaded"] = loaded

    if "vc_loaded" in st.session_state:
        loaded = st.session_state["vc_loaded"]
        if not loaded:
            st.error("Aucun test valide chargé.")
        else:
            st.subheader(f"📊 Résultats des {len(loaded)} tests")
            n_cols = 2
            for row_start in range(0, len(loaded), n_cols):
                row_items = loaded[row_start : row_start + n_cols]
                cols = st.columns(n_cols)
                for col_idx, item in enumerate(row_items):
                    with cols[col_idx]:
                        hr  = item["hr"]
                        spd = item["spd"]
                        dur_str = seconds_to_hms(item["dur_s"])
                        dist_str = f"{item['dist_m']/1000:.2f} km" if item["dist_m"] else "—"
                        avg_v = (item["dist_m"] / item["dur_s"]) if item["dist_m"] and item["dur_s"] > 0 else None
                        st.markdown(f'<div class="test-card"><h4>🔵 {item["name"]}</h4>', unsafe_allow_html=True)
                        m1,m2,m3 = st.columns(3)
                        m1.metric("Durée", dur_str)
                        m2.metric("Distance", dist_str)
                        m3.metric("Vitesse moy.", f"{avg_v:.2f} m/s" if avg_v else "—")
                        if hr.get("available"):
                            st.markdown("**Fréquence cardiaque**")
                            h1,h2,h3 = st.columns(3)
                            h1.metric("FC max (P95)", f"{hr['fc_max']} bpm")
                            h2.metric("FC moy.", f"{hr['fc_avg']:.0f} bpm")
                            h3.metric("Dérive", f"+{hr['drift_abs']:.1f} bpm")
                            st.caption(f"Seuil estimé ~{hr['seuil_estime']} bpm · Fiabilité : {hr['reliability']}")
                            df_act = item["df"]
                            if "heart_rate" in df_act.columns and df_act["heart_rate"].notna().any():
                                hr_s = smooth_hr(df_act["heart_rate"].ffill(), window=15)
                                fig_hr, ax_hr = plt.subplots(figsize=(5,2))
                                ax_hr.plot(df_act["elapsed_s"]/60.0, hr_s, color="#d62728", lw=1.5)
                                ax_hr.set_xlabel("Temps (min)"); ax_hr.set_ylabel("FC (bpm)")
                                ax_hr.set_title(f"FC — {item['name']}", fontsize=9)
                                ax_hr.grid(alpha=0.3); fig_hr.tight_layout()
                                st.pyplot(fig_hr); plt.close(fig_hr)
                        else:
                            st.caption("FC non disponible dans ce fichier.")
                        if spd.get("available"):
                            pace_avg = pace_str(1000.0/spd["speed_avg_ms"]) if spd["speed_avg_ms"]>0 else "—"
                            st.caption(f"Vitesse moy. : {spd['speed_avg_ms']:.2f} m/s ({pace_avg}/km) | r={spd['r_value']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📐 Vitesse Critique (D = VC × T + D')")
            vc_points = [(item["dist_m"], item["dur_s"])
                         for item in loaded if item["dist_m"] is not None and item["dur_s"] > 0]
            if len(vc_points) >= 2:
                dists_vc = [p[0] for p in vc_points]
                durs_vc  = [p[1] for p in vc_points]
                vc, d_prime, r2 = compute_vc(dists_vc, durs_vc)
                cv1,cv2,cv3,cv4 = st.columns(4)
                if vc and vc > 0:
                    cv1.metric("Vitesse Critique (VC)", f"{vc:.2f} m/s")
                    cv2.metric("Allure VC", pace_str(1000.0/vc)+"/km")
                    cv3.metric("D' (réserve anaérobie)", f"{d_prime:.0f} m" if d_prime else "—")
                    cv4.metric("R² régression", f"{r2:.3f}")
                    if r2 < 0.90:
                        st.warning(f"⚠️ R²={r2:.3f} — régression faible.")
                    else:
                        st.success(f"✅ R²={r2:.3f} — bonne qualité.")

                    fig_vc, ax_vc = plt.subplots(figsize=(7,4))
                    T_arr = np.array(durs_vc); D_arr = np.array(dists_vc)
                    T_line = np.linspace(T_arr.min()*0.8, T_arr.max()*1.2, 100)
                    D_line = vc * T_line + d_prime
                    ax_vc.scatter(T_arr/60, D_arr/1000, s=80, color="#1f77b4", zorder=5, label="Tests réels")
                    ax_vc.plot(T_line/60, D_line/1000, color="#d62728", lw=2,
                               label=f"VC = {vc:.2f} m/s | D' = {d_prime:.0f} m")
                    ax_vc.set_xlabel("Durée (min)"); ax_vc.set_ylabel("Distance (km)")
                    ax_vc.set_title("Modèle D = VC × T + D'"); ax_vc.legend(); ax_vc.grid(alpha=0.3)
                    st.pyplot(fig_vc); plt.close(fig_vc)

                    st.markdown("---")
                    st.subheader("📋 Table des temps de maintien")
                    st.caption(
                        "**80 % → 100 % VC** : modèle Riegel calé sur vos références. "
                        "**100 % → 120 % VC** : modèle D' (réserve anaérobie). "
                        "Paliers de 4 % de VC."
                    )
                    refs_for_table = st.session_state.get("refs_fit_vc", [])
                    if not refs_for_table:
                        refs_for_table = [
                            {"distance": item["dist_m"], "temps": item["dur_s"]}
                            for item in loaded if item["dist_m"] and item["dur_s"] > 0
                        ]
                    K_for_table = st.session_state.get("K_riegel_vc", 1.06)
                    df_hold = build_holding_table(vc, d_prime, refs_for_table, K_for_table)
                    if not df_hold.empty:
                        # Mise en évidence de la ligne VC (100 %)
                        def style_vc_row(row):
                            if row["% VC"] == "100 %":
                                return ["background-color: #e8f0fe; font-weight: bold"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            df_hold.style.apply(style_vc_row, axis=1),
                            use_container_width=True,
                        )

                        # Graphique en % VC
                        fig_hold, ax_hold = plt.subplots(figsize=(9, 4))
                        pcts = [float(p.replace(" %", "")) for p in df_hold["% VC"]]
                        mask_ri = df_hold["Modèle"] == "Riegel"
                        mask_dp = df_hold["Modèle"] == "Modèle D'"
                        if mask_ri.any():
                            ax_hold.plot(
                                [pcts[i] for i in df_hold.index[mask_ri]],
                                df_hold.loc[mask_ri, "Durée (min)"],
                                "o-", color="#1f77b4", lw=2.5, ms=6,
                                label="Riegel (80 %–100 % VC)"
                            )
                        if mask_dp.any():
                            ax_hold.plot(
                                [pcts[i] for i in df_hold.index[mask_dp]],
                                df_hold.loc[mask_dp, "Durée (min)"],
                                "o--", color="#d62728", lw=2.5, ms=6,
                                label="Modèle D' (100 %–120 % VC)"
                            )
                        ax_hold.axvline(100, color="gray", lw=1.5, ls=":",
                                        label=f"VC = 100 % ({pace_str(1000/vc)}/km)")
                        # Annoter chaque point avec l'allure
                        for i, row in df_hold.iterrows():
                            ax_hold.annotate(
                                row["Allure (/km)"],
                                xy=(pcts[i], row["Durée (min)"]),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=7, color="#444"
                            )
                        ax_hold.set_xlabel("% de la Vitesse Critique")
                        ax_hold.set_ylabel("Temps de maintien (min)")
                        ax_hold.set_title("Temps de maintien par pourcentage de VC")
                        ax_hold.set_xticks([float(p.replace(" %","")) for p in df_hold["% VC"]])
                        ax_hold.set_xticklabels([p for p in df_hold["% VC"]], rotation=45, fontsize=8)
                        ax_hold.legend(); ax_hold.grid(alpha=0.3); ax_hold.set_ylim(0)
                        fig_hold.tight_layout()
                        st.pyplot(fig_hold); plt.close(fig_hold)

                    st.markdown("---")
                    st.subheader("📄 Export PDF")
                    if st.button("Générer le PDF", key="btn_pdf_vc"):
                        buf = io.BytesIO()
                        with PdfPages(buf) as pdf:
                            fig_p1, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
                            axes[0].scatter(T_arr/60, D_arr/1000, s=80, color="#1f77b4", label="Tests réels", zorder=5)
                            axes[0].plot(T_line/60, D_line/1000, color="#d62728", lw=2,
                                         label=f"VC = {vc:.2f} m/s | D' = {d_prime:.0f} m")
                            axes[0].set_xlabel("Durée (min)"); axes[0].set_ylabel("Distance (km)")
                            axes[0].set_title("Modèle Vitesse Critique"); axes[0].legend(); axes[0].grid(alpha=0.3)
                            if not df_hold.empty:
                                pcts_pdf = [float(p.replace(" %","")) for p in df_hold["% VC"]]
                                if mask_ri.any():
                                    axes[1].plot(
                                        [pcts_pdf[i] for i in df_hold.index[mask_ri]],
                                        df_hold.loc[mask_ri, "Durée (min)"],
                                        "o-", color="#1f77b4", lw=2, label="Riegel")
                                if mask_dp.any():
                                    axes[1].plot(
                                        [pcts_pdf[i] for i in df_hold.index[mask_dp]],
                                        df_hold.loc[mask_dp, "Durée (min)"],
                                        "o--", color="#d62728", lw=2, label="Modèle D'")
                                axes[1].axvline(100, color="gray", lw=1.5, ls=":")
                                axes[1].set_xlabel("% VC"); axes[1].set_ylabel("Durée (min)")
                                axes[1].set_title("Temps de maintien par % VC")
                                axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0)
                            fig_p1.tight_layout(); pdf.savefig(fig_p1); plt.close(fig_p1)
                            for item in loaded:
                                df_act = item["df"]
                                if "heart_rate" not in df_act.columns or not df_act["heart_rate"].notna().any():
                                    continue
                                fig_fc, ax_fc = plt.subplots(figsize=(8.27, 4))
                                hr_s = smooth_hr(df_act["heart_rate"].ffill(), window=15)
                                ax_fc.plot(df_act["elapsed_s"]/60.0, hr_s, color="#d62728", lw=1.5)
                                ax_fc.set_xlabel("Temps (min)"); ax_fc.set_ylabel("FC (bpm)")
                                ax_fc.set_title(f"FC — {item['name']}"); ax_fc.grid(alpha=0.3); fig_fc.tight_layout()
                                pdf.savefig(fig_fc); plt.close(fig_fc)
                        buf.seek(0)
                        st.download_button("⬇️ Télécharger le rapport PDF",
                                           data=buf, file_name="rapport_vc.pdf", mime="application/pdf")
            else:
                st.info("Chargez au moins 2 tests pour calculer la Vitesse Critique.")


# ══════════════════════════════════════════════════════════════
# ONGLET 2 — ANALYSE ENTRAÎNEMENT (avec analyse fractionné)
# ══════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.title("⚙️  Analyse d'entraînement")
    st.markdown("""
Chargez une activité (sortie longue, fartlek, tempo...) pour analyser la **dérive cardiaque**,
la **cinétique de vitesse** et la **qualité de l'effort**.
Utilisez l'**analyse fractionné** pour isoler et comparer chaque répétition.
""")

    if not HAS_FITDECODE:
        st.warning("⚠️ `fitdecode` non installé — `pip install fitdecode`")

    act_file = st.file_uploader("📂 Importer une activité (FIT, GPX, TCX ou CSV)",
                                 type=["fit","gpx","tcx","csv"], key="entr_file")

    with st.expander("⚙️ Options d'analyse"):
        col_o1,col_o2 = st.columns(2)
        with col_o1:
            hr_smooth_win = st.slider("Lissage FC (fenêtre points)", 3, 61, 15, 2)
            # ── Début d'analyse au format hh:mm:ss ──
            trim_start_hms = hms_input("Écarter début (hh:mm:ss)", "0:00:00",
                                        key="trim_start_hms",
                                        help="Ignorer les N premières minutes (chauffe)")
            trim_start_s = float(hms_to_seconds(trim_start_hms))
        with col_o2:
            show_speed = st.checkbox("Afficher la cinétique de vitesse", value=True)
            # ── Fin d'analyse au format hh:mm:ss ──
            trim_end_hms = hms_input("Écarter fin (hh:mm:ss)", "0:00:00",
                                      key="trim_end_hms",
                                      help="Ignorer les N dernières minutes (récup)")
            trim_end_s = float(hms_to_seconds(trim_end_hms))

    if act_file is not None:
        with st.spinner("Chargement de l'activité..."):
            df_entr_raw = load_activity(act_file)

        if df_entr_raw is None or df_entr_raw.empty:
            st.error("Impossible de lire ce fichier.")
        else:
            t_max_raw = float(df_entr_raw["elapsed_s"].max())
            # Durée totale affichée pour aider l'utilisateur
            st.info(f"⏱ Durée totale de l'activité : **{seconds_to_hms(t_max_raw)}** "
                    f"({t_max_raw/60:.1f} min) — utilisez ce repère pour vos plages de temps.")

            t_start = trim_start_s
            t_end   = t_max_raw - trim_end_s
            if t_start > 0 or trim_end_s > 0:
                df_entr = df_entr_raw[(df_entr_raw["elapsed_s"] >= t_start) & (df_entr_raw["elapsed_s"] <= t_end)].copy()
                df_entr["elapsed_s"] = df_entr["elapsed_s"] - t_start
            else:
                df_entr = df_entr_raw.copy()

            if df_entr.empty:
                st.error("Après troncature, il ne reste plus de données. Réduisez les marges.")
            else:
                dur_s  = float(df_entr["elapsed_s"].max())
                if df_entr["distance_m"].notna().any():
                    d_valid_e = df_entr["distance_m"].dropna()
                    dist_m = float(d_valid_e.iloc[-1] - d_valid_e.iloc[0])
                    if dist_m <= 0:
                        dist_m = float(d_valid_e.max() - d_valid_e.min())
                else:
                    dist_m = None

                st.subheader("📊 Vue d'ensemble")
                mg1,mg2,mg3,mg4 = st.columns(4)
                mg1.metric("Durée analysée", seconds_to_hms(dur_s))
                mg2.metric("Distance", f"{dist_m/1000:.2f} km" if dist_m else "—")
                v_avg = dist_m/dur_s if dist_m and dur_s>0 else None
                mg3.metric("Vitesse moy.", f"{v_avg:.2f} m/s" if v_avg else "—")
                mg4.metric("Allure moy.", pace_str(1000.0/v_avg)+"/km" if v_avg else "—")

                # ── Analyse FC globale ──
                hr_stats = analyze_heart_rate(df_entr)
                st.subheader("💓 Analyse de la fréquence cardiaque")
                if not hr_stats.get("available"):
                    st.info("Données FC non disponibles dans ce fichier.")
                else:
                    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
                    fc1.metric("FC max (P95)",  f"{hr_stats['fc_max']} bpm")
                    fc2.metric("FC moyenne",    f"{hr_stats['fc_avg']:.0f} bpm")
                    fc3.metric("FC mini (P5)",  f"{hr_stats['fc_min']} bpm")
                    fc4.metric("Dérive (Q3−Q1)", f"+{hr_stats['drift_abs']:.1f} bpm ({hr_stats['drift_pct']:.1f}%)")
                    fc5.metric("Seuil estimé",  f"~{hr_stats['seuil_estime']} bpm")
                    rel = hr_stats["reliability"]
                    if "haute" in rel:    st.success(f"✅ Fiabilité : {rel}")
                    elif "moyenne" in rel: st.warning(f"⚠️ Fiabilité : {rel}")
                    else:                 st.error(f"❌ Fiabilité : {rel}")

                    hr_s = smooth_hr(df_entr["heart_rate"].ffill(), window=int(hr_smooth_win))
                    fig_hr, ax_hr = plt.subplots(figsize=(11,3))
                    ax_hr.plot(df_entr["elapsed_s"]/60.0, df_entr["heart_rate"],
                               color="#d62728", alpha=0.2, lw=1, label="FC brute")
                    ax_hr.plot(df_entr["elapsed_s"]/60.0, hr_s,
                               color="#d62728", lw=2, label="FC lissée")
                    n = len(df_entr)
                    q1_t = df_entr["elapsed_s"].iloc[int(n*0.25)]/60.0
                    q3_t = df_entr["elapsed_s"].iloc[int(n*0.75)]/60.0
                    fc_q1_mean = float(df_entr["heart_rate"].iloc[:int(n*0.25)].mean())
                    fc_q3_mean = float(df_entr["heart_rate"].iloc[int(n*0.75):].mean())
                    ax_hr.axvline(q1_t, color="steelblue", lw=1, ls="--", alpha=0.7)
                    ax_hr.axvline(q3_t, color="steelblue", lw=1, ls="--", alpha=0.7)
                    ax_hr.axhline(fc_q1_mean, color="green",  lw=1, ls=":", alpha=0.8, label=f"Moy Q1 {fc_q1_mean:.0f}")
                    ax_hr.axhline(fc_q3_mean, color="orange", lw=1, ls=":", alpha=0.8, label=f"Moy Q3 {fc_q3_mean:.0f}")
                    ax_hr.set_xlabel("Temps (min)"); ax_hr.set_ylabel("FC (bpm)")
                    ax_hr.set_title("Profil de fréquence cardiaque"); ax_hr.legend(fontsize=8); ax_hr.grid(alpha=0.3)
                    st.pyplot(fig_hr); plt.close(fig_hr)

                    drift = hr_stats["drift_abs"]
                    if drift < 5:
                        st.success(f"✅ Dérive faible (+{drift:.1f} bpm).")
                    elif drift < 12:
                        st.warning(f"⚠️ Dérive modérée (+{drift:.1f} bpm).")
                    else:
                        st.error(f"❌ Dérive élevée (+{drift:.1f} bpm).")

                # ── Cinétique de vitesse ──
                if show_speed:
                    st.subheader("🏃 Cinétique de vitesse")
                    spd_stats = analyze_speed_kinetics(df_entr)
                    if not spd_stats.get("available"):
                        st.info("Données de vitesse non disponibles.")
                    else:
                        sv1,sv2,sv3 = st.columns(3)
                        sv1.metric("Vitesse moy.",  f"{spd_stats['speed_avg_ms']:.2f} m/s")
                        sv2.metric("Vitesse max (P95)", f"{spd_stats['speed_max_ms']:.2f} m/s")
                        sv3.metric("Corrélation dérive", f"r = {spd_stats['r_value']:.3f}")
                        spd_s = df_entr["speed_ms"].rolling(int(hr_smooth_win), center=True, min_periods=1).mean()
                        fig_spd, ax_spd = plt.subplots(figsize=(11,3))
                        ax_spd.plot(df_entr["elapsed_s"]/60.0, df_entr["speed_ms"],
                                    color="#1f77b4", alpha=0.2, lw=1, label="Vitesse brute")
                        ax_spd.plot(df_entr["elapsed_s"]/60.0, spd_s,
                                    color="#1f77b4", lw=2, label="Vitesse lissée")
                        x_arr = df_entr["elapsed_s"].values
                        v_arr = df_entr["speed_ms"].dropna().values
                        if len(v_arr) == len(x_arr):
                            sl, ic_r, _, _, _ = sp_stats.linregress(x_arr, v_arr)
                            trend_line = sl * x_arr + ic_r
                            ax_spd.plot(x_arr/60.0, trend_line, color="red", lw=1.5, ls="--",
                                        label=f"Tendance (pente={sl*60:.4f} m/s/min)")
                        ax_spd.set_xlabel("Temps (min)"); ax_spd.set_ylabel("Vitesse (m/s)")
                        ax_spd.set_title("Profil de vitesse"); ax_spd.legend(fontsize=8); ax_spd.grid(alpha=0.3)
                        st.pyplot(fig_spd); plt.close(fig_spd)

                # ══════════════════════════════════════════════════════════
                # ANALYSE FRACTIONNÉ (nouveau bloc)
                # ══════════════════════════════════════════════════════════
                st.markdown("---")
                st.subheader("🔁 Analyse des intervalles (fractionné)")
                st.markdown("""
Définissez jusqu'à **6 intervalles** en saisissant les bornes de début et fin au format **hh:mm:ss**,
relatifs au **début de l'activité** (après troncature éventuelle).
Chaque intervalle est analysé indépendamment : FC, vitesse, dérive, allure.
""")
                # Repère visuel de durée
                st.caption(f"🕐 Durée disponible après troncature : **{seconds_to_hms(dur_s)}**")

                if "n_intervals" not in st.session_state:
                    st.session_state["n_intervals"] = 3

                col_ia, col_ib = st.columns(2)
                with col_ia:
                    if st.button("➕ Ajouter un intervalle") and st.session_state["n_intervals"] < 6:
                        st.session_state["n_intervals"] += 1
                with col_ib:
                    if st.button("➖ Retirer un intervalle") and st.session_state["n_intervals"] > 1:
                        st.session_state["n_intervals"] -= 1

                interval_defs = []
                for iv in range(st.session_state["n_intervals"]):
                    with st.expander(f"⏱ Intervalle {iv+1}", expanded=(iv < 2)):
                        col_n, col_s, col_e = st.columns([2, 2, 2])
                        with col_n:
                            iv_name = st.text_input("Nom", value=f"Rép. {iv+1}", key=f"iv_name_{iv}")
                        with col_s:
                            iv_start = hms_input("Début (hh:mm:ss)", "0:00:00",
                                                  key=f"iv_start_{iv}",
                                                  help="Temps relatif depuis le début de l'activité")
                        with col_e:
                            iv_end = hms_input("Fin (hh:mm:ss)", "0:05:00",
                                                key=f"iv_end_{iv}",
                                                help="Temps relatif depuis le début de l'activité")
                        # Validation rapide
                        s_s = hms_to_seconds(iv_start)
                        e_s = hms_to_seconds(iv_end)
                        if e_s <= s_s:
                            st.warning("⚠️ La fin doit être postérieure au début.")
                        elif e_s > dur_s:
                            st.warning(f"⚠️ La fin ({iv_end}) dépasse la durée analysée ({seconds_to_hms(dur_s)}).")
                        interval_defs.append({"name": iv_name, "start": iv_start, "end": iv_end})

                if st.button("🔍 Analyser les intervalles", type="primary", key="btn_analyze_intervals"):
                    results_intervals = []
                    for iv_def in interval_defs:
                        df_int = extract_interval_df(df_entr, iv_def["start"], iv_def["end"])
                        res_iv = analyze_interval(df_int, iv_def["name"])
                        results_intervals.append(res_iv)
                    st.session_state["interval_results"] = results_intervals

                if "interval_results" in st.session_state:
                    results_intervals = st.session_state["interval_results"]
                    valid_ivs = [r for r in results_intervals if r.get("valid")]

                    if not valid_ivs:
                        st.error("Aucun intervalle valide. Vérifiez les bornes.")
                    else:
                        st.subheader(f"📈 Résultats — {len(valid_ivs)} intervalles analysés")

                        # ── Tableau récapitulatif ──
                        recap_rows = []
                        for r in valid_ivs:
                            hr = r["hr"]
                            spd = r["spd"]
                            recap_rows.append({
                                "Intervalle": r["name"],
                                "Durée": seconds_to_hms(r["dur_s"]),
                                "Distance": f"{r['dist_m']/1000:.2f} km" if r["dist_m"] else "—",
                                "Allure moy.": pace_str(1000.0/r["avg_speed"])+"/km" if r["avg_speed"] else "—",
                                "FC moy. (bpm)": f"{hr['fc_avg']:.0f}" if hr.get("available") else "—",
                                "FC max P95": f"{hr['fc_max']}" if hr.get("available") else "—",
                                "Dérive FC": f"+{hr['drift_abs']:.1f} bpm" if hr.get("available") else "—",
                                "Fiabilité FC": hr.get("reliability","—") if hr.get("available") else "—",
                            })
                        df_recap = pd.DataFrame(recap_rows)
                        st.dataframe(df_recap, use_container_width=True)

                        # ── Graphiques FC superposés ──
                        has_hr_data = any(r["hr"].get("available") for r in valid_ivs)
                        if has_hr_data:
                            st.subheader("💓 Profils FC superposés")
                            fig_comp, ax_comp = plt.subplots(figsize=(11, 4))
                            colors_list = ["#d62728","#1f77b4","#2ca02c","#ff7f0e","#9467bd","#8c564b"]
                            for idx_r, r in enumerate(valid_ivs):
                                df_iv = r["df"]
                                if "heart_rate" not in df_iv.columns or not df_iv["heart_rate"].notna().any():
                                    continue
                                hr_sv = smooth_hr(df_iv["heart_rate"].ffill(), window=15)
                                col_iv = colors_list[idx_r % len(colors_list)]
                                ax_comp.plot(df_iv["elapsed_s"]/60.0, hr_sv,
                                             lw=2, color=col_iv, label=r["name"])
                            ax_comp.set_xlabel("Temps depuis début de l'intervalle (min)")
                            ax_comp.set_ylabel("FC (bpm)")
                            ax_comp.set_title("Comparaison FC — intervalles superposés")
                            ax_comp.legend(); ax_comp.grid(alpha=0.3)
                            st.pyplot(fig_comp); plt.close(fig_comp)

                        # ── Graphiques vitesse superposés ──
                        has_spd_data = any(r["spd"].get("available") for r in valid_ivs)
                        if has_spd_data and show_speed:
                            st.subheader("🏃 Profils de vitesse superposés")
                            fig_spd2, ax_spd2 = plt.subplots(figsize=(11, 4))
                            for idx_r, r in enumerate(valid_ivs):
                                df_iv = r["df"]
                                if "speed_ms" not in df_iv.columns or not df_iv["speed_ms"].notna().any():
                                    continue
                                spd_sv = df_iv["speed_ms"].rolling(15, center=True, min_periods=1).mean()
                                col_iv = colors_list[idx_r % len(colors_list)]
                                ax_spd2.plot(df_iv["elapsed_s"]/60.0, spd_sv,
                                             lw=2, color=col_iv, label=r["name"])
                            ax_spd2.set_xlabel("Temps depuis début de l'intervalle (min)")
                            ax_spd2.set_ylabel("Vitesse (m/s)")
                            ax_spd2.set_title("Comparaison vitesse — intervalles superposés")
                            ax_spd2.legend(); ax_spd2.grid(alpha=0.3)
                            st.pyplot(fig_spd2); plt.close(fig_spd2)

                        # ── Évolution des métriques clés d'un intervalle à l'autre ──
                        if len(valid_ivs) >= 2:
                            st.subheader("📉 Évolution inter-répétitions")
                            fig_ev, axes_ev = plt.subplots(1, 2, figsize=(12, 4))
                            iv_labels = [r["name"] for r in valid_ivs]
                            # Allures
                            allures = [1000.0/r["avg_speed"] if r["avg_speed"] else None for r in valid_ivs]
                            allures_valid = [a for a in allures if a is not None]
                            if allures_valid:
                                axes_ev[0].plot(range(len(iv_labels)), allures, "o-",
                                                color="#1f77b4", lw=2, ms=7)
                                axes_ev[0].set_xticks(range(len(iv_labels))); axes_ev[0].set_xticklabels(iv_labels, rotation=20)
                                axes_ev[0].set_ylabel("Allure (s/km)")
                                axes_ev[0].set_title("Allure moyenne par répétition")
                                axes_ev[0].invert_yaxis(); axes_ev[0].grid(alpha=0.3)
                            # FC moyenne
                            fc_avgs = [r["hr"]["fc_avg"] if r["hr"].get("available") else None for r in valid_ivs]
                            fc_avgs_valid = [f for f in fc_avgs if f is not None]
                            if fc_avgs_valid:
                                axes_ev[1].plot(range(len(iv_labels)), fc_avgs, "o-",
                                                color="#d62728", lw=2, ms=7)
                                axes_ev[1].set_xticks(range(len(iv_labels))); axes_ev[1].set_xticklabels(iv_labels, rotation=20)
                                axes_ev[1].set_ylabel("FC moyenne (bpm)")
                                axes_ev[1].set_title("FC moyenne par répétition")
                                axes_ev[1].grid(alpha=0.3)
                            fig_ev.tight_layout()
                            st.pyplot(fig_ev); plt.close(fig_ev)

                            # Interprétation automatique
                            if allures_valid and len(allures_valid) >= 2:
                                trend_allure = allures_valid[-1] - allures_valid[0]
                                if trend_allure > 15:
                                    st.warning(f"⚠️ Dégradation d'allure : +{trend_allure:.0f} s/km entre la 1ère et la dernière répétition — fatigue progressive ou dosage excessif.")
                                elif trend_allure < -5:
                                    st.success(f"✅ Accélération inter-répétitions : {abs(trend_allure):.0f} s/km de gain — bonne montée en puissance.")
                                else:
                                    st.success("✅ Allure stable entre les répétitions — excellent contrôle de l'effort.")

                            if fc_avgs_valid and len(fc_avgs_valid) >= 2:
                                trend_fc = fc_avgs_valid[-1] - fc_avgs_valid[0]
                                if trend_fc > 8:
                                    st.warning(f"⚠️ Dérive de FC inter-répétitions : +{trend_fc:.0f} bpm — accumulation de fatigue cardiovasculaire.")
                                else:
                                    st.success(f"✅ FC stable entre les répétitions (+{trend_fc:.1f} bpm) — récupération bien gérée.")

                # ── Recalibration météo ──
                with st.expander("🌡️ Recalibration conditions météo (simplifié)"):
                    st.caption("Estime l'équivalent performance dans des conditions idéales.")
                    entr_t  = st.number_input("Température réelle (°C)", value=18.0, step=0.5, key="entr_t")
                    entr_h  = st.number_input("Humidité réelle (%)", value=65.0, step=5.0, key="entr_h")
                    entr_wbgt = wbgt_simplified(entr_t, entr_h)
                    entr_mult = temp_multiplier(entr_wbgt, sb_opt_temp,
                                                sb_k_temp_cold, sb_k_temp_hot, 0.10)
                    st.markdown(f"**WBGT calculé :** {entr_wbgt:.1f}°C | "
                                f"**Multiplicateur thermique :** {entr_mult:.3f} | "
                                f"**Pénalité :** +{(entr_mult-1)*100:.1f}%")
                    if dist_m and dur_s > 0:
                        dur_ideal_s = dur_s / entr_mult
                        entr_pace   = 1000.0 / (dist_m / dur_s)
                        ideal_pace  = 1000.0 / (dist_m / dur_ideal_s)
                        col_entr1, col_entr2 = st.columns(2)
                        col_entr1.metric("Temps réel",   seconds_to_hms(dur_s),
                                          delta=f"Allure {pace_str(entr_pace)}/km")
                        col_entr2.metric("Équivalent conditions idéales", seconds_to_hms(dur_ideal_s),
                                          delta=f"Allure {pace_str(ideal_pace)}/km")

    else:
        st.info("⬆️ Chargez une activité pour commencer l'analyse.")
        st.markdown("""
**Formats acceptés :**
- **FIT** (Garmin, Polar, Suunto, etc.) — nécessite `fitdecode`
- **GPX** — données GPS
- **TCX** — format Garmin Training Center
- **CSV** — colonnes : `elapsed_s`, `heart_rate`, `speed_ms`, `distance_m`, `altitude_m`
""")
