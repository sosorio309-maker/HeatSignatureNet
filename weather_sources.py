"""
weather_sources.py
===================

Wetterdaten-Beschaffung für pcnn_model.py, aus zwei unabhängigen Quellen:

1. MeteoSchweiz (Open Government Data, SwissMetNet) -- echte Stationsmessungen.
   Zwei Wege, die Station zu wählen:
     a) direkt per Kürzel, z. B. station_abbr="STG" (St. Gallen)
     b) automatisch: Koordinaten (lat, lon) angeben, `nearest_station()` sucht die
        nächstgelegene Station per Haversine-Distanz.
   Nur innerhalb der Schweiz sinnvoll (SwissMetNet-Netz).

2. Open-Meteo (open-meteo.com) -- Reanalyse-Wetterdaten für beliebige Koordinaten,
   weltweit, ohne Stationswahl. Einfacher, aber Modell- statt Stationsdaten.

Beide Funktionen liefern ein pandas.DataFrame mit stündlichem DatetimeIndex und
den Spalten T_a [°C], W_s [m/s], I_g [W/m²] -- direkt verwendbar mit
pcnn_model.PCNNModel.fit(Q_h, T_a, W_s, I_g).

Quellen / Dokumentation:
- MeteoSchweiz OGD: https://opendatadocs.meteoswiss.ch/
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

Autor: Santiago Rojo Osorio
"""

from io import StringIO

import numpy as np
import pandas as pd
import requests

METEOSWISS_BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

_STATION_COLS = {
    "station_abbr": "station_abbr",
    "station_name": "station_name",
    "station_canton": "canton",
    "station_coordinates_wgs84_lat": "lat",
    "station_coordinates_wgs84_lon": "lon",
    "station_height_masl": "height_masl",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1) MeteoSchweiz -- Stationswahl per Kürzel ODER per nächstgelegener Station
# ─────────────────────────────────────────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    """Distanz [km] auf der Kugeloberfläche zwischen (lat1, lon1) und Arrays (lat2, lon2)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - lat1)
    dlmb = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def list_meteoswiss_stations():
    """
    Lädt die offizielle Stationsliste aller MeteoSchweiz-Automatikstationen
    (Kürzel, Name, Kanton, Koordinaten, Höhe).
    """
    url = f"{METEOSWISS_BASE}/ogd-smn_meta_stations.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), sep=";")
    df = df[list(_STATION_COLS.keys())].rename(columns=_STATION_COLS)
    return df


def nearest_station(lat, lon, stations=None, n=1):
    """
    Findet die n nächstgelegenen MeteoSchweiz-Stationen zu einer Koordinate.

    Parameters
    ----------
    lat, lon : float
        Zielkoordinate (WGS84).
    stations : pd.DataFrame, optional
        Ergebnis von `list_meteoswiss_stations()`. Wird sie nicht übergeben, wird
        sie hier neu geladen (ein zusätzlicher Netzwerkaufruf).
    n : int
        Anzahl zurückzugebender Stationen, sortiert nach Entfernung.

    Returns
    -------
    pd.DataFrame mit zusätzlicher Spalte `distanz_km`, aufsteigend sortiert.
    """
    if stations is None:
        stations = list_meteoswiss_stations()
    d = _haversine_km(lat, lon, stations["lat"].values, stations["lon"].values)
    out = stations.copy()
    out["distanz_km"] = d
    return out.sort_values("distanz_km").head(n).reset_index(drop=True)


def fetch_meteoswiss_hourly(station_abbr, start, end, verbose=True):
    """
    Lädt stündliche Wetterdaten (T_a, W_s, I_g) einer MeteoSchweiz-Station für
    den Zeitraum [start, end].

    Die MeteoSchweiz-OGD-Struktur teilt die Historie einer Station in
    Jahrzehnt-Dateien (`..._h_historical_1990-1999.csv`, ...) plus eine
    rollierende `..._h_recent.csv`-Datei für die jüngste Vergangenheit. Diese
    Funktion lädt automatisch alle Jahrzehnt-Dateien, die den angefragten
    Zeitraum überschneiden, sowie die `recent`-Datei, hängt sie zusammen und
    entfernt doppelte Zeitstempel (die `recent`-Version gewinnt bei Überlappung).

    Parameters
    ----------
    station_abbr : str
        Stationskürzel, z. B. "STG" (Groß-/Kleinschreibung egal).
    start, end : str oder pd.Timestamp
        Zeitraum (inklusive).
    verbose : bool
        Gibt aus, welche Dateien geladen bzw. übersprungen wurden.

    Returns
    -------
    pd.DataFrame, DatetimeIndex (stündlich), Spalten T_a [°C], W_s [m/s], I_g [W/m²]
    """
    abbr = station_abbr.lower()
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    decades = sorted({(y // 10) * 10 for y in range(start.year, end.year + 1)})
    urls = [f"{METEOSWISS_BASE}/{abbr}/ogd-smn_{abbr}_h_historical_{d}-{d + 9}.csv" for d in decades]
    urls.append(f"{METEOSWISS_BASE}/{abbr}/ogd-smn_{abbr}_h_recent.csv")

    frames = []
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            frames.append(pd.read_csv(StringIO(r.text), sep=";"))
            if verbose:
                print(f"  geladen: {url.split('/')[-1]}")
        except Exception as e:
            if verbose:
                print(f"  übersprungen ({e.__class__.__name__}): {url.split('/')[-1]}")

    if not frames:
        raise RuntimeError(
            f"Keine Wetterdaten für Station '{station_abbr}' gefunden. "
            "Kürzel korrekt? (siehe list_meteoswiss_stations())"
        )

    df = pd.concat(frames, ignore_index=True)
    df["reference_timestamp"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
    df = (
        df.drop_duplicates(subset="reference_timestamp", keep="last")
        .set_index("reference_timestamp")
        .sort_index()
    )

    out = df[["tre200h0", "fu3010h0", "gre000h0"]].rename(
        columns={"tre200h0": "T_a", "fu3010h0": "W_s", "gre000h0": "I_g"}
    )
    out["W_s"] = out["W_s"] / 3.6  # km/h -> m/s
    return out.loc[start:end]


# ─────────────────────────────────────────────────────────────────────────────
# 2) Open-Meteo -- nur Koordinaten, keine Stationswahl, weltweit
# ─────────────────────────────────────────────────────────────────────────────
def fetch_openmeteo_hourly(lat, lon, start, end, timezone="UTC"):
    """
    Lädt stündliche Wetterdaten (T_a, W_s, I_g) von Open-Meteo für beliebige
    Koordinaten -- keine Stationswahl nötig, funktioniert weltweit (Reanalyse-
    /Modelldaten statt Stationsmessung, siehe open-meteo.com/en/docs/historical-weather-api).

    Parameters
    ----------
    lat, lon : float
    start, end : str ("YYYY-MM-DD") oder pd.Timestamp
    timezone : str, Default "UTC"
        An "auto" übergeben, um die lokale Zeitzone der Koordinate zu verwenden.

    Returns
    -------
    pd.DataFrame, DatetimeIndex (stündlich), Spalten T_a [°C], W_s [m/s], I_g [W/m²]
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timezone": timezone,
    }
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(
        columns={
            "temperature_2m": "T_a",
            "wind_speed_10m": "W_s",
            "shortwave_radiation": "I_g",
        }
    )
    return df
