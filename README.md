# PCNN Tool -- Physically Consistent Neural Network für Gebäude-Wärmebedarf

Ein eigenständiges, wiederverwendbares Werkzeug zur thermischen Parameteridentifikation
von Wohngebäuden aus Wärmebedarfs- und Wetterdaten, mittels eines Physically Consistent
Neural Network (PCNN).

Dieses Tool ist im Rahmen der Masterarbeit *"Thermische Gebäudemodellierung zur
Parameteridentifikation und energetischen Charakterisierung von Wohngebäuden"*
(FH Vorarlberg, Nachhaltige Energiesysteme) entstanden, in Kooperation mit der
Energieagentur St. Gallen. Es stellt eines der vier in der Arbeit verglichenen Modelle
(PCNN, Kapitel 2.2.3) als eigenständiges Werkzeug bereit, unabhängig vom
Modellvergleich der Arbeit selbst.

- Masterarbeit (Volltext): https://opus.fhv.at/frontdoor/index/index/docId/7508
- Autor: Santiago Rojo Osorio -- https://www.linkedin.com/in/srojosorio/

## Was ist ein PCNN?

Ein Physically Consistent Neural Network kombiniert ein trainierbares neuronales Netz
mit einer festen physikalischen Gleichung. Statt frei zu lernen, wie Wärmebedarf und
Wetter zusammenhängen, lernt das Netz nur die Parameter einer vorgegebenen physikalischen
Gleichung, der Energy-Signature-Gleichung:

```
Q_h(t) = UA0 * (Ti - Ta(t)) + UAw * Ws(t) * (Ti - Ta(t)) - gA * Ig(t) - Phi0
```

| Symbol | Einheit    | Bedeutung |
|--------|------------|-----------|
| Q_h    | W          | Wärmebedarf des Heizsystems (Zielgröße) |
| UA0    | W/K        | Transmissionsleitwert der Gebäudehülle |
| Ti     | °C         | Innentemperatur / Basistemperatur des Heizsystems |
| Ta     | °C         | Außentemperatur |
| UAw    | W/(K·m/s)  | windabhängiger Leitwert (Infiltration) |
| Ws     | m/s        | Windgeschwindigkeit |
| gA     | m²         | solare Apertur (Energiedurchlassgrad x Fensterfläche) |
| Ig     | W/m²       | Globalstrahlung |
| Phi0   | W          | interne Wärmegewinne (Personen, Geräte, Beleuchtung) |

Ein Encoder (mehrschichtiges neuronales Netz) bildet die Wetterdaten der letzten
Stunden auf den Parametervektor `{UA0, UAw, gA, Ti, Phi0}` ab. Ein fester, nicht
trainierbarer Decoder setzt diese Parameter in die Gleichung oben ein und berechnet
daraus `Q_hat`, die Modellvorhersage. Trainiert wird ausschließlich über den
Huber-Verlust zwischen `Q_hat` und dem gemessenen Wärmebedarf `Q_h`. Die Physik ist
damit kein Verlustterm, sondern die Berechnungsstruktur des Modells selbst -- das
Netz kann keine physikalisch unsinnigen Zusammenhänge lernen.

## Funktionsweise des Tools

Das Tool besteht aus zwei Modulen:

**`pcnn_model.py`** -- das eigentliche Modell.

- `PCNNModel().fit(Q_h, T_a, W_s, I_g)` trainiert das Netz und gibt ein `PCNNResult`
  zurück (identifizierte Parameter UA0, UAw, gA, Ti, Phi0 sowie RMSE, MAE, R²,
  CV-RMSE, Bias auf einer Validierungsperiode).
- `.predict(T_a, W_s, I_g)` berechnet Q_hat für neue Wetterdaten.
- `.energy_balance(T_a, W_s, I_g)` zerlegt den Wärmebedarf einer Periode in seine vier
  physikalischen Anteile (Transmission, Wind/Infiltration, solare und interne Gewinne)
  -- die Energieverteilung des Gebäudes.

**`weather_sources.py`** -- Wetterdaten-Beschaffung, unabhängig vom Modell nutzbar.

- `list_meteoswiss_stations()`, `nearest_station(lat, lon)`, `fetch_meteoswiss_hourly(...)`
  für MeteoSchweiz-Stationsdaten.
- `fetch_openmeteo_hourly(lat, lon, start, end)` für Open-Meteo-Daten.

## Die zwei Beispiel-Notebooks

Beide Notebooks nutzen dieselbe Beispiel-Heizdaten-Datei (`data/heizdaten_beispiel.csv`)
und denselben `pcnn_model.py`; sie unterscheiden sich ausschließlich in der Wetterdatenquelle:

| | `example_01_meteoswiss.ipynb` | `example_02_openmeteo.ipynb` |
|---|---|---|
| Wetterquelle | MeteoSchweiz (SwissMetNet) | Open-Meteo |
| Datentyp | echte Stationsmessung | Reanalyse-/Modelldaten |
| Geografische Abdeckung | nur Schweiz | weltweit |
| Stationswahl | per Kürzel (z. B. `"STG"`) **oder** automatisch über die nächstgelegene Station zu gegebenen Koordinaten | keine Stationswahl -- nur Koordinaten |
| Wann sinnvoll | Gebäude in der Schweiz, wenn eine echte Messstation in der Nähe genutzt werden soll | Gebäude außerhalb der Schweiz, oder wenn eine einfache Koordinaten-Abfrage reicht |

Beide Notebooks führen dieselben Schritte aus: Heizdaten laden, Wetterdaten laden,
zusammenführen, `PCNNModel().fit(...)` aufrufen, Ergebnis und Energieverteilung ausgeben.

## Format der Heizdaten

`PCNNModel.fit()` benötigt vier gleich lange, stündliche Zeitreihen: Wärmebedarf sowie
die drei Wettergrößen Ta, Ws, Ig. Die Heizdaten selbst müssen als CSV mit mindestens
zwei Spalten vorliegen:

- eine **Zeitstempel-Spalte** (mit `pandas.to_datetime` parsebar, z. B. `2026-03-01 14:00:00`)
- eine **Wärmeleistungs-Spalte** in Watt

```
timestamp,Waermeleistung_gemessen
2026-03-01 00:00:00,1842.3
2026-03-01 00:01:00,1798.1
...
```

Liegen die Daten nicht stündlich vor (z. B. minütlich, wie im mitgelieferten Beispiel),
resamplen beide Notebooks automatisch auf Stundenmittel (`.resample("1h").mean()`),
bevor sie mit den Wetterdaten zusammengeführt werden. Spaltenname und Dateipfad werden
in der Konfigurationszelle jedes Notebooks über `QCOL` bzw. `DATA_PATH` eingestellt --
der Rest des Notebooks muss dafür nicht angepasst werden.

## Beispieldaten

`data/heizdaten_beispiel.csv` enthält reale, im Rahmen der Masterarbeit erfasste
Wärmeleistungsdaten (`Waermeleistung`, `Waermeleistung_gemessen`) eines Wohngebäudes in
St. Gallen, minütlich, Februar bis Mai 2026. Es ist eine auf die zwei tatsächlich vom
Tool genutzten Spalten reduzierte Version von `Data_processed.csv`; die übrigen rund
20 Messgrößen der Originaldatei (Zonentemperaturen, Speichertemperaturen, Vor-/Rücklauf,
Massenströme etc.) sind für dieses Tool nicht relevant und wurden entfernt.

## Installation

```bash
git clone https://github.com/sosorio309-maker/pcnn-tool.git
cd pcnn-tool
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Nutzung

```bash
jupyter notebook
```

`example_01_meteoswiss.ipynb` oder `example_02_openmeteo.ipynb` öffnen, die
Konfigurationszelle am Anfang anpassen (Koordinaten bzw. Stationskürzel, Zeitraum,
Datenpfad) und der Reihe nach ausführen. Für eigene Daten `pcnn_model.py` direkt
importieren:

```python
from pcnn_model import PCNNModel, print_pcnn_result

model = PCNNModel()
result = model.fit(Q_h, T_a, W_s, I_g)
print_pcnn_result(result)
```

## Struktur

```
pcnn-tool/
├── pcnn_model.py                   # PCNN-Modell (Encoder/Decoder, Fit, Predict, Energieverteilung)
├── weather_sources.py              # Wetterdaten: MeteoSchweiz und Open-Meteo
├── example_01_meteoswiss.ipynb     # Beispiel mit MeteoSchweiz-Stationsdaten
├── example_02_openmeteo.ipynb      # Beispiel mit Open-Meteo-Koordinatendaten
├── data/
│   └── heizdaten_beispiel.csv      # reale Beispiel-Heizdaten (reduziert auf 2 Spalten)
└── requirements.txt
```

## Danksagung

Dieses Tool ist im Rahmen der oben verlinkten Masterarbeit an der FH Vorarlberg
entstanden, in Kooperation mit der Energieagentur St. Gallen.

## Autor

Santiago Rojo Osorio
FH Vorarlberg -- Nachhaltige Energiesysteme (MSc.)
LinkedIn: https://www.linkedin.com/in/srojosorio/
Masterarbeit: https://opus.fhv.at/frontdoor/index/index/docId/7508
