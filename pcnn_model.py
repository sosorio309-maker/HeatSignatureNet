"""
pcnn_model.py
==============

Physically Consistent Neural Network (PCNN) zur thermischen Parameteridentifikation
von Wohngebäuden.

Physikalischer Hintergrund
---------------------------
Alle Berechnungen basieren auf der Energy-Signature-Gleichung (Rojo Osorio, 2026,
Masterarbeit FH Vorarlberg, Abschnitt 2.1, Gl. 2.1):

    Q_h(t) = UA0 * (Ti - Ta(t)) + UAw * Ws(t) * (Ti - Ta(t)) - gA * Ig(t) - Phi0

mit
    Q_h   [W]        Wärmebedarf des Heizsystems (Zielgröße)
    UA0   [W/K]       Transmissionsleitwert der Gebäudehülle
    Ti    [°C]        Innentemperatur / Basistemperatur des Heizsystems
    Ta    [°C]        Außentemperatur
    UAw   [W/(K·m/s)] windabhängiger Leitwert (Infiltration)
    Ws    [m/s]        Windgeschwindigkeit
    gA    [m²]         solare Apertur (Energiedurchlassgrad x Fensterfläche)
    Ig    [W/m²]        Globalstrahlung
    Phi0  [W]         interne Wärmegewinne (Personen, Geräte, Beleuchtung)

Das PCNN setzt diese Gleichung als festen, nicht trainierbaren "Decoder" ein
(Abschnitt 2.2.3). Ein trainierbarer Encoder (MLP) bildet die Wetter-Zeitreihen
[Ta, Ws, Ig] (mit einem Lag-Fenster vergangener Zeitschritte) auf den
Parametervektor theta = {UA0, UAw, gA, Ti, Phi0} ab. Diese Parameter werden pro
Zeitschritt neu geschätzt, deterministisch in die Physikgleichung eingesetzt und
ergeben Q_hat. Trainiert wird ausschließlich über den Huber-Loss zwischen Q_hat
und dem gemessenen/abgeleiteten Q_h -- die Physik selbst ist keine Verlustfunktion,
sondern die Berechnungsstruktur des Modells.

Nutzung (Kurzfassung)
----------------------
    from pcnn_model import PCNNModel, print_pcnn_result

    model = PCNNModel()
    result = model.fit(Q_h, T_a, W_s, I_g)     # Q_h, T_a, W_s, I_g: stündliche Zeitreihen
    print_pcnn_result(result)

    energie = model.energy_balance(T_a_periode, W_s_periode, I_g_periode)
    # -> Aufteilung des Wärmebedarfs in Transmission / Wind / Solar / Intern

Siehe README.md für ein vollständiges Beispiel und die beiden Demo-Notebooks
(example_01_synthetic_data.ipynb, example_02_scenario_data.ipynb).

Autor: Santiago Rojo Osorio
Quelle: Masterarbeit "Thermische Gebäudemodellierung zur Parameteridentifikation
        und energetischen Charakterisierung von Wohngebäuden", FH Vorarlberg, 2026.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────
def _huber_delta(residuen, floor=50.0):
    """Datengetriebener Huber-Schwellwert delta = max(1.4826*MAD(r), floor)."""
    r = np.asarray(residuen, dtype=float)
    mad = np.median(np.abs(r - np.median(r)))
    return max(1.4826 * mad, floor)


def _metrics(Q_true, Q_pred):
    """RMSE, MAE, R², CV(RMSE) [%] und Bias zwischen Messung und Vorhersage."""
    Q_true = np.asarray(Q_true, dtype=float)
    Q_pred = np.asarray(Q_pred, dtype=float)
    r = Q_true - Q_pred
    rmse = float(np.sqrt(np.mean(r**2)))
    mae = float(np.mean(np.abs(r)))
    denom = np.sum((Q_true - np.mean(Q_true)) ** 2)
    r2 = float(1 - np.sum(r**2) / denom) if denom > 0 else float("nan")
    cvrmse = float(rmse / np.mean(Q_true) * 100) if np.mean(Q_true) != 0 else float("nan")
    bias = float(np.mean(r))
    return rmse, mae, r2, cvrmse, bias


def make_features(T_a, W_s, I_g, lag=6):
    """
    Baut die Lag-Feature-Matrix [Ta(t), Ws(t), Ig(t), Ta(t-1), ...] für die
    letzten `lag` Zeitschritte. Die ersten `lag` Zeilen der Eingangsreihen werden
    verworfen (kein vollständiges Fenster verfügbar).

    Returns
    -------
    X : np.ndarray, shape (N - lag, 3*lag), dtype float32
    """
    T_a = np.asarray(T_a, dtype=np.float32)
    W_s = np.asarray(W_s, dtype=np.float32)
    I_g = np.asarray(I_g, dtype=np.float32)
    feats = []
    for l in range(lag):
        feats.append(np.roll(T_a, l))
        feats.append(np.roll(W_s, l))
        feats.append(np.roll(I_g, l))
    X = np.stack(feats, axis=1)[lag:]
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Netzarchitektur: trainierbarer Encoder + fester physikalischer Decoder
# ─────────────────────────────────────────────────────────────────────────────
class _PCNNNet(nn.Module):
    """
    Encoder (trainierbar, 3 Hidden-Layer, Tanh) -> theta = [UA0, UAw, gA, Ti, Phi0]
    Decoder (physikalisch, fest) -> Q_hat nach Gleichung (2.1).

    Die sigmoid-skalierten Ausgaben halten alle Parameter innerhalb physikalisch
    plausibler Grenzen (siehe *_scale / Ti_lo / Ti_hi Argumente).
    """

    def __init__(self, n_features, UA0_scale=200.0, UA0_offset=10.0,
                 UAw_scale=50.0, gA_scale=20.0, Ti_lo=18.0, Ti_hi=22.0,
                 Phi0_scale=2000.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 5),  # -> [UA0, UAw, gA, Ti, Phi0]
        )
        self.UA0_scale, self.UA0_offset = UA0_scale, UA0_offset
        self.UAw_scale = UAw_scale
        self.gA_scale = gA_scale
        self.Ti_lo, self.Ti_span = Ti_lo, (Ti_hi - Ti_lo)
        self.Phi0_scale = Phi0_scale

    def forward(self, x, T_a, W_s, I_g):
        raw = self.encoder(x)
        UA0 = torch.sigmoid(raw[:, 0]) * self.UA0_scale + self.UA0_offset
        UAw = torch.sigmoid(raw[:, 1]) * self.UAw_scale
        gA = torch.sigmoid(raw[:, 2]) * self.gA_scale
        Ti = torch.sigmoid(raw[:, 3]) * self.Ti_span + self.Ti_lo
        Phi0 = torch.sigmoid(raw[:, 4]) * self.Phi0_scale

        dT = Ti - T_a
        Q_hat = UA0 * dT + UAw * W_s * dT - gA * I_g - Phi0
        return Q_hat, UA0, UAw, gA, Ti, Phi0


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PCNNResult:
    UA0: float = np.nan       # W/K        Transmissionsleitwert
    UAw: float = np.nan       # W/(K·m/s)  windabhängiger Leitwert
    gA: float = np.nan        # m²         solare Apertur
    Ti: float = np.nan        # °C         Innentemperatur (Basistemperatur)
    Phi0: float = np.nan      # W          interne Gewinne
    rmse: float = np.nan      # W          (auf Validierungsmenge)
    mae: float = np.nan       # W
    r2: float = np.nan
    cvrmse: float = np.nan    # %
    bias: float = np.nan      # W
    Q_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_true: np.ndarray = field(default_factory=lambda: np.array([]))
    n_train: int = 0
    n_val: int = 0
    success: bool = False
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Modellklasse
# ─────────────────────────────────────────────────────────────────────────────
class PCNNModel:
    """
    Physically Consistent Neural Network für die thermische Parameteridentifikation
    eines Gebäudes aus Wärmebedarfs- und Wetterdaten.

    Parameters
    ----------
    lag : int
        Anzahl vergangener Stunden, die als Feature-Fenster verwendet werden.
    epochs, batch_size, lr : Trainings-Hyperparameter (Adam-Optimierer).
    validation_fraction : float
        Anteil der Daten, der als chronologischer Holdout verwendet wird, wenn
        `fit()` keine separate Validierungsperiode erhält.
    patience : int
        Geduld des ReduceLROnPlateau-Schedulers auf dem Validierungsverlust.
    phi0_penalty : float
        Gewicht der leichten Regularisierung auf Phi0 (verhindert, dass das Netz
        alle Verluste in konstante interne Gewinne "versteckt").
    seed : int
        Zufalls-Seed für PyTorch/NumPy (Reproduzierbarkeit).
    """

    def __init__(self, lag=6, epochs=1000, batch_size=64, lr=1e-3,
                 validation_fraction=0.2, patience=20, phi0_penalty=0.005,
                 seed=42, verbose=True):
        self.lag = lag
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.phi0_penalty = phi0_penalty
        self.seed = seed
        self.verbose = verbose

        self.net = None
        self.X_mean = None
        self.X_std = None
        self.result_ = None

    # -- internes Feature-Preprocessing -----------------------------------
    def _features(self, T_a, W_s, I_g, fit_scaler=False):
        X = make_features(T_a, W_s, I_g, self.lag)
        if fit_scaler:
            self.X_mean = X.mean(0)
            self.X_std = X.std(0) + 1e-8
        return (X - self.X_mean) / self.X_std

    # -- Training -----------------------------------------------------------
    def fit(self, Q_h, T_a, W_s, I_g, Q_val=None, Ta_val=None, Ws_val=None, Ig_val=None):
        """
        Trainiert das PCNN auf stündlichen Zeitreihen.

        Parameters
        ----------
        Q_h, T_a, W_s, I_g : array-like, gleiche Länge N
            Wärmebedarf [W], Außentemperatur [°C], Windgeschwindigkeit [m/s],
            Globalstrahlung [W/m²] der Trainingsperiode.
        Q_val, Ta_val, Ws_val, Ig_val : array-like, optional
            Separate Validierungsperiode. Fehlt sie, werden die letzten
            `validation_fraction` der übergebenen Reihen chronologisch als
            Holdout abgetrennt (kein zufälliges Mischen -- Zeitreihen bleiben
            zusammenhängend).

        Returns
        -------
        PCNNResult
            Identifizierte Parameter (Median über die Validierungsmenge) und
            Gütekennwerte. Auch über `self.result_` zugänglich.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Q_h = np.asarray(Q_h, dtype=np.float32)
        T_a = np.asarray(T_a, dtype=np.float32)
        W_s = np.asarray(W_s, dtype=np.float32)
        I_g = np.asarray(I_g, dtype=np.float32)

        if Q_val is None:
            n = len(Q_h)
            n_tr = int(n * (1 - self.validation_fraction))
            if n_tr <= self.lag or (n - n_tr) <= self.lag:
                raise ValueError(
                    "Zeitreihe zu kurz für lag=%d und validation_fraction=%.2f"
                    % (self.lag, self.validation_fraction)
                )
            Q_val, Ta_val, Ws_val, Ig_val = Q_h[n_tr:], T_a[n_tr:], W_s[n_tr:], I_g[n_tr:]
            Q_h, T_a, W_s, I_g = Q_h[:n_tr], T_a[:n_tr], W_s[:n_tr], I_g[:n_tr]
        else:
            Q_val = np.asarray(Q_val, dtype=np.float32)
            Ta_val = np.asarray(Ta_val, dtype=np.float32)
            Ws_val = np.asarray(Ws_val, dtype=np.float32)
            Ig_val = np.asarray(Ig_val, dtype=np.float32)

        lag = self.lag

        X_tr = self._features(T_a, W_s, I_g, fit_scaler=True)
        y_tr = Q_h[lag:]
        n_tr = min(len(X_tr), len(y_tr))
        X_tr, y_tr = X_tr[:n_tr], y_tr[:n_tr]

        X_va = self._features(Ta_val, Ws_val, Ig_val, fit_scaler=False)
        y_va = Q_val[lag:]
        n_va = min(len(X_va), len(y_va))
        X_va, y_va = X_va[:n_va], y_va[:n_va]

        Ta_tr_t = torch.tensor(T_a[lag:lag + n_tr])
        Ws_tr_t = torch.tensor(W_s[lag:lag + n_tr])
        Ig_tr_t = torch.tensor(I_g[lag:lag + n_tr])
        Ta_va_t = torch.tensor(Ta_val[lag:lag + n_va])
        Ws_va_t = torch.tensor(Ws_val[lag:lag + n_va])
        Ig_va_t = torch.tensor(Ig_val[lag:lag + n_va])

        X_tr_t = torch.tensor(X_tr)
        y_tr_t = torch.tensor(y_tr)
        X_va_t = torch.tensor(X_va)
        y_va_t = torch.tensor(y_va)

        delta = _huber_delta(y_tr)
        criterion = nn.HuberLoss(delta=delta)

        self.net = _PCNNNet(n_features=X_tr.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t, Ta_tr_t, Ws_tr_t, Ig_tr_t),
            batch_size=self.batch_size, shuffle=True,
        )

        best_state, best_loss = None, np.inf

        if self.verbose:
            print(f"┌─ PCNN -- Encoder/Decoder, Adam, Huber-Loss (delta={delta:.1f}) ──────")
            print(f"│  n_train={n_tr}h  n_val={n_va}h  lag={lag}h  epochs={self.epochs}")
            print(f"└──────────────────────────────────────────────────────────────────")

        for epoch in range(self.epochs):
            self.net.train()
            for xb, yb, ta, ws, ig in loader:
                optimizer.zero_grad()
                Q_hat, _, _, _, _, Phi0_b = self.net(xb, ta, ws, ig)
                loss = criterion(Q_hat, yb) + self.phi0_penalty * Phi0_b.mean()
                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                Q_val_hat, *_ = self.net(X_va_t, Ta_va_t, Ws_va_t, Ig_va_t)
                val_loss = criterion(Q_val_hat, y_va_t).item()
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if self.verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}  val_loss={val_loss:.1f}")

        self.net.load_state_dict(best_state)
        self.net.eval()

        with torch.no_grad():
            Q_pred_t, UA0, UAw, gA, Ti, Phi0 = self.net(X_va_t, Ta_va_t, Ws_va_t, Ig_va_t)
        Q_pred = Q_pred_t.numpy()
        rmse, mae, r2, cvrmse, bias = _metrics(y_va, Q_pred)

        self.result_ = PCNNResult(
            UA0=UA0.mean().item(), UAw=UAw.mean().item(), gA=gA.mean().item(),
            Ti=Ti.mean().item(), Phi0=Phi0.mean().item(),
            rmse=rmse, mae=mae, r2=r2, cvrmse=cvrmse, bias=bias,
            Q_pred=Q_pred, Q_true=y_va, n_train=n_tr, n_val=n_va,
            success=True, message="OK",
        )
        return self.result_

    # -- Vorhersage -----------------------------------------------------------
    def predict(self, T_a, W_s, I_g):
        """
        Berechnet Q_hat für beliebige neue Wetterdaten mit dem trainierten Netz.
        Achtung: Das Netz schätzt weiterhin pro Zeitschritt eigene [UA0, UAw, gA,
        Ti, Phi0] -- für die *gemittelten* Parameter des trainierten Modells
        siehe `self.result_`, für die physikalische Aufteilung `energy_balance()`.
        """
        if self.net is None:
            raise RuntimeError("Modell wurde noch nicht trainiert (zuerst fit() aufrufen).")
        T_a = np.asarray(T_a, dtype=np.float32)
        W_s = np.asarray(W_s, dtype=np.float32)
        I_g = np.asarray(I_g, dtype=np.float32)
        X = self._features(T_a, W_s, I_g, fit_scaler=False)
        n = len(X)
        Ta_t = torch.tensor(T_a[self.lag:self.lag + n])
        Ws_t = torch.tensor(W_s[self.lag:self.lag + n])
        Ig_t = torch.tensor(I_g[self.lag:self.lag + n])
        self.net.eval()
        with torch.no_grad():
            Q_hat, *_ = self.net(torch.tensor(X), Ta_t, Ws_t, Ig_t)
        return Q_hat.numpy()

    # -- Energieverteilung ------------------------------------------------
    def energy_balance(self, T_a, W_s, I_g, dt_h=1.0):
        """
        Zerlegt den Wärmebedarf einer Wetterperiode in die vier physikalischen
        Anteile aus Gleichung (2.1) -- die "Energieverteilung" -- auf Basis der
        über `fit()` identifizierten (gemittelten) Parameter UA0, UAw, gA, Ti, Phi0:

            Q_transmission = UA0 * (Ti - Ta)
            Q_wind         = UAw * Ws * (Ti - Ta)
            Q_solar        = gA * Ig            (Gewinn)
            Q_internal     = Phi0               (Gewinn, konstant)

        Parameters
        ----------
        T_a, W_s, I_g : array-like, gleiche Länge
            Wetterdaten der Periode, für die die Energiebilanz berechnet werden soll
            (z. B. derselbe Zeitraum wie die Validierungsmenge, oder ein neuer Monat).
        dt_h : float
            Zeitschrittweite in Stunden (Default 1.0 für stündliche Daten).

        Returns
        -------
        dict mit:
            n_hours                      Anzahl Stunden
            Q_transmission_kWh, Q_wind_kWh, Q_solar_kWh, Q_internal_kWh, Q_h_kWh
                                          Summen der vier Anteile + Netto-Wärmebedarf
            anteil_transmission_pct, anteil_wind_pct
                                          Anteile an den Verlusten (Trans+Wind = 100 %)
            anteil_solar_pct, anteil_internal_pct
                                          Anteile an den Gewinnen (Solar+Intern = 100 %)
            Q_h_W                        Zeitreihe des Netto-Wärmebedarfs [W]
        """
        if self.result_ is None:
            raise RuntimeError("Modell wurde noch nicht trainiert (zuerst fit() aufrufen).")

        T_a = np.asarray(T_a, dtype=float)
        W_s = np.asarray(W_s, dtype=float)
        I_g = np.asarray(I_g, dtype=float)
        n = min(len(T_a), len(W_s), len(I_g))
        T_a, W_s, I_g = T_a[:n], W_s[:n], I_g[:n]

        r = self.result_
        dT = r.Ti - T_a
        Q_trans = r.UA0 * dT
        Q_wind = r.UAw * W_s * dT
        Q_solar = r.gA * I_g
        Q_internal = np.full(n, r.Phi0)
        Q_h_W = Q_trans + Q_wind - Q_solar - Q_internal  # kein max() -- vorzeichenbehaftet

        to_kwh = lambda a: float(np.sum(a)) * dt_h / 1000.0
        Q_transmission_kWh = to_kwh(Q_trans)
        Q_wind_kWh = to_kwh(Q_wind)
        Q_solar_kWh = to_kwh(Q_solar)
        Q_internal_kWh = to_kwh(Q_internal)
        Q_h_kWh = to_kwh(Q_h_W)

        verluste = Q_transmission_kWh + Q_wind_kWh
        gewinne = Q_solar_kWh + Q_internal_kWh
        pct = lambda x, base: float(x / base * 100) if base else float("nan")

        return dict(
            n_hours=n,
            Q_transmission_kWh=Q_transmission_kWh,
            Q_wind_kWh=Q_wind_kWh,
            Q_solar_kWh=Q_solar_kWh,
            Q_internal_kWh=Q_internal_kWh,
            Q_h_kWh=Q_h_kWh,
            anteil_transmission_pct=pct(Q_transmission_kWh, verluste),
            anteil_wind_pct=pct(Q_wind_kWh, verluste),
            anteil_solar_pct=pct(Q_solar_kWh, gewinne),
            anteil_internal_pct=pct(Q_internal_kWh, gewinne),
            Q_h_W=Q_h_W,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Formatierte Ausgabe
# ─────────────────────────────────────────────────────────────────────────────
def print_pcnn_result(r: PCNNResult):
    print("\n══════════════════════════════════════════════")
    print("  PCNN -- Ergebnisse")
    print("══════════════════════════════════════════════")
    print(f"  Status       : {r.message}")
    print("──────────────────────────────────────────────")
    print(f"  UA0          = {r.UA0:.2f}   W/K")
    print(f"  UAw          = {r.UAw:.4f}  W/(K·m/s)")
    print(f"  gA           = {r.gA:.4f}  m²")
    print(f"  Ti           = {r.Ti:.2f}   °C   (Innentemperatur)")
    print(f"  Φ0           = {r.Phi0:.1f}   W    (interne Gewinne)")
    print("──────────────────────────────────────────────")
    print(f"  RMSE         = {r.rmse:.1f}   W    (Validierung, n={r.n_val}h)")
    print(f"  MAE          = {r.mae:.1f}   W")
    print(f"  R²           = {r.r2:.4f}")
    print(f"  CV-RMSE      = {r.cvrmse:.2f}   %")
    print(f"  Bias         = {r.bias:+.1f}   W")
    print("══════════════════════════════════════════════")


def print_energy_balance(en: dict, label: str = ""):
    W = 60
    print(f"\n{'=' * W}")
    title = f"Energieverteilung -- {label}" if label else "Energieverteilung"
    print(f"  {title}")
    print(f"{'=' * W}")
    print(f"  Zeitraum          : {en['n_hours']} h  ({en['n_hours'] / 24:.1f} Tage)")
    print(f"{'-' * W}")
    print("  VERLUSTE")
    print(f"    Transmission    : {en['Q_transmission_kWh']:>9.1f}  kWh  "
          f"({en['anteil_transmission_pct']:5.1f} % der Verluste)")
    print(f"    Wind/Infiltr.   : {en['Q_wind_kWh']:>9.1f}  kWh  "
          f"({en['anteil_wind_pct']:5.1f} % der Verluste)")
    print(f"{'-' * W}")
    print("  GEWINNE")
    print(f"    Solar (gA*Ig)   : {en['Q_solar_kWh']:>9.1f}  kWh  "
          f"({en['anteil_solar_pct']:5.1f} % der Gewinne)")
    print(f"    Intern (Phi0)   : {en['Q_internal_kWh']:>9.1f}  kWh  "
          f"({en['anteil_internal_pct']:5.1f} % der Gewinne)")
    print(f"{'-' * W}")
    print(f"  NETTO Q_h         : {en['Q_h_kWh']:>9.1f}  kWh")
    print(f"{'=' * W}")


# ─────────────────────────────────────────────────────────────────────────────
# Selbsttest mit synthetischen Daten (siehe example_01_synthetic_data.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)

    N = 24 * 60  # 60 Tage stündlich
    t = np.arange(N)

    T_a = -2 + 8 * np.sin(2 * np.pi * (t - 8 * 24) / (24 * 60)) \
        + 5 * np.sin(2 * np.pi * t / 24 - np.pi) + 1.0 * np.random.randn(N)
    W_s = np.abs(2 + 1.5 * np.random.randn(N))
    I_g = np.maximum(0, 400 * np.sin(np.pi * (t % 24) / 24 - np.pi / 6)) \
        * (0.5 + 0.5 * np.random.rand(N))

    # Wahre Parameter (zum Vergleich mit der Identifikation)
    UA0_t, UAw_t, gA_t, Ti_t, Phi0_t = 180.0, 6.0, 5.0, 20.0, 300.0
    dT_t = Ti_t - T_a
    Q_h = np.maximum(
        0,
        UA0_t * dT_t + UAw_t * W_s * dT_t - gA_t * I_g - Phi0_t
        + 80 * np.random.randn(N),
    )

    print("Selbsttest: PCNN auf synthetischen Daten mit bekannten Parametern\n")
    model = PCNNModel(epochs=400, verbose=True)
    result = model.fit(Q_h, T_a, W_s, I_g)
    print_pcnn_result(result)

    print("\n── Parametervergleich (wahr vs. identifiziert) ─────────────────")
    print(f"  {'':10s}  {'Wahr':>10s}  {'PCNN':>10s}")
    print(f"  {'UA0 [W/K]':10s}  {UA0_t:10.2f}  {result.UA0:10.2f}")
    print(f"  {'UAw':10s}  {UAw_t:10.4f}  {result.UAw:10.4f}")
    print(f"  {'gA [m²]':10s}  {gA_t:10.4f}  {result.gA:10.4f}")
    print(f"  {'Ti [°C]':10s}  {Ti_t:10.2f}  {result.Ti:10.2f}")
    print(f"  {'Phi0 [W]':10s}  {Phi0_t:10.2f}  {result.Phi0:10.2f}")

    en = model.energy_balance(T_a[-model.result_.n_val:], W_s[-model.result_.n_val:],
                               I_g[-model.result_.n_val:])
    print_energy_balance(en, label="Selbsttest (Validierungsperiode)")
