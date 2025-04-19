import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import yaml

@dataclass
class Physiology:
    weight_kg: float = 70.0
    V_blood: float = 5.0
    V_liver: float = 1.8
    Q_liver: float = 90.0
    V_muscle: float = 29.0
    Q_muscle: float = 450.0
    V_fat: float = 18.0
    Q_fat: float = 30.0
    CLint_liver: float = 20.0

@dataclass
class Compound:
    name: str = "DrugX"
    Kp_liver: float = 10.0
    Kp_muscle: float = 3.0
    Kp_fat: float = 50.0
    k_abs: float = 1.0

@dataclass
class DosingEvent:
    type: str
    amount: float
    time: float
    duration: float = 0.0

class PBPKQSPSimulator:
    def __init__(self, phys: Physiology, cmpd: Compound, qsp_params=None):
        self.phys = phys
        self.cmpd = cmpd
        self.qsp = qsp_params
        self.names = ['Gut','Blood','Liver','Muscle','Fat']
        self.vols = [1.0, phys.V_blood, phys.V_liver, phys.V_muscle, phys.V_fat]
        self.flows = [0.0, 0.0, phys.Q_liver, phys.Q_muscle, phys.Q_fat]
        self.kps   = [1.0, 1.0, cmpd.Kp_liver, cmpd.Kp_muscle, cmpd.Kp_fat]
        self.clint = [0.0, 0.0, phys.CLint_liver, 0.0, 0.0]
        self.n_pk = len(self.names)
        self.n_qsp = 3 if qsp_params else 0

    def odes(self, t, y, events):
        inj = np.zeros(self.n_pk)
        for ev in events:
            if ev.type=='iv_bolus' and abs(t-ev.time)<self.dt/2:
                inj[1] += ev.amount/self.dt
            elif ev.type=='iv_infusion' and ev.time <= t < ev.time+ev.duration:
                inj[1] += ev.amount/ev.duration
            elif ev.type=='oral' and abs(t-ev.time)<self.dt/2:
                inj[0] += ev.amount/self.dt

        Agut, Ab, Aliv, Amusc, Afat = y[:5]
        dAgut = -self.cmpd.k_abs*Agut + inj[0]
        Cb = Ab/self.vols[1]
        dAb = self.cmpd.k_abs*Agut + inj[1]
        dAliv = dAmusc = dAfat = 0.0
        for idx, (vol, flow, kp, cl) in enumerate(zip(self.vols, self.flows, self.kps, self.clint)):
            if idx<2: continue
            Ci = y[idx]/vol
            flux = flow*(Cb - Ci/kp)
            dAb   -= flux
            if idx==2: dAliv = flux - cl*Ci
            elif idx==3: dAmusc = flux
            elif idx==4: dAfat  = flux

        dPK = [dAgut, dAb, dAliv, dAmusc, dAfat]

        if self.qsp:
            kon, koff, Rtot, kprod, kdeg = self.qsp
            Rf, Rc, M = y[5:8]
            bind = kon*Cb*Rf - koff*Rc
            dRf = -bind
            dRc = bind
            dM  = kprod*Rc - kdeg*M
            return dPK + [dRf, dRc, dM]

        return dPK

    def simulate(self, events, t_end=24.0, dt=0.1):
        y0 = [0.0]*self.n_pk
        if self.qsp:
            y0 += [self.qsp[2], 0.0, 0.0]
        self.dt = dt
        t_eval = np.arange(0, t_end+dt, dt)
        sol = solve_ivp(
            fun=lambda t,y: self.odes(t,y,events),
            t_span=(0, t_end), y0=y0, t_eval=t_eval, method='RK45'
        )
        cols = self.names.copy()
        if self.qsp:
            cols += ['Free_Receptor','Drug_Receptor_Complex','Biomarker']
        df = pd.DataFrame(sol.y.T, columns=cols)
        df.insert(0, 'Time_h', sol.t)
        return df

def compute_pk_metrics(df, comp='Blood'):
    conc = df[comp]
    time = df['Time_h']
    cmax = np.max(conc)
    tmax = time[np.argmax(conc)]
    auc = np.trapz(conc, time)
    return {'Cmax': cmax, 'Tmax': tmax, 'AUC': auc}

def load_config(path='pbpk_config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# Statistical Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from prophet import Prophet

# Tree-Based & Ensemble Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

# Regression Models
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, BayesianRidge

# Deep Learning Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def pbpk_predict(phys, cmpd, qsp_tuple, dosing_events, t_end, dt):
    sim = PBPKQSPSimulator(phys, cmpd, qsp_params=qsp_tuple)
    df = sim.simulate(dosing_events, t_end=t_end, dt=dt)
    pk = compute_pk_metrics(df, comp='Blood')
    return pk['AUC']  # Example: use AUC as the prediction

def ensemble_predict(time_series):
    preds = []
    # Holt-Winters
    try:
        model = ExponentialSmoothing(time_series)
        fit = model.fit()
        preds.append(fit.forecast(1)[0])
    except Exception:
        pass
    # ARIMA
    try:
        model = ARIMA(time_series, order=(1,1,1))
        fit = model.fit()
        preds.append(fit.forecast(1)[0])
    except Exception:
        pass
    # SARIMA
    try:
        model = SARIMAX(time_series, order=(1,1,1), seasonal_order=(1,1,1,12))
        fit = model.fit(disp=False)
        preds.append(fit.forecast(1)[0])
    except Exception:
        pass
    # VAR (requires multivariate)
    try:
        if isinstance(time_series, pd.DataFrame) and time_series.shape[1] > 1:
            model = VAR(time_series)
            fit = model.fit()
            preds.append(fit.forecast(time_series.values[-fit.k_ar:], steps=1)[0][0])
    except Exception:
        pass
    # Prophet
    try:
        df = pd.DataFrame({'ds': np.arange(len(time_series)), 'y': time_series})
        m = Prophet()
        m.fit(df)
        future = pd.DataFrame({'ds': [len(time_series)]})
        forecast = m.predict(future)
        preds.append(forecast['yhat'].values[0])
    except Exception:
        pass
    # XGBoost
    try:
        X = np.arange(len(time_series)).reshape(-1,1)
        y = np.array(time_series)
        model = XGBRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # LightGBM
    try:
        model = LGBMRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # CatBoost
    try:
        model = CatBoostRegressor(verbose=0)
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Random Forest
    try:
        model = RandomForestRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Gradient Boosting
    try:
        model = GradientBoostingRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Extra Trees
    try:
        model = ExtraTreesRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Histogram Gradient Boosting
    try:
        model = HistGradientBoostingRegressor()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # SVR
    try:
        model = SVR()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Ridge
    try:
        model = Ridge()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Lasso
    try:
        model = Lasso()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # Bayesian Ridge
    try:
        model = BayesianRidge()
        model.fit(X, y)
        preds.append(model.predict([[len(time_series)]])[0])
    except Exception:
        pass
    # LSTM
    try:
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1,1))
        X_dl = np.array([y_scaled[i-5:i] for i in range(5, len(y_scaled))])
        y_dl = y_scaled[5:]
        model = Sequential([LSTM(10, input_shape=(5,1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_dl, y_dl, epochs=10, verbose=0)
        pred = model.predict(y_scaled[-5:].reshape(1,5,1))
        preds.append(scaler.inverse_transform(pred)[0][0])
    except Exception:
        pass
    # GRU
    try:
        model = Sequential([GRU(10, input_shape=(5,1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_dl, y_dl, epochs=10, verbose=0)
        pred = model.predict(y_scaled[-5:].reshape(1,5,1))
        preds.append(scaler.inverse_transform(pred)[0][0])
    except Exception:
        pass
    # CNN
    try:
        model = Sequential([Conv1D(16, 2, activation='relu', input_shape=(5,1)), Flatten(), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_dl, y_dl, epochs=10, verbose=0)
        pred = model.predict(y_scaled[-5:].reshape(1,5,1))
        preds.append(scaler.inverse_transform(pred)[0][0])
    except Exception:
        pass
    # Average all model predictions
    return np.mean(preds) if preds else np.nan

def main():
    cfg = load_config()
    phys = Physiology(**{k: float(v) for k, v in cfg['physiology'].items()})
    cmpd = Compound(**{k: v if k == 'name' else float(v) for k, v in cfg['compound'].items()})
    qsp_tuple = tuple(float(v) for v in cfg['qsp'].values())
    dosing_events = [DosingEvent(**d) for d in cfg['dosing']]
    t_end = 24.0
    dt = 0.1

    pbpk_pred = pbpk_predict(phys, cmpd, qsp_tuple, dosing_events, t_end, dt)
    # For demonstration, use simulated PBPK output as time series
    sim = PBPKQSPSimulator(phys, cmpd, qsp_params=qsp_tuple)
    df = sim.simulate(dosing_events, t_end=t_end, dt=dt)
    blood_ts = df['Blood'].values
    ensemble_pred = ensemble_predict(blood_ts)
    final_pred = 0.95 * pbpk_pred + 0.05 * ensemble_pred
    print(f"Final Prediction: {final_pred}")

if __name__ == "__main__":
    main()
