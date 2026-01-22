import numpy as np
import mne
from scipy.stats import kurtosis
from .base_analyzer import BaseAnalyzer

class EpochStabilityAnalyzer(BaseAnalyzer):
    def __init__(self, picks=None, 
                 tmin=-1.0, tmax=3.0, 
                 detrend=True, 
                 threshold_z=3.0, 
                 **kwargs):
        super().__init__(picks, **kwargs)
        self.tmin = tmin
        self.tmax = tmax
        self.detrend = detrend
        self.threshold_z = threshold_z

    def analyze(self, raw: mne.io.BaseRaw):
        # 1. Preprocesamiento mínimo
        local_raw = raw.copy().pick(self.picks)
        if self.detrend:
            local_raw.filter(l_freq=1.0, h_freq=None, verbose=False)

        # 2. Segmentación por Cues
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        epochs = mne.Epochs(local_raw, events, tmin=self.tmin, tmax=self.tmax, 
                            baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data() * 1e6 # Microvoltios
        n_epochs, n_channels, n_times = data.shape

        # 3. Cálculo de métricas (Matrices de n_epochs x n_channels)
        metrics_data = {
            "Peak-to-Peak": np.ptp(data, axis=-1),
            "Kurtosis": kurtosis(data, axis=-1),
            "Hjorth Activity": np.var(data, axis=-1)
        }
        
        # Cálculo de Hjorth Mobility y Complexity (basado en derivadas)
        diff1 = np.diff(data, axis=-1)
        var_d1 = np.var(diff1, axis=-1)
        mobility = np.sqrt(var_d1 / (metrics_data["Hjorth Activity"] + 1e-9))
        
        diff2 = np.diff(diff1, axis=-1)
        var_d2 = np.var(diff2, axis=-1)
        mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-9))
        complexity = mob_d1 / (mobility + 1e-9)
        
        metrics_data["Hjorth Mobility"] = mobility
        metrics_data["Hjorth Complexity"] = complexity

        # 4. DETERMINACIÓN DE ÉPOCAS MALAS (Lógica por Canal)
        bad_trial_indices = set()
        trial_tags = [[] for _ in range(n_epochs)]

        # Analizamos métrica por métrica
        for m_name, m_matrix in metrics_data.items():
            # SOLO evaluamos estabilidad en PtP y Kurtosis (Hjorth es informativo)
            if m_name not in ["Peak-to-Peak", "Kurtosis"]: continue
            
            # ITERAMOS POR CANAL (Independencia total)
            for j in range(n_channels):
                ch_values = m_matrix[:, j]
                
                # Calculamos la estadística propia de este canal
                mu = np.mean(ch_values)
                sigma = np.std(ch_values)
                
                # Buscamos trials donde este canal se volvió "loco"
                # Comparando contra su propio promedio
                z_scores = np.abs((ch_values - mu) / (sigma + 1e-9))
                outliers = np.where(z_scores > self.threshold_z)[0]
                
                for ep_idx in outliers:
                    bad_trial_indices.add(ep_idx)
                    ch_name = self.picks[j]
                    trial_tags[ep_idx].append(f"{ch_name}: {m_name}_outlier")

        # 5. Construcción del Reporte
        # Estructura de visuals: {Metrica: {Canal: [Valores]}}
        visuals = {}
        for m_name, m_matrix in metrics_data.items():
            visuals[m_name] = {ch: m_matrix[:, j].tolist() for j, ch in enumerate(self.picks)}

        bad_indices = sorted(list(bad_trial_indices))
        
        return {
            "summary": (f"Análisis de Estabilidad por Canal Finalizado.\n"
                        f"Trials Marcados: {len(bad_indices)} / {n_epochs}\n"
                        f"Umbral Z: {self.threshold_z} (Local por Canal)"),
            "visuals": visuals,
            "artifacts": {
                "bad_indices": bad_indices,
                "bad_tags": [trial_tags[i] for i in bad_indices],
                "tmin": self.tmin,
                "tmax": self.tmax
            }
        }