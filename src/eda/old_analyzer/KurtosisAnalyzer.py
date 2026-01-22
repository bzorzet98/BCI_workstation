import numpy as np
import mne
from scipy.stats import kurtosis
from .base_analyzer import BaseAnalyzer

class KurtosisAnalyzer(BaseAnalyzer):
    def __init__(self, picks=None, tmin=-1.0, tmax=3.0, 
                 threshold=3.0, detrend=True,
                 log_scale=False, z_score=True, z_mad=False, **kwargs):
        super().__init__(picks, **kwargs)
        self.params = {'tmin': tmin, 'tmax': tmax, 'detrend': detrend, 'thresh': threshold,
                       'log': log_scale, 'z': z_score, 'mad': z_mad}

    def analyze(self, raw: mne.io.BaseRaw):
        local_raw = raw.copy().pick(self.picks)
        if self.params['detrend']: local_raw.filter(1.0, None, verbose=False)
        
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        epochs = mne.Epochs(local_raw, events, tmin=self.params['tmin'], tmax=self.params['tmax'], 
                            baseline=None, preload=True, verbose=False)
        
        # Calculate Metric
        raw_metric = kurtosis(epochs.get_data(), axis=-1) # (n_epochs, n_chans)
        
        # Apply Transforms
        processed_metric = self._apply_transforms(raw_metric)
        
        return self._build_report(processed_metric, "Kurtosis", epochs, events)

    def _apply_transforms(self, data):
        if self.params['log']: data = np.log10(np.maximum(data, 1e-12))
        if self.params['z']:
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
        elif self.params['mad']:
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            data = (data - med) / (mad * 1.4826 + 1e-9)
        return data

    def _build_report(self, data, name, epochs, events):
        bad_indices = set()
        tags = [[] for _ in range(len(epochs))]
        for j, ch in enumerate(self.picks):
            outliers = np.where(np.abs(data[:, j]) > self.params['thresh'])[0]
            for idx in outliers:
                bad_indices.add(idx)
                tags[idx].append(f"{ch}: high_{name}")
        
        idx_list = sorted(list(bad_indices))
        
        return {
            "summary": f"{name} Analysis: {len(idx_list)} trials flagged.",
            "visuals": {name: {ch: data[:, i].tolist() for i, ch in enumerate(self.picks)}},
            "artifacts": {"bad_indices": idx_list, "bad_tags": [tags[i] for i in idx_list],
                          "event_types": events[:, 2].tolist(),
                          "tmin": self.params['tmin'], "tmax": self.params['tmax']}
        }