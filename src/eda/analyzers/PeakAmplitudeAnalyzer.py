import numpy as np
import mne
from .base_analyzer import BaseAnalyzer

class PeakAmplitudeAnalyzer(BaseAnalyzer):
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
        
        # Peak-to-Peak Math
        ptp_data = np.ptp(epochs.get_data() * 1e6, axis=-1) 
        transformed = self._apply_transforms(ptp_data)
        
        bad_indices = set()
        tags = [[] for _ in range(len(epochs))]
        for j, ch in enumerate(self.picks):
            outliers = np.where(np.abs(transformed[:, j]) > self.params['thresh'])[0]
            for idx in outliers:
                bad_indices.add(idx)
                tags[idx].append(f"{ch}: peak_outlier")

        idx_list = sorted(list(bad_indices))
        return {
            "summary": f"Peak Analysis: {len(idx_list)} trials flagged.",
            "visuals": {"Peak-to-Peak": {ch: transformed[:, i].tolist() for i, ch in enumerate(self.picks)}},
            "artifacts": {"bad_indices": idx_list, "bad_tags": [tags[i] for i in idx_list],
                          "tmin": self.params['tmin'], "tmax": self.params['tmax']}
        }

    def _apply_transforms(self, data):
        if self.params['log']: data = np.log10(np.maximum(data, 1e-12))
        if self.params['z']:
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
        elif self.params['mad']:
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            data = (data - med) / (mad * 1.4826 + 1e-9)
        return data