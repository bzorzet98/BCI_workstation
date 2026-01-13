import numpy as np
import mne
from .base_analyzer import BaseAnalyzer

class HjorthAnalyzer(BaseAnalyzer):
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
        data = epochs.get_data()
        
        # Hjorth Math
        act = np.var(data, axis=-1)
        diff1 = np.diff(data, axis=-1)
        mob = np.sqrt(np.var(diff1, axis=-1) / (act + 1e-12))
        
        diff2 = np.diff(diff1, axis=-1)
        mob_d1 = np.sqrt(np.var(diff2, axis=-1) / (np.var(diff1, axis=-1) + 1e-12))
        comp = mob_d1 / (mob + 1e-12)

        visuals = {}
        bad_indices = set()
        tags = [[] for _ in range(len(epochs))]

        for m_data, m_name in zip([act, mob, comp], ["Activity", "Mobility", "Complexity"]):
            transformed = self._apply_transforms(m_data)
            visuals[f"Hjorth {m_name}"] = {ch: transformed[:, i].tolist() for i, ch in enumerate(self.picks)}
            
            # Flagging logic (Using transformed values)
            for j, ch in enumerate(self.picks):
                outliers = np.where(np.abs(transformed[:, j]) > self.params['thresh'])[0]
                for idx in outliers:
                    bad_indices.add(idx)
                    tags[idx].append(f"{ch}: Hjorth_{m_name}")

        idx_list = sorted(list(bad_indices))
        return {
            "summary": f"Hjorth Analysis complete. {len(idx_list)} trials flagged.",
            "visuals": visuals,
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