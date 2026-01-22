import numpy as np
import mne
from ..analyzers.base_analyzer import BaseAnalyzer

class EMGBurstAnalyzer(BaseAnalyzer):
    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"], 
                 emg_filter=(30.0, 95.0), 
                 threshold_ratio=3.0,
                 threshold_uv=50.0,
                 tmin=-1.0, tmax=3.0,
                 baseline_window=(-0.5, 0),
                 task_window=(0.5, 2.5), **kwargs):
        super().__init__(picks, **kwargs)
        self.params = {
            'filt': emg_filter, 'ratio': threshold_ratio, 'uv': threshold_uv,
            'tmin': tmin, 'tmax': tmax, 'base': baseline_window, 'task': task_window
        }

    def analyze(self, raw: mne.io.BaseRaw):
        # 1. Shadow Preprocessing
        local_raw = raw.copy().pick(self.picks).filter(self.params['filt'][0], self.params['filt'][1], verbose=False)
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        epochs = mne.Epochs(local_raw, events, tmin=self.params['tmin'], tmax=self.params['tmax'], 
                            baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data() * 1e6 # to uV
        times = epochs.times
        
        # 2. Windowing
        idx_base = np.where((times >= self.params['base'][0]) & (times <= self.params['base'][1]))[0]
        idx_task = np.where((times >= self.params['task'][0]) & (times <= self.params['task'][1]))[0]
        
        # 3. Metrics
        var_base = np.var(data[:, :, idx_base], axis=-1)
        var_task = np.var(data[:, :, idx_task], axis=-1)
        ratios = var_task / (var_base + 1e-9)
        peaks = np.max(np.abs(data[:, :, idx_task]), axis=-1)

        # 4. Tagging
        bad_indices = []
        trial_tags = [[] for _ in range(len(epochs))]
        for i in range(len(epochs)):
            for j, ch in enumerate(self.picks):
                if ratios[i, j] > self.params['ratio']:
                    trial_tags[i].append(f"{ch}: Burst_Var")
                if peaks[i, j] > self.params['uv']:
                    trial_tags[i].append(f"{ch}: Burst_Peak")
            if trial_tags[i]: bad_indices.append(i)
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        event_types = events[:, 2] # The 3rd column is the ID

        return {
            "summary": f"Burst Detection: {len(bad_indices)} trials flagged.",
            "visuals": {
                "Variance Ratio": {ch: ratios[:, i].tolist() for i, ch in enumerate(self.picks)},
                "Peak Amplitude (uV)": {ch: peaks[:, i].tolist() for i, ch in enumerate(self.picks)}
            },
            "artifacts": {"bad_indices": bad_indices, "bad_tags": [trial_tags[i] for i in bad_indices],
                          "event_types": event_types.tolist(),
                          "tmin": self.params['tmin'], "tmax": self.params['tmax']}
        }