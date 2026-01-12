import numpy as np
import mne
from .base_analyzer import BaseAnalyzer

class FalseMIAnalyzer(BaseAnalyzer):
    def __init__(self,picks=None, 
                 emg_filter=(30.0, 100.0), 
                 notch_filter=50.0,
                 tmin=-1.0, 
                 tmax=3.0,
                 threshold_ratio=3.0,
                 threshold_amplitude_uv=50.0,
                 baseline_window=(-0.5, 0),
                 task_window=(0.5, 2.5),
                 **kwargs):
        super().__init__(picks, **kwargs)
        # Internal mini-pipeline parameters
        self.emg_filter = emg_filter # Filter to isolate EMG activity
        self.notch_filter = notch_filter # e.g., 50 or 60 Hz
        
        self.epoch_tmin = tmin
        self.epoch_tmax = tmax
        self.unit_factor = 1e6  # Convert to microvolts
        
        # Thresholds for tagging
        self.threshold_ratio = threshold_ratio
        self.threshold_amplitude_uv = threshold_amplitude_uv # in microvolts

        # Windows for baseline and task
        self.baseline_win = baseline_window
        self.task_win = task_window

    def analyze(self, raw: mne.io.BaseRaw):
        # --- Step 1: Internal Preprocessing (Shadow) ---
        local_raw = raw.copy().pick(self.picks)
        
        # High-pass filter to isolate EMG activity
        local_raw.filter(l_freq=self.emg_filter[0], 
                         h_freq=self.emg_filter[1], 
                         picks =self.picks,
                         fir_design='firwin', 
                        verbose=False)
        # Notch filter to remove power line noise
        local_raw.notch_filter(freqs=self.notch_filter,
                                picks =self.picks, 
                                verbose=False)
        
        # --- Step 2: Internal Epoching ---
        events = mne.find_events(raw, stim_channel=None, 
                                 output='onset', consecutive='increasing',
                                 min_duration=0, shortest_event=1, 
                                 mask=None, uint_cast=False, mask_type='and', 
                                 initial_event=False, verbose=None)
        
        # Create epochs with a window that includes rest
        epochs = mne.Epochs(local_raw, events, tmin=self.epoch_tmin, tmax=self.epoch_tmax, 
                            baseline=None, preload=True, verbose=False)

        # --- Step 3: CALCULO DE MÉTRICAS ---
        data = epochs.get_data() * self.unit_factor 
        times = epochs.times
        
        # Window indices
        idx_base = np.where((times >= self.baseline_win[0]) & (times <= self.baseline_win[1]))[0]
        idx_task = np.where((times >= self.task_win[0]) & (times <= self.task_win[1]))[0]
        
        # 1. Calculate variance per channel
        # Result shape: (n_trials, n_channels)
        v_base_per_ch = np.var(data[:, :, idx_base], axis=-1)
        v_task_per_ch = np.var(data[:, :, idx_task], axis=-1)

        # 2. Calculate ratio per channel
        # Result shape: (n_trials, n_channels)
        ratios_per_ch = v_task_per_ch / (v_base_per_ch + 1e-9)
        
        # 3. Peak amplitude per channel
        # Result shape: (n_trials, n_channels)
        peaks_per_ch = np.max(np.abs(data[:, :, idx_task]), axis=-1)

        # --- REDUCTION: Find the "worst" channel per trial ---
        
        # We take the MAX across channels (axis=1)
        # Result shape: (n_trials,)
        max_ratios = np.max(ratios_per_ch, axis=1)
        max_peaks = np.max(peaks_per_ch, axis=1)
        
        # Identify which channel was the worst (optional but great for forensics)
        worst_ch_idx_ratio = np.argmax(ratios_per_ch, axis=1)
        worst_ch_idx_peak = np.argmax(peaks_per_ch, axis=1)
        
        # --- Tagging Logic ---
        mask_ratio = max_ratios > self.threshold_ratio
        mask_burst = max_peaks > self.threshold_amplitude_uv

        # Bad Trials
        bad_trials = np.where(mask_ratio | mask_burst)[0].tolist()
        channel_visuals = {
                            "Variance Ratio": {ch: ratios_per_ch[:, i].tolist() for i, ch in enumerate(self.picks)},
                            "Peak Amplitude (uV)": {ch: peaks_per_ch[:, i].tolist() for i, ch in enumerate(self.picks)}
                        }

        # 2. En el etiquetado, incluimos explícitamente el nombre del canal
        trial_tags = []
        for i in range(len(max_ratios)):
            tags = []
            # Buscamos qué canales específicos superaron el umbral en este trial
            ch_ratio_fail = np.where(ratios_per_ch[i, :] > self.threshold_ratio)[0]
            ch_peak_fail = np.where(peaks_per_ch[i, :] > self.threshold_amplitude_uv)[0]
            
            for c_idx in ch_ratio_fail:
                tags.append(f"{self.picks[c_idx]}: High Var")
            for c_idx in ch_peak_fail:
                tags.append(f"{self.picks[c_idx]}: Burst")
            trial_tags.append(tags)
            
        # --- 4. Final Dictionary Construction ---
        return {
                "summary": f"Analysis Completed.\nMarked Trials: {np.sum(mask_ratio | mask_burst)}",
                "visuals": channel_visuals,
                "artifacts": {
                    "bad_trials": bad_trials,
                    "tags": trial_tags,
                    "tmin": self.epoch_tmin,
                    "tmax": self.epoch_tmax,
                }
            }
        
    def _get_worst_channel_stats(self, worst_indices, mask):
        """
        Counts how many times each channel was the "worst" one 
        among the trials that were actually flagged.
        """
        # 1. Filter indices: only keep those from trials that failed the threshold
        bad_indices = worst_indices[mask]
        
        if len(bad_indices) == 0:
            return {}

        # 2. Count occurrences of each channel index
        unique_idx, counts = np.unique(bad_indices, return_counts=True)
        
        # 3. Map indices back to channel names
        stats = {}
        for idx, count in zip(unique_idx, counts):
            ch_name = self.picks[idx]
            stats[ch_name] = int(count) # Convert to int for JSON serialization
            
        return stats
    
    def get_analyzer_info(self) -> dict:
        return {
            "analyzer_name":  self.__class__.__name__,
            "picks": self.picks,
            "params": {
                "emg_filter": self.emg_filter,
                "notch_filter": self.notch_filter,
                "epoch_tmin": self.epoch_tmin,
                "epoch_tmax": self.epoch_tmax,
                "unit_factor": self.unit_factor,
                "threshold_ratio": self.threshold_ratio,
                "threshold_amplitude_uv": self.threshold_amplitude_uv,
                "baseline_window": self.baseline_win,
                "task_window": self.task_win
            }
        }