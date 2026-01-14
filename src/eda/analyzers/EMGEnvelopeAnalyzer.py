import numpy as np
import mne
from scipy.signal import butter, filtfilt
from .base_analyzer import BaseAnalyzer

class EMGEnvelopeAnalyzer(BaseAnalyzer):
    """
    Modular EMG Artifact Detection, this analyzer computes:
     * Contamination Rate (%): Percentage of the trial spent above the threshold.
     * Peak-to-Baseline Ratio (Local SNR):  Max(Envelope_Task) / Mean(Envelope_Baseline), how "loud" the muscle 
        was compared to that specific subject's rest level. It normalizes the subject's "noise floor.
    * Envelope Integral (Area under the curve): Total energy of the envelope during the task window.
    
    """
    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"], 
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 tmin=-1.0, tmax=3.0, 
                 lp_cut=40.0,
                 threshold_multiplier=5.0,
                 window_duration_s=0.05,
                 contamination_percent=50.0,
                 baseline_window=(-0.5, 0.0),
                 **kwargs):
        super().__init__(picks, **kwargs)
        self.params = {
            'bp': bandpass_filter, 'notch': notch_filter, 'tmin': tmin, 'tmax': tmax,
            'lp': lp_cut, 'mult': threshold_multiplier, 'win_s': window_duration_s,
            'limit': contamination_percent, 'base_win': baseline_window
        }

    def analyze(self, raw: mne.io.BaseRaw):
        # 1. Shadow Preprocessing on Raw
        local_raw = raw.copy().pick(self.picks)
        local_raw.filter(self.params['bp'][0], self.params['bp'][1], picks=self.picks, verbose=False)
        local_raw.notch_filter(freqs=self.params['notch'], picks=self.picks, verbose=False)
        
        # 2. Epoching
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        epochs = mne.Epochs(local_raw, events, tmin=self.params['tmin'], 
                            tmax=self.params['tmax'], baseline=None, 
                            preload=True, verbose=False)
        
        segments = epochs.get_data() # (n_epochs, n_channels, n_samples)
        sfreq = raw.info['sfreq']

        # 3. Modular Processing Steps
        envelope = self._get_envelope(segments, sfreq)
        thresholds = self._calculate_thresholds(envelope, sfreq)
        
        # 4. Vectorized Sliding Window Analysis
        # Reshape data to calculate all window means at once (Performance Fix)
        win_size = int(self.params['win_s'] * sfreq)
        n_epochs, n_chans, n_samples = envelope.shape
        n_windows = n_samples // win_size
        
        # Truncate to fit exact windows and reshape to (Epochs, Chans, Windows, SamplesPerWin)
        truncated_env = envelope[:, :, :n_windows * win_size]
        reshaped_env = truncated_env.reshape(n_epochs, n_chans, n_windows, win_size)
        
        # Calculate mean of every window across all epochs/channels in one operation
        window_means = np.mean(reshaped_env, axis=-1) 
        
        # Compare against thresholds (broadcasting thresholds across epochs and windows)
        # thresholds is (n_chans,), window_means is (n_epochs, n_chans, n_windows)
        active_mask = window_means > np.array(thresholds)[None, :, None]
        
        # 5. Classification Logic
        contamination_rates = (np.sum(active_mask, axis=-1) / n_windows) * 100
        
        bad_indices = []
        trial_tags = [[] for _ in range(n_epochs)]

        for i in range(n_epochs):
            for j, ch_name in enumerate(self.picks):
                rate = contamination_rates[i, j]
                if rate >= self.params['limit']:
                    trial_tags[i].append(f"{ch_name}: Contaminated ({int(rate)}%)")
            
            if trial_tags[i]:
                bad_indices.append(i)
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        event_types = events[:, 2] # The 3rd column is the ID
        return {
            "summary": (f"Sliding Window Result:\n"
                        f"Found {len(bad_indices)} contaminated trials.\n"
                        f"Threshold: {self.params['mult']}x SD of Baseline."),
            "visuals": {
                "Active Windows (%)": {ch: contamination_rates[:, j].tolist() for j, ch in enumerate(self.picks)},
                "Peak Envelope (uV)": {ch: (np.max(envelope[:, j, :], axis=-1) * 1e6).tolist() for j, ch in enumerate(self.picks)}
            },
            "threshold_lines": thresholds, 
            "artifacts": {
                "bad_indices": bad_indices,
                "bad_tags": [trial_tags[i] for i in bad_indices],
                "event_types": event_types.tolist(),
                "tmin": self.params['tmin'],
                "tmax": self.params['tmax']
            }
        }

    def _get_envelope(self, segments, sfreq):
        """Step 3: Zero-mean -> Rectify -> Low-pass."""
        # Zero-mean per segment and channel
        segments = np.abs(segments - np.mean(segments, axis=-1, keepdims=True))
        
        # 3rd order zero-phase LP Butterworth
        b, a = butter(3, self.params['lp'] / (sfreq/2), btype='low')
        return filtfilt(b, a, segments, axis=-1)

    def _calculate_thresholds(self, envelope, sfreq):
        """Step 4: SD of Baseline (Rest) segment."""
        # Map time window to indices
        # Baseline start index relative to tmin
        start_idx = int((self.params['base_win'][0] - self.params['tmin']) * sfreq)
        end_idx = int((self.params['base_win'][1] - self.params['tmin']) * sfreq)
        
        # Safety check for indices
        start_idx = max(0, start_idx)
        
        thresholds = []
        for j in range(envelope.shape[1]):
            # Use baseline data from all trials for a robust subject-specific threshold
            rest_data = envelope[:, j, start_idx:end_idx].flatten()
            sd_rest = np.std(rest_data)
            thresholds.append(sd_rest * self.params['mult'])
        return thresholds