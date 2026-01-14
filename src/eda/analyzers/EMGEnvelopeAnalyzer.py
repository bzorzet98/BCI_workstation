import numpy as np
import mne
from scipy.signal import butter, filtfilt
from .base_analyzer import BaseAnalyzer

class EMGEnvelopeAnalyzer(BaseAnalyzer):
    """
    Advanced Modular EMG Artifact Detection Framework.
    
    This analyzer identifies muscle contamination in EEG trials by extracting 
    physiological features from the EMG envelope. It uses an adaptive 
    threshold based on the subject's noise floor at rest.

    Analyzed Metrics:
    ----------------
    1. Contamination Rate (%): Total percentage of the trial duration where 
       the EMG envelope exceeds the adaptive threshold.
    2. Local SNR (Peak-to-Baseline): Ratio between the maximum task amplitude 
       and the mean rest amplitude. Quantifies the intensity of the contraction.
    3. Envelope Integral: The cumulative energy (Area Under the Curve) of the 
       rectified/filtered signal during the task window.
    4. Sustained Duration (ms): The longest consecutive time window where 
       EMG activity remained above the threshold. Critical for discriminating 
       muscle tension from short ECG spikes or electrical pops.
    """

    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"], 
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 tmin=-1.0, tmax=3.0, 
                 lp_cut=40.0,
                 threshold_multiplier=5.0,
                 window_duration_s=0.05,
                 contamination_cutoff_percent=50.0,
                 baseline_window=(-0.5, 0.0),
                 **kwargs):
        super().__init__(picks, **kwargs)
        self.params = {
            'bp': bandpass_filter, 'notch': notch_filter, 'tmin': tmin, 'tmax': tmax,
            'lp': lp_cut, 'mult': threshold_multiplier, 'win_s': window_duration_s,
            'limit': contamination_cutoff_percent, 'base_win': baseline_window
        }

    def analyze(self, raw: mne.io.BaseRaw):
        """
        Main execution pipeline. Returns a detailed report for BCIWorkstation 
        diagnostics and future decision matrix integration.
        """
        # 1. Shadow Preprocessing (Filtered Raw Segments)
        local_raw = raw.copy().pick(self.picks)
        local_raw.filter(self.params['bp'][0], self.params['bp'][1], 
                         picks=self.picks, verbose=False)
        local_raw.notch_filter(freqs=self.params['notch'], 
                         picks=self.picks, verbose=False)
        
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        epochs = mne.Epochs(local_raw, events, tmin=self.params['tmin'], 
                            tmax=self.params['tmax'], baseline=None, 
                            preload=True, verbose=False)
        
        raw_segments = epochs.get_data() # (n_epochs, n_channels, n_samples)
        sfreq = raw.info['sfreq']

        # 2. Envelope and Threshold Estimation
        envelope = self._extract_envelope(raw_segments, sfreq)
        thresholds = self._calculate_adaptive_thresholds(envelope, sfreq)

        # 3. Compute Individual Metrics
        active_mask, window_means = self._get_active_windows_mask(envelope, thresholds, sfreq)
        
        metrics = {
            "Contamination Rate": self._compute_contamination_rate(active_mask),
            "Local SNR": self._compute_local_snr(envelope, sfreq),
            "Envelope Integral": self._compute_integral(envelope),
            "Sustained Duration": self._compute_sustained_duration(active_mask, sfreq)
        }

        # 4. Generate Decision Logic (Bad Trials)
        bad_indices, trial_tags = self._perform_tagging(metrics)

        # 5. Build Final Output Contract
        return self._build_report(metrics, raw_segments, envelope, thresholds, 
                                  events, bad_indices, trial_tags)

    # --- INTERNAL METRIC METHODS ---

    def _extract_envelope(self, segments, sfreq):
        """Zero-mean -> Rectification -> 3rd order LP Butterworth."""
        # Remove DC offset per segment
        segments = np.abs(segments - np.mean(segments, axis=-1, keepdims=True))
        b, a = butter(3, self.params['lp'] / (sfreq/2), btype='low')
        return filtfilt(b, a, segments, axis=-1)

    def _calculate_adaptive_thresholds(self, envelope, sfreq):
        """Calculates 5x SD of the specific baseline window relative to tmin."""
        start_idx = int((self.params['base_win'][0] - self.params['tmin']) * sfreq)
        end_idx = int((self.params['base_win'][1] - self.params['tmin']) * sfreq)
        
        thresholds = []
        for j in range(envelope.shape[1]):
            rest_data = envelope[:, j, max(0, start_idx):end_idx].flatten()
            thresholds.append(np.std(rest_data) * self.params['mult'])
        return thresholds

    def _get_active_windows_mask(self, envelope, thresholds, sfreq):
        """Vectorized sliding window check."""
        win_size = int(self.params['win_s'] * sfreq)
        n_epochs, n_chans, n_samples = envelope.shape
        n_windows = n_samples // win_size
        
        truncated = envelope[:, :, :n_windows * win_size]
        reshaped = truncated.reshape(n_epochs, n_chans, n_windows, win_size)
        window_means = np.mean(reshaped, axis=-1)
        
        mask = window_means > np.array(thresholds)[None, :, None]
        return mask, window_means

    def _compute_contamination_rate(self, mask):
        """% of trial spent above threshold."""
        return (np.sum(mask, axis=-1) / mask.shape[-1]) * 100

    def _compute_local_snr(self, envelope, sfreq):
        """Max Envelope / Mean Baseline."""
        start_idx = int((self.params['base_win'][0] - self.params['tmin']) * sfreq)
        end_idx = int((self.params['base_win'][1] - self.params['tmin']) * sfreq)
        
        snr_matrix = np.zeros((envelope.shape[0], envelope.shape[1]))
        for i in range(envelope.shape[0]):
            for j in range(envelope.shape[1]):
                baseline_mean = np.mean(envelope[i, j, max(0, start_idx):end_idx]) + 1e-9
                task_max = np.max(envelope[i, j, :])
                snr_matrix[i, j] = task_max / baseline_mean
        return snr_matrix

    def _compute_integral(self, envelope):
        """Area under the envelope curve."""
        return np.trapezoid(envelope, axis=-1)

    def _compute_sustained_duration(self, mask, sfreq):
        """Longest consecutive string of active windows (ms)."""
        n_epochs, n_chans, n_windows = mask.shape
        max_durations = np.zeros((n_epochs, n_chans))
        
        win_ms = self.params['win_s'] * 1000
        
        for i in range(n_epochs):
            for j in range(n_chans):
                # Efficient way to find longest run of True in boolean array
                m = mask[i, j, :]
                if not np.any(m): continue
                
                # Identify changes in state
                bounded = np.hstack(([False], m, [False]))
                diffs = np.diff(bounded.view(np.int8))
                run_starts = np.where(diffs == 1)[0]
                run_ends = np.where(diffs == -1)[0]
                
                max_run = np.max(run_ends - run_starts)
                max_durations[i, j] = max_run * win_ms
        return max_durations

    def _perform_tagging(self, metrics):
        """Identifies bad indices based on the contamination cutoff."""
        contam = metrics["Contamination Rate"]
        n_epochs, n_chans = contam.shape
        bad_indices = []
        tags = [[] for _ in range(n_epochs)]
        
        for i in range(n_epochs):
            for j, ch in enumerate(self.picks):
                if contam[i, j] >= self.params['limit']:
                    tags[i].append(f"{ch}: Contaminated ({int(contam[i,j])}%)")
            if tags[i]: bad_indices.append(i)
        return bad_indices, tags

    def _build_report(self, metrics, raw_seg, env_seg, thresholds, events, bad_idx, tags):
        """Constructs the high-fidelity dictionary for BCIWorkstation."""
        return {
            "summary": (f"Analyzer: EMG Envelope\n"
                        f"Flagged Trials: {len(bad_idx)} / {len(tags)}\n"
                        f"Threshold: {self.params['mult']}x SD Rest"),
            "artifacts": {
                "bad_indices": bad_idx,
                "bad_tags": [tags[i] for i in bad_idx],
                "event_types": events[:, 2].tolist(),
                "tmin": self.params['tmin'],
                "tmax": self.params['tmax']
            },
            "metrics": {
                "Contamination Rate": {"values": {ch: metrics["Contamination Rate"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "%"},
                "Local SNR": {"values": {ch: metrics["Local SNR"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "ratio"},
                "Sustained Duration": {"values": {ch: metrics["Sustained Duration"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "ms"},
                "Envelope Integral": {"values": {ch: metrics["Envelope Integral"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "area"}
            },
            "threshold_lines": {ch: thresholds[j] for j, ch in enumerate(self.picks)},
            "inspection_data": {
                "raw_traces": {ch: raw_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)},
                "envelope_traces": {ch: env_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)}
            }
        }
