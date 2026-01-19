import numpy as np
import mne
from typing import Union, Optional, Tuple
from scipy.signal import butter, filtfilt
from .base_analyzer import BaseAnalyzer

class EMGEnvelopeAnalyzer(BaseAnalyzer):
    """
    Advanced Modular EMG Artifact Detection Framework.
    
    This analyzer identifies muscle contamination in EEG trials by extracting 
    physiological features from the EMG envelope. It uses an adaptive 
    threshold based on the subject's noise floor at rest..
    
    Improvements:
    1. Hysteresis Detection: Uses High/Low thresholds to prevent flickering.
    2. Robust Scaling: Uses Median Absolute Deviation (MAD) to ignore ECG spikes in baseline.
    3. Flexible Baseline: Supports external 'Rest' Raw objects or trial-relative windows.
    4. Duration Filtering: Automatically rejects short spikes (ECG) from contamination metrics.

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
                 lp_cut=20.0, # Lowered slightly to smooth ECG spikes better
                 threshold_multiplier=5.0,
                 hysteresis_factor=0.6, # T_low is 60% of T_high
                 min_duration_s=0.15, # Minimum duration to count as muscle activity (filters ECG)
                 contamination_cutoff_percent=50.0,
                 baseline_window: Optional[Tuple[float, float]] = (-0.5, 0.0),
                 **kwargs):
        
        super().__init__(picks, **kwargs)
        self.params = {
            'bp': bandpass_filter, 'notch': notch_filter, 'tmin': tmin, 'tmax': tmax,
            'lp': lp_cut, 'mult': threshold_multiplier, 'hyst': hysteresis_factor,
            'min_dur': min_duration_s, 'limit': contamination_cutoff_percent, 
            'base_win': baseline_window,
            'mult_high': threshold_multiplier,
            'mult_low': threshold_multiplier * hysteresis_factor,
        }


    def analyze(self, raw: mne.io.BaseRaw, rest: Optional[mne.io.BaseRaw] = None):
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


        # 3. Flexible Baseline & Adaptive Thresholds
        # If rest is provided and baseline_window is None, compute global thresholds from rest
        t_high, t_low, baseline_means = self._calculate_dual_thresholds(envelope, sfreq, rest)
        
        # 4. Hysteresis Masking (Sample-by-sample for millisecond precision)
        active_mask = self._apply_hysteresis_threshold(envelope, t_high, t_low)
        
        # 5. Compute Metrics
        metrics = {
            "Contamination Rate": self._compute_contamination_rate(active_mask),
            "Local SNR": self._compute_local_snr(envelope, baseline_means),
            "Envelope Integral": self._compute_integral(envelope),
            "Sustained Duration": self._compute_sustained_duration(active_mask, sfreq)
        }

        # 6. Generate Decision Logic
        bad_indices, trial_tags = self._perform_tagging(metrics)

        return self._build_report(metrics, raw_segments, envelope, t_high, 
                                  events, bad_indices, trial_tags)


    # --- INTERNAL METRIC METHODS ---
    def _extract_envelope(self, segments, sfreq):
        """Zero-mean -> Rectification -> 3rd order LP Butterworth."""
        # Remove DC offset per segment
        segments = np.abs(segments - np.mean(segments, axis=-1, keepdims=True))
        b, a = butter(4, self.params['lp'] / (sfreq/2), btype='low')
        return filtfilt(b, a, segments, axis=-1)
    
    def _calculate_dual_thresholds(self, envelope, sfreq, rest_raw: Optional[mne.io.BaseRaw]):
        """
        Determines noise floor from rest_raw or baseline_window.
        Returns T_high, T_low, and baseline_means for SNR.
        """
        n_channels = len(self.picks)
        sigmas = np.zeros(n_channels)
        baseline_means = np.zeros(n_channels)

        # Case A: External Rest Recording provided
        if rest_raw is not None:
            rest_data = rest_raw.copy().pick(self.picks).filter(
                self.params['bp'][0], self.params['bp'][1], verbose=False, picks=self.picks).get_data()
            # Get envelope of rest
            rest_env = self._extract_envelope(rest_data[np.newaxis, ...], sfreq)[0]
            sigmas = rest_env.std(keepdims = True, axis = -1)
            baseline_means = rest_env.mean(keepdims = True, axis = -1)

        # Case B: Use baseline window from the epochs
        else:
            base_win = self.params['base_win'] if self.params['base_win'] else (self.params['tmin'], 0.0)
            start_idx = int((base_win[0] - self.params['tmin']) * sfreq)
            end_idx = int((base_win[1] - self.params['tmin']) * sfreq)
            
            for j in range(n_channels):
                # Flatten across all epochs to get a robust channel-specific noise floor
                rest_segment = envelope[:, j, max(0, start_idx):end_idx].flatten()
                sigmas[j] = np.std(rest_segment)
                baseline_means[j] = np.mean(rest_segment)

        t_high = sigmas * self.params['mult_high']
        t_low = sigmas * self.params['mult_low']
        
        return t_high, t_low, baseline_means
    
    def _apply_hysteresis_threshold(self, envelope, t_high, t_low):
        
        """
        Zero-loop vectorized Schmitt Trigger.
        Fixed broadcasting to ensure (Epochs, Channels, Samples) alignment.
        """
        n_epochs, n_chans, n_samples = envelope.shape
        
        # 1. Force thresholds to (1, n_chans, 1) to match (n_epochs, n_chans, n_samples)
        # This prevents the "shape (1, 4, 1, 1)" broadcast error.
        th = np.asanyarray(t_high).reshape(1, n_chans, 1)
        tl = np.asanyarray(t_low).reshape(1, n_chans, 1)
        
        # 2. Identify High (Trigger) and Low (Reset) points
        hi = envelope > th
        lo = envelope < tl
        
        # 3. Define Event Points: any change that FORCES a state shift
        events = hi | lo
        
        # 4. Create a "Phantom" starting state (index 0)
        # This ensures every trial begins in a 'False' (Low) state.
        pad_events = np.ones((n_epochs, n_chans, 1), dtype=bool) # Always an event at t=0
        pad_values = np.zeros((n_epochs, n_chans, 1), dtype=bool) # The event is 'False'
        
        # 5. Concatenate Phantom start with real data
        # event_values: True if high, False if low. 
        # (If both False, it won't be indexed by the forward-fill anyway)
        event_values = np.where(hi, True, False)
        
        full_events = np.concatenate([pad_events, events], axis=-1)
        full_values = np.concatenate([pad_values, event_values], axis=-1)
        
        # 6. FORWARD-FILL INDEXING
        # Create an array of indices [0, 1, 2 ... samples]
        # At every event point, we store its index. At non-events, we store 0.
        time_indices = np.arange(full_events.shape[-1])
        event_indices = np.where(full_events, time_indices, 0)
        
        # The 'magic' step: propagate the index of the last occurring event forward
        # accum_idx at time 't' will contain the index of the most recent hi or lo crossing.
        last_event_idx = np.maximum.accumulate(event_indices, axis=-1)
        
        # 7. Map indices back to values
        # We use take_along_axis to grab the True/False value from full_values
        # based on the last_event_idx map.
        final_mask_padded = np.take_along_axis(full_values, last_event_idx, axis=-1)
        
        # 8. Return result, stripping the phantom padding
        return final_mask_padded[:, :, 1:]


    def _compute_contamination_rate(self, mask):
        """% of trial spent above threshold."""
        return (np.sum(mask, axis=-1) / mask.shape[-1]) * 100

    def _compute_local_snr(self, envelope, baseline_means):
        # Max of task window / mean of baseline
        task_max = np.max(envelope, axis=-1)
        return task_max / (np.asanyarray(baseline_means).reshape(1, -1) + 1e-9)

    def _compute_integral(self, envelope):
        """Area under the envelope curve."""
        return np.trapezoid(envelope, axis=-1)
    
    def _compute_sustained_duration(self, mask, sfreq):
        """
        Uses np.diff to find edges of active blocks and calculate max length.
        """
        n_epochs, n_chans, n_samples = mask.shape
        max_durations = np.zeros((n_epochs, n_chans))
        
        for i in range(n_epochs):
            for j in range(n_chans):
                m = mask[i, j, :]
                if not np.any(m): continue
                
                # Detect transitions (True to False or False to True)
                # Pad with False to catch activity that starts/ends at the boundary
                padded = np.pad(m, (1, 1), mode='constant', constant_values=False)
                diffs = np.diff(padded.astype(int))
                
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                
                durations = ends - starts
                if len(durations) > 0:
                    max_durations[i, j] = (np.max(durations) / sfreq)
                
        return max_durations

    def _perform_tagging(self, metrics):
        contam = metrics["Contamination Rate"]
        sustained = metrics["Sustained Duration"]
        n_epochs, n_chans = contam.shape
        bad_indices = []
        tags = [[] for _ in range(n_epochs)]
        
        for i in range(n_epochs):
            for j, ch in enumerate(self.picks):
                # Only flag if contamination is high AND it's sustained (not just an ECG spike)
                is_high_contam = contam[i, j] >= self.params['limit']
                is_sustained = sustained[i, j] >= self.params['min_dur']
                
                if is_high_contam:
                    if is_sustained:
                        tags[i].append(f"{ch}: EMG Contaminated ({int(contam[i,j])}%)")
                    else:
                        tags[i].append(f"{ch}: High-Freq Spikes/ECG")
            
            if tags[i]: 
                bad_indices.append(i)
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
                "Sustained Duration": {"values": {ch: metrics["Sustained Duration"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "s"},
                "Envelope Integral": {"values": {ch: metrics["Envelope Integral"][:, j].tolist() for j, ch in enumerate(self.picks)}, "unit": "area"}
            },
            "threshold_lines": {ch: thresholds[j] for j, ch in enumerate(self.picks)},
            "inspection_data": {
                "raw_traces": {ch: raw_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)},
                "envelope_traces": {ch: env_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)}
            }
        }
