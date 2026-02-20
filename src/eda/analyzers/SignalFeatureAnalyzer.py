import numpy as np
import mne
from typing import Union
from scipy.stats import kurtosis
from scipy.signal import envelope
from .base_analyzer import BaseAnalyzer
from src.eda.eda_report import EDAReport, EDAArtifacts, EDAMetric

class SignalFeatureAnalyzer(BaseAnalyzer):
    """
    Standard Feature Extraction Analyzer.
    Calculates statistical and spectral metrics to characterize signal properties
    across trials and channels. 
    
    Metrics:
    - Variance: Signal power in the time domain.
    - Kurtosis: Measures "peakedness" or presence of outliers (transients).
    - Band Power: Mean power in a specific frequency range (e.g., Alpha/Beta).
    """
    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"], 
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 powerband=(8, 30), 
                 tmin=0.0, 
                 tmax=2.0, 
                 n_pseudo_trials = None,
                 zc_threshold=1e-6,
                 **kwargs):
        
        super().__init__(picks, **kwargs)
        self.params = {
            'bandpass_filter':bandpass_filter,
            'notch_filter':notch_filter,
            'powerband': powerband,   
            'tmin': tmin,
            'tmax': tmax,
            'n_pseudo': n_pseudo_trials, # Target trials if no events found
            'zc_eps': zc_threshold,
            'visuals': ['hist', 'mapping', 'scatter','inspect']
        }

    def analyze(self, inst: Union[mne.io.BaseRaw, mne.Epochs]) -> dict:
        """
        Executes feature extraction and returns a standardized report dictionary.
        """

        inst.filter(self.params['bandpass_filter'][0], self.params['bandpass_filter'][1], 
                         picks=self.picks, verbose=False)
        inst.notch_filter(freqs=self.params['notch_filter'], 
                         picks=self.picks, verbose=False)
        
        # 1. Standardize Input to Epochs
        epochs = self._ensure_epochs(inst)
        data = epochs.get_data() # (n_trials, n_chans, n_samples)
        ch_names = epochs.ch_names
        sfreq = epochs.info['sfreq']
        
        # 2. Compute Metrics via Individual Methods
        # All methods return (n_trials, n_chans)
        results = {
            "Kurtosis":(self._compute_kurtosis(data), ""),
            "MAV": (self._compute_mav(data), "µV"),
            "Zero Crossing": (self._compute_zc(data), "count"),
            "Waveform Length": (self._compute_wl(data), "dist"),
            "Variance": (self._compute_variance(data), "V²"),
            "Band Power": (self._compute_band_power(epochs), "dB"),
            "RMS": (self._compute_RMS(data), "V"),
        }

        # Add Hjorth parameters (Activity, Mobility, Complexity)
        h_act, h_mob, h_com = self._compute_hjorth(data, sfreq)
        results["Hjorth Activity"] = (h_act, "var")
        results["Hjorth Mobility"] = (h_mob, "ratio")
        results["Hjorth Complexity"] = (h_com, "ratio")
        
        signal_envelopes, integral_envelope = self._compute_envelope_integral(data)
        results["Envelope Integral"] = (integral_envelope, "V·s")
        results["Signal Envelope"] = (signal_envelopes, "V")

        # 3. Construct the Standardized Report Dictionary
        return self._build_report(epochs, results, ch_names, sfreq)

    def _ensure_epochs(self, inst) -> mne.Epochs:
        """
        Logic: Try standard epoching. If it fails (0 events), 
        trigger the sliding window splitter.
        """
        # A. Already Epoched
        if isinstance(inst, mne.Epochs):
            return inst.copy().pick(self.picks) if self.picks else inst
        
        try:
            events = mne.find_events(inst, stim_channel='stim', verbose=False)
        except:
            events = np.array([])

        if len(events) > 0:
            # Standard Epoching
            return mne.Epochs(inst, events, tmin=self.params['tmin'], 
                              tmax=self.params['tmax'], baseline=None, 
                              preload=True, verbose=False, picks=self.picks)
        else:
            # --- FALLBACK: PSEUDO-EPOCHING ---
            print(f"[Analyzer] No events found. Generating {self.params['n_pseudo']} pseudo-epochs.")
            return self._create_pseudo_epochs(inst.copy().pick(self.picks))
        
    def _create_pseudo_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        """
        Splits raw signal into N trials using sliding windows if length is insufficient.
        """
        sfreq = raw.info['sfreq']   
        data = raw.get_data() # (n_channels, n_samples)
        n_chans, L = data.shape
        
        # Window size in samples
        W = int((self.params['tmax'] - self.params['tmin']) * sfreq)
        N = self.params['n_pseudo']

        if L < W:
            raise ValueError(f"Signal length ({L}) is shorter than window size ({W}).")

        # CASE A: Signal is long enough for non-overlapping windows
        # CASE B: Signal is short, we calculate sliding step (S)
        # Formula: L = W + (N-1)*S  =>  S = (L-W)/(N-1)
        
        start_indices = np.linspace(0, L - W, N).astype(int)
        
        # Slice the data
        epochs_data = []
        for start in start_indices:
            epochs_data.append(data[:, start : start + W])
        
        epochs_data = np.array(epochs_data) # (N, n_channels, W)

        # Create MNE EpochsArray to maintain compatibility
        events = np.column_stack([
            np.arange(N) * W, # Fake time latency
            np.zeros(N, dtype=int), 
            -1*np.ones(N, dtype=int) # All events have ID -1
        ])
        
        event_id = {'pseudo_trial': -1}
        
        return mne.EpochsArray(epochs_data, raw.info, events=events, 
                               tmin=self.params['tmin'], event_id=event_id, 
                               verbose=False)
    
     # --- INDIVIDUAL METRIC METHODS ---
    def _compute_kurtosis(self, data):
        """Kurtosis"""
        return kurtosis(data,-1)
    
    def _compute_mav(self, data):
        """Mean Absolute Value: Average of absolute samples."""
        return np.mean(np.abs(data), axis=-1)

    def _compute_zc(self, data):
        """Zero Crossing: Number of times signal crosses zero."""
        # We use a small epsilon to avoid counting noise as crossings
        eps = self.params['zc_eps']
        # Find points where sign changes
        diff_sign = np.diff(np.sign(data), axis=-1)
        # Count crossings that are significant
        return np.sum((diff_sign != 0) & (np.abs(np.diff(data, axis=-1)) > eps), axis=-1)

    def _compute_wl(self, data):
        """Waveform Length: Cumulative length of the waveform over the time segment."""
        return np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)

    def _compute_variance(self, data):
        """Variance: Measure of signal power/spread."""
        return np.var(data, axis=-1)

    def _compute_band_power(self, epochs):
        """Average Power in the specified frequency band."""
        psd = epochs.compute_psd(fmin=self.params['powerband'][0], fmax=self.params['powerband'][1], verbose=False, picks = self.picks)
        return np.mean(psd.get_data(picks = self.picks), axis=-1)

    def _compute_hjorth(self, data, sfreq):
        """
        Calculates Hjorth Parameters: Activity, Mobility, and Complexity.
        """
        # Activity: Simply the variance
        activity = np.var(data, axis=-1)

        # Mobility: Sqrt(var(diff(x)) / var(x))
        # We approximate the derivative with np.diff
        d1 = np.diff(data, axis=-1) * sfreq
        mobility = np.sqrt(np.var(d1, axis=-1) / (activity + 1e-9))

        # Complexity: Mobility(diff(x)) / Mobility(x)
        d2 = np.diff(d1, axis=-1) * sfreq
        mobility_d1 = np.sqrt(np.var(d2, axis=-1) / (np.var(d1, axis=-1) + 1e-9))
        complexity = mobility_d1 / (mobility + 1e-9)

        return activity, mobility, complexity
    
    def _compute_RMS(self, data):
        """Root Mean Square (RMS)"""
        return np.sqrt(np.mean(data**2, axis=-1))
    
    def _compute_envelope_integral(self, data):
        signal_envelopes = envelope(data, residual=None)
        integral_envelope = np.trapezoid(signal_envelopes, axis=-1)
        return signal_envelopes, integral_envelope
    
    def _build_report(self, epochs, results, ch_names, sfreq):
        report_metrics = {}
        for name, (vals, unit) in results.items():
            if name != "Signal Envelope":
                report_metrics[name] = EDAMetric(
                    values={ch: vals[:, i].tolist() for i, ch in enumerate(ch_names)},
                    unit=unit
                )
                
        data = epochs.get_data(copy=False)
        
        return EDAReport(
            summary=f"Feature Analysis: {len(epochs)} trials | SFreq: {sfreq}Hz",
            visuals=["hist", "scatter", "mapping", "inspect"],
            # No artifacts generated by this specific feature analyzer, so we leave it empty
            artifacts=EDAArtifacts(bad_indices=[], bad_tags=[]),
            metrics=report_metrics,
            metadata={
                "sfreq": sfreq,
                "ch_names": ch_names,
                "event_types": epochs.events[:, 2].tolist() if epochs.events is not None else [],
                "tmin": self.params.get('tmin', 0),
                "tmax": self.params.get('tmax', 1)
            },
            inspection_data={
                "x-axis": np.linspace(self.params['tmin'], self.params['tmax'], data.shape[-1]).tolist(),
                "raw_traces": {ch: data[:, j, :].tolist() for j, ch in enumerate(self.picks)},
                "envelope_traces": {ch: results['Signal Envelope'][0][:, j, :].tolist() for j, ch in enumerate(self.picks)}
            }
        )