import numpy as np
import mne
from typing import Union
from scipy.stats import kurtosis
from .base_analyzer import BaseAnalyzer

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
    def __init__(self, picks=None, band=(8, 30), tmin=0.0, tmax=2.0, **kwargs):
        super().__init__(picks, **kwargs)
        self.params = {
            'band': band,
            'tmin': tmin,
            'tmax': tmax
        }

    def analyze(self, inst: Union[mne.io.BaseRaw, mne.Epochs]) -> dict:
        """
        Executes feature extraction and returns a standardized report dictionary.
        """
        # 1. Standardize Input to Epochs
        epochs = self._ensure_epochs(inst)
        data = epochs.get_data() # (n_trials, n_chans, n_samples)
        ch_names = epochs.ch_names
        
        # 2. Extract Features
        # Shapes: (n_trials, n_chans)
        var_vals = np.var(data, axis=-1)
        kurt_vals = kurtosis(data, axis=-1)
        
        # Band Power calculation using MNE's compute_psd
        psd = epochs.compute_psd(
            fmin=self.params['band'][0], 
            fmax=self.params['band'][1], 
            picks=self.picks,
            verbose=False
        )
        # Average across the frequency dimension
        power_vals = np.mean(psd.get_data(picks = self.picks), axis=-1) 
        
        # 3. Construct the Standardized Report Dictionary
        return self._build_report(epochs, var_vals, kurt_vals, power_vals, ch_names)

    def _ensure_epochs(self, inst) -> mne.Epochs:
        """Helper to convert Raw to Epochs if necessary."""
        if isinstance(inst, mne.io.BaseRaw):
            # Apply picks from initialization
            if 'stim' not in inst.ch_names:
                raise ValueError("Stimulus channel 'stim' not found in Raw data.")
            
            # Simple epoching based on stimulus channel
            events = mne.find_events(inst, stim_channel='stim', verbose=False)
            return mne.Epochs(
                inst, events, 
                tmin=self.params['tmin'], 
                tmax=self.params['tmax'], 
                picks = self.picks,
                baseline=None, 
                preload=True, 
                verbose=False
            )
        
        # If already epochs, just filter by picks
        return inst.copy().pick(self.picks) if self.picks else inst

    def _build_report(self, epochs, var_vals, kurt_vals, power_vals, ch_names):

        # Structure: { Metric: { values: { Ch: [list] }, unit: str } }
        metrics_report = {
            "Variance": {
                "values": {ch: var_vals[:, i].tolist() for i, ch in enumerate(ch_names)},
                "unit": "V²"
            },
            "Kurtosis": {
                "values": {ch: kurt_vals[:, i].tolist() for i, ch in enumerate(ch_names)},
                "unit": "excess"
            },
            f"Power ({self.params['band'][0]}-{self.params['band'][1]}Hz)": {
                "values": {ch: power_vals[:, i].tolist() for i, ch in enumerate(ch_names)},
                "unit": "dB/Hz"
            }
        }
        
        return {
            "summary": (f"Analyzer: Signal Features\n"
                        f"Trials: {len(epochs)}\n"
                        f"Band: {self.params['band']} Hz\n"
                        f"Avg Var: {np.mean(var_vals):.2e}"),
            "visuals": ["hist"], # Note: no 'inspect'
            "artifacts": {
                "bad_indices": [],   # This analyzer doesn't reject by default
                "bad_tags": [],
                "event_types": epochs.events[:, 2].tolist(),
                "tmin": self.params['tmin'],
                "tmax": self.params['tmax']
            },
            
            "metrics": metrics_report,

            "metadata": {
                "sfreq": epochs.info['sfreq'],
                "ch_names": ch_names
            }
        }