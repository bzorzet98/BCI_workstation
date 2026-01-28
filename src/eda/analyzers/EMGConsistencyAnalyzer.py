import numpy as np
import mne
from typing import Union, Optional
from scipy.stats import norm
from scipy.signal import envelope

from .base_analyzer import BaseAnalyzer

class EMGConsistencyAnalyzer(BaseAnalyzer):
    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"],
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 **kwargs):
        super().__init__(picks, **kwargs)

    def analyze(self, 
                data: Union[mne.io.BaseRaw, mne.Epochs], 
                reference: Union[mne.io.BaseRaw, mne.Epochs], 
                baseline: Optional[mne.io.BaseRaw] = None):
        
        # 1. Preprocess All Recordings
        data = self._preprocess(data)
        reference = self._preprocess(reference)
        baseline = self._preprocess(baseline)

        # 2. Compute the RMS and Envelope Integrals
        rms_mi = self._calculate_rms(data.get_data(picks=self.picks))
        rms_ref = self._calculate_rms(reference.get_data(picks=self.picks))
        rms_baseline = self._calculate_rms(baseline.get_data(picks=self.picks))
        
        envint_mi = self._calculate_envelope_integral(data.get_data(picks=self.picks))
        envint_ref = self._calculate_envelope_integral(reference.get_data(picks=self.picks))
        envint_baseline = self._calculate_envelope_integral(baseline.get_data(picks=self.picks))
        
        # 3. Normalize Metrics by Baseline
        rms_mi_norm = rms_mi / (rms_baseline + 1e-9)
        rms_ref_norm = rms_ref / (rms_baseline + 1e-9)
        envint_mi_norm = envint_mi / (envint_baseline + 1e-9)
        envint_ref_norm = envint_ref / (envint_baseline + 1e-9)
        
        # 4. Build the distributions of the metrics
        rms_mi_dist = (rms_mi_norm.mean(axis=-1) , rms_mi_norm.std(axis=-1))
        rms_ref_dist = (rms_ref_norm.mean(axis=-1) , rms_ref_norm.std(axis=-1))
        
        envint_mi_dist = (envint_mi_norm.mean(axis=-1), envint_mi_norm.std(axis=-1))
        envint_ref_dist = (envint_ref_norm.mean(axis=-1), envint_ref_norm.std(axis=-1))
        
        # 1. Get Task Duration (to match rest slicing)
        # Assuming MI and ME have the same trial length
        sfreq = data.info['sfreq']
        events_mi = mne.find_events(data, stim_channel='stim', verbose=False)
        trial_len_samples = int(2.0 * sfreq) # Or detect from your tmin/tmax

        # 2. Extract 
        return {}

    def _preprocess(self, raw):
        # Returns data array (n_chans, n_samples)
        raw.filter(self.params['bandpass_filter'][0], self.params['bandpass_filter'][1], 
                         picks=self.picks, verbose=False)
        raw.notch_filter(freqs=self.params['notch_filter'], 
                         picks=self.picks, verbose=False)
        return raw
    
    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data**2, axis=-1))
    
    def _calculate_envelope_integral(self, data):
        signal_envelopes = envelope(data, residual=None)
        integral_envelope = np.trapezoid(signal_envelopes, axis=-1)
        return integral_envelope