import numpy as np
import mne
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
                raw_mi: Union[mne.io.BaseRaw, mne.Epochs], 
                reference: Union[mne.io.BaseRaw, mne.Epochs], 
                baseline: Union[mne.io.BaseRaw, mne.Epochs]):
        
        # 1. Preprocess All Recordings
        raw_mi = self._preprocess(raw_mi)
        reference = self._preprocess(reference)
        baseline = self._preprocess(baseline)

        # 2. Compute the RMS and Envelope Integrals
        rms_mi = self._calculate_rms(raw_mi.get_data(picks=self.picks))
        rms_ref = self._calculate_rms(reference.get_data(picks=self.picks))
        rms_baseline = self._calculate_rms(baseline.get_data(picks=self.picks))
        
        envint_mi = self._calculate_envelope_integral(raw_mi.get_data(picks=self.picks))
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
        sfreq = raw_mi.info['sfreq']
        events_mi = mne.find_events(raw_mi, stim_channel='stim', verbose=False)
        trial_len_samples = int(2.0 * sfreq) # Or detect from your tmin/tmax

        # 2. Extract RMS for Rest (The Baseline Reference)
        # Slice 2min Rest into windows of same size as trials
        rest_data = self._preprocess(raw_rest) 
        rest_chunks = self._slice_into_pseudo_trials(rest_data, trial_len_samples)
        rms_rest_dist = np.sqrt(np.mean(rest_chunks**2, axis=-1)) # (n_chunks, n_chans)
        mu_rest = np.mean(rms_rest_dist, axis=0) # Average noise floor per channel

        # 3. Extract and Normalize ME (The 100% Reference Distribution)
        me_data = self._get_epoched_data(raw_me) # (n_me_trials, n_chans, n_samples)
        rms_me_raw = np.sqrt(np.mean(me_data**2, axis=-1))
        rms_me_norm = rms_me_raw / mu_rest # Normalized by rest mean
        
        # ME Reference Stats
        mu_me_ref = np.mean(rms_me_norm, axis=0)
        std_me_ref = np.std(rms_me_norm, axis=0)

        # 4. Extract and Normalize MI (The Target Trials)
        mi_data = self._get_epoched_data(raw_mi)
        rms_mi_raw = np.sqrt(np.mean(mi_data**2, axis=-1))
        rms_mi_norm = rms_mi_raw / mu_rest

        # 5. Probability Calculation: "Does this MI look like an ME?"
        # We calculate the likelihood of the MI trial under the ME Gaussian distribution
        probs = np.zeros(rms_mi_norm.shape)
        for ch in range(len(self.picks)):
            # Probability that a value is at least as 'ME-like' as this one
            z_scores = (rms_mi_norm[:, ch] - mu_me_ref[ch]) / (std_me_ref[ch] + 1e-9)
            # Two-tailed probability
            probs[:, ch] = norm.sf(np.abs(z_scores)) * 2

        return self._build_report(rms_mi_norm, rms_me_norm, mu_rest, probs)

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