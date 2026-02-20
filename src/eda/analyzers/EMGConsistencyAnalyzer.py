from src.eda.eda_report import EDAReport, EDAArtifacts, EDAMetric
from .base_analyzer import BaseAnalyzer
import numpy as np
import mne
from typing import Union, Optional
from scipy.stats import median_abs_deviation
from scipy.signal import envelope

class EMGConsistencyAnalyzer(BaseAnalyzer):
    def __init__(self, picks=["EMG1", "EMG2", "EMG3", "EMG4"],
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 z_threshold=3.5,
                 n_pseudo_trials=40,
                 **kwargs):
        # Correctly merge parameters into self.params
        params = {
            'bandpass_filter': bandpass_filter,
            'notch_filter': notch_filter,
            'z_threshold': z_threshold,
            'tmin': kwargs.get('tmin', 0),
            'tmax': kwargs.get('tmax', 1.0),
            'n_pseudo_trials': n_pseudo_trials,
            **kwargs
        }
        super().__init__(picks, **params)

    def analyze(self, 
                data: Union[mne.io.BaseRaw, mne.Epochs], 
                reference: Union[mne.io.BaseRaw, mne.Epochs], 
                baseline: Optional[mne.io.BaseRaw] = None):
        
        # 1. Preprocess (Note: .copy() prevents modifying original data)
        data_proc = self._preprocess(data.copy())
        ref_proc = self._preprocess(reference.copy())
        base_proc = self._preprocess(baseline.copy())

        # 2. Compute Metrics
        # data.get_data() returns (n_epochs, n_channels, n_times)
        rms_mi = self._calculate_rms(data_proc.get_data(picks=self.picks))
        rms_ref = self._calculate_rms(ref_proc.get_data(picks=self.picks))
        rms_baseline = self._calculate_rms(base_proc.get_data(picks=self.picks))
        
        envint_mi, env_traces = self._calculate_envelope_integral(data_proc.get_data(picks=self.picks))
        envint_ref, _ = self._calculate_envelope_integral(ref_proc.get_data(picks=self.picks))
        envint_baseline, _ = self._calculate_envelope_integral(base_proc.get_data(picks=self.picks))
        
        # 3. Normalize Metrics by Mean Baseline
        # We take the mean across time/epochs of the baseline to get a scalar per channel
        b_rms = np.mean(rms_baseline, axis=0) + 1e-9
        b_env = np.mean(envint_baseline, axis=0) + 1e-9
        
        
        rms_ref_norm = np.log(rms_ref / (b_rms + 1e-9) + 1e-9)
        rms_mi_norm  = np.log(rms_mi  / (b_rms + 1e-9) + 1e-9)

        env_ref_norm = np.log(envint_ref / (b_env + 1e-9) + 1e-9)
        env_mi_norm  = np.log(envint_mi  / (b_env + 1e-9) + 1e-9)
        
        # 4. Calculate Robust Reference Distributions (Median & MAD)
        # 0.6745 is the scaling factor to make MAD consistent with SD of a normal distribution
        ref_rms_med = np.median(rms_ref_norm, axis=0)
        ref_rms_mad = median_abs_deviation(rms_ref_norm, axis=0) * 1.4826 

        ref_env_med = np.median(env_ref_norm, axis=0)
        ref_env_mad = median_abs_deviation(env_ref_norm, axis=0) * 1.4826

        # 5. Calculate Modified Z-Scores
        # This tells us: "How many robust-standard-deviations is this MI trial from the Reference Median?"
        z_rms = (rms_mi_norm - ref_rms_med) / (ref_rms_mad + 1e-9)
        z_env = (env_mi_norm - ref_env_med) / (ref_env_mad + 1e-9)

        # 4. Perform Tagging (fixing the ambiguity error)
        bad_idx, tags = self._perform_tagging(z_rms, z_env)

        # 5. Build Report (Matching your specific Schema)
        metrics_payload = {
            "RMS Z-Score": z_rms,
            "Envelope Integral Z-Score": z_env
        }
        
        events = data_proc.events[:, 2] if data_proc.events is not None else []
        
        return self._build_report(
            metrics=metrics_payload,
            raw_seg=data_proc.get_data(picks=self.picks),
            env_seg=env_traces,
            bad_idx=bad_idx,
            tags=tags,
            events=events
        )

    
    def _preprocess(self, raw):
        # Returns data array (n_chans, n_samples)
        raw.filter(self.params['bandpass_filter'][0], self.params['bandpass_filter'][1], 
                         picks=self.picks, verbose=False)
        raw.notch_filter(freqs=self.params['notch_filter'], 
                         picks=self.picks, verbose=False)
        epochs = self._ensure_epochs(raw)
        data = epochs.get_data() # (n_trials, n_chans, n_samples)
        
        self.ch_names = epochs.ch_names
        self.sfreq = epochs.info['sfreq']
        return epochs
    
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
            print(f"[Analyzer] No events found. Generating {self.params['n_pseudo_trials']} pseudo-epochs.")
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
        N = self.params['n_pseudo_trials']

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
        
    
    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data**2, axis=-1))
    
    def _calculate_envelope_integral(self, data):
        signal_envelopes = envelope(data, residual=None)
        integral_envelope = np.trapezoid(signal_envelopes, axis=-1)
        return integral_envelope, signal_envelopes
    
    def _perform_tagging(self, z_rms, z_env):
        n_epochs = z_rms.shape[0]
        bad_indices = []
        tags = [[] for _ in range(n_epochs)]
        thresh = self.params['z_threshold']

        for i in range(n_epochs):
            for j, ch in enumerate(self.picks):
                # Access individual floats to avoid ambiguity error
                z_r = z_rms[i, j]
                z_e = z_env[i, j]
                
                if abs(z_r) > thresh or abs(z_e) > thresh:
                    if z_r > thresh and z_e > thresh:
                        tags[i].append(f"{ch}: Outlier Dist (Z_rms={z_r:.1f})")
                    elif z_r > thresh:
                        tags[i].append(f"{ch}: High-Freq Noise")
                    else:
                        tags[i].append(f"{ch}: Baseline Drift")

            if tags[i]:
                bad_indices.append(i)
        return bad_indices, tags

    def _build_report(self, metrics, raw_seg, env_seg, bad_idx, tags, events):
        # Convert dictionary metrics into EDAMetric dataclasses
        report_metrics = {
            name: EDAMetric(
                values={ch: m[:, j].tolist() for j, ch in enumerate(self.picks)},
                unit="Z"
            ) for name, m in metrics.items()
        }

        return EDAReport(
            summary=(f"Analyzer: EMG Consistency\n"
                     f"Flagged Trials: {len(bad_idx)} / {len(tags)}\n"
                     f"Method: Robust Modified Z-Score"),
            visuals=["hist", "scatter", "mapping", "inspect"],
            artifacts=EDAArtifacts(
                bad_indices=bad_idx,
                bad_tags=[tags[i] for i in bad_idx]
            ),
            metrics=report_metrics,
            metadata={
                "event_types": events,  # <-- Added Metadata!
                "tmin": self.params.get('tmin', 0),
                "tmax": self.params.get('tmax', 1)
            },
            inspection_data={
                "x-axis": np.linspace(self.params['tmin'], self.params['tmax'], raw_seg.shape[-1]).tolist(),
                "raw_traces": {ch: raw_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)},
                "envelope_traces": {ch: env_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)}
            }
        )
        