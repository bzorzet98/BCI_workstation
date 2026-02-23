from src.eda.eda_report import EDAReport, EDAArtifacts, EDAMetric
from .base_analyzer import BaseAnalyzer
import numpy as np
import mne
from typing import Union, Optional
from scipy.stats import median_abs_deviation
from scipy.signal import envelope

class EMGConsistencyAnalyzer(BaseAnalyzer):
    def __init__(self, picks=["EMG1","EMG3", ],
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 percentage_threshold=0.15,
                 n_pseudo_trials=40,
                 **kwargs):
        
        params = {
            'bandpass_filter': bandpass_filter,
            'notch_filter': notch_filter,
            'percentage_threshold': percentage_threshold,
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
        
        # 1. Preprocess
        data_proc = self._preprocess(data.copy())
        ref_proc = self._preprocess(reference.copy())
        base_proc = self._preprocess(baseline.copy())

        # 2. Compute Metrics
        rms_data = self._calculate_rms(data_proc.get_data(picks=self.picks))
        rms_ref = self._calculate_rms(ref_proc.get_data(picks=self.picks))
        rms_baseline = self._calculate_rms(base_proc.get_data(picks=self.picks))
        
        envint_data, env_data_traces = self._calculate_envelope_integral(data_proc.get_data(picks=self.picks))
        envint_ref, _ = self._calculate_envelope_integral(ref_proc.get_data(picks=self.picks))
        envint_baseline, _ = self._calculate_envelope_integral(base_proc.get_data(picks=self.picks))
        
        # 3. Dynamic Range Calculation (Medians)
        # Using medians makes the scaling robust against signal artifacts
        med_base_rms = np.median(rms_baseline, axis=0)
        med_base_env = np.median(envint_baseline, axis=0)

        ref_labels = ref_proc.events[:, 2]
        unique_labels = np.unique(ref_labels)
        med_ref_rms = {}
        med_ref_env = {}
        for i in unique_labels:
            med_ref_rms[i] = np.median(rms_ref[ref_labels == i], axis=0)
            med_ref_env[i] = np.median(envint_ref[ref_labels == i], axis=0)

        # 4. Perform Tagging (Percentage-based Policing)
        bad_idx, tags, rms_norm, env_norm = self._perform_tagging(
            rms_data, 
            med_base_rms,
            med_ref_rms
        )
            
        # 5. Build Report
        metrics_payload = {
            "RMS_Percentage": rms_norm,
            "Envelope_Percentage": env_norm,
        }
        events = data_proc.events[:, 2] if data_proc.events is not None else []
        
        return self._build_report(
            metrics=metrics_payload,
            raw_seg=data_proc.get_data(picks=self.picks),
            env_seg=env_data_traces,
            bad_idx=bad_idx,
            tags=tags,
            events=events
        )


    def _perform_tagging(self, rms_data, b_rms, ref_rms):
        """
        rms_data: (n_epochs, n_channels) - Raw MI metrics
        b_rms: (n_channels,) - Median of rest
        ref_rms: Dict {label: array(n_channels,)} - Median of movement per class
        """
        n_epochs = rms_data.shape[0]
        bad_indices = []
        tags = [[] for _ in range(n_epochs)]
        lambda_p = self.params.get('percentage_threshold', 0.15)
        
        # 1. Rescale the data for every label
        # This creates a dict where each value is an (n_epochs, n_channels) matrix
        # 0.0 = Rest, 1.0 = Median of that specific Movement Class
        rms_data_norm = {}
        for label, m_rms in ref_rms.items():
            # Vectorized normalization across all epochs and channels for this label
            rms_data_norm[label] = (rms_data - b_rms) / (m_rms - b_rms + 1e-9)

        # 2. Policing: Flag if any trial exceeds lambda_p for ANY class-reference
        # Logic: If EMG1 looks like 15% of 'Left Movement' OR EMG3 looks like 15% of 'Right Movement'...
        for label, norm_matrix in rms_data_norm.items():
            # Find where any channel in any epoch exceeds the threshold
            # combined_mask shape: (n_epochs, n_channels)
            mask = norm_matrix >= lambda_p
            
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                ch_name = self.picks[c]
                percentage = norm_matrix[r, c] * 100
                tag_msg = f"{ch_name}: {percentage:.1f}% of Label {label} ME"
                
                if tag_msg not in tags[r]:
                    tags[r].append(tag_msg)
                
                if r not in bad_indices:
                    bad_indices.append(r)

        # 3. For the final metrics payload, we need a single representative matrix.
        # Usually, we take the maximum normalization across labels to show 'worst case' leakage.
        all_norms = np.array(list(rms_data_norm.values())) # (n_labels, n_epochs, n_channels)
        max_rms_norm = np.max(all_norms, axis=0)
        
        return sorted(bad_indices), tags, max_rms_norm, max_rms_norm # Returning norms for the report
    

    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data**2, axis=-1))
    
    def _calculate_envelope_integral(self, data):
        signal_envelopes = envelope(data, residual=None)
        integral_envelope = np.trapezoid(signal_envelopes, axis=-1)
        return integral_envelope, signal_envelopes
    
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
        