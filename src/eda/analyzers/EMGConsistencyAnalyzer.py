from src.eda.eda_report import EDAReport, EDAArtifacts, EDAMetric
from .base_analyzer import BaseAnalyzer
import numpy as np
import mne
from typing import Union, Optional
from scipy.stats import median_abs_deviation
from scipy.signal import envelope

class EMGConsistencyAnalyzer(BaseAnalyzer):
    def __init__(self, picks=["EMG1", "EMG3"],
                 bandpass_filter=(10.0, 95.0),
                 notch_filter=[50.0],
                 tmin = 0.0,
                 tmax= 3.0,
                 percentage_threshold=0.15,
                 n_pseudo_trials=40,
                 channel_mapping={1: ["EMG1",], 2: ["EMG3",]}, # Updated to lists for safety
                 **kwargs):
        
        params = {
            'bandpass_filter': bandpass_filter,
            'notch_filter': notch_filter,
            'percentage_threshold': percentage_threshold,
            'tmin': kwargs.get('tmin', 0),
            'tmax': kwargs.get('tmax', 1.0),
            'n_pseudo_trials': n_pseudo_trials,
            'channel_mapping': channel_mapping,
            **kwargs
        }
        super().__init__(picks, **params)
        
        self.channel_map = channel_mapping
        self.metrics_lib = {
            "RMS": lambda x: np.sqrt(np.mean(x**2, axis=-1)),
            "EnvInt": lambda x: np.trapezoid(envelope(x, residual=None), axis=-1), 
        }
        
    def analyze(self, data, reference, baseline):
        # 1. Preprocess and get MNE Epochs objects
        data_proc = self._preprocess(data.copy())
        ref_proc = self._preprocess(reference.copy())
        base_proc = self._preprocess(baseline.copy())

        # Extract raw data arrays (n_epochs, n_channels, n_times)
        d_mi = data_proc.get_data(picks=self.picks)
        d_ref = ref_proc.get_data(picks=self.picks)
        d_base = base_proc.get_data(picks=self.picks)

        # Extract Envelope Traces explicitly for the report visuals
        _, env_seg = self._calculate_envelope_integral(d_mi)
        
        # 2. Compute metrics dynamically based on metrics_list
        mi_m, ref_m, base_m = {}, {}, {}
        for m_name in self.metrics_lib.keys():
            func = self.metrics_lib[m_name]
            mi_m[m_name] = func(d_mi)
            ref_m[m_name] = func(d_ref)
            base_m[m_name] = func(d_base)
            
        # 3. Extract labels from MNE events
        mi_labels = data_proc.events[:, 2] if data_proc.events is not None else []
        ref_labels = ref_proc.events[:, 2] if ref_proc.events is not None else []

        # 4. Perform Tagging and get 0-1 Normalized Metrics
        bad_idx, tags, mi_metrics_norm = self._perform_tagging(mi_m, ref_m, base_m, ref_labels)
        
        # 5. Build Report
        return self._build_report(
            mi_metrics_raw=mi_m,
            mi_metrics_norm=mi_metrics_norm,
            raw_seg=d_mi,
            env_seg=env_seg,
            bad_idx=bad_idx,
            tags=tags,
            events=mi_labels
        )
        
    def _perform_tagging(self, mi_metrics, ref_metrics, base_metrics, ref_labels):
        n_epochs = next(iter(mi_metrics.values())).shape[0]
        tags = [[] for _ in range(n_epochs)]
        bad_indices = set()
        alpha = self.params.get('percentile', 5.0)
        
        # Dictionary to store the 0-1 normalized data to send back to the report
        mi_metrics_norm = {m_name: np.zeros_like(m_data) for m_name, m_data in mi_metrics.items()}
        
        for label, target_channels in self.channel_map.items():
            # Get numerical indices of channels for this specific limb
            ch_indices = [self.picks.index(ch) for ch in target_channels if ch in self.picks]
            if not ch_indices: continue
            
            # Mask to find Reference trials corresponding to this limb's movement
            m_ref = (ref_labels == label)
            if not np.any(m_ref): continue

            for m_name in mi_metrics.keys():
                # 1. Calculate baseline and reference medians using ONLY valid trials
                b_val = np.median(base_metrics[m_name][:, ch_indices], axis=0)
                r_val = np.median(ref_metrics[m_name][m_ref][:, ch_indices], axis=0)
                denom = (r_val - b_val) + 1e-9

                # 2. Calculate the threshold based on Reference data
                ref_norm = (ref_metrics[m_name][m_ref][:, ch_indices] - b_val) / denom
                threshold_ratio = self.params.get('percentage_threshold', 0.15)
                thresh_vals = np.ones(len(ch_indices)) * threshold_ratio
                
                # 3. Apply normalization to ALL Imagery trials for these channels
                # This ensures we have the 0-1 value for reporting regardless of the trial's class
                mi_norm_channels = (mi_metrics[m_name][:, ch_indices] - b_val) / denom
                mi_metrics_norm[m_name][:, ch_indices] = mi_norm_channels

                # 4. Cross-Vigilance Check
                # Check where ANY MI trial exceeded the threshold for these specific channels
                for local_idx, ch_idx in enumerate(ch_indices):
                    thresh = thresh_vals[local_idx]
                    ch_name = self.picks[ch_idx]
                    
                    # Find trials where this specific channel failed
                    fails = np.where(mi_norm_channels[:, local_idx] >= thresh)[0]
                    
                    for idx in fails:
                        val_01 = mi_norm_channels[idx, local_idx]
                        tags[idx].append(f"{ch_name}:{m_name} Overlap (Val={val_01:.2f})")
                        bad_indices.add(idx)

        return sorted(list(bad_indices)), tags, mi_metrics_norm            

    def _build_report(self, mi_metrics_raw, mi_metrics_norm, raw_seg, env_seg, bad_idx, tags, events):
        # 1. Package metrics for EDAMetric
        # Create strict metric objects
        report_metrics = {
            "RMS": EDAMetric(
                values={ch: mi_metrics_raw["RMS"][:, j].tolist() for j, ch in enumerate(self.picks)},
                unit="uV"
            ),
            "Envelope Int": EDAMetric(
                values={ch: mi_metrics_raw["EnvInt"][:, j].tolist() for j, ch in enumerate(self.picks)},
                unit="area"
            ),
            "RMS norm": EDAMetric(
                values={ch: mi_metrics_norm["RMS"][:, j].tolist() for j, ch in enumerate(self.picks)},
                unit="0-1 Norm"
            ),
            "Envelope Int": EDAMetric(
                values={ch: mi_metrics_norm["EnvInt"][:, j].tolist() for j, ch in enumerate(self.picks)},
                unit="area"
            )
        }

        summary = (
            f"Analyzer: EMG Consistency (Class-Specific Scaling)\n"
            f"Flagged Trials: {len(bad_idx)} / {len(tags)}\n"
            f"Percentage Threshold: {self.params.get('percentage_threshold', 5.0)}%\n"
            f"Channels Monitored: {', '.join(self.picks)}"
        )

        return EDAReport(
            summary=summary,
            visuals=["hist", "scatter", "mapping", "inspect"],
            artifacts=EDAArtifacts(
                bad_indices=bad_idx,
                bad_tags=[tags[i] for i in bad_idx]
            ),
            metrics=report_metrics,
            metadata={
                "event_types": events.tolist() if hasattr(events, 'tolist') else events,
                "tmin": self.params.get('tmin', 0),
                "tmax": self.params.get('tmax', 1),
                "channel_mapping": self.channel_map
            },
            inspection_data={
                "x-axis": np.linspace(self.params.get('tmin', 0), self.params.get('tmax', 1), raw_seg.shape[-1]).tolist(),
                "raw_traces": {ch: raw_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)},
                "envelope_traces": {ch: env_seg[:, j, :].tolist() for j, ch in enumerate(self.picks)}
            }
        )

    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data**2, axis=-1))
    
    def _calculate_envelope_integral(self, data):
        signal_envelopes = envelope(data, residual=None)
        integral_envelope = np.trapezoid(signal_envelopes, axis=-1)
        return integral_envelope, signal_envelopes
    
    def _preprocess(self, raw):
        raw.filter(self.params['bandpass_filter'][0], self.params['bandpass_filter'][1], 
                         picks=self.picks, verbose=False)
        raw.notch_filter(freqs=self.params['notch_filter'], 
                         picks=self.picks, verbose=False)
        epochs = self._ensure_epochs(raw)
        
        self.ch_names = epochs.ch_names
        self.sfreq = epochs.info['sfreq']
        return epochs
    
    def _ensure_epochs(self, inst) -> mne.Epochs:
        if isinstance(inst, mne.Epochs):
            return inst.copy().pick(self.picks) if self.picks else inst
        
        try:
            events = mne.find_events(inst, stim_channel='stim', verbose=False)
        except:
            events = np.array([])

        if len(events) > 0:
            return mne.Epochs(inst, events, tmin=self.params['tmin'], 
                              tmax=self.params['tmax'], baseline=None, 
                              preload=True, verbose=False, picks=self.picks)
        else:
            print(f"[Analyzer] No events found. Generating {self.params['n_pseudo_trials']} pseudo-epochs.")
            return self._create_pseudo_epochs(inst.copy().pick(self.picks))
    
    def _create_pseudo_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        sfreq = raw.info['sfreq']   
        data = raw.get_data() 
        n_chans, L = data.shape
        W = int((self.params['tmax'] - self.params['tmin']) * sfreq)
        N = self.params['n_pseudo_trials']

        if L < W:
            raise ValueError(f"Signal length ({L}) is shorter than window size ({W}).")

        start_indices = np.linspace(0, L - W, N).astype(int)
        epochs_data = np.array([data[:, start : start + W] for start in start_indices])

        events = np.column_stack([
            np.arange(N) * W, 
            np.zeros(N, dtype=int), 
            -1*np.ones(N, dtype=int)
        ])
        
        event_id = {'pseudo_trial': -1}
        
        return mne.EpochsArray(epochs_data, raw.info, events=events, 
                               tmin=self.params['tmin'], event_id=event_id, 
                               verbose=False)