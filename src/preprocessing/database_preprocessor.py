import os
import numpy as np
import json
from .preprocess_eeg import EEGPreprocessor
from src.utils import save_dict_as_json
from global_config import DATABASES_PATH
import os
import json
import numpy as np
import mne
from src.utils.json_utils import save_dict_as_json


class EEGDatabasePreprocessor:
    def __init__(self, dataset, preprocessing_name, preprocessing_config):
        self.dataset = dataset
        self.preprocessing_name = preprocessing_name
        self.config = preprocessing_config
        self.output_path = os.path.join(DATABASES_PATH, 'preprocessed_databases', self.preprocessing_name, self.dataset.__class__.__name__)
        os.makedirs(self.output_path, exist_ok=True)
        self.worker = self._init_preprocessor()
        self._description_saved =False

    def _init_preprocessor(self):
        return EEGPreprocessor(
            raw_pipeline=self.config.get('raw_pipeline'),
            epoching_config=self.config.get('epoching'),
            epoch_pipeline=self.config.get('epoch_pipeline')
        )

    def run(self):
        # 1. Save Initial Metadata (Original Info + Config)
        self._save_dataset_metadata(results=None)
        
        description_updated = False
        
        # 2. Get list of subjects
        subjects_list = self.dataset.get_subjects()
        print(f"Starting preprocessing for {len(subjects_list)} subjects...")

        for subject in subjects_list:
            
            try:
                full_subject_data = self.dataset.get_data(subjects=[subject])
                if not full_subject_data or subject not in full_subject_data:
                    continue
                data_subject = full_subject_data[subject]
            except Exception as e:
                print(f"  CRITICAL ERROR loading subject {subject}: {e}")
                continue

            if 'extra_info' in data_subject:
                extra_info = data_subject.pop('extra_info')
                save_dict_as_json(os.path.join(self.output_path, f"subject_{subject}"), extra_info, f"subject_{subject}_extra_info.json")

            for session_label, session_content in data_subject.items():
                for type_signal, runs in session_content.items():
                    for run_label, raw_data in runs.items():
                        
                        # --- PROCESS ---
                        try:
                            results = self.worker.process(raw_data)
                        except Exception as e:
                            print(f"ERROR processing run: {e}")
                            continue
                        
                        # --- UPDATE GLOBAL DESCRIPTION (Run Once) ---
                        if not description_updated:
                            self._save_dataset_metadata(results=results)
                            description_updated = True

                        # --- SAVE RESULTS ---
                        self._save_results(
                            results, 
                            base_path=os.path.join(self.output_path, f"subject_{subject}", session_label), 
                            prefix=f"subject_{subject}_{session_label}_{type_signal}_{run_label}"
                        )
        print("--- Preprocessing Finished ---")


    def _save_dataset_metadata(self, results=None):
        """
        Generates and saves the 'dataset_description.json' and 'preprocessing_pipeline.json'.
        If 'results' is None:
            Saves only the original dataset attributes (before processing).
        If 'results' is provided:
            Updates the file with the ACTUAL output properties (sfreq, channels, events)
            extracted from the processed data.
        """
        # 1. Base Information (Always present)
        info = {
            "dataset_code": getattr(self.dataset, 'code', 'Unknown'),
            "subject_list": self.dataset.get_subjects(),
            "data_availables": getattr(self.dataset, 'data_availables', None),
            # Original attributes (for reference)
            "original_properties": {
                "sfreq": getattr(self.dataset, 'sfreq', None),
                "n_channels": len(getattr(self.dataset, 'ch_names', [])),
                "ch_names": getattr(self.dataset, 'ch_names', None),
                "montage": getattr(self.dataset, 'standard_montage', None),
                "event_id": getattr(self.dataset, 'event_id', {}),
                "unit_factor": getattr(self.dataset, 'unit_factor', 1.0)
            }
        }
        # 2. Update with Processed Info (The "Ground Truth" of the output)
        if results is not None:
            print("Updating dataset description with actual output properties...")

            mne_object = True if isinstance(results, mne.io.BaseRaw) or isinstance(results, mne.Epochs) else False
            
            if mne_object:
                # Extract from MNE Object
                info['output_properties'] = {
                    'sfreq': mne_object.info['sfreq'],
                    'n_channels': len(mne_object.ch_names),
                    'ch_names': mne_object.ch_names,
                    'montage': mne_object.get_montage().name if mne_object.get_montage() else "Unknown",
                    'event_id': getattr(mne_object, 'event_id', None)
                }
            else:
                # Extract from Dictionary (e.g. from extract_data_from_epochs)
                info['output_properties'] = {
                    'sfreq': results['sfreq'],
                    'n_channels': len(results['ch_names']),
                    'ch_names': results['ch_names'],
                    'montage': results.get('montage', "Unknown"),
                    'event_id': results.get('event_id', None)
                }

        # 3. Save to Disk
        # We merge existing file content if it exists to avoid losing data? 
        # Actually, it's safer to overwrite with the complete object we just built.
        save_dict_as_json(self.output_path, info, "dataset_description.json")
        
        # 4. Save Pipeline Config (Only need to do this once, but safe to repeat)
        if results is None:
            save_dict_as_json(self.output_path, self.config, "preprocessing_pipeline.json")
            print(f"Saved metadata to {self.output_path}")
        
    def _save_results(self, results, base_path, prefix):
        
        os.makedirs(base_path, exist_ok=True)
        
        # 1. Save Raw
        if 'raw' in results and 'epochs' not in results and 'data' not in results:
            results['raw'].save(os.path.join(base_path, f"{prefix}_raw.fif"), overwrite=True, verbose=False)

        # 2. Save Epochs
        if 'epochs' in results and 'data' not in results:
            results['epochs'].save(os.path.join(base_path, f"{prefix}_epoch.fif"), overwrite=True, verbose=False)

        # 3. Save Extracted Data
        if 'data' in results:
            data_content = results['data']
            if isinstance(data_content, dict):
                for key, val in data_content.items():
                    if isinstance(val, np.ndarray):
                        np.save(os.path.join(base_path, f"{prefix}_{key}.npy"), val)
            elif isinstance(data_content, np.ndarray):
                np.save(os.path.join(base_path, f"{prefix}_data.npy"), data_content)

        # 4. Save Labels
        if 'labels' in results and isinstance(results['labels'], np.ndarray):
             if len(results['labels']) > 0:
                np.save(os.path.join(base_path, f"{prefix}_labels.npy"), results['labels'])
