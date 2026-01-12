# src/eda/eda_database_processor.py
import os
import json
from src.eda.eda_processor import EDAProcessor
from src.utils.json_utils import save_dict_as_json
from global_config import DATABASES_PATH

class EDADatabaseProcessor:
    def __init__(self, dataset, eda_name, eda_pipeline):
        self.dataset = dataset
        self.eda_name = eda_name
        self.eda_pipeline = eda_pipeline
        
        # Mirroring your output path structure
        self.output_path = os.path.join(
            DATABASES_PATH, 'eda_reports', 
            self.eda_name, self.dataset.__class__.__name__
        )
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize the worker
        self.worker = EDAProcessor(self.eda_pipeline)

    def run(self):
        subjects_list = self.dataset.get_subjects()
        print(f"--- Starting EDA Diagnostic for {len(subjects_list)} subjects ---")

        for subject in subjects_list:
            try:
                full_subject_data = self.dataset.get_data(subjects=[subject])
                if not full_subject_data or subject not in full_subject_data:
                    continue
                data_subject = full_subject_data[subject]
            except Exception as e:
                print(f"  CRITICAL ERROR loading subject {subject} for EDA: {e}")
                continue

            # Nested loops following your DatabasePreprocessor structure
            for session_label, session_content in data_subject.items():
                for type_signal, runs in session_content.items():
                    for run_label, raw_data in runs.items():
                            # 1. RUN EDA (FalseMI usually runs on RAW to see baseline)
                            report = self.worker.run(raw_data)
                            
                            # # 2. SAVE DIAGNOSTIC JSON
                            # prefix = f"subject_{subject}_{session_label}_{type_signal}_{run_label}"
                            # save_path = os.path.join(self.output_path, f"subject_{subject}", session_label)
                            
                            # self._save_report(report, save_path, f"{prefix}_eda.json")
        
        print("--- EDA Diagnostic Finished ---")

    def _save_report(self, report, base_path, filename):
        os.makedirs(base_path, exist_ok=True)
        full_path = os.path.join(base_path, filename)
        with open(full_path, 'w') as f:
            json.dump(report, f, indent=4)