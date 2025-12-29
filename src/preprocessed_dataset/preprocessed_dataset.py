import os 
import numpy as np
import pandas as pd

from global_config import DATABASES_PATH
from src.utils.json_utils import load_json_file
from joblib import Parallel, delayed

class PreprocessedDataset():
    def __init__(self, db_name, config, channels=None, classes_to_return=None):
        
        self.db_name = db_name
        self.config = config
        self.code = db_name
        self.path_to_load = os.path.join(DATABASES_PATH, 'preprocessed_databases', config, db_name)
        
        self.sessions = None
        self.n_sessions = None
        self.sessions_per_subject = None
        self.subject_list = None
        self.sfreq = None
        self.montage = None
        self.channels = None
        self.classes_to_return = classes_to_return 
        self.channels_order = None
        self.event_id = None
        self.interval = None
        self.unit_factor = None
        self.__load_info_db()
        idx = None
        if channels:
            if all(ch in self.channels for ch in channels):
                # First we need to obtain the index of the channels
                idx = [self.channels.index(ch) for ch in channels]
        self.selected_channels =  idx
        
    def __load_info_db(self):
        path = os.path.join(self.path_to_load, 'information_db.json')
        info_db = load_json_file(path)
        
        self.subject_list = info_db['subject_list'] if 'subject_list' in info_db else list(range(1, info_db['nsubjects']+1)) 
        self.sessions = list(range(1, info_db['nsessions']+1))
        self.n_sessions = info_db['nsessions']
        self.sessions_per_subject = info_db['nsessions']
        self.nruns = info_db['nruns']
        self.sfreq = info_db['sfreq']
        self.channels = info_db['channels']
        self.channels_order = info_db['channels']
        self.events = info_db['event_id']
        self.event_id = info_db['event_id']
        if self.classes_to_return is not None:
            self.classes_to_return = [self.event_id[cls] for cls in self.classes_to_return if cls in self.event_id]
        self.interval = info_db['interval'] if 'interval' in info_db else None
        self.unit_factor = info_db['unit_factor']
        
    def select_channels(self, channels):
        if all(ch in self.channels for ch in channels):
            idx = [self.channels.index(ch) for ch in channels]
            self.selected_channels = idx

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
            """
            Get the path to the data for a given subject.

            Args:
                subject (int): The subject number to download data for.
                path (str): Not used in this implementation.
                force_update (bool): Not used in this implementation.
                update_path (str): Not used in this implementation.
                verbose (bool): Not used in this implementation.

            Returns:
                dict: A dict with the file paths to the downloaded data for the specified subject.
            """
            if subject not in self.subject_list:
                raise (ValueError("Invalid subject number"))

            """Only implement the load data from a local path"""
            subject_paths = {}
            for session in range(1, self.n_sessions+1):
                subject_paths[session] = {}
                for run in range(1, self.nruns+1):
                    path_eeg=os.path.join(self.path_to_load, f"subject_{subject}_session_{session}_run_{run}_eeg_preprocessed.npy")
                    path_label=os.path.join(self.path_to_load, f"subject_{subject}_session_{session}_run_{run}_labels.npy")
                    subject_paths[session][run] = {'eeg': path_eeg, 'labels': path_label}
            return subject_paths

    def _load_run_data(self, subject, file_path_dict, session, run):
        run_dict = file_path_dict[session][run]
        eeg_data = np.load(run_dict['eeg'])
        labels_eeg = np.load(run_dict['labels'])

        if self.selected_channels:
            eeg_data = eeg_data[:, self.selected_channels, :]

        if self.classes_to_return is not None:
            mask = np.isin(labels_eeg, self.classes_to_return)
            eeg_data = eeg_data[mask]
            labels_eeg = labels_eeg[mask]

        metadata = pd.DataFrame({
            'subject': subject,
            'session': f"session_{session}",
            'run': f"run_{run}",
            'labels': labels_eeg
        })

        return {
            'data_eeg': eeg_data,
            'labels_eeg': labels_eeg,
            'metadata': metadata
        }


    def _get_single_subject_data(
        self, subject, return_raw=False, return_epochs=False, return_as_dict=True, sessions=None
                                ):
        
        if return_raw or return_epochs:
            raise NotImplementedError("Return type not implemented.")

        file_path_dict = self.data_path(subject)

        if return_as_dict:
            sessions_dict = {}
            for session in range(1, self.n_sessions + 1):
                session_key = f"session_{session}"
                sessions_dict[session_key] = {}
                for run in range(1, self.nruns + 1):
                    run_data = self._load_run_data(subject, file_path_dict, session, run)
                    sessions_dict[session_key][f"run_{run}"] = run_data
            return sessions_dict
        else:
            X, labels, metadata = [], [], []
            session_indices = (
                range(1, self.n_sessions + 1)
                if sessions is None
                else sessions
            )

            for session in session_indices:
                for run in range(1, self.nruns + 1):
                    run_data = self._load_run_data(subject, file_path_dict, session, run)
                    X.append(run_data['data_eeg'])
                    labels.append(run_data['labels_eeg'])
                    metadata.append(run_data['metadata'])

            return (
                np.concatenate(X, axis=0),
                np.concatenate(labels, axis=0),
                pd.concat(metadata, ignore_index=True),
            )

    def get_data(self, subjects=None, sessions=None, return_as_dict=False, n_jobs=1):
        """
        Parámetros adicionales:
        - n_jobs: int
            Número de procesos en paralelo. Usá -1 para usar todos los núcleos disponibles.
        """
        if subjects is None:
            subjects = self.subject_list

        if return_as_dict:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._get_single_subject_data)(s, return_as_dict=True)
                for s in subjects
            )
            return {f"subject_{s}": r for s, r in zip(subjects, results)}

        # Versión plana (X, y, metadata)
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._get_single_subject_data)(s, return_as_dict=False, sessions=sessions)
            for s in subjects
        )

        X, y, metadata = zip(*results)  # descomprime resultados
        return (
            np.concatenate(X, axis=0),
            np.concatenate(y, axis=0),
            pd.concat(metadata, ignore_index=True),
        )



