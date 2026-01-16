import os
from abc import ABC, abstractmethod

class BaseEEGDataset(ABC):
    """
    Abstract Base Class for all EEG Datasets.
    Enforces a standard interface for loading data with specific contexts.
    """
    def __init__(self, dataset_path, subject_list, sessions, event_id=None, code=None):
        """
        Parameters:
        -----------
        dataset_path : str
            Absolute path to the dataset folder.
        subject_list : list of int
            List of valid subject IDs.
        event_id : dict
            Mapping of event names to integer codes (e.g., {'left_hand': 1}).
        code : str
            Unique identifier for the dataset (e.g., 'Cho2017').
        """
        self.dataset_path = dataset_path
        self.subject_list = subject_list
        self.sessions = sessions
        self.event_id = event_id if event_id is not None else {}
        self.code = code if code is not None else self.__class__.__name__
        self.subjects_metadata = None
        self.data_availables = None
        # Verify path exists
        if not os.path.exists(self.dataset_path):
            print(f"WARNING: Dataset path does not exist: {self.dataset_path}")

    def _validate_inputs(self, subjects, data_to_load):
        """
        Helper to standardize inputs.
        """
        if subjects is None:
            subjects = self.subject_list
        
        # Ensure subjects is a list
        if not isinstance(subjects, list):
            if isinstance(subjects, (int, float)):
                subjects = [int(subjects)]
            else:
                subjects = list(subjects)

        # Ensure data_to_load is a list
        if data_to_load is None:
            raise ValueError("Data to load cannot be None. Please specify (e.g., ['MI', 'rest']).")
        if isinstance(data_to_load, str):
            data_to_load = [data_to_load]

        return subjects, data_to_load

    @abstractmethod
    def get_data(self, subjects=None, data_to_load=None):
        """
        Main method to load data. Must be implemented by child classes.
        
        Returns:
        --------
        dict : {
            "subject_X": {
                "session_Y": {
                    "data_type_Z": {
                        "run_W": mne.io.Raw
                    }
                }
            }
        }
        """
        pass
    
    def get_subject_metadata(self):
        """Returns the pandas DataFrame with subject info."""
        if hasattr(self, 'subjects_metadata'):
            return self.subjects_metadata
        elif hasattr(self, 'metadata'): # Fallback if you named it 'metadata'
            return self.metadata
        return None
    
    def get_subjects(self):
        return self.subject_list
    
    def get_data_available(self):
        return self.data_availables
    
    def get_sessions(self):
        return self.sessions