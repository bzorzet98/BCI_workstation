import mne
from typing import List, Dict, Optional, Any, Union, Tuple
from src.eeg_datasets import Cho2017, Dreyer2023A, Dreyer2023B, Dreyer2023C

# Global mapping for dynamic instantiation
DATASETS_AVAILABLE = {
    "Cho2017": Cho2017,
    "Dreyer2023A": Dreyer2023A,
    "Dreyer2023B": Dreyer2023B,
    "Dreyer2023C": Dreyer2023C,
}

class DataManager:
    """
    Factory class responsible for managing, instantiating, and retrieving EEG data.
    
    It serves as an abstraction layer between the User Interface (GUI) and the specific 
    EEG dataset classes. Its primary role is to ensure that data requested by the app 
    is delivered in a standardized format (Dictionary of MNE Raw objects), regardless 
    of the original database's internal hierarchy.

    Attributes:
        active_dataset_instance (Any): The instantiated dataset class currently in use.
        active_dataset_name (str): The string identifier of the selected dataset.
        DATASETS (dict): Mapping of dataset names to their respective classes.
    """

    def __init__(self):
        self.active_dataset_instance: Any = None
        self.active_dataset_name: Optional[str] = None
        self.DATASETS = DATASETS_AVAILABLE

    def get_available_datasets(self) -> List[str]:
        """
        Retrieves the list of dataset names implemented in the system.
        
        Returns:
            List[str]: A list of dataset identifiers (e.g., ['Cho2017', 'Dreyer2023A']).
        """
        return list(self.DATASETS.keys())

    def load_dataset_class(self, name: str) -> Any:
        """
        Dynamically instantiates the selected dataset class.
        
        Args:
            name (str): The identifier of the dataset to be initialized.
            
        Returns:
            Any: The instantiated dataset class.
            
        Raises:
            ValueError: If the name does not match any entry in DATASETS_AVAILABLE.
        """
        if name not in self.DATASETS:
            raise ValueError(f"Dataset '{name}' is not recognized in the configuration.")
        
        if self.active_dataset_name != name:
            print(f"[DataManager] Initializing dataset class: {name}")
            self.active_dataset_instance = self.DATASETS[name]()
            self.active_dataset_name = name
            
        return self.active_dataset_instance

    def get_metadata(self) -> Optional[Dict[str, List[Union[int, str]]]]:
        """
        Extracts available subject IDs and context labels from the active dataset.
        
        Returns:
            Optional[Dict]: A dictionary containing 'subjects' (List of ints) and 
                           'contexts' (List of strings). Returns None if no dataset is loaded.
        """
        if not self.active_dataset_instance:
            return None
            
        # Support both 'data_availables' and 'contexts_availables' attribute names
        contexts = getattr(self.active_dataset_instance, 'data_availables', []) 
        if not contexts:
            contexts = getattr(self.active_dataset_instance, 'contexts_availables', [])
            
        return {
            "subjects": self.active_dataset_instance.get_subjects(),
            "contexts": contexts,
            "sessions": self.active_dataset_instance.get_sessions()
            }

    def load_data(self, subject_id: int, contexts: List[str], session_id: Optional[str] = None) -> Tuple[Dict[str, mne.io.BaseRaw], Dict[str, Any]]:
        """
        Loads multiple signals and FLATTENS sub-structures (Runs/Noise Types) 
        into individual workspace entries.
        """
        if not self.active_dataset_instance:
            raise RuntimeError("No dataset loaded.")

        full_data_dict = self.active_dataset_instance.get_data(
            subjects=[subject_id], 
            data_to_load=contexts
        )
        
        if not full_data_dict or subject_id not in full_data_dict:
            return {}, {}

        subject_entry = full_data_dict[subject_id]
        extra_info = subject_entry.get('extra_info', {})

        # Resolve Session
        if session_id is None:
            available_sessions = [k for k in subject_entry.keys() if k.startswith('session')]
            session_id = available_sessions[0] if available_sessions else None

        if not session_id or session_id not in subject_entry:
            return {}, extra_info

        session_content = subject_entry[session_id]
        workspace_raws = {}

        # --- NEW DEEP-FLATTENING LOGIC ---
        for context_label in contexts:
            if context_label in session_content:
                runs_dict = session_content[context_label]
                
                # Check how many sub-items (runs/noise types) this context has
                sub_items = list(runs_dict.keys())
                
                for sub_key in sub_items:
                    raw_obj = runs_dict[sub_key]
                    workspace_name = f"{context_label}:{sub_key}"
                    
                    # Extraction
                    if isinstance(raw_obj, dict) and 'recording' in raw_obj:
                        workspace_raws[workspace_name] = raw_obj['recording']
                    elif isinstance(raw_obj, mne.io.BaseRaw):
                        workspace_raws[workspace_name] = raw_obj
                        
        return workspace_raws, extra_info