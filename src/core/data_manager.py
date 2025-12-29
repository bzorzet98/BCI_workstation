from src.eeg_datasets import Cho2017
from src.eeg_datasets import Dreyer2023A, Dreyer2023B, Dreyer2023C
# Import your preprocessed loader here too if you have it

DATASETS_AVAILABLES = {
    "Cho2017": Cho2017,
    "Dreyer2023A": Dreyer2023A,
    "Dreyer2023B": Dreyer2023B,
    "Dreyer2023C": Dreyer2023C,
}

class DataManager:
    """
    Factory class to manage dataset instantiation and data retrieval.
    Acts as the bridge between the App and the Dataset Classes.
    """

    def __init__(self):
        self.active_dataset_instance = None
        self.active_dataset_name = None
        self.DATASETS = DATASETS_AVAILABLES

    def get_available_datasets(self):
        return list(self.DATASETS.keys())

    def load_dataset_class(self, name):
        """Instantiates the dataset class (e.g., Cho2017)."""
        if name not in self.DATASETS:
            raise ValueError(f"Dataset {name} not found.")
        
        print(f"Initializing {name}...")
        self.active_dataset_instance = self.DATASETS[name]()
        self.active_dataset_name = name
        return self.active_dataset_instance

    def load_dataset_class(self, name):
        """Instantiates the dataset class and returns it."""
        if name not in self.DATASETS:
            raise ValueError(f"Dataset {name} not found.")
        
        # Only instantiate if we haven't already, or if we switched datasets
        if self.active_dataset_name != name:
            print(f"Initializing {name}...")
            self.active_dataset_instance = self.DATASETS[name]()
            self.active_dataset_name = name
            
        return self.active_dataset_instance

    def get_metadata(self):
        """Returns subject list and contexts for the active dataset."""
        if not self.active_dataset_instance:
            return None
            
        # Inspect the instance for context info
        # We check both naming conventions you used
        contexts = getattr(self.active_dataset_instance, 'data_availables', []) 
        if not contexts:
            contexts = getattr(self.active_dataset_instance, 'contexts_availables', [])
            
        return {
            "subjects": self.active_dataset_instance.get_subjects(),
            "contexts": contexts
        }

    def load_data(self, subject_id, data_to_load):
        """
        Loads the data and returns the FIRST available Raw object found.
        
        Returns:
            mne.io.BaseRaw or None
        """
        if not self.active_dataset_instance:
            raise RuntimeError("No dataset selected.")

        print(f"Loading Subject {subject_id}, Context {data_to_load}...")
        
        # 1. Call the dataset's get_data method
        # This returns the complex dict: {sub: {sess: {ctx: {run: raw}}}}
        data_to_load = data_to_load if isinstance(data_to_load, str) else data_to_load
        full_data = self.active_dataset_instance.get_data(
            subjects=[subject_id], 
            data_to_load=[data_to_load] # or data_to_load=[data_to_load] depending on your base class
        )
        
        if not full_data or subject_id not in full_data:
            return None

        # 2. Drill down to find the Raw object
        # We assume the structure: Session -> Data Type -> Run
        # We return the first valid Raw object we find.
        try:
            sessions = full_data[subject_id]
            first_sess = list(sessions.values())[0] # Session 1
            
            data_type_data = first_sess.get(data_to_load)
            if not data_type_data: return None
            
            first_run = list(data_type_data.values())[0] # Run 1 or 'recording'
            
            # Handle the specific wrapper dict {'recording': raw} if it exists
            if isinstance(first_run, dict) and 'recording' in first_run:
                return first_run['recording']
            elif isinstance(first_run, dict):
                 # Fallback for nested dicts not following exact structure
                 return list(first_run.values())[0]
            
            return first_run # It is the Raw object
            
        except Exception as e:
            print(f"Error drilling down into data dict: {e}")
            return None