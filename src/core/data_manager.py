from typing import List, Dict, Union, Optional, Any
import mne
from src.eeg_datasets import Cho2017, Dreyer2023A, Dreyer2023B, Dreyer2023C

# Diccionario de mapeo para instanciación dinámica
DATASETS_AVAILABLES = {
    "Cho2017": Cho2017,
    "Dreyer2023A": Dreyer2023A,
    "Dreyer2023B": Dreyer2023B,
    "Dreyer2023C": Dreyer2023C,
}

class DataManager:
    """
    Factory Class encargada de la gestión, instanciación y recuperación de datos de EEG.
    
    Actúa como una capa de abstracción entre la Interfaz de Usuario (GUI) y las clases
    específicas de cada base de datos. Su responsabilidad principal es garantizar que
    los datos solicitados se entreguen en un formato estandarizado (Diccionario de Raws)
    independientemente de la estructura interna de la base de datos original.

    Attributes:
        active_dataset_instance (Any): Instancia de la clase de dataset seleccionada.
        active_dataset_name (str): Nombre identificador del dataset en uso.
        DATASETS (dict): Mapeo de nombres a clases de datasets disponibles.
    """

    def __init__(self):
        self.active_dataset_instance: Any = None
        self.active_dataset_name: Optional[str] = None
        self.DATASETS = DATASETS_AVAILABLES

    def get_available_datasets(self) -> List[str]:
        """
        Retorna la lista de nombres de datasets implementados en el sistema.
        
        Returns:
            List[str]: Nombres de los datasets (ej: ['Cho2017', 'Dreyer2023A']).
        """
        return list(self.DATASETS.keys())

    def load_dataset_class(self, name: str) -> Any:
        """
        Instancia dinámicamente la clase del dataset seleccionado.
        
        Args:
            name (str): Nombre del dataset a cargar.
            
        Returns:
            Any: Instancia de la clase del dataset.
            
        Raises:
            ValueError: Si el nombre no existe en DATASETS_AVAILABLES.
        """
        if name not in self.DATASETS:
            raise ValueError(f"Dataset '{name}' no reconocido en la configuración.")
        
        if self.active_dataset_name != name:
            print(f"[DataManager] Inicializando clase de dataset: {name}")
            self.active_dataset_instance = self.DATASETS[name]()
            self.active_dataset_name = name
            
        return self.active_dataset_instance

    def get_metadata(self) -> Optional[Dict[str, List[Any]]]:
        """
        Extrae la información de sujetos y contextos disponibles del dataset activo.
        
        Returns:
            Optional[Dict]: Diccionario con llaves 'subjects' (List[int]) y 
                           'contexts' (List[str]). Retorna None si no hay dataset cargado.
        """
        if not self.active_dataset_instance:
            return None
            
        # Normalización de nombres de atributos según la implementación de base_dataset
        contexts = getattr(self.active_dataset_instance, 'data_availables', []) 
        if not contexts:
            contexts = getattr(self.active_dataset_instance, 'contexts_availables', [])
            
        return {
            "subjects": self.active_dataset_instance.get_subjects(),
            "contexts": contexts
        }

    def load_data(self, subject_id: int, contexts: List[str]) -> Dict[str, mne.io.BaseRaw]:
        """
        Carga múltiples señales (contextos) para un sujeto específico.
        
        Este método maneja la perforación (drilling) de la estructura anidada 
        típica de los datasets {Sujeto: {Sesión: {Contexto: {Run: Raw}}}}.
        
        Args:
            subject_id (int): ID numérico del sujeto.
            contexts (List[str]): Lista de etiquetas de contexto a cargar 
                                 (ej: ['motor_imagery', 'rest', 'real_movement']).
            
        Returns:
            Dict[str, mne.io.BaseRaw]: Un diccionario donde la llave es el nombre 
                                       del contexto y el valor es el objeto Raw de MNE.
                                       Ej: {'rest': RawObject, 'motor_imagery': RawObject}
                                       
        Raises:
            RuntimeError: Si no se ha seleccionado un dataset previamente.
        """
        if not self.active_dataset_instance:
            raise RuntimeError("Error de flujo: Debe llamar a load_dataset_class primero.")

        print(f"[DataManager] Cargando señales para Sub {subject_id}: {contexts}")
        
        # 1. Llamada a la lógica interna del dataset (puede cargar múltiples contextos)
        full_data_dict = self.active_dataset_instance.get_data(
            subjects=[subject_id], 
            data_to_load=contexts
        )
        
        if not full_data_dict or subject_id not in full_data_dict:
            print(f"[DataManager] Advertencia: No se encontraron datos para el sujeto {subject_id}")
            return {}

        workspace_raws: Dict[str, mne.io.BaseRaw] = {}

        # 2. Procesamiento y aplanamiento de la estructura anidada
        try:
            sessions = full_data_dict[subject_id]
            # Por simplicidad en la Workstation, tomamos la primera sesión disponible
            first_sess_label = list(sessions.keys())[0]
            session_content = sessions[first_sess_label]
            
            for ctx_label in contexts:
                if ctx_label in session_content:
                    runs = session_content[ctx_label]
                    # Tomamos el primer run (usualmente 'run_1' o 'recording')
                    first_run_label = list(runs.keys())[0]
                    raw_obj = runs[first_run_label]
                    
                    # Verificamos si el objeto está envuelto en otro dict {'recording': raw}
                    if isinstance(raw_obj, dict) and 'recording' in raw_obj:
                        workspace_raws[ctx_label] = raw_obj['recording']
                    elif isinstance(raw_obj, mne.io.BaseRaw):
                        workspace_raws[ctx_label] = raw_obj
                    else:
                        print(f"[DataManager] Error: El objeto para {ctx_label} no es un MNE Raw válido.")

            return workspace_raws
            
        except Exception as e:
            print(f"[DataManager] Error crítico procesando estructura de datos: {e}")
            return {}


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