import inspect
import os
import mne
from src.preprocessing import preprocessing_functions
import pkgutil
import importlib
from src.utils import import_class

class Registry:
    """
    Knowledge base for available processing methods and their arguments.
    """
    
    ALLOWED_RAW_METHODS = [
        'filter', 'notch_filter', 'resample', 'pick_channels', 'drop_channels', 
        'set_eeg_reference', 'crop', 'set_annotations'
    ]
    
    ALLOWED_EPOCH_METHODS = [
        'filter', 'pick_channels', 'drop_channels', 'resample', 'crop', 
        'apply_baseline', 'shift_time'
    ]

    CUSTOM_FUNCTIONS = [
        name for name, obj in inspect.getmembers(preprocessing_functions) 
        if inspect.isfunction(obj) and not name.startswith('_')
    ]

    @staticmethod
    def get_available_steps(data_type):
        """
        Returns a list of available actions based on whether data is Raw or Epochs.
        """
        methods = []
        if isinstance(data_type, mne.io.BaseRaw) or data_type == 'raw':
            methods = [{'name': m, 'type': 'method'} for m in Registry.ALLOWED_RAW_METHODS]
            
            # EXPLICITLY ADD EPOCHING AS A SPECIAL ACTION FOR RAW DATA
            methods.append({'name': 'Epochs', 'type': 'class_init'}) 

        elif isinstance(data_type, mne.Epochs) or data_type == 'epochs':
            methods = [{'name': m, 'type': 'method'} for m in Registry.ALLOWED_EPOCH_METHODS]
            
        functions = [{'name': f, 'type': 'function'} for f in Registry.CUSTOM_FUNCTIONS]
        
        return methods + functions
    
    @staticmethod
    def get_analyzer_params(class_name, module_path):
        """
        Inspecciona dinámicamente el __init__ de una clase para obtener
        sus argumentos y valores por defecto.
        """
        try:
            cls = import_class(class_name, module_path)
            # Obtenemos la firma del método __init__
            sig = inspect.signature(cls.__init__)
            
            params = {}
            for name, param in sig.parameters.items():
                # Ignoramos argumentos internos del sistema
                if name in ['self', 'kwargs', 'args']:
                    continue
                
                # Si no tiene valor por defecto, ponemos None
                val = param.default if param.default is not inspect.Parameter.empty else None
                params[name] = val
                
            return params
        except Exception as e:
            print(f"Error en Registry: {e}")
            return {}
    
    @staticmethod
    def discover_eda_analyzers():
        """
        Escanea la carpeta src/eda/analyzers y devuelve un diccionario
        con las clases encontradas {NombreClase: RutaModulo}.
        """
        analyzers = {}
        # Ruta física de la carpeta de analizadores
        pkg_path = os.path.join("src", "eda", "analyzers")
        
        if not os.path.exists(pkg_path):
            return analyzers

        # pkgutil recorre todos los archivos .py de la carpeta
        for loader, module_name, is_pkg in pkgutil.iter_modules([pkg_path]):
            if module_name in ["base_analyzer", "__init__"]: 
                continue # Saltamos la base y el init
            
            # Importamos el módulo dinámicamente
            full_module_path = f"src.eda.analyzers.{module_name}"
            module = importlib.import_module(full_module_path)
            
            # Buscamos clases dentro del módulo
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Verificamos que la clase esté definida en ese archivo 
                # y no sea una importada (como BaseAnalyzer)
                if obj.__module__ == full_module_path:
                    analyzers[name] = full_module_path
                    
        return analyzers
    @staticmethod
    def get_defaults(name, step_type):
        """
        Introspects the function/method signature to return a dict of default kwargs.
        """
        func_obj = None
        
        if step_type == 'method':
            if hasattr(mne.io.BaseRaw, name):
                func_obj = getattr(mne.io.BaseRaw, name)
            elif hasattr(mne.Epochs, name):
                func_obj = getattr(mne.Epochs, name)
                
        elif step_type == 'function':
            if hasattr(preprocessing_functions, name):
                func_obj = getattr(preprocessing_functions, name)

        # --- SPECIAL CASE: EPOCHING ---
        elif name == 'Epochs' or (step_type == 'class_init' and name == 'Epochs'):
            func_obj = mne.Epochs.__init__
        if func_obj is None:
            return {}

        # Use Inspection to get arguments
        sig = inspect.signature(func_obj)
        defaults = {}
        
        for param_name, param in sig.parameters.items():
            # Filter out internal arguments we don't want the user to touch
            if param_name in ['self', 'raw', 'events', 'epochs', 'data', 'inst']: 
                continue 
            
            # HANDLE **kwargs
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # We name it 'kwargs' explicitly so the GUI knows what it is
                defaults['kwargs'] = {} 
                continue
            
            # HANDLE *args (usually skippable for GUI, but can default to empty tuple)
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue

            val = param.default if param.default is not inspect.Parameter.empty else None
            defaults[param_name] = val
            
        return defaults