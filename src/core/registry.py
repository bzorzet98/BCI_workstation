import inspect
import os
import mne
from src.preprocessing import preprocessing_functions
import pkgutil
import importlib
from src.utils import import_class

class Registry:
    """Registry
    The `Registry` class serves as a centralized knowledge base for managing and discovering 
    available processing methods, custom functions, and their respective arguments. It is 
    designed to dynamically inspect and provide metadata about processing steps, analyzers, 
    and their parameters, enabling flexible and extensible workflows for EEG/MEG data 
    processing.
    ### Attributes:
    - `ALLOWED_RAW_METHODS` (list of str): 
        A predefined list of methods that can be applied to `mne.io.Raw` objects. These methods 
        include common preprocessing steps such as filtering, resampling, and channel selection.
    - `ALLOWED_EPOCH_METHODS` (list of str): 
        A predefined list of methods that can be applied to `mne.Epochs` objects. These methods 
        include operations like baseline correction, time shifting, and channel selection.
    - `CUSTOM_FUNCTIONS` (list of str): 
        Dynamically discovered custom preprocessing functions from the `preprocessing_functions` 
        module. Only functions that do not start with an underscore (`_`) are included.
    ### Methods:
    1. `get_available_steps(data_obj)`:
         - **Purpose**: Determines the available processing steps for a given data object 
             (e.g., `Raw` or `Epochs`).
         - **Returns**: A list of dictionaries, where each dictionary represents a method or 
             function with its name and type (`method`, `function`, or `class_init`).
    2. `get_analyzer_params(class_name, module_path)`:
         - **Purpose**: Dynamically inspects the `__init__` method of a class to extract its 
             arguments and their default values.
         - **Parameters**:
             - `class_name` (str): The name of the class to inspect.
             - `module_path` (str): The module path where the class is defined.
         - **Returns**: A dictionary mapping argument names to their default values. If an 
             argument has no default, its value is set to `None`.
    3. `discover_eda_analyzers()`:
         - **Purpose**: Scans the `src/eda/analyzers` directory to discover and list all 
             available analyzer classes.
         - **Returns**: A dictionary where keys are class names and values are their respective 
             module paths.
    4. `get_defaults(name, step_type)`:
         - **Purpose**: Introspects the signature of a method, function, or class initializer 
             to extract its arguments and their default values.
         - **Parameters**:
             - `name` (str): The name of the method, function, or class initializer.
             - `step_type` (str): The type of the step (`method`, `function`, or `class_init`).
         - **Returns**: A dictionary mapping argument names to their default values. Special 
             handling is applied for certain cases, such as setting `preload=True` for the 
             `Epochs` class.
    ### Use Cases:
    - **Dynamic Discovery**: The `Registry` class enables dynamic discovery of available 
        processing steps and their parameters, making it easier to build flexible and 
        user-configurable pipelines.
    - **Analyzer Management**: It provides tools to discover and manage custom analyzers 
        defined in the `src/eda/analyzers` directory.
    - **GUI Integration**: The metadata provided by the `Registry` can be used to populate 
        graphical user interfaces (GUIs) with available processing options and their default 
        parameters.
    ### Notes:
    - The class relies heavily on Python's `inspect` module for dynamic introspection.
    - It assumes a specific directory structure and naming conventions for analyzers and 
        preprocessing functions.
    - Error handling is implemented to ensure robustness when inspecting classes or modules 
        that may not exist or have unexpected structures.
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
    def get_available_steps(data_obj):
        """
        Returns a list of available actions based on data type.
        """
        methods = []
        
        # Usamos nombres de clase en string para máxima compatibilidad
        type_str = str(type(data_obj)).lower()
        
        methods = []

        if "raw" in type_str:
            methods = [{'name': m, 'type': 'method'} for m in Registry.ALLOWED_RAW_METHODS]
            methods.append({'name': 'Epochs', 'type': 'class_init'}) 
        elif "epochs" in type_str:
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
            
            if name == 'Epochs' and param_name == 'preload':
                val = True # Set default to True specifically for Epoching
                
            defaults[param_name] = val
            
        return defaults