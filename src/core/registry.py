import inspect
import mne
from src.preprocessing import preprocessing_functions

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
                
            val = param.default if param.default is not inspect.Parameter.empty else None
            
            # --- OVERRIDE DEFAULTS FOR GUI CONVENIENCE ---
            if name == 'Epochs' and param_name == 'preload':
                val = True
                
            defaults[param_name] = val
            
        return defaults