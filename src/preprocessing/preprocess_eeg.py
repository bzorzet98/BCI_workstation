from typing import List, Dict, Any, Callable, Any, Optional, Union, Tuple
import mne
import numpy as np
from copy import deepcopy
from src.utils.imports import import_class

class EEGPreprocessor:
    """
    Worker class that applies a sequence of processing steps to EEG data.
    It supports partial execution (e.g., Raw only, or Raw + Epoching).

    Parameters
    ----------
    raw_pipeline : List[Dict[str, Any]], optional
        Steps for continuous data (e.g., filtering). 
        Can use 'type': 'method' (MNE object method) or 'function' (external).
    
    epoching_config : Dict[str, Any], optional
        Strategy for cutting data. If None, the process stops after Raw.
        Must contain: 'method' ('events' or 'sliding') and 'kwargs'.
        
    epoch_pipeline : List[Dict[str, Any]], optional
        Steps for epoched data (e.g., normalization). If None, returns Epochs object.
    """

    def __init__(self, 
                 raw_pipeline: Optional[List[Dict]] = None, 
                 epoching_config: Optional[Dict] = None, 
                 epoch_pipeline: Optional[List[Dict]] = None):
        
        # Initialize as empty list if None, to safely iterate
        self.raw_pipeline = raw_pipeline if raw_pipeline is not None else []
        self.epoching_config = epoching_config
        self.epoch_pipeline = epoch_pipeline if epoch_pipeline is not None else []
        
    def _apply_steps(self, data, pipeline_steps):
        """Helper to apply a list of steps (Method or Function)."""
        for step in pipeline_steps:
            data = self._apply_single_step(data, step)
        return data
    
    def _apply_single_step(self, data, step):
        step_type = step.get('type')
        method_name = step['name']
        module_name = step.get('module_name')
        kwargs = step.get('kwargs', {}).copy()
        
        if step_type == 'method':
            if hasattr(data, method_name):
                method = getattr(data, method_name)
                # MNE methods often return the object itself (builder pattern)
                # but some might modify inplace and return None.
                # Safe approach: assume inplace modification unless return is new object
                res = method(**kwargs)
                if res is not None: 
                    data = res
            else:
                raise AttributeError(f"Object {type(data)} has no method '{method_name}'")
        
        elif step_type in ['function', 'class', 'class_init']:
            if method_name == 'Epochs':
                if 'events' not in kwargs:
                    # We reuse our robust event finder
                    events = self._obtain_events(data)
                    module_name = "mne"
                    kwargs['events'] = events
                    if len(events) == 0:
                        raise ValueError("No events found in data. Cannot epoch.")
            module_name = module_name if module_name is not None else "mne"
            # 1. Dynamic Import
            method = import_class(method_name, module_name)
            data = method(data, **kwargs)
        return data
    
    def apply_single_step(self, data, step):
        """
        Public wrapper for internal logic (used by SessionState).
        """
        return self._apply_single_step(data, step)
     
    def process(self, raw: mne.io.BaseRaw) -> Dict[str, Any]:
        """
        Executes the pipeline stages defined in init.
        
        Returns
        -------
        result : 
        """
        results = {}
        # Work on a copy to avoid modifying the original dataset in memory
        current_raw_data = raw.copy()

        # --- STAGE 1: RAW TRANSFORMS ---
        if self.raw_pipeline:
            current_raw_data = self._apply_steps(current_raw_data, self.raw_pipeline)
            
        # Check if we should stop here
        if not self.epoching_config:
            return current_raw_data
        
        # --- STAGE 2: EPOCHING ---
        # 1. Calculate Events dynamically (Robustly)
        events = self._obtain_events(current_raw_data)
        
        # 2. Package the Epoching Step
        # We construct a step dict that fits _apply_single_step structure
        # We inject 'events' into kwargs
        epoch_kwargs = self.epoching_config.get('kwargs', {}).copy()
        epoch_kwargs['events'] = events
        
        epoch_step = {
            "type":"class",
            "name": "Epochs",
            "module_name": "mne", # Explicitly state module
            "kwargs": epoch_kwargs
        }
        # 3. Execute via standard logic
        current_data = self._apply_single_step(current_data, epoch_step)
        
        # Check if we should stop here
        if not self.epoch_pipeline:
            return current_epoch_data

        # --- STAGE 3: EPOCH TRANSFORMS 
        
        if self.epoch_pipeline:
            current_epoch_data = self._apply_steps(current_epoch_data, self.epoch_pipeline)
        
        return current_epoch_data


    def _obtain_events(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """
        Obtains the events from the given raw data.

        Parameters:
        - raw: The raw data from which to obtain the events.

        Returns:
        - events: The obtained events.
        """
        # find the events
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        return events
