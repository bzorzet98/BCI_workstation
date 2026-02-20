import numpy as np
import json
import mne
import copy
from typing import List, Dict, Optional, Any, Union, Tuple 
from src.preprocessing import EEGPreprocessor

class SessionState:
    """
    The SessionState class serves as the core model for managing and processing 
    multi-signal EEG workspaces. It provides a structured and modular approach 
    to handle raw EEG data, preprocessing pipelines, and event-based epoching.

    Key Responsibilities:
    - Workspace Management: Organizes EEG signals into a dictionary structure 
      where each signal is associated with a unique context label. This allows 
      for efficient access and manipulation of individual signals or groups of signals.
    - Logical Grouping: Supports logical grouping of signals (e.g., grouping multiple 
      runs of a task like 'motor_imagery') to streamline preprocessing and analysis.
    - Preprocessing Pipelines: Maintains a structured pipeline for each context, 
      enabling step-by-step transformations such as filtering, epoching, and artifact removal.
    - Event Handling: Provides robust methods to extract events and event IDs from 
      raw signals or epochs, supporting both annotations and stimulus channels.
    - Undo and Reset: Allows reverting the last preprocessing step or resetting 
      signals to their original state, ensuring flexibility during iterative analysis.
    - JSON Integration: Supports loading and applying preprocessing pipelines from 
      JSON files, enabling reproducibility and external configuration.

    This class is designed to work seamlessly with MNE-Python objects (e.g., Raw, Epochs) 
    and integrates with an EEGPreprocessor worker to apply transformations. It is 
    particularly suited for EEG/MEG workflows that require handling multiple signals 
    and maintaining a clear history of preprocessing steps.
    """

    def __init__(self):
        # Data: {context_label: mne_object}
        self.workspace: Dict[str, Union[mne.io.BaseRaw, mne.Epochs]] = {}
        self.original_workspace: Dict[str, mne.io.BaseRaw] = {}
        
        self.active_context: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.subject_id: Optional[int] = None
        self.selected_channels: Optional[List[str]] = None
        self.extra_info: Dict[str, Any] = {}
        
        # Source of Truth for processing: {context_label: structured_pipeline_dict}
        self.logical_keys: List[str] = [] 
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        
        self.eda_report: Optional[Dict[str, Any]] = None
        self.worker = EEGPreprocessor() # Assuming it's imported

    @property
    def current_data(self) -> Optional[Union[mne.io.BaseRaw, mne.Epochs]]:
        """
        Smart pointer to the data. 
        If active_context is 'motor_imagery' (logical), it returns 
        the data for 'motor_imagery:run_1' (physical).
        """
        if not self.active_context:
            return None
            
        # 1. Try direct match (for Noise or single signals)
        if self.active_context in self.workspace:
            return self.workspace[self.active_context]
            
        # 2. Try group match (Find the first run that belongs to this group)
        # Search for any key that starts with "logical_key:"
        prefix = f"{self.active_context}:"
        for key in self.workspace.keys():
            if key.startswith(prefix):
                return self.workspace[key]
                
        return None

    @property
    def current_pipeline(self):
        return self.pipelines.get(self.active_context, {})

    def load_data(self, data_dict: Dict[str, mne.io.BaseRaw], ds_name: str, sub_id: int, extra_info: Dict):
        """Initializes the workspace and sets up logical pipeline groups."""
        self.original_workspace = {ctx: raw.copy() for ctx, raw in data_dict.items()}
        self.workspace = data_dict
        self.extra_info = extra_info
        
        # 1. Calcular llaves lógicas para Preprocesamiento
        self._calculate_logical_keys()
        
        # 2. Inicializar Pipelines usando las llaves lógicas
        self.pipelines = {
            lk: {"raw_pipeline": [], "epoching": None, "epoch_pipeline": []} 
            for lk in self.logical_keys
        }
        
        # Foco inicial (la primera llave lógica)
        self.active_context = self.logical_keys[0] if self.logical_keys else None
        
        self.dataset_name, self.subject_id = ds_name, sub_id
        self.selected_channels, self.eda_report = None, None

        
    def _calculate_logical_keys(self):
        """
        Internal method:
        - Tasks (MI, Movement, Rest): Grouped by base name.
        - Noise: Kept individual with colon (noise:blink).
        """
        unique_keys = set()
        for key in self.workspace.keys():
            if ":" in key and "noise" not in key.lower():
                # motor_imagery:run_1 -> motor_imagery
                unique_keys.add(key.split(":")[0])
            else:
                # noise:eye_blink o rest -> se queda igual
                unique_keys.add(key)
        
        self.logical_keys = sorted(list(unique_keys))

    def run_transform(self, step_config: Dict[str, Any], apply_to_all: bool = False) -> bool:
        if not self.workspace: return False
        
        # Identificamos qué señales físicas se verán afectadas
        if apply_to_all:
            physical_targets = list(self.workspace.keys())
            # Si aplicamos a todo, usamos una llave genérica para el log o la primera
            pipeline_key = self.logical_keys[0] 
        else:
            # Si el foco es 'motor_imagery', targets son 'motor_imagery:run_1', 'run_2'...
            selected = self.active_context
            if selected in self.workspace: # Caso de Noise o señal única
                physical_targets = [selected]
            else: # Caso de Grupo (MI/Movement)
                physical_targets = [k for k in self.workspace.keys() if k.startswith(f"{selected}:")]
            
            pipeline_key = selected

        stage = step_config.get('pipeline_stage', 'raw_pipeline')

        try:
            for ctx in physical_targets:
                current_config = copy.deepcopy(step_config)
                
                # --- MANEJO DE EVENTOS (Independiente por señal) ---
                if stage == "epoching":
                    events, event_id = self.obtain_events_for_data(self.workspace[ctx])
                    if len(events) == 0: continue
                    current_config['kwargs']['events'] = events
                    current_config['kwargs']['event_id'] = event_id

                # APLICAR TRANSFORMACIÓN
                result = self.worker.apply_single_step(self.workspace[ctx], current_config)
                
                if result is not None:
                    # --- FIX PARA EL CRASH DE MNE (Length is unknown) ---
                    if isinstance(result, mne.BaseEpochs):
                        result.drop_bad() 
                    
                    self.workspace[ctx] = result

            # ACTUALIZAR PIPELINE (JSON) - Solo una vez por grupo lógico
            if stage == 'epoching':
                clean_config = copy.deepcopy(step_config)
                clean_config['kwargs']['events'] = None # Para que el JSON sea ligero
                self.pipelines[pipeline_key]['epoching'] = clean_config
            else:
                self.pipelines[pipeline_key][stage].append(step_config)
            
            return True
        except Exception as e:
            print(f"[Session] Transform Error: {e}")
            return False

    def obtain_events_for_data(self, data):
        """Helper robusto para obtener eventos de cualquier señal del workspace."""
        try:
            events, event_id = mne.events_from_annotations(data, verbose=False)
        except:
            # Fallback a stim channel
            try:
                events = mne.find_events(data, stim_channel='stim', verbose=False)
                u_ids = np.unique(events[:, 2]) if len(events) > 0 else []
                event_id = {str(uid): int(uid) for uid in u_ids}
            except:
                return np.array([]), {}
        return events, event_id

    
    def get_history_flat(self) -> List[str]:
        """
        Generates the list of strings for the GUI by flattening the pipeline dict.
        This removes the need for a separate 'history' variable.
        """
        if not self.active_context: return []
        
        pipe = self.current_pipeline
        flat_list = []
        
        # We follow the logical order: Raw -> Epoching -> Epoch Pipeline
        for step in pipe['raw_pipeline']:
            flat_list.append(f"[Raw] {step['name']}")
            
        if pipe['epoching']:
            flat_list.append(f"[Epoching] {pipe['epoching']['name']}")
            
        for step in pipe['epoch_pipeline']:
            flat_list.append(f"[Epochs] {step['name']}")
            
        return flat_list

    def undo_last_step(self) -> bool:
        """
        Removes the last step from the active pipeline and replays.
        """
        pipe = self.current_pipeline
        if not any([pipe['raw_pipeline'], pipe['epoching'], pipe['epoch_pipeline']]):
            return False

        # 1. Logic to find and remove the 'last' logical step
        if pipe['epoch_pipeline']:
            pipe['epoch_pipeline'].pop()
        elif pipe['epoching']:
            pipe['epoching'] = None
        else:
            pipe['raw_pipeline'].pop()

        # 2. Replay strategy to restore state
        ctx = self.active_context
        self.workspace[ctx] = self.original_workspace[ctx].copy()
        
        # Apply Raw
        for step in pipe['raw_pipeline']:
            self.workspace[ctx] = self.worker.apply_single_step(self.workspace[ctx], step)
        # Apply Epoching
        if pipe['epoching']:
            self.workspace[ctx] = self.worker.apply_single_step(self.workspace[ctx], pipe['epoching'])
        # Apply Epochs
        for step in pipe['epoch_pipeline']:
            self.workspace[ctx] = self.worker.apply_single_step(self.workspace[ctx], step)
            
        return True

    def obtain_events(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Robustly extracts events and event_id mapping from the ACTIVE context.
        
        It attempts to find events via:
        1. MNE Annotations (if Raw).
        2. Stimulus Channels (if Raw and no annotations).
        3. Internal event attributes (if Epochs).

        Returns:
            Tuple: (Numpy events array (N, 3), Dictionary {Label: ID}).
        """
        data = self.current_data
        if data is None:
            return np.array([]), {}

        events, event_id = None, None

        # 1. Logic for RAW signals
        if isinstance(data, mne.io.BaseRaw):
            # A. Try Annotations first (standard in many modern formats)
            try:
                events, event_id = mne.events_from_annotations(data, verbose=False)
            except Exception:
                pass
            
            # B. Fallback to STIM channel detection
            if events is None or len(events) == 0:
                stim_chans = mne.utils._get_stim_channel(None, data.info, raise_error=False)
                if stim_chans:
                    events = mne.find_events(data, verbose=False)
                    # Generate generic ID mapping if labels are missing
                    unique_ids = np.unique(events[:, 2])
                    event_id = {str(uid): int(uid) for uid in unique_ids}

        # 2. Logic for EPOCHED signals
        elif isinstance(data, mne.Epochs):
            events = data.events
            event_id = data.event_id

        # Final safety check
        if events is None or len(events) == 0:
            return np.array([]), {}
            
        return events, event_id

    def reset_pipeline(self, only_active: bool = True) -> bool:
        """
        Reverts signals to their original state loaded from disk.
        
        Args:
            only_active (bool): If True, only resets the currently focused signal.
                                If False, resets the entire workspace.
        """
        if not self.original_workspace:
            return False
        
        targets = [self.active_context] if only_active else list(self.workspace.keys())
        
        print(f"[Session] Resetting pipeline for: {targets}")
        
        for ctx in targets:
            # Restore from clean checkpoint
            self.workspace[ctx] = self.original_workspace[ctx].copy()
            # Wipe processing history
            self.pipelines[ctx] = {"raw_pipeline": [], "epoching": None, "epoch_pipeline": []}
        
        self.eda_report = None
        return True

    def load_pipeline_from_json(self, file_path: str, apply_to_all: bool = False) -> bool:
        """
        Loads a preprocessing JSON and applies it to the Workspace.
        
        Args:
            file_path (str): Path to the .json pipeline file.
            apply_to_all (bool): If True, the entire workspace will follow this pipeline.
                                If False, only the active signal is transformed.
        """
        if self.current_data is None:
            print("[Session] Error: No data loaded. Cannot apply pipeline.")
            return False

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Extract the pipeline dictionary (handles full exports or partial dicts)
            pipeline = config.get('preprocessing_pipeline', config)
            
            print(f"[Session] Applying Pipeline: {config.get('preprocessing_name', 'External JSON')}")

            # Define the logical order of stages
            stages = [
                ('raw_pipeline', 'list'),
                ('epoching', 'dict'),
                ('epoch_pipeline', 'list')
            ]

            for stage_name, data_type in stages:
                steps = pipeline.get(stage_name)
                if not steps:
                    continue

                # Ensure steps are in a list format for iteration
                steps_to_run = steps if data_type == 'list' else [steps]

                for step in steps_to_run:
                    # Inject the stage information so run_transform knows where to store it
                    step['pipeline_stage'] = stage_name
                    
                    if not self.run_transform(step, apply_to_all=apply_to_all):
                        print(f"[Session] Pipeline aborted: Error in stage '{stage_name}'")
                        return False
            
            return True

        except Exception as e:
            print(f"[Session] Failed to load JSON pipeline: {e}")
            return False
        
    def load_workspace_pipelines_from_json(self, pipeline_data: Dict[str, Any]) -> bool:
        """
        Applies pipelines to multiple signals in the workspace based on a JSON structure.
        
        Structure expected:
        { "context_name": { "raw_pipeline": [], "epoching": {}, "epoch_pipeline": [] }, ... }
        """
        success_overall = True
        
        for ctx_name, pipeline in pipeline_data.items():
            # Skip if this context isn't loaded in the current workspace
            if ctx_name not in self.workspace:
                print(f"[Session] Skipping {ctx_name}: Not found in current workspace.")
                continue

            print(f"[Session] Replaying pipeline for context: {ctx_name}")
            
            # Reset this specific signal before applying the new pipeline
            self.workspace[ctx_name] = self.original_workspace[ctx_name].copy()
            self.pipelines[ctx_name] = {"raw_pipeline": [], "epoching": None, "epoch_pipeline": []}

            # Apply stages sequentially
            stages = [('raw_pipeline', 'list'), ('epoching', 'dict'), ('epoch_pipeline', 'list')]
            
            for stage_name, data_type in stages:
                steps = pipeline.get(stage_name)
                if not steps: continue
                
                steps_to_run = steps if data_type == 'list' else [steps]
                for step in steps_to_run:
                    # Temporarily set focus to ensure correct signal is transformed
                    old_focus = self.active_context
                    self.active_context = ctx_name
                    
                    if not self.run_transform(step, apply_to_all=False):
                        print(f"  [!] Failed at {stage_name} for {ctx_name}")
                        success_overall = False
                    
                    self.active_context = old_focus
                    
        return success_overall
    
    def get_available_raw_contexts(self) -> List[str]:
        """Returns a list of keys in the workspace that are MNE Raw objects."""
        return [k for k, v in self.workspace.items() if isinstance(v, mne.io.BaseRaw)]