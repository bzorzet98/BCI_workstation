import numpy as np
import mne
import json
from src.preprocessing import EEGPreprocessor

class SessionState:
    """
    The Model. Holds the state of the current analysis session.
    Maintains a structured pipeline history live.
    """
    def __init__(self):
        # Data
        self.current_data = None 
        self.original_data = None # CHECKPOINT: The clean loaded data
    
        # Metadata
        self.dataset_name = None
        self.subject_id = None
        self.context = None
        self.selected_channels = None
        self.extra_info = {}
        
        # --- EDA State ---
        self.eda_report = None
        
        # --- STRUCTURED HISTORY ---
        # We initialize the structure expected by the batch preprocessor
        self.pipeline = {
            "raw_pipeline": [],
            "epoching": None,      # This will be a single dict, not a list
            "epoch_pipeline": []
        }
        
        self.worker = EEGPreprocessor()

    def load_data(self, data, ds_name, sub_id, ctx):
        """
        Sets new data into the session.
        We keep a SAFE COPY of the original data to allow Undo/Reset.
        """
        # 1. Store Original (Clean Checkpoint)
        # We copy it so 'current_data' modifications don't touch 'original_data'
        self.original_data = data.copy()
        
        # 2. Set Working Copy
        self.current_data = data # This is the object MNE will modify in-place
        
        self.dataset_name = ds_name
        self.subject_id = sub_id
        self.context = ctx
        
        # 3. Reset State
        self.selected_channels = None
        self.history = []
        self.eda_report = None
        self.pipeline = {"raw_pipeline": [], "epoching": None, "epoch_pipeline": []}
        
        print(f"Session started: {ds_name} | Sub {sub_id} | {ctx}")
        
    def run_transform(self, step_config):
        """Executes a step and updates history."""
        if self.current_data is None: return False
        
        step_name = step_config.get('name', 'Unknown')
        print(f"Running: {step_name}...")
        
        try:
            # 1. Delegate to worker
            new_data = self.worker.apply_single_step(self.current_data, step_config)
            
            # 2. Update State
            self.current_data = new_data
            
            # 3. Add to History
            self._add_to_history(step_config)
            
            # 4. Update Channels if needed
            if self.selected_channels and hasattr(new_data, 'ch_names'):
                self.selected_channels = [ch for ch in self.selected_channels if ch in new_data.ch_names]
            
            return True
        except Exception as e:
            print(f"Transform '{step_name}' failed: {e}")
            return False

    def obtain_events(self):
        """
        Robustly extracts events and event_id mapping from the current data.
        Returns:
            events (np.ndarray): The events array (N, 3).
            event_id (dict): Mapping {Label: ID}.
        """
        if self.current_data is None:
            return np.array([]), {}

        # Check cache if you implemented caching, otherwise calculation is fast enough
        
        events = None
        event_id = None

        # 1. Try Annotations (Preferred for Raw)
        if isinstance(self.current_data, mne.io.BaseRaw):
            try:
                events, event_id = mne.events_from_annotations(self.current_data, verbose=False)
            except Exception:
                # Fallback if no annotations
                pass
            
            # 2. If no annotations, try Stim Channel
            if events is None or len(events) == 0:
                stim_channels = mne.utils._get_stim_channel(None, self.current_data.info, raise_error=False)
                if len(stim_channels) > 0:
                    events = mne.find_events(self.current_data, verbose=False)
                    # Create generic labels for Stim events if no dict exists
                    unique_ids = np.unique(events[:, 2])
                    event_id = {str(uid): int(uid) for uid in unique_ids}

        # 3. Handle Epochs
        elif isinstance(self.current_data, mne.Epochs):
            events = self.current_data.events
            event_id = self.current_data.event_id

        if events is None:
            return np.array([]), {}
            
        return events, event_id
    
    def reset_pipeline(self):
        """
        Reverts data to the original state loaded from disk.
        """
        if self.original_data is None: return False
        
        print("Resetting pipeline to original state...")
        # Restore from checkpoint (copying again to keep checkpoint safe)
        self.current_data = self.original_data.copy()
        
        # Clear history
        self.history = []
        self.eda_report = None
        self.pipeline = {"raw_pipeline": [], "epoching": None, "epoch_pipeline": []}
        return True
    
    def load_pipeline_from_json(self, file_path):
        """
        Loads a JSON file and applies the pipeline steps to the current data.
        """
        if self.current_data is None:
            print("Error: No data loaded. Cannot apply pipeline.")
            return False

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Extract the pipeline dict
            # Handle both formats: full config output vs just the pipeline dict
            if 'preprocessing_pipeline' in config:
                pipeline = config['preprocessing_pipeline']
            else:
                # Assume the file IS the pipeline dict
                pipeline = config
            
            print(f"Loading Pipeline: {config.get('preprocessing_name', 'Unnamed')}")

            # 1. Apply Raw Steps
            raw_steps = pipeline.get('raw_pipeline', [])
            for step in raw_steps:
                # Ensure the stage tag is correct for history
                step['pipeline_stage'] = 'raw_pipeline'
                if not self.run_transform(step):
                    print("Pipeline stopped due to error in raw step.")
                    return False

            # 2. Apply Epoching
            epoch_step = pipeline.get('epoching')
            if epoch_step:
                # Epoching is usually a dict in the config
                # Sometimes it might be wrapped in a list in custom JSONs, handle both
                if isinstance(epoch_step, list) and len(epoch_step) > 0:
                    epoch_step = epoch_step[0]
                
                epoch_step['pipeline_stage'] = 'epoching'
                if not self.run_transform(epoch_step):
                    print("Pipeline stopped due to error in epoching.")
                    return False

            # 3. Apply Epoch Pipeline
            epoch_pipe = pipeline.get('epoch_pipeline', [])
            for step in epoch_pipe:
                step['pipeline_stage'] = 'epoch_pipeline'
                if not self.run_transform(step):
                    print("Pipeline stopped due to error in epoch step.")
                    return False
            
            return True

        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            return False
        
    def undo_last_step(self):
        """
        Removes the last step and re-calculates state.
        """
        if not self.history:
            print("History empty.")
            return False
            
        print("Undoing last step...")
        
        # 1. Pop last step
        removed_step = self.history.pop()
        
        # 2. Clean up structured pipeline dict
        self._remove_from_pipeline_structure(removed_step)
        
        # 3. REPLAY STRATEGY
        # Restore original
        self.current_data = self.original_data.copy()
        
        # Re-apply all remaining steps
        # We temporarily disable history appending to avoid duplicating the list
        temp_history = list(self.history)
        self.history = [] # Will be rebuilt by run_transform logic or we handle manually
        
        # Actually, simpler to just iterate and apply, skipping the history add logic
        # But we need run_transform logic. 
        # Let's iterate and call worker directly.
        
        try:
            for step in temp_history:
                print(f"  Re-applying: {step['name']}")
                self.current_data = self.worker.apply_single_step(self.current_data, step)
            
            # Restore the list
            self.history = temp_history
            return True
            
        except Exception as e:
            print(f"CRITICAL ERROR during Undo Replay: {e}")
            # If replay fails, we are in a bad state. Reset.
            self.reset_pipeline()
            return False

    # --- Internal Helpers ---

    def _add_to_history(self, step_config):
        """Adds step to flat list and structured dict."""
        self.history.append(step_config)
        
        stage = step_config.get('pipeline_stage', 'raw_pipeline')
        
        # Clean config for JSON export (remove GUI artifacts if any)
        clean = {k:v for k,v in step_config.items() if k != 'pipeline_stage'}
        clean['pipeline_stage'] = stage # Keep stage for logic
        
        if stage == 'epoching':
            self.pipeline['epoching'] = clean
        elif stage in self.pipeline:
            self.pipeline[stage].append(clean)
        else:
            self.pipeline['raw_pipeline'].append(clean)

    def _remove_from_pipeline_structure(self, step_config):
        """Removes a specific step from the structured dict."""
        stage = step_config.get('pipeline_stage', 'raw_pipeline')
        
        if stage == 'epoching':
            self.pipeline['epoching'] = None
        elif stage in self.pipeline and self.pipeline[stage]:
            # Pop the last item (assuming LIFO order match)
            self.pipeline[stage].pop()

    def get_history_flat(self):
        """Returns strings for GUI display."""
        return [f"[{s.get('pipeline_stage','raw')}] {s['name']}" for s in self.history]
    
    