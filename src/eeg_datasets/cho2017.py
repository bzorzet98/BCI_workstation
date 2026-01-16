import os
import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from global_config import DATABASES_PATH
from .base import BaseEEGDataset

class Cho2017(BaseEEGDataset):
    """
    Motor Imagery dataset from Cho et al. 2017 (GigaScience).
    
    This dataset contains EEG and EMG recordings from 52 subjects performing Left/Right 
    hand Motor Imagery (MI) and real movement tasks.

    References
    ----------
    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034
    
    Data Structure & Channels
    -------------------------
    The raw data is loaded from MATLAB structures ('*.mat').
    *   **Sampling Rate:** 512 Hz
    *   **Channels 1-64:** EEG (10-20 System)
    *   **Channels 65-68:** EMG (EMG1, EMG2, EMG3, EMG4)

    Experiment Protocols & Trials
    -----------------------------
    The dataset includes four distinct experimental contexts. The trial structure 
    and event timing are detailed below:

    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | Context           | Condition / Class     | Trials (Count)    | Description                               |
    +===================+=======================+===================+===========================================+
    | **Motor Imagery** | Left Hand MI          | 100 or 120        | Subject imagines left hand movement.      |
    | (MI)              | Right Hand MI         | 100 or 120        | Subject imagines right hand movement.     |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Real Movement** | Left Hand Mov.        | 20                | Subject performs real left hand grasp.    |
    |                   | Right Hand Mov.       | 20                | Subject performs real right hand grasp.   |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Resting State** | Rest                  | 1 (Continuous)    | Resting state with eyes-open condition.   |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Noise**         | Eye Blinking          | 2 (5 sec each)    | Intentional eye blinking artifacts.       |
    | (Artifacts)       | Eyeball Up/Down       | 2 (5 sec each)    | Intentional vertical eye movements.       |
    |                   | Eyeball Left/Right    | 2 (5 sec each)    | Intentional horizontal eye movements.     |
    |                   | Jaw Clenching         | 2 (5 sec each)    | Intentional jaw clenching artifacts.      |
    |                   | Head Movement         | 2 (5 sec each)    | Intentional head movement artifacts.      |
    +-------------------+-----------------------+-------------------+-------------------------------------------+

    Metadata Fields
    ---------------
    The `extra_info` dictionary returned by `get_data` contains:
    *   **senloc/psenloc:** 3D sensor locations and spherical projections.
    *   **bad_trial_idx_mi:** Indices of MI trials rejected due to EMG activity.
    *   **bad_trial_idx_voltage:** Indices of MI trials rejected due to voltage > 100uV.
    *   **n_imagery_trials:** Exact count of MI trials for the subject.
    """
    def __init__(self, data_to_load = None):
        # 1. Define specific configuration for Cho
        path = os.path.join(DATABASES_PATH, 'MNE-gigadb-data', 'gigadb-datasets', 
                            'live', 'pub', '10.5524', '100001_101000', '100295','mat_data')
        subjects = list(range(1, 53))
        event_id = {'left_hand': 1, 'right_hand': 2}
        
        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, sessions = ['session_1'], event_id=event_id, code='Cho2017')
        
        # 3. Database-specific attributes
        self.sfreq = 512
        self.standard_montage = "standard_1005"
        self.ch_names = {"eeg":[
                            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
                            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
                            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
                            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
                            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
                            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
                            ],
                        "emg":["EMG1", "EMG2", "EMG3", "EMG4"]}
        self.data_availables = ['motor_imagery', 'movement', 'rest', 'noise']
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata(path = os.path.join(DATABASES_PATH, 'MNE-gigadb-data', 'gigadb-datasets', 
                                                                                'live', 'pub', '10.5524', '100001_101000', '100295'))
    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load

    def _load_subjects_metadata(self, path=None):
        # Check if the file exists
        if os.path.exists(os.path.join(path, 'database_information.csv')):
            try:
                subjects_metadata = pd.read_csv(os.path.join(path, 'database_information.csv'))
            except Exception as e:
                print(f"Error loading subjects metadata: {e}")
                subjects_metadata = None
        else:
            subjects_metadata = None
        return subjects_metadata
    
    def get_data(self, subjects = None, data_to_load = None):
        data_to_load = data_to_load if data_to_load is not None else self.data_to_load
        # Use Base helper to validate
        subjects, data_to_load = self._validate_inputs(subjects, data_to_load)
        data_dict = {}
        for subject in subjects:
            if subject not in self.subject_list:
                continue
            data_dict[subject] = self._get_single_subject_data(subject, data_to_load)
        return data_dict

    def _get_single_subject_data(self, subject, data_to_load):
        file_path = os.path.join(self.dataset_path, f's{subject:02d}.mat')
        if not os.path.exists(file_path):
            print(f"Warning: File not found for subject {subject}")
            return None
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_struct = mat['eeg']
        subject_data = {'session_1':{}}
        for data in data_to_load:
            subject_data['session_1'][data] = {}
            if data == "noise":
                noises = getattr(eeg_struct, data)
                for i, noise in enumerate(['eye_blink', 'eyeball_movement_ud', 
                                            'eyeball_movement_lr', 'jaw_clenching', 
                                            'head_movement_lr']):
                    data_array = noises[i]
                    raw = self._create_raw_simple(data_array)
                    subject_data['session_1'][data][noise] = raw
            elif data == 'motor_imagery' or data == 'movement':
                raw = self._create_raw_task(eeg_struct, data)
                subject_data['session_1'][data] = {'run_1': raw}
            else:
                data_array = getattr(eeg_struct, data)
                raw = self._create_raw_simple(data_array)
                subject_data['session_1'][data] = {'recording': raw}

        # Find other possible essential data here if needed 
        subject_data['extra_info'] = self._obtain_extra_info(eeg_struct)
        return subject_data 

    def _create_raw_simple(self, data_array):
        ch_names = self.ch_names['eeg']+ self.ch_names['emg'] 
        ch_types = ["eeg"]*len(self.ch_names['eeg']) + ["emg"]*len(self.ch_names['emg'])
        montage = mne.channels.make_standard_montage(self.standard_montage)
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.sfreq)
        raw = mne.io.RawArray(data_array * self.unit_factor, info=info, verbose=False)
        raw.set_montage(montage)
        return raw
    
    def _create_raw_task(self, eeg_data, context):
        ctx = 'imagery' if context == 'motor_imagery' else 'movement'
        
        event = getattr(eeg_data, f'{ctx}_event')
        
        left = np.vstack([getattr(eeg_data, f'{ctx}_left') * self.unit_factor, event * self.event_id['left_hand']])
        right = np.vstack([getattr(eeg_data, f'{ctx}_right') * self.unit_factor, event * self.event_id['right_hand']])
        data_array =  np.hstack([left, np.zeros((left.shape[0], 500)), right])
        
        ch_names = self.ch_names['eeg']+ self.ch_names['emg'] + ['stim']
        ch_types = ["eeg"]*len(self.ch_names['eeg']) + ["emg"]*len(self.ch_names['emg']) + ['stim']
        montage = mne.channels.make_standard_montage(self.standard_montage)
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.sfreq)
        raw = mne.io.RawArray(data_array , info=info, verbose=False)
        raw.set_montage(montage)
        return raw

    def _obtain_extra_info(self, eeg_data):
        extra_info = {}
        # First we have to obtain the bad trials correlated with EMG artifacts
        left_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_mi')[0]
        right_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_mi')[1] + getattr(eeg_data, 'n_imagery_trials') # Offset for right trials
        extra_info['bad_trial_idx_mi'] = np.hstack([left_bad_trial_idx, right_bad_trial_idx])
        
        # First we have to obtain the bad trials for above 100 uV
        left_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_voltage')[0] 
        right_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_voltage')[1] + getattr(eeg_data, 'n_imagery_trials') # Offset for right trials
        extra_info['bad_trial_idx_voltage'] = np.hstack([left_bad_trial_idx, right_bad_trial_idx])

        extra_info['senloc'] = getattr(eeg_data, 'senloc')
        extra_info['psenloc'] = getattr(eeg_data, 'psenloc')
        
        extra_info['n_imagery_trials'] = getattr(eeg_data, 'n_imagery_trials')
        extra_info['n_movement_trials'] = getattr(eeg_data, 'n_movement_trials')
        return extra_info
        
        
        
        
        