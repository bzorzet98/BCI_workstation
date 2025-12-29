# init_paths.py
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))

if os.getenv('CONDA_DEFAULT_ENV') == 'bci_workstation':
    sys.path.append(os.path.join(current_path, 'src' , 'preprocessing'))
    sys.path.append(os.path.join(current_path, 'src' , 'preprocessed_dataset'))
    sys.path.append(os.path.join(current_path, 'src' , 'interface'))
    sys.path.append(os.path.join(current_path, 'src' , 'core'))
    sys.path.append(os.path.join(current_path, 'src' , 'utils'))
    sys.path.append(os.path.join(current_path, 'src' , 'eeg_datasets'))