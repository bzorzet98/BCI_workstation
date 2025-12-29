import numpy as np
import mne
from mne import Epochs
from mne.io import BaseRaw, BaseRaw


############################ Preprocessing functions #############################################
def set_events_relative_to_reference(raw, new_events, stim_channel='stim', **kwargs):
    """
    Creates new events at specific times relative to a reference event.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw data.
    new_events : list of dict
        Definitions for the new events to create. Each dict must contain:
        - 'event_label' (str): The label for the new event.
        - 'reference_id' (int): The ID of the existing event to use as the time anchor (t=0).
        - 'new_id' (int): The ID for the new event.
        - 'delta_t' (float): Time difference in seconds relative to reference.
          (negative = before, positive = after).
    **kwargs :
        Ignored.
        
    Returns
    -------
    mne.io.BaseRaw
        Raw object with new annotations added.
    """
    # 1. Get existing events (This usually checks Stim channel first, then Annotations)
    events = mne.find_events(raw, stim_channel=stim_channel, verbose=False)
    
    sfreq = raw.info['sfreq']
    
    events_to_add = []
    # 2. Create new events for each definition
    for definition in new_events:
        reference_id = definition['reference_id']
        new_id = definition['new_id']
        delta_t = definition['delta_t']
        
        # Find reference events of this type        
        mask = events[:, 2] == reference_id
        ref_events = events[mask]
        ref_samples = ref_events[:, 0] # Array of sample indices
        
        # Calculate sample offset
        offset_samples = int(delta_t * sfreq)
        
        # Calculate new absolute sample positions
        new_samples = ref_samples + offset_samples
        
        # Create small event array: [sample, 0, new_id]
        # Valid samples only
        valid_mask = new_samples >= 0
        valid_samples = new_samples[valid_mask]
        
        events_to_add.append(np.concatenate(
            [valid_samples[:, np.newaxis],
             np.zeros((len(valid_samples), 1), dtype=int),
             np.full((len(valid_samples), 1), new_id, dtype=int)],
            axis=1
            ))

    # 3. Add to Stim Channel
    if len(events_to_add) > 0:
        events_to_add = np.concatenate(events_to_add, axis=0)
        # Note: replace=False adds values if they overlap!
        raw.add_events(events_to_add, stim_channel=stim_channel, replace=False)
        
    return raw

def pad_raw(raw, pad_before=0.0, pad_after=0.0, mode='constant', **kwargs):
    """
    Adds padding to the beginning and/or end of the Raw object using numpy.pad.
    Shifts existing events by manipulating sample indices.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw data.
    pad_before : float
        Seconds to add before the start.
    pad_after : float
        Seconds to add after the end.
    mode : str
        The padding mode to use (e.g., 'constant', 'edge', 'reflect').
    **kwargs : 
        Additional arguments passed to numpy.pad.

    Returns
    -------
    mne.io.BaseRaw
        New Raw object with padding applied and shifted events.
    """
    if pad_before == 0 and pad_after == 0:
        return raw

    # 1. Get underlying data and info
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # 2. Calculate padding samples
    n_pad_start = int(round(pad_before * sfreq))
    n_pad_end = int(round(pad_after * sfreq))
    
    # 3. Apply Padding to Data
    pad_width = ((0, 0), (n_pad_start, n_pad_end))
    new_data = np.pad(data, pad_width, mode=mode, **kwargs)
    
    # 4. Create new Raw object
    new_raw = mne.io.RawArray(new_data, raw.info.copy(), verbose=False)
    
    # 5. Handle Events via Sample Shifting
    # Extract events from the original raw object
    # checks stim channels first, then annotations
    events = mne.find_events(raw, stim_channel='stim', verbose=False)
    
    if len(events) > 0:
        # Shift the sample indices (column 0)
        events[:, 0] += n_pad_start
        # Note: replace=False adds values if they overlap!
        new_raw.add_events(events, stim_channel='stim', replace=True)
    return new_raw

def extract_data_from_epochs(epochs, **kwargs):
    """ 
    Extracts data, labels, and metadata from an Epochs object.
    
    Parameters
    ----------
    epochs : mne.Epochs
    **kwargs : 
        Arguments passed to epochs.get_data() (e.g., units, tmin, tmax).
        
    Returns
    -------
    dict
        {'data': np.array, 'labels': np.array, 'ch_names': list, 'sfreq': float}
    """
    result = {}
    
    # Obtain the frequency information
    result['sfreq'] = epochs.info['sfreq']
    
    # Obtain the channels names 
    result['ch_names'] = epochs.ch_names
    
    # Obtain the data
    result['data'] = epochs.get_data(**kwargs)
    
    # Obtain the events_id
    result['event_id'] = epochs.event_id
    
    # Obtain labels (Handle case where events might be missing/None)
    if epochs.events is not None:
        result['labels'] = epochs.events[:, -1]
    else:
        result['labels'] = np.array([])

    return result

def normalize_by_mean_std(data):
    if isinstance(data, dict) and 'data' in data:
        data['data'] = (data['data'] - data['data'].mean(axis=-1, keepdims=True)) / data['data'].std(axis=-1, keepdims=True)
        return data
    elif isinstance(data, mne.Epochs):
        data_array = data.get_data(copy=True)
        norm_data = (data_array - data_array.mean(axis=-1, keepdims=True)) / data_array.std(axis=-1, keepdims=True)
        epoch = mne.EpochsArray(norm_data, data.info, events=data.events, event_id=data.event_id, tmin=data.tmin)
        return epoch
    elif isinstance(data, mne.io.BaseRaw):
        data_array = data.get_data()
        norm_data = (data_array - data_array.mean(axis=-1, keepdims=True)) / data_array.std(axis=-1, keepdims=True)
        
        return norm_data
    else:
        raise TypeError("Data type not supported for normalization.")