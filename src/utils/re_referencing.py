from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from mne.io.edf.edf import RawEDF


def rref_remove(raw: RawEDF) -> RawEDF:
    raw_rref = raw.copy()
    raw_rref, _ = mne.set_eeg_reference(raw, [])
    return raw_rref

def rref_average(raw: RawEDF, projection: Optional[bool] = False) -> RawEDF:
    """
    Re-referencing using AVERAGE method.
    """
    raw_rref = raw.copy()
    raw_rref.set_eeg_reference('average', projection=projection)
    return raw_rref

def rref_CSD(raw: RawEDF) -> RawEDF:
    """
    Re-referencing using CSD method.
    """
    raw_rref = raw.copy()
    raw_rref = raw_rref.pick_types(meg=False, eeg=True, eog=False, ecg=False, stim=False,
                     exclude=raw.info['bads']).load_data()
    raw_rref.set_eeg_reference(projection=True).apply_proj()
    raw_rref = mne.preprocessing.compute_current_source_density(raw_rref)
    return raw_rref

def rref_REST(raw: RawEDF) -> RawEDF:
    """
    Re-referencing using REST method.
    """
    raw_rref = raw.copy()
    sphere = mne.make_sphere_model('auto', 'auto', raw_rref.info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, exclude=30.,
                                        pos=15., verbose=False)  # large "pos" just for speed!
    forward = mne.make_forward_solution(raw_rref.info, trans=None, src=src, bem=sphere, verbose=False)
    raw_rref.set_eeg_reference('REST', forward=forward, verbose=False)
    return raw_rref

def create_virtual_channel(raw: RawEDF, chs_names: List) -> List:
    """
    Create a virtual channel from chs_names
    """
    from .data_filtering import filter_band_array_to_array
    
    raw = raw.copy()
    chs_data = [raw[ch_name][0] for ch_name in chs_names]
    ocular_data = np.mean(np.concatenate(chs_data), axis=0)
    ocular_virtual = filter_band_array_to_array(
        data=ocular_data, 
        band=[0.1, 5], 
        fs=raw.info["sfreq"], 
        order=2
    )
    return ocular_virtual
