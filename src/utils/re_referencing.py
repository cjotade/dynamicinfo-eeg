from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
from mne.io.edf.edf import RawEDF

def rref_remove(raw: RawEDF) -> RawEDF:
    raw = raw.copy()
    raw, _ = mne.set_eeg_reference(raw, [])
    return raw

def rref_average(raw: RawEDF, projection: Optional[bool] = False) -> RawEDF:
    """
    Re-referencing using AVERAGE method.
    """
    raw = raw.copy()
    raw.set_eeg_reference('average', projection=projection)
    return raw

def rref_CSD(raw: RawEDF) -> RawEDF:
    """
    Re-referencing using CSD method.
    """
    raw = raw.copy()
    raw = raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                     exclude=raw.info['bads']).load_data()
    raw.set_eeg_reference(projection=True).apply_proj()
    raw = mne.preprocessing.compute_current_source_density(raw)
    return raw

def rref_REST(raw: RawEDF) -> RawEDF:
    """
    Re-referencing using REST method.
    """
    raw = raw.copy()
    sphere = mne.make_sphere_model('auto', 'auto', raw.info, verbose=False)
    src = mne.setup_volume_source_space(sphere=sphere, exclude=30.,
                                        pos=15., verbose=False)  # large "pos" just for speed!
    forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere, verbose=False)
    raw.set_eeg_reference('REST', forward=forward, verbose=False)
    return raw