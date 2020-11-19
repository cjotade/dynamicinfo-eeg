from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
from mne.io.edf.edf import RawEDF

def read_data(data_path: str) -> RawEDF:
    raw = mne.io.read_raw_edf(data_path, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    return raw
