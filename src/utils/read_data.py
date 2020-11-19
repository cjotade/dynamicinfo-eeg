from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
from mne.io.edf.edf import RawEDF

def read_data(data_path: str) -> RawEDF:
    raw = mne.io.read_raw_edf(data_path, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    return raw

def create_windows(raw_edf: RawEDF, time: Optional[int] = 5, band: str = 'alpha') -> Tuple[List, List]:
    """
    Return data filtered by band.
    """
    # Filter all data by band
    all_data = filter_band(raw_edf, band)
    # Creating windows
    n_samples = int(raw_edf.info['sfreq']*time)
    n_windows = int(all_data.shape[-1]/n_samples)
    data_windows = np.array([all_data[:, (i_window*n_samples):((i_window+1)*n_samples)] 
                    for i_window in range(n_windows)])
    return data_windows, all_data
