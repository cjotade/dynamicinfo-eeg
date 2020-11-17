import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from scipy.signal import butter, lfilter

import mne
from mne.io.edf.edf import RawEDF

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

BANDS = {
    "alpha": [8, 12.5],
    "beta": [13, 30],
    "gamma": [30, 45],
    "theta": [4.5, 7.5]
}

def filter_band_to_array(raw_edf: RawEDF, band: Optional[Union[str, List]] = 'alpha', verbose: Optional[bool] = False) -> List:
    """mpy as nps
    Method description.
    """
    
    if not band:
        logger.info(f"Please use a band in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    if isinstance(band, list):
        l_freq, h_freq = band    
    else:
        l_freq, h_freq  = BANDS.get(band.lower(), (None, None))
    
    if (not h_freq) or (not l_freq):
        logger.info(f"Check your band={band}, not in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    return mne.filter.filter_data(
        raw_edf.get_data(), raw_edf.info['sfreq'],
        l_freq=l_freq, 
        h_freq=h_freq,
        method='iir',
        iir_params=dict(order=2, ftype='butter'),
        verbose=verbose
    )

def filter_band_to_raw(raw_edf: RawEDF, band: Optional[Union[str, List]] = 'alpha') -> RawEDF:
    if not band:
        logger.info(f"Please use a band in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    if isinstance(band, list):
        l_freq, h_freq = band    
    else:
        l_freq, h_freq  = BANDS.get(band.lower(), (None, None))
    
    if (not h_freq) or (not l_freq):
        logger.info(f"Check your band={band}, not in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    return raw_edf.filter(l_freq=l_freq, h_freq=h_freq)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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