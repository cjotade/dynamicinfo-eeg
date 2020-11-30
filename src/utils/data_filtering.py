import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from mne.io.edf.edf import RawEDF
from scipy.signal import butter, lfilter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

BANDS = {
    "alpha": [8, 12.5],
    "beta": [13, 30],
    "gamma": [30, 45],
    "theta": [4.5, 7.5]
}

def filter_band_raw_to_array(raw: RawEDF, band: Optional[Union[str, List]] = "alpha", verbose: Optional[bool] = False) -> List:
    """
    Filter butter by band which receives a RawEDF object and return a numpy ndarray.
    """
    raw_edf = raw.copy()
    if not band:
        logger.info(f"Please use a band in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    if isinstance(band, list):
        l_freq, h_freq = band    
    else:
        l_freq, h_freq  = BANDS.get(band.lower(), (None, None))
    
    if (not h_freq) and (not l_freq):
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

def filter_band_raw_to_raw(raw: RawEDF, band: Optional[Union[str, List]] = "alpha") -> RawEDF:
    """
    Filter butter by band which receives a RawEDF object and return a RawEDF object.
    """
    raw_edf = raw.copy()
    if not band:
        logger.info(f"Please use a band in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    if isinstance(band, list):
        l_freq, h_freq = band    
    else:
        l_freq, h_freq  = BANDS.get(band.lower(), (None, None))
    
    if (not h_freq) and (not l_freq):
        logger.info(f"Check your band={band}, not in filter_bands={BANDS.keys()}")
        return raw_edf.get_data()
    
    return raw_edf.filter(l_freq=l_freq, h_freq=h_freq)

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
    """
    Butter bandpass.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_band_array_to_array(data: List, band: Union[str, List], fs: float, order: int = 2):
    """
    Filter butter by band which receives a numpy ndarray and return a numpy ndarray.
    """
    if isinstance(band, list):
        l_freq, h_freq = band    
    else:
        l_freq, h_freq  = BANDS.get(band.lower(), (None, None))
    b, a = butter_bandpass(l_freq, h_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y

def eliminate_blink_corr_electrodes(corr_matrix: List, threshold: float = 0.6):
    """
    Return indexes for deletting the channels that are high correlated (by threshold) with the first component of corr_matrix.
    Usually first component of corr_matrix is the mean of Fp1 and Fp2 EEG channels.
    """
    corr_s_diag = corr_matrix - np.eye(len(corr_matrix))
    # find electrodes 'e' with corr(e, ocular_virtual) >= th
    to_eliminate = np.where(corr_s_diag[0] >= threshold)[0] - 1 
    return to_eliminate

def create_windows(raw: RawEDF, time: Optional[int] = 5, band: str = 'alpha') -> Tuple[List, List, List, List]:
    """
    Return data filtered by band.
    """
    raw_edf = raw.copy()
    # Filter all data by band
    all_data = filter_band_raw_to_array(raw_edf, band)
    all_data_not_filt = filter_band_raw_to_array(raw_edf, [0.1, 100])
    # Creating windows
    n_samples = int(raw_edf.info['sfreq']*time)
    n_windows = int(all_data.shape[-1]/n_samples)
    data_windows = np.array([all_data[:, (i_window*n_samples):((i_window+1)*n_samples)] 
                    for i_window in range(n_windows)])
    data_windows_not_filt = np.array([all_data_not_filt[:, (i_window*n_samples):((i_window+1)*n_samples)] 
                    for i_window in range(n_windows)])
    return data_windows, all_data, data_windows_not_filt, all_data_not_filt

def clean_windows_artifacts(raw: RawEDF, window_time: Optional[int] = 10, band: Optional[Union[str, List]] = "alpha", std_threshold: Optional[float] = 0.6, rref_fn: Optional[Callable] = None):
    """
    Create windows and then clean artifacts by threshold. 
    """
    raw_open_eyes = raw.copy()
    # Create Windows
    if rref_fn is not None:
        windows, all_data, windows_not_filt, _ = create_windows(rref_fn(raw_open_eyes), time=window_time, band=band)
    else: 
        windows, all_data, windows_not_filt, _ = create_windows(raw_open_eyes, time=window_time, band=band)
    # Clean artifacts
    windows_cls = [] # windows with close eyes
    rejected = [] # windows rejected idxs
    for window_n, window in enumerate(windows):
        # index 4 is for channel "O2" and index 9 is for channel "O1".
        if (window[4,:].std() > std_threshold*all_data[4].std()) and (window[9,:].std() > std_threshold*all_data[9].std()):
            windows_cls.append(windows_not_filt[window_n])
        else: 
            rejected.append(window_n)
            logger.info(f'window number {window_n} rejected')
    return np.array(windows_cls), rejected

def reconstruct_signal(raw: RawEDF, threshold: float, ch_names: Optional[List] = ["Fp1", "Fp2"]):
    """
    Reconstruct signal by removing ICA components that are high correlated with ocular virtual channel.
    """
    from .re_referencing import create_virtual_channel
    reconst_raw = raw.copy()
    # Butter filter (order 2) on mean of Fp1 and Fp2 EEG signals
    ocular_virtual = create_virtual_channel(reconst_raw, ch_names)
    # Decomposition ICA using raw_rref
    raw_ica = raw.copy()
    ica = mne.preprocessing.ICA(random_state=96)
    ica.fit(raw_ica, reject=dict(eeg=150e-6), flat=dict(eeg=2e-6)) # 150 µV for reject and 2 µV for flat
    # Concatenate ocular_virtual with ICA decompositioned data 
    concat_virtual_ica = np.concatenate([np.expand_dims(ocular_virtual, 0), raw_ica.get_data()], axis=0)
    corr_matrix = np.corrcoef(concat_virtual_ica)
    # Find electrodes 'e' with corr(e, ocular_virtual) > threshold
    to_eliminate = eliminate_blink_corr_electrodes(corr_matrix, threshold=threshold)
    print(f'Threshold: {threshold} => Eliminate ICA component(s): {to_eliminate}.')
    # Reconstruct signal and remove selected components from the signal
    ica.apply(reconst_raw, exclude=to_eliminate) 
    return reconst_raw, ica, to_eliminate
