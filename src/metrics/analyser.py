from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.results import ResultsNetworkInference, ResultsSingleProcessAnalysis
from mne.io.edf.edf import RawEDF


def analyse_network_by_metric(
    data_window: List, 
    channels: Union[int, List[int]],
    metric: str,
    target: Optional[int] = None,
    **settings: Optional[Dict[str, float]]) -> Union[ResultsSingleProcessAnalysis, ResultsNetworkInference]:
    """
    Analyse nentwork for metrics calculation.
    """
    from utils.data_filtering import get_sources
    # Configure settings
    if not settings:
        settings = {
                'cmi_estimator':  'JidtGaussianCMI', #JidtGaussianCMI, JidtKraskovCMI
            }
        if metric == "AIS":
            settings.update({
                'max_lag': 5,
                'local_values': True,
            })
        elif metric == "TE":
            settings.update({
                'max_lag_sources': 3,
                'min_lag_sources': 1
            })
        else:
            raise Exception('Metric parameter must be AIS or TE')
    # Network creation
    if metric == "AIS":
        data = Data(data=data_window[channels,:].reshape(1, -1), dim_order='ps') 
        network_analysis = ActiveInformationStorage()
    elif metric == "TE":
        data = Data(data=data_window[channels,:].reshape(len(channels), data_window.shape[-1]), dim_order='ps') 
        network_analysis = MultivariateTE()
    else:
        raise Exception('Metric parameter must be AIS or TE')
    # Return analysis and data filtered
    if target is not None:
        sources = get_sources(channels, target)
        return network_analysis.analyse_network_single_target(settings=settings, data=data, target=target, sources=sources)
    else:
        return network_analysis.analyse_network(settings=settings, data=data)