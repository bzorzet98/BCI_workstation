from abc import ABC, abstractmethod
from typing import Union
import mne
from src.eda.eda_report import EDAReport

class BaseAnalyzer(ABC):
    def __init__(self, picks: list = None, **kwargs):
        self.picks = picks
        self.params = kwargs

    @abstractmethod
    def analyze(self, data: Union[mne.io.BaseRaw, mne.Epochs], **kwargs) -> EDAReport:
        """ 
        MUST return an EDAReport instance. 
        Example: return EDAReport(summary="Done", metrics={...})
        """    
        pass
    
# class BaseAnalyzer(ABC):
#     def __init__(self, picks: list = None, **kwargs):
#         self.picks = picks
#         self.params = kwargs

#     @abstractmethod
#     def analyze(self, data: Union[mne.io.BaseRaw, mne.Epochs]) -> dict:
#         """ The dictionary should contain at least:
#                 {
#                 "summary": str,              
#                 "visuals":: ["hist", "scatter", "mapping", "inspect"],
#                 "artifacts": {                
#                     "bad_trials": [index],  
#                     "tags": [tag_list]   
#                 }, 
#                 "metrics": {                  
#                     "metric_name": {"ch": value, ...},  
#                 },
                
#             "inspection_data": {
#                 "raw": {ch: raw_data},
#                 ...
#             }
#             }
#         """    
#         pass