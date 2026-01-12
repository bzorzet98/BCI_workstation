from abc import ABC, abstractmethod
from typing import Union
import mne

class BaseAnalyzer(ABC):
    def __init__(self, picks: list = None, **kwargs):
        self.picks = picks
        self.params = kwargs

    @abstractmethod
    def analyze(self, data: Union[mne.io.BaseRaw, mne.Epochs]) -> dict:
        """ The dictionary should contain at least:
                {
                "summary": str,              
                "visuals": {                  
                    "Plot name 1": [...], 
                    "Plot name 2": [...]
                },
                "artifacts": {                
                    "bad_trials": [index],  
                    "tags": [tag_list]   
                }
            }
        """    
        pass