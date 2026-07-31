from dataclasses import dataclass, replace
import numpy as np
import ipywidgets as W


@dataclass(frozen=True)
class P:
    """One scalar parameter plus the metadata that decides what may be done to it."""
    lo: float | None = None
    hi: float | None = None
    Slider_step: float | None = None
    value: float | None = None
    name: str = None          # Name of the variable
    tf: str = "linear"          # "linear" or "log" interpolation inside [lo, hi]
    note: str = ""
    
 
    def __post_init__(self):
        if self.lo is None or self.hi is None:
            raise ValueError(f"a fitted parameter needs bounds: {self}")
        if self.tf == "log" and self.lo <= 0:
            raise ValueError(f"log transform needs lo > 0: {self}")
 
    def return_Slider(self):
        """Return a Floatslider that can be used to adjust this parameter 
        in an interactive plot"""
    
        return W.FloatSlider(min=self.lo, max=self.hi, step=self.Slider_step, value=self.value,
                                           description=self.note, continuous_update=False)
        
        