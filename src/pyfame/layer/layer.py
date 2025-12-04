from pydantic import BaseModel, NonNegativeFloat
from typing import Callable, Optional, Any
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
import copy

class TimingConfiguration(BaseModel):
    onset_time:Optional[NonNegativeFloat] = None
    offset_time:Optional[int] = None
    rise_time:NonNegativeFloat = 500.0
    fall_time:NonNegativeFloat = 500.0
    rise_curve:Callable[...,float] = timing_linear
    fall_curve:Callable[...,float] = timing_linear
    rise_curve_kwargs:Optional[dict] = None
    fall_curve_kwargs:Optional[dict] = None

class Layer(ABC): 
    """ An abstract base class to be extended by pyfame's manipulation layer classes. """

    def __init__(self, configuration:TimingConfiguration|None = None):
        
        # if config is none, populate with defaults
        self.config = configuration or TimingConfiguration()

        self.onset_t = self.config.onset_time
        self.offset_t = self.config.offset_time
        self.rise = self.config.rise_time
        self.fall = self.config.fall_time
        self.rise_fn = self.config.rise_curve
        self.fall_fn = self.config.fall_curve
        self.rise_kwargs = self.config.rise_curve_kwargs
        self.fall_kwargs = self.config.fall_curve_kwargs
    
    def _snapshot_state(self):
        self._initial_state = copy.deepcopy(self.__dict__)
    
    def _reset_state(self):
        init_state = copy.deepcopy(self._initial_state)
        init_state["_initial_state"] = self._initial_state
        self.__dict__ = init_state
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        # Handle None case for kwargs list
        if supports_weight:
            if dt <= (self.onset_t + self.rise):
                # off -> rise transition
                if self.rise_kwargs is not None:
                    return self.rise_fn(dt, self.onset_t, (self.onset_t + self.rise), True, **self.rise_kwargs)
                else:
                    return self.rise_fn(dt, self.onset_t, (self.onset_t + self.rise), True)
            elif dt >= (self.offset_t - self.fall):
                # fall transition -> off
                if self.fall_kwargs is not None:
                    return self.fall_fn(dt, (self.offset_t - self.fall), self.offset_t, False, **self.fall_kwargs)
                else:
                    return self.fall_fn(dt, (self.offset_t - self.fall), self.offset_t, False)
            else:
                # sustain at max
                return 1.0
        else:
            if self.onset_t <= dt <= self.offset_t:
                return 1.0
            else:
                return 0.0

    @abstractmethod
    def supports_weight(self) -> bool:
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        pass

    @abstractmethod
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:MatLike, dt:float) -> MatLike:
        pass