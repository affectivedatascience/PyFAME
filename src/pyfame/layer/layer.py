from pydantic import BaseModel, NonNegativeFloat
from typing import Callable, Optional
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_constant
import copy

class TimingConfiguration(BaseModel):
    """
    Configuration model defining the temporal behaviour of a manipulation `Layer`.

    The `TimingConfiguration` class specifies when a layer's manipulation becomes
    active, how long it remains active, and how smoothly it transitions on and off.
    It supports both binary and dynamic activation windows via configurable rise 
    and fall curves.

    This class inherits from pydantic's `BaseModel` to provide validation and 
    default handling of timing parameters.

    Attributes
    ----------
    onset_time_msec : NonNegativeFloat or None, optional
        Time (in milliseconds) at which the layer begins to activate.
        If `None`, the onset time will be inferred or handled by the calling
        context.
    offset_time_msec : float or None, optional
        Time (in milliseconds) at which the layer fully deactivates.
        If `None`, the offset time will be inferred or handled by the calling
        context.
    rise_time_msec : NonNegativeFloat, default=500.0
        Duration (in milliseconds) over which the layer transitions from
        inactive to fully active.
    fall_time_msec : NonNegativeFloat, default=500.0
        Duration (in milliseconds) over which the layer transitions from
        fully active to inactive.
    rise_curve : callable, default=timing_linear
        Function defining the temporal interpolation during the rise phase.
        The callable must return a scalar weight in the range `[0.0, 1.0]`.
    fall_curve : callable, default=timing_linear
        Function defining the temporal interpolation during the fall phase.
        The callable must return a scalar weight in the range `[0.0, 1.0]`.
    rise_curve_kwargs : dict or None, optional
        Optional keyword arguments passed to `rise_curve`.
    fall_curve_kwargs : dict or None, optional
        Optional keyword arguments passed to `fall_curve`.
    
    Notes
    -----
    - Rise and fall curves are expected to follow the PyFAME timing function
      signature: `fn(t, start, end, rising, **kwargs) -> float`.
    - If rise/fall times are set to `0`, transitions become instantaneous.
    - Timing behavior is agnostic to the underlying manipulation and can be
      reused across different layer types.
    """

    onset_time_msec:Optional[NonNegativeFloat] = None
    offset_time_msec:Optional[float] = None
    rise_time_msec:NonNegativeFloat = 500.0
    fall_time_msec:NonNegativeFloat = 500.0
    rise_curve:Callable[...,float] = timing_constant
    fall_curve:Callable[...,float] = timing_constant
    rise_curve_kwargs:Optional[dict] = None
    fall_curve_kwargs:Optional[dict] = None

class Layer(ABC): 
    """
    Abstract base class for all PyFAME manipulation layers.

    A `Layer` represents a manipulation that can be applied to a video frame 
    or static image. For video files, a `Layer` optionally supports smooth 
    temporal weighting (i.e., rise/fall curves) to allow gradual onset and 
    offset of effects.

    All concrete manipulation layers must inherit from this class and
    implement the abstract methods defined here.

    Parameters
    ----------
    configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset, 
        offset, rise/fall durations, and weighting curves. If `None`, a 
        default `TimingConfiguration` is instantiated.

    Attributes
    ----------
    config : TimingConfiguration
        Timing configuration used by the layer.
    onset_t : float
        Onset time (in milliseconds) at which the layer becomes active.
    offset_t : float
        Offset time (in milliseconds) at which the layer deactivates.
    rise : float
        Duration (in milliseconds) of the rise phase.
    fall : float
        Duration (in milliseconds) of the fall phase.
    rise_fn : callable
        Function defining the temporal rise curve.
    fall_fn : callable
        Function defining the temporal fall curve.
    rise_kwargs : dict or None
        Optional keyword arguments passed to `rise_fn`.
    fall_kwargs : dict or None
        Optional keyword arguments passed to `fall_fn`.
    """

    def __init__(self, configuration:TimingConfiguration|None = None):
        """
        Initialise the manipulation layer.
        
        Parameters
        ----------
        configuration : TimingConfiguration or None, optional
            A pydantic model containing timing configurations controlling onset, 
            offset, rise/fall durations, and weighting curves. If `None`, a 
            default `TimingConfiguration` is instantiated. The default 
            instantiation assumes a linear rise and fall transition, onset at 
            0.0 and offset at the video's duration.
        """
        # if config is none, populate with defaults
        self.config = configuration or TimingConfiguration()

        self.onset_t = self.config.onset_time_msec
        self.offset_t = self.config.offset_time_msec
        self.rise = self.config.rise_time_msec
        self.fall = self.config.fall_time_msec
        self.rise_fn = self.config.rise_curve
        self.fall_fn = self.config.fall_curve
        self.rise_kwargs = self.config.rise_curve_kwargs
        self.fall_kwargs = self.config.fall_curve_kwargs
    
    def _snapshot_state(self):
        """
        Snapshot the current internal state of the layer.

        This method stores a deep copy of the layer's `__dict__` and is
        intended to be called before applying the layer across frames,
        allowing the state to be restored later.
        """
        self._initial_state = copy.deepcopy(self.__dict__)
    
    def _reset_state(self):
        """
        Reset the layer to its previously snapshotted state.

        Restores the internal state saved by `_snapshot_state`. This is
        useful when layers maintain internal state that should not persist
        across independent videos or trials.
        """
        init_state = copy.deepcopy(self._initial_state)
        init_state["_initial_state"] = self._initial_state
        self.__dict__ = init_state
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        """
        Compute the temporal weight of the layer at time `dt`.

        If the layer supports weighting, the returned value smoothly
        transitions according to the configured rise and fall curves.
        Otherwise, a binary on/off weight is returned.

        Parameters
        ----------
        dt : float
            Current time (in milliseconds).
        supports_weight : bool
            Whether the layer supports continuous temporal weighting.

        Returns
        -------
        float
            A scalar weight in the range `[0.0, 1.0]` indicating the
            strength of the layer at time `dt`.

        Notes
        -----
        - During the rise phase, `rise_fn` is used to interpolate from
          inactive to fully active.
        - During the fall phase, `fall_fn` interpolates from active to
          inactive.
        - Outside the active window, the weight is `0.0`.
        - If weighting is unsupported, the weight is binary (discrete 0.0 or 1.0).
        """
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
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            `True` if the layer supports continuous rise/fall weighting,
            `False` if it operates as a binary on/off manipulation.
        """
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method should expose all configurable parameters required
        to reproduce the layer's behavior (excluding timing parameters,
        which are handled separately).

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values.
        """
        pass

    @abstractmethod
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:MatLike, dt:float) -> MatLike:
        """
        Apply the layer's manipulation to a single frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the manipulation is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The manipulated frame.
        """
        pass