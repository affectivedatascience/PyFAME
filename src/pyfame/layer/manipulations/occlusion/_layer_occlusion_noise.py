from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeFloat, PositiveInt
from typing import Union, List, Tuple, Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np
from skimage.util import *

class NoiseParameters(BaseModel):
    """
    Configuration model defining the control parameters for applying
    a noise-based occlusion to a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation
    and default handling of noise occlusion parameters.

    Attributes
    ----------
    random_seed : int or None
        An optional seed for the random number generator, enabling
        reproducible noise patterns across runs. If ``None``, the
        generator is seeded non-deterministically.
    noise_method : str or int
        The noise algorithm to apply. Accepted string values are
        ``"pixelate"``, ``"salt and pepper"``, and ``"gaussian"``.
        Accepted integer values are ``18`` (pixelate), ``19``
        (salt and pepper), and ``20`` (gaussian). Integer inputs are
        normalised to their string equivalents on validation.
    noise_probability : float
        The probability that any given pixel is affected by salt and
        pepper noise. Must be non-negative. Only used when
        ``noise_method`` is ``"salt and pepper"``.
    pixel_size : int
        The downsampling factor used for pixelation. The frame is
        downscaled by this factor before being upscaled back to its
        original dimensions. Must be a positive integer >= 4. Only
        used when ``noise_method`` is ``"pixelate"``.
    gaussian_mean : float
        The mean of the Gaussian noise distribution. Only used when
        ``noise_method`` is ``"gaussian"``.
    gaussian_deviation : float
        The standard deviation of the Gaussian noise distribution.
        Must be non-negative. Variance is derived as the square of
        this value. Only used when ``noise_method`` is ``"gaussian"``.
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the
        region(s) in which noise will be applied.
    """

    random_seed:Optional[int]
    noise_method:Union[int,str]
    noise_probability:NonNegativeFloat
    pixel_size:PositiveInt
    gaussian_mean:float
    gaussian_deviation:NonNegativeFloat
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("noise_method", mode="before")
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        noise_method_mapping = {18:"pixelate", 19:"salt and pepper", 20:"gaussian"}

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"pixelate", "salt and pepper", "gaussian"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {18,19,20}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return noise_method_mapping.get(value)
        
        raise TypeError(f"{field_name} provided an invalid type. Must be one of int, str.")
    
    @field_validator("pixel_size")
    @classmethod
    def check_compatible_size(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value < 4:
            raise ValueError(f"{field_name} requires a size >= 4.")
        
        return value

class LayerOcclusionNoise(Layer):
    """
    Manipulation layer that applies a noise-based occlusion within
    landmark-defined facial regions.

    This layer obscures a region of interest using one of three noise
    methods: pixelation, salt and pepper noise, or Gaussian noise.
    Pixelation reduces spatial resolution by downsampling and upsampling
    the frame, removing fine detail while preserving coarse structure.
    Salt and pepper noise randomly sets pixels to black or white with a
    configurable probability. Gaussian noise adds normally distributed
    intensity perturbations to each pixel, with controllable mean and
    standard deviation.

    An optional random seed enables reproducible noise patterns across
    independent runs or frames.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, and rise/fall
        durations.
    noise_parameters : NoiseParameters
        Configuration model specifying the noise method, method-specific
        parameters, landmark region(s), and optional random seed.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    noise_params : NoiseParameters
        Noise-specific configuration parameters.
    rand_seed : int or None
        Seed for the random number generator. If ``None``, noise patterns
        are non-deterministic across runs.
    noise_method : str
        The noise algorithm to apply (one of ``"pixelate"``,
        ``"salt and pepper"``, or ``"gaussian"``).
    noise_probability : float
        Per-pixel noise probability used by the salt and pepper method.
    pixel_size : int
        Downsampling factor used by the pixelation method.
    mean : float
        Mean of the Gaussian noise distribution.
    standard_deviation : float
        Standard deviation of the Gaussian noise distribution.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region(s) in which noise is applied.

    Notes
    -----
    - This layer does not support temporal weighting; noise is applied
      as a binary on/off effect governed solely by onset and offset times.
    """

    def __init__(self, timing_configuration:TimingConfiguration, noise_parameters:NoiseParameters):
        """
        Initialize a noise occlusion manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the noise effect is
            applied.
        noise_parameters : NoiseParameters
            Parameters defining the noise method, method-specific
            configuration, target landmark region(s), and optional
            random seed.

        Notes
        -----
        - A snapshot of the initial state is taken after initialization to
          allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.noise_params = noise_parameters

        # Initialising superclass
        super().__init__(self.time_config)

        # Defining class parameters
        self.rand_seed = self.noise_params.random_seed
        self.noise_method = self.noise_params.noise_method
        self.noise_probability = self.noise_params.noise_probability
        self.pixel_size = self.noise_params.pixel_size
        self.mean = self.noise_params.gaussian_mean
        self.standard_deviation = self.noise_params.gaussian_deviation
        self.landmark_paths = self.noise_params.landmark_paths

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            ``False``, as noise occlusion operates as a binary on/off
            effect and does not support continuous rise/fall weighting.
        """
        return False

    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method exposes all configurable parameters required to reproduce
        the layer's behavior.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values,
            combining both timing and noise occlusion configuration fields.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.noise_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)

    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the noise occlusion manipulation to a single frame.

        A binary mask is derived from the configured landmark paths to
        isolate the region of interest. Noise is generated and applied to
        a copy of the full frame using the configured method, then the
        noised result is composited over the original frame within the
        masked region.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the noise occlusion is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The frame with noise applied within the landmark-defined region
            and original pixel values preserved outside it.
        """
        # This layer does not support weight; weight will always be 0.0 or 1.0
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Create an rng instance to help generate random noise
            rng = None
            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            # Mask out the roi
            mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            output_frame = frame.copy()

            match self.noise_method:
                case "pixelate":
                    height, width = frame.shape[:2]
                    h = frame.shape[0]//self.pixel_size
                    w = frame.shape[1]//self.pixel_size

                    # resizing the pixels of the image in the region of interest
                    temp = cv.resize(frame, (w, h), None, 0, 0, cv.INTER_LINEAR)
                    output_frame = cv.resize(temp, (width, height), None, 0, 0, cv.INTER_NEAREST)

                    output_frame = np.where(mask == 255, output_frame, frame)
                
                case "salt and pepper":
                    # Divide prob in 2 for "salt" and "pepper"
                    thresh = self.noise_probability
                    noise_prob = self.noise_probability/2
                    
                    # Use numpy's random number generator to generate a random matrix in the shape of the frame
                    rdm = rng.random(frame.shape[:2])

                    # Create boolean masks 
                    pepper_mask = rdm < noise_prob
                    salt_mask = (rdm >= noise_prob) & (rdm < thresh)
                    
                    # Apply boolean masks
                    output_frame[pepper_mask] = [0,0,0]
                    output_frame[salt_mask] = [255,255,255]

                    output_frame = np.where(mask == 255, output_frame, frame)
                
                case "gaussian":
                    var = self.standard_deviation**2

                    # scikit-image's random_noise function works with floating point images; we need to pre-convert our frames to float64
                    output_frame = img_as_float64(output_frame)
                    output_frame = random_noise(image=output_frame, mode='gaussian', rng=rng, mean=self.mean, var=var)
                    output_frame = img_as_ubyte(output_frame)

                    output_frame = np.where(mask == 255, output_frame, frame)
            
            return output_frame

def layer_occlusion_noise(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, noise_method:int|str = "gaussian", 
                          noise_probability:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_deviation:float = 0.5, random_seed:int|None = None) -> LayerOcclusionNoise:
    """
    Factory function for the noise occlusion manipulation layer.
    `LayerOcclusionNoise` obscures one or more landmark-defined facial
    regions using one of three noise methods: pixelation, salt and pepper,
    or Gaussian noise. An optional random seed enables reproducible noise
    patterns across independent runs.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset
        and offset. If ``None``, a default ``TimingConfiguration`` is
        instantiated. The default instantiation assumes onset at 0.0 and
        offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the
        region(s) in which noise will be applied.
    noise_method : str or int, default="gaussian"
        The noise algorithm to apply. Accepted string values are
        ``"pixelate"``, ``"salt and pepper"``, and ``"gaussian"``.
        Accepted integer values are ``18`` (pixelate), ``19``
        (salt and pepper), and ``20`` (gaussian).
    noise_probability : float, default=0.5
        The per-pixel noise probability used by the salt and pepper method.
        Must be non-negative. Ignored by other noise methods.
    pixel_size : int, default=32
        The downsampling factor used by the pixelation method. Must be a
        positive integer >= 4. Ignored by other noise methods.
    mean : float, default=0.0
        The mean of the Gaussian noise distribution. Ignored by other
        noise methods.
    standard_deviation : float, default=0.5
        The standard deviation of the Gaussian noise distribution. Must be
        non-negative. Ignored by other noise methods.
    random_seed : int or None, default=None
        An optional seed for the random number generator. If provided,
        noise patterns are reproducible across runs. If ``None``, noise
        is sampled non-deterministically.

    Returns
    -------
    LayerOcclusionNoise
        An instance of the noise occlusion manipulation layer.

    Raises
    ------
    ValueError
        When provided invalid, out-of-range, or unrecognized parameter values.
    """

    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = NoiseParameters(
            random_seed=random_seed, 
            noise_method=noise_method, 
            noise_probability=noise_probability, 
            pixel_size=pixel_size, 
            gaussian_mean=mean, 
            gaussian_deviation=standard_deviation, 
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionNoise.__name__}: {e}")
    
    return LayerOcclusionNoise(time_config, params)

__all__ = ["layer_occlusion_noise", "NoiseParameters"]