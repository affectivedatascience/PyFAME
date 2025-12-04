import numpy as np

# Defining useful timing functions
def timing_constant(time_delta:float, time_start:float, time_end:float, positive_slope:bool, **kwargs) -> float:
    """ Constant timing function. Always returns 1.0, regardless of input.
    
    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated.

    time_start: float
        The time at which the function begins to rise or fall.
    
    time_end: float
        The time at which the function stops transitioning once it has reached its maximum 
        or minimum weight.
    
    positive_slope: bool
        A boolean flag indicating whether the slope of the function is rising or falling.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0.0, 1.0].
    """

    if time_start <= time_delta <= time_end:
        return 1.0
    else:
        return 0.0

def timing_linear(time_delta:float, time_start:float, time_end:float, positive_slope:bool, **kwargs) -> float:
    """ Normalised linear timing function.

    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated.

    time_start: float
        The time at which the function begins to rise or fall.
    
    time_end: float
        The time at which the function stops transitioning once it has reached its maximum 
        or minimum weight.
    
    positive_slope: bool
        A boolean flag indicating whether the slope of the function is rising or falling.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0.0, 1.0].
    """
    
    duration = time_end - time_start

    if time_start <= time_delta <= time_end:
        if positive_slope:
            weight = (time_delta - time_start) / duration
            return np.clip(weight, 0.0, 1.0)
        else:
            weight = 1 - ((time_delta - (time_end - duration)) / duration)
            return np.clip(weight, 0.0, 1.0)
    else:
        return 0.0

def timing_sigmoid(time_delta:float, time_start:float, time_end:float, positive_slope:bool, **kwargs) -> float:
    """ Returns the value of the sigmoid function evaluated at time t. If paramater k (growth_rate) is 
    not provided in kwargs, it will be set to 10.
    
    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated. 
    
    time_start: float
        The time at which the function begins to rise or fall.
    
    time_end: float
        The time at which the function stops transitioning once it has reached its maximum 
        or minimum weight.
    
    positive_slope: bool
        A boolean flag indicating whether the slope of the function is rising or falling.
    
    growth_rate: float
        The slope or growth rate parameter, controls how quickly the sigmoid function transitions
        from zero to one. 
    
    returns
    -------

    weight: float
        A normalised weight in the range [0.0, 1.0].
    """

    def scaled_sigmoid(t, k):
        raw = 1 / (1 + np.exp(-k * (t-0.5)))
        min_val = 1 / (1 + np.exp(k*0.5))
        max_val = 1 / (1 + np.exp(-k*0.5))
        term = (raw - min_val) / (max_val - min_val)
        return np.clip(term, 0.0, 1.0)
    
    duration = time_end - time_start
    k = 10.0

    if kwargs.get("growth_rate") is not None:
        k = kwargs.get("growth_rate")
    elif kwargs.get("k") is not None:
        k = kwargs.get("k")
    
    if time_start <= time_delta <= time_end:
        if positive_slope:
            cur_eval = (time_delta - time_start) / duration
            return scaled_sigmoid(cur_eval, k)
        else:
            cur_eval = 1 - ((time_delta - (time_end - duration)) / duration)
            return scaled_sigmoid(cur_eval, k)
    else:
        return 0.0

def timing_gaussian(time_delta:float, time_start:float, time_end:float, positive_slope:bool, **kwargs) -> float:
    """ Normalized gaussian timing function

    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated. 
    
    time_start: float
        The time at which the function begins to rise or fall.
    
    time_end: float
        The time at which the function stops transitioning once it has reached its maximum 
        or minimum weight.
    
    positive_slope: bool
        A boolean flag indicating whether the slope of the function is rising or falling.
  
    variance (sigma): float
        Controls the steepness of the curve's transition.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0.0, 1.0].
    """
    def half_gaussian(x, sigma, positive):
        if positive:
            # map to left half of distribution
            t = 0.5 * x
            raw = np.exp(-((t - 0.5) ** 2) / (2 * sigma**2))
            min_val = np.exp(-((0.0 - 0.5) ** 2) / (2 * sigma**2))
            max_val = 1.0
        else:
            # map to right half of distribution
            t = 0.5 + 0.5 * x
            raw = np.exp(-((t - 0.5) ** 2) / (2 * sigma**2))
            min_val = np.exp(-((1.0 - 0.5) ** 2) / (2 * sigma**2))
            max_val = 1.0

        return np.clip(((raw - min_val) / (max_val - min_val)), 0.0, 1.0)

    duration = time_end - time_start
    sigma = 1.0

    if kwargs.get("variance") is not None:
        sigma = kwargs.get("variance")
    elif kwargs.get("sigma") is not None:
        sigma = kwargs.get("sigma")

    if time_start <= time_delta <= time_end:
        cur_eval = (time_delta - time_start) / duration
        return half_gaussian(cur_eval, sigma, positive_slope)
    else:
        return 0.0