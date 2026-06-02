from .layer import Layer
from cv2.typing import MatLike
from pyfame.layer.manipulations.colour.layer_colour_recolour import LayerColourRecolour
from pyfame.layer.manipulations.overlay.layer_overlay import LayerOverlay

class LayerPipeline:
    """
    A container class that resolves layer ordering and sequentially 
    applies layers to a given input image/frame. 

    Notes
    -----
    - `enforce_layer_ordering()` ensures that weighted full-frame 
    manipulations are applied first, and that unweighted (binary) 
    manipulations are applied second. This is done to ensure that 
    colour and spatial transforms are accurately represented prior
    to the addition of occlusions or regional masking.
    - `apply_layers()` provides distinct behavour for 
    `LayerColourRecolour` and `LayerOverlay` as they require 
    blendshapes from the mediapipe FaceLandmarker task.
    """
    
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:Layer):
        self.layers.append(layer)
    
    def add_layers(self, layers:list[Layer]):
        if not self.layers:
            self.layers = layers
        else:
            self.layers.extend(layers)
    
    def enforce_layer_ordering(self) -> list[Layer]:
        # put non weighted manipulations like masking and occlusion
        # at the end, to not disrupt the full-frame manipulations
        weighted = []
        non_weighted = []

        for layer in self.layers:
            if layer.supports_weight():
                weighted.append(layer)
            else:
                non_weighted.append(layer)
        
        # Non-weighted manipulations order of precidence
        # Masking, blurring/noise, 
        
        weighted.extend(non_weighted)
        return weighted
    
    def apply_layers(self, landmarker_coordinates:list[tuple[int,int]], frame:MatLike, dt:float, **kwargs) -> MatLike:
        self.layers = self.enforce_layer_ordering()
        blendshapes = kwargs.get("blendshapes", None)

        for layer in self.layers:
            if isinstance(layer, (LayerColourRecolour, LayerOverlay)):
                frame = layer.apply_layer(
                    landmarker_coordinates = landmarker_coordinates,
                    frame = frame, 
                    dt = dt, 
                    blendshapes = blendshapes
                )
            else:
                frame = layer.apply_layer(
                    landmarker_coordinates = landmarker_coordinates, 
                    frame = frame, 
                    dt = dt
                )
        return frame

__all__ = ["LayerPipeline"]