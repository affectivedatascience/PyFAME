# About PyFAME?

PyFAME is a python package providing tools for targeted facial manipulation and analysis
using facial landmark-based computer vision techniques.

Some of the packages capabilities include changing the colour of user-specified facial regions using industry-standard colour spaces (Lab, HSV, BGR), occlusion of user-defined facial regions (e.g., eyes, nose, mouth, hemi-face), and isolation of the head from background scene through video matting. Facial modifications can be further transitioned on and off through a range of timing functions (e.g., linear, sigmoid, Gaussian etc).

!!! info
    Want to see it in action? Skip ahead to the [Quick Start](./index.md#quick-start)

---

## Full List of Features

<div class="grid cards" markdown>

- :material-face-recognition: **Landmark-based facial regions**
- :material-timer-outline: **Precise temporal modulation**
- :material-palette: **Colouring, brightness and saturation based manipulations**
- :material-eye: **Region-based masking and occlusion**
- :material-image-edit: **Object overlays and stylisation**
- :material-move-resize: **Spatial manipulations**
- :simple-googleanalytics: **Colour and motion-based video analyses**

</div>

---

## Statement of Need

Currently, there are no available tools for performing these types of pixel-level operations over videos. Existing research has commonly used general image editing tools (such as Photoshop). However, these tools apply changes to the entire image, causing noticeable background artifacts. PyFAME provides users the ability to selectively modify specific regions of the face, for both static images and videos. Our package also seamlessly integrates temporal functions into it's video processing, allowing users to specify how and when pixel-level operations will be applied.