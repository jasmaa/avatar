# Avatar

Inspired by [avatarify](https://github.com/alievk/avatarify)

## Installation
**NOTE: Dlib wheel only works with Python 3.6**
  1. `pip install -r requirements.txt`
  2. Download and unzip [facial landmark model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2)
  3. Install OBS
  4. Install OBS VirtualCam plugin
  
## Running
  1. `python app.py`
  2. Open OBS
    - Add a new `Window Capture` source linked to the output feed
	- Start VirtualCam under Tools -> VirtualCam
	- Adjust screen in OBS accordingly
  3. Open Zoom
    - Set video source to `OBS Camera`