# 👺😷 The Masked Man 🎃🥸
<p>
An application that overlays a mask on the human face using mediapipe face mesh model.
</p>

## Description :scroll:
** TODO **

## General Requirements :mage_man:
1. Record a video of person or enable a live cam.
2. Place the camera horizontally facing the person.
3. Higher the resolution of video, higher the quality of estimation and measurement.

## Code Requirements :mage_woman:
```python
pip install -r requirements.txt
```

## How to run :running_man:
```python
python mrmask.py --video <path to video file> --face <face mode> --mask <mask number> --mesh <face mesh>
```
Note:<br />

*<path to video file\>* - Path to the input video file. If not provided, enable webcam.<br />
*<face mode\>* - Face mode mentioning single face or multiple faces. <b>Choices: "single" or "multi". Default: single </b><br />
*<mask number\>* - Mask number corresponding to mask key of MASK dict in utils/const.py. Random mask if argument not provided. <br />
*<face mesh\>* - Add face mesh on detected faces. <b>Choices: "True" or "False". Default: False </b><br />

## Results :bar_chart:

```python
python mrmask.py --video <path to video file> --mask 6 --face multi --mesh True
```
Note:<br />
--mask 6 -> 6 corresponds to pikachu.png in MASK dict.

https://user-images.githubusercontent.com/36563521/189902888-7f92f2fe-45da-481b-b0f3-5bf1fdef51c8.mp4

Input video source [here](https://www.pexels.com/video/girl-friends-posing-for-selfies-5935550/).
---

```python
python mrmask.py
```

** TODO: Add result video **

## References :page_facing_up:

* [Mediapipe by Google](https://github.com/google/mediapipe).

Happy Learning! 😄
