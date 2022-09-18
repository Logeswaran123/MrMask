# 👺😷 The Masked Man 🎃🥸
<p>
An application that overlays a mask on the human face using mediapipe face mesh model.
</p>

## Description :scroll:
* Use [MakeSense](https://www.makesense.ai/) annotation tool to annaotate the mask.
* Refer [Mask annotated image](https://github.com/Logeswaran123/The-Masked-Man/blob/main/data/mask_points_info/gal_gadot_annotated.jpg) for all the key points.
* Refer utils/const.py for three modes of mask annotations and their corresponding points (FULL_FACE, HALF_FACE, BEARD_FACE). <br />
FULL_FACE - For masks that cover entire face. <br />
HALF_FACE - For masks that cover half of the face. <br />
BEARD_FACE - For masks thar cover beard area. <br />
* Annotate the mask based on the required mode of mask.
* Place the mask png image and annotations csv files in /data/masks dir.
* Ensure the mask image and annotations csv files have same filename.
* Add the mask png image filename in dict in utils/const.py.

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
