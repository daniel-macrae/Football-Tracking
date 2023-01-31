## Tracking football player positions from a single camera feed
In this play project I attempt to track the positions of players on a football pitch over time. The ability to do so would allow for a multitude of visualisation and team analysis possiblities at a low cost (i.e. without multiple cameras or expensive analysis software suites).

My approach is to utilise pretrained Mask-R-CNN models (pytorch) to identify players at the beginning of a sequence of frames, identify the players' teams and generate labels accordingly, and then apply object tracking techniques (open-cv2) to follow the players' positions throughout the video clip.


An early example output:
![](https://github.com/daniel-macrae/Football-Tracking/blob/main/output_gif.gif)
