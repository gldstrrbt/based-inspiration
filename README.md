# based-inspiration
Detecting inspirational Lil B quotes overlayed in his music videos. The majority, if not all, of his video quotes are not documented in text. This is an attempt to create an everlasting archive of the inspirational quotes he has provided to the world, via his music videos.

### Requirements
1. `pip3 install youtube-dl`
2. `pip3 install pillow`
3. `pip3 install opencv-python`

#### Run
1. Download Lil B's videos from youtube `youtube-dl -ciw -f "mp4" https://www.youtube.com/user/lilbpack1`
2. Export frames with Lil B quotes. This part takes quite a long time, as the text detection tends to detect shapes and forms objects or lighting of objects as text. `python3 text_detection_video.py`


#### Credits
Credit to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) @ [pyimagesearch.com](https://www.pyimagesearch.com), for the text detection [tutorial](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/). I've made modifications to the original code, for example detecting the changes of frames and higher levels of prediction confidences, to lower the number of false positives, also the exporting of frames.

Adrian's teachings are invaluable, as they're incredibly thorough and comprehensible. If you're looking to learn more about computer vision/opencv with Python, I highly recommend checking out his tutorials.

