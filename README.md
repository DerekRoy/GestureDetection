# GestureDetection
![Detector In Action](/images/Detector.gif)
https://youtu.be/RCMpGXbhoY8

Detect 5 different hand gestures shown on a live webcam.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Peace Hand Sign](/images/peace.jpg) ![Left Hand Sign](/images/left.jpg) ![Right Hand Sign](/images/right.jpg) ![Forks Hand Sign](/images/forks.jpg) ![Stop Hand Sign](/images/stop.jpg) 

- Peace
- Left
- Right
- Forks 
- Stop

## Required modules

You will need to pip install several python modules to run this code: 
- `pip install numpy`
- `pip install tensorflow`
- `pip install keras`
- `pip install opencv-python`
- `pip install imutils` 

If there are any issues installing these modules please use `sudo pip install` instead. 

## Notes

The hand detection aspect of this code was taken from Victor Dibia, here is the [Github Repository](https://github.com/victordibia/handtracking) if you are interested. 

While the gesture detector works in some regards it is not adequate enough as is. The model I used was trained on 28,000 images of my hands making these gestures. The pictures were taken in different environments, and with different hand angles and positions. If you would like to work more with this project here is the [data](https://drive.google.com/file/d/1Q9KPq5pb_Sp_9FPD0CUbS7wlxTnyTPo5/view?usp=sharing). If you can not access the data feel free to contact me. 
