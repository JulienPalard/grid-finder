# Goal of this project is to find a grid (chessboard) in the given image.

http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration_square_chess/camera_calibration_square_chess.html

```
$ git clone https://github.com/Itseez/opencv.git
$ cd opencv
$ git checkout 3.0.0-rc1
$ cd cmake
$ apt-get install libgstreamer-plugins-base1.0-dev libgstreamer-plugins-base0.10-dev
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_opencv_java=OFF -D PYTHON_EXECUTABLE=/usr/bin/python3.4 ..
$ cd ..
$ make -j 4
$ PYTHONPATH=opencv/lib/python3/ ./grid_finder.py
```

We want to warp an image to the "airplane" 2d view.

We do this in two steps:

 - Find 4 points around something that should be a rectangle
 - Use warpPerspective to warp it

To find those 4 points, several methods:

 - Detect lines, detect columns, apply and AND:
   - Work only if lines are already horizontal and columns already vertical,
     that's not our case.
 - findChessboardCorners
   - Not bad but need to know the chessboard size, that's not our case.
 - Detect lines with houghlines2
   - That's we're doing for now.
 - What else ?
