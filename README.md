## Advanced Lane Finding
The Project
---

The goals / steps of this project are the following:

* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Usage:

### Run the pipeline:
```bash
chmod +x shell_script.sh
# 1 For debug mode, 0 for normal mode
./shell_script.sh --video "challenge_video.mp4" "output.mp4" 1
./shell_script.sh --photo "test_images/challenge_video_frame_1.jpg" "output.jpg" 1
```
