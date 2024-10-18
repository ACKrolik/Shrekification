# Shrekification with OpenCV and MediaPipe

In one of the Shrek movies, Shrek turns into a human. Boring. Let's alter the narrative and instead turn everyone into Shrek. Using real-time face detection and emotion recognition via MediaPipe, and a good amount of OpenCV, my dream is now reality. Since everyone loves portrait mode on cameras, I applied a gaussian blur to the background so Shrek can really be the star of the show. A color histogram is shown for nerds to enjoy. The project allows you to toggle the Shrek overlay on and off while interacting with the camera feed. (Multiple Shreks supported, but as with portrait mode, only one will be unblurred)

### Features
- Face Detection: Real-time detection using MediaPipe's face mesh.
- Emotion Detection: Detects happy, sad, and surprised expressions based on facial landmarks.
- Shrek Overlay: Depending on the detected emotion, a Shrek face is overlaid on top of the user’s face.
- Background Blur: The background behind the user is blurred in a "portrait mode" style.
- Dynamic Histograms: A live, updating histogram of the color channels in the video feed.
- Command-Line Interface: Use argparse to specify input options, such as toggling features.

### Chapters Represented from Practical Python and OpenCV:
**Chapter 3: Loading, Displaying, and Saving Images** \
Functions: cv2.imread(), cv2.imshow(), cv2.imwrite()

**Chapter 5: Drawing**\
Functions: cv2.circle(), used to visualize face landmarks and Shrek face overlay

**Chapter 6: Image Processing**\
Functions: cv2.GaussianBlur(), used to blur the background

**Chapter 7: Histograms**\
Functions: cv2.calcHist(), dynamic color histograms for live frames

**Chapter 8**: Smoothing and Blurring\
Functions: cv2.GaussianBlur(), cv2.bitwise_and()

**Bonus: Mediapipe**\
Functions: Convex hull used to define the face boundary for the blur mask


### Ensure you have the required dependencies:
`pip install opencv-python mediapipe matplotlib numpy argparse`


### Usage
Run the program with:


`python shrek_overlay.py --input <path_to_video> --shrek_overlay --blur --histogram`


##### Command-Line Arguments
`--input`: Path to the video feed (default: live webcam).\
`--shrek_overlay`: Enable or disable the Shrek face overlay.\
`--blur`: Enable background blurring.\
`--histogram`: Display real-time histograms of color channels.

##### Key Controls
Spacebar: Toggle Shrek overlay on/off.\
q: Quit the application.


### Future Improvements
- Deep Learning Based Emotion Detection
- GAN Based Real-Time Shrek Morphing 
- The AR Shrek Experience
