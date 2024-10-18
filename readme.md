# <span style="color:green">Shrekification with OpenCV and MediaPipe</span>

In one of the Shrek movies, Shrek turns into a human. Boring. Let's alter the narrative and instead turn **everyone** into Shrek! Using real-time face detection and emotion recognition via **<span style="color:blue">MediaPipe</span>**, and a generous helping of **<span style="color:red">OpenCV</span>**, this dream has become reality.

Since everyone loves portrait mode on cameras, I added a **<span style="color:purple">gaussian blur</span>** to the background so Shrek can truly be the star of the show. A live color histogram is shown for the curious nerds out there to enjoy. The project lets you **<span style="color:orange">toggle</span>** the Shrek overlay on and off while interacting with the camera feed. _(Multiple Shreks supported, but as with portrait mode, only one will be unblurred!)_

---

### <span style="color:green">Features</span>
- **<span style="color:blue">Face Detection</span>**: Real-time detection using **<span style="color:orange">MediaPipe's face mesh</span>**.
- **<span style="color:red">Emotion Detection</span>**: Detects happy, sad, and surprised expressions based on facial landmarks.
- **<span style="color:green">Shrek Overlay</span>**: Depending on the detected emotion, a Shrek face is overlaid on top of the user’s face.
- **<span style="color:purple">Background Blur</span>**: The background behind the user is blurred in a "portrait mode" style.
- **<span style="color:orange">Dynamic Histograms</span>**: A live, updating histogram of the color channels in the video feed.
- **<span style="color:blue">Command-Line Interface</span>**: Use argparse to specify input options, such as toggling features.

---

### <span style="color:purple">Chapters Represented from Practical Python and OpenCV</span>:
**Chapter 3: Loading, Displaying, and Saving Images**  
Functions: `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`

**Chapter 5: Drawing**  
Functions: `cv2.circle()`, used to visualize face landmarks and Shrek face overlay

**Chapter 6: Image Processing**  
Functions: `cv2.GaussianBlur()`, used to blur the background

**Chapter 7: Histograms**  
Functions: `cv2.calcHist()`, dynamic color histograms for live frames

**Chapter 8: Smoothing and Blurring**  
Functions: `cv2.GaussianBlur()`, `cv2.bitwise_and()`

**Bonus: Mediapipe**  
Functions: Convex hull used to define the face boundary for the blur mask

---


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

--- 

### Future Improvements
- Deep Learning Based Emotion Detection
- GAN Based Real-Time Shrek Morphing 
- The AR Shrek Experience

--- 

⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ \
⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ \
⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ \
⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ \
⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ \
⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ \
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠛⠉
