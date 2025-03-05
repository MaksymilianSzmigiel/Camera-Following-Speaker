from picamera2 import Picamera2
import time
import cv2

# Initialize camera
camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (160, 120)}))

# Camera settings
camera.set_controls({"AnalogueGain": 5.0, "ExposureTime": 100000})  
camera.set_controls({"AwbMode": 1, "AeEnable": True})

camera.start()
time.sleep(2)

# Load face detectors
frontal_cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

if frontal_cascade.empty() or profile_cascade.empty():
    print("❌ Error: Could not load face detector XML files!")
    exit()

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Live preview with face detection
while True:
    image = camera.capture_array()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 120))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect frontal faces
    faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))

    # Detect profile faces (side view)
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
    
    
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for front

    for (x, y, w, h) in profiles:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for profile
     # Draw rectangles around detected upper bodies
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for upper body

    # Display image
    cv2.imshow("Camera Feed", image)

    # Check if window was closed
    if cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_AUTOSIZE) < 0:
        break

    # Exit on "q" key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
0
