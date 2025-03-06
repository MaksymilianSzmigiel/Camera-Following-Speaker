import time
import cv2

# Initialize camera
camera = cv2.VideoCapture(0)

# Camera settings (some settings may not be available on laptops)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# Load face detectors
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

if frontal_cascade.empty() or profile_cascade.empty() or upperbody_cascade.empty():
    print("❌ Error: Could not load face detector XML files!")
    exit()

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Live preview with face detection
while True:
    ret, image = camera.read()
    if not ret:
        print("❌ Error: Could not capture image!")
        break
    
    image = cv2.resize(image, (160, 120))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.bilateralFilter(gray, 5, 50, 50)
    edges = cv2.Canny(gray, 20, 150)

# Sumowanie obrazu z krawędziami, aby wzmocnić kontrast
    enhanced_gray = cv2.addWeighted(gray, 0.6, edges, 0.4, 0)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #gray = clahe.apply(gray)


    # Detect frontal faces
    # faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))

    # Detect profile faces (side view)
    # profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
    
    # Detect upper bodies
    upperbodies = upperbody_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
    
    # # Draw rectangles around detected faces and upper bodies
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for front

    # for (x, y, w, h) in profiles:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for profile
     
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(enhanced_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for upper body

    # Display image
    cv2.imshow("Camera Feed",enhanced_gray)

    # Exit on "q" key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
