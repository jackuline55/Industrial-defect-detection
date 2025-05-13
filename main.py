import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
uploaded = files.upload()
import numpy as np
import io
file_name = next(iter(uploaded))
image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)
cv2_imshow(image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_image)
edges = cv2.Canny(image, threshold1=100, threshold2=200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Green color contours
cv2_imshow(image_with_contours)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define range of blue color in HSV
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Create a mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=mask)

# Show mask and result
cv2_imshow(mask)
cv2_imshow(result)
# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

