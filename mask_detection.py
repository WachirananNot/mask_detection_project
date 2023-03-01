import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
model = tf.keras.models.load_model('mask_detection_model.h5')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        # Preprocess the face for the mask detection model
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0
        # Make a prediction with the mask detection model
        prediction = model.predict(face)

        # Determine if the person is wearing a mask or not
        if prediction[0][0] > 0.5:
            label = 'no mask'
            color = (0, 0, 255)
        else:
            label = 'mask'
            color = (0, 255, 0)
        
        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show the frame
    cv2.imshow('Mask Detection', frame)
    
    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()