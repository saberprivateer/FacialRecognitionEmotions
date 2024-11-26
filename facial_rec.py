import cv2
import numpy as np
print("this is cv2's version", cv2.__version__)

# (Optional) Load Deepface model for emotion recognition
from deepface import DeepFace
model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Convert frame to grayscale for face detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the grayscale frame
  #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  # Process each detected face
  for (x, y, w, h) in faces:
    # Extract the face region of interest (ROI)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    # (Optional) Use Deepface for emotion recognition
    #if DeepFace is not None:
      # Predict emotions using Deepface
      #emotions = DeepFace.model(roi_color)
      # Display the predicted emotions on the frame

    # Resize the face ROI to match the input shape of the model
    #resized_face = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize the resized face image
    #normalized_face = resized_face / 255.0

    # Reshape the image to match the input shape of the model
    #reshaped_face = normalized_face.reshape(1, 48, 48, 1)

    #preds = model.predict(roi_gray)[0]
    #emotion_idx = preds.argmax()
    #emotion = emotion_labels[emotion_idx]
    result = DeepFace.analyze(frame, enforce_detection=False, actions=['emotion'])
    result_gray = DeepFace.analyze(frame, enforce_detection=False, actions=['emotion'])
    print('the emotion', result)
    print('lets try again')
    print('the dominant emotion', result[0]['dominant_emotion'])
    #objs = DeepFace.analyze(roi_gray, actions=['emotion'])
    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, "check it!", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
    cv2.putText(frame, result[0]['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # cv2.putText(frame,)
    # Display the resulting frame
  cv2.imshow('Facial Emotion Recognition', frame)

  # Exit loop on 'q' key press
  if cv2.waitKey(1) == ord('q'):
    break

# Release capture and close all windows
cap.release()
cv2.destroyAllWindows()