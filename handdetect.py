# Importing Libraries
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Define connections between landmarks
connections = [
    (0, 1), (0, 5), (0,17), (0, 9), (0, 13),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
l_conf_sum = 0
r_conf_sum = 0
l_conf_len = 0
r_conf_len = 0
l_conf_avg = 0
r_conf_avg = 0
while True:
    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Resize the image
    # Resize the image while maintaining aspect ratio
    width = 1024
    height = 600
    aspect_ratio = img.shape[1] / img.shape[0]
    if width / aspect_ratio <= height:
        new_size = (int(width), int(width / aspect_ratio))
    else:
        new_size = (int(height * aspect_ratio), int(height))
    img = cv2.resize(img, new_size)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Return whether it is Right or Left Hand
            label = MessageToDict(hand_info)['classification'][0]['label']
            confidence = MessageToDict(hand_info)['classification'][0]['score']

            # Draw landmarks on the image
            h, w, _ = img.shape
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks_list.append((x, y))
                cv2.circle(img, (x, y), 8, (255, 0, 0), -1)

            # Draw lines between landmarks
            for connection in connections:
                start_point = landmarks_list[connection[0]]
                end_point = landmarks_list[connection[1]]
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)

            # Confidence value and text
            rounded_confidence = round(confidence * 100, 2)
            confidence_text = f'Confidence: {rounded_confidence}%'

            if label == 'Left':
                # Display 'Left Hand' on the left side of the window
                cv2.putText(img, label+' Hand',
                            (20, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.9, (0, 255, 0), 2)
                # Display confidence value on the image (right-aligned)
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)[0]
                text_x = img.shape[1] - 20 - text_size[0]
                cv2.putText(img, confidence_text, (text_x, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                l_conf_sum += rounded_confidence
                l_conf_len += 1
                l_conf_avg = l_conf_sum / l_conf_len
            elif label == 'Right':
                # Display 'Right Hand' on the right side of the window
                cv2.putText(img, label+' Hand', (20, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.9, (0, 255, 0), 2)
                #  Display confidence value on the image (right-aligned)
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)[0]
                text_x = img.shape[1] - 20 - text_size[0]
                cv2.putText(img, confidence_text, (text_x, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                r_conf_sum += rounded_confidence
                r_conf_len += 1
                r_conf_avg = r_conf_sum / r_conf_len

    # Display Video and exit when 'q' is pressed
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Display the average confidence value of both hands
        print(f'Left Hand Confidence Average: {l_conf_avg}%')
        print(f'Right Hand Confidence Average: {r_conf_avg}%')
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
