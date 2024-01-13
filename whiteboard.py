import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize variables
cap = cv2.VideoCapture(0)
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White canvas
drawing = False
prev_x, prev_y = 0, 0
selected_color = (0, 0, 0)  # Default color is black

# Define color boxes at the top
color_boxes = [
    {
        'name': 'Blue',  # Blue
        'code': (0, 0, 80, 50),
    },
    {
        'name': 'Green',  # Green
        'code': (80, 0, 160, 50),
    },
    {
        'name': 'Red',  # Red
        'code': (160, 0, 240, 50),
    },
]

# Define clear box
clear_box = (240, 0, 320, 50)

while True:
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract index finger and thumb coordinates
            index_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ih, iw, _ = frame.shape
            index_finger_x, index_finger_y = int(index_finger_tip.x * iw), int(
                index_finger_tip.y * ih
            )
            thumb_x, thumb_y = int(thumb_tip.x * iw), int(thumb_tip.y * ih)

            # Check if index finger and thumb are touching the clear box
            if clear_box[0] < thumb_x < clear_box[2] and clear_box[1] < thumb_y < clear_box[3] \
                and clear_box[0] < index_finger_x < clear_box[2] and clear_box[1] < index_finger_y < clear_box[3]:
                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Clear the canvas

            for i, color_box in enumerate(color_boxes):
                x1, y1, x2, y2 = color_box['code']
                if x1 < thumb_x < x2 and x1 < index_finger_x < x2 and y1 < thumb_y < y2 and y1 < index_finger_y < y2:
                    selected_color = color_box['code'][:3]  # Use color code from 'code'

            # Start drawing when index finger is pointed
            if index_finger_y < ih - 50:
                if not drawing:
                    prev_x, prev_y = index_finger_x, index_finger_y
                    drawing = True

                # Draw a line between previous and current fingertip position with selected color
                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (index_finger_x, index_finger_y),
                    selected_color,
                    5,
                )
                prev_x, prev_y = index_finger_x, index_finger_y
            else:
                drawing = False

    # Draw color boxes at the top on the canvas
    for i, color_box in enumerate(color_boxes):
        x1, y1, x2, y2 = color_box['code']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_box['code'][:3], -1)  # Use color code from 'code'
        
        # Add text on the color box
        color_name = color_box['name']
        cv2.putText(canvas, color_name, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    

    
    # Draw clear box on the canvas
    cv2.rectangle(
        canvas,
        (clear_box[0], clear_box[1]),
        (clear_box[2], clear_box[3]),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        canvas,
        "CLEAR",
        (clear_box[0] + 10, clear_box[1] + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )

    # Add the canvas to the frame
    frame = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

    # Display the frame
    cv2.imshow("Hand Gesture Drawing", frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
