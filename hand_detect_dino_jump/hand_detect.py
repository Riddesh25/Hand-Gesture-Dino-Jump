import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Initialize PyAutoGUI
pyautogui.PAUSE = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)  # You may need to change the index if you have multiple cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize font settings for text display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)
font_thickness = 2
jump_text_position = (50, 50)  # Adjust the position where "Jump" text will be displayed

# Initialize a flag to track if "Jump" text has been displayed
jump_text_displayed = False

while cap.isOpened():  # Check if the camera is opened successfully
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Initialize landmarks and hull to None
    landmarks = None
    hull = None

    # Initialize is_open_hand flag to False
    is_open_hand = False

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract the landmarks of the hand
            landmarks_list = []
            for point in landmarks.landmark:
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                landmarks_list.append((x, y))

            # Calculate the area of the convex hull of the hand
            hull = cv2.convexHull(np.array(landmarks_list), returnPoints=True)
            area = cv2.contourArea(hull)

            # Check if the hand is open (area above a threshold)
            open_hand_threshold = 3000  # Adjust this threshold based on your setup
            is_open_hand = area > open_hand_threshold

            # Check if the hand is in a position that corresponds to the jump gesture
            # You may need to adjust these coordinates based on your setup
            if (
                is_open_hand and
                landmarks_list[8][1] < landmarks_list[7][1] and
                landmarks_list[12][1] < landmarks_list[11][1] and
                landmarks_list[16][1] < landmarks_list[15][1] and
                landmarks_list[20][1] < landmarks_list[19][1]
            ):
                if not jump_text_displayed:
                    pyautogui.press('space')
                    jump_text_displayed = True
            else:
                jump_text_displayed = False

    # Draw gesture tracking lines on the frame
    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw "Jump" text on the frame if the hand is open
    if is_open_hand and jump_text_displayed:
        cv2.putText(frame, 'Jump', jump_text_position, font, font_scale, font_color, font_thickness)

    # Display the frame with gesture tracking lines and "Jump" text
    cv2.imshow("Dino Game Control", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()