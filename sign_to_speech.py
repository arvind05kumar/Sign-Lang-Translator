import cv2
import mediapipe as mp
import pyttsx3

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

spoken = False

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get y-coordinates of fingertips (indexes: 8, 12, 16, 20)
            tip_ids = [8, 12, 16, 20]
            fingers_up = 0

            for tip in tip_ids:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers_up += 1

            if fingers_up == 0 and not spoken:
                cv2.putText(img, "Hello", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                engine.say("Hello")
                engine.runAndWait()
                spoken = True
            elif fingers_up > 0:
                spoken = False

    cv2.imshow("Sign Language to Speech", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
