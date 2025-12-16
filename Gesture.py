import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- CANVAS ---------------- #
canvas = None
prev_x, prev_y = 0, 0

# ---------------- COLORS ---------------- #
colors = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 255, 255),   # Yellow
    (0, 0, 0)        # Eraser
]
color_names = ["PURPLE", "BLUE", "GREEN", "YELLOW", "ERASER"]
current_color = colors[0]

# ---------------- UI SETTINGS ---------------- #
PALETTE_ALPHA = 0.5

# ---------------- FPS ---------------- #
start_time = time.time()
frame_count = 0
fps = 0

# ---------------- FINGER DETECTION ---------------- #
def fingers_up(hand):
    index_up = hand.landmark[8].y < hand.landmark[6].y
    middle_up = hand.landmark[12].y < hand.landmark[10].y
    return index_up, middle_up

# ---------------- DRAW TRANSPARENT PALETTE ---------------- #
def draw_palette(img):
    overlay = img.copy()
    h, w, _ = img.shape
    box_w = w // len(colors)

    cv2.rectangle(overlay, (0, 0), (w, 60), (40, 40, 40), -1)

    for i, col in enumerate(colors):
        x1 = i * box_w
        x2 = (i + 1) * box_w
        cv2.rectangle(overlay, (x1, 0), (x2, 60), col, -1)
        cv2.putText(
            overlay, color_names[i], (x1 + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    cv2.addWeighted(overlay, PALETTE_ALPHA, img, 1 - PALETTE_ALPHA, 0, img)

# ---------------- MAIN LOOP ---------------- #
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "NONE"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index_up, middle_up = fingers_up(hand)

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # -------- COLOR SELECTION -------- #
            if index_up and middle_up:
                mode = "SELECT"
                prev_x, prev_y = 0, 0

                if y < 60:
                    box_w = w // len(colors)
                    idx = x // box_w
                    if idx < len(colors):
                        current_color = colors[idx]

                cv2.circle(frame, (x, y), 15, current_color, cv2.FILLED)

            # -------- DRAW / ERASE -------- #
            elif index_up and not middle_up:
                mode = "DRAW"
                cv2.circle(frame, (x, y), 10, current_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                thickness = 40 if current_color == (0, 0, 0) else 8
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # -------- MERGE CANVAS -------- #
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # -------- BRUSH INDICATOR -------- #
    thickness = 40 if current_color == (0, 0, 0) else 8
    cv2.circle(frame, (w - 50, 100), thickness, current_color, cv2.FILLED)
    cv2.putText(frame, "BRUSH", (w - 90, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # -------- FPS COUNTER -------- #
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (w - 140, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # -------- UI TEXT -------- #
    cv2.putText(frame, f"Mode: {mode}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(
        frame,
        "Index: Draw | Index+Middle: Select | C: Clear | S: Save | Q: Quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
    )

    cv2.imshow("Real-Time Hand Gesture–Controlled Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('s'):
        cv2.imwrite("real_time_hand_gesture_controlled_canvas.png", canvas)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
