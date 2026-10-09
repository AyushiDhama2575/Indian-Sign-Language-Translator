#collect handland marks for model/camera
import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = 'data'

actions = np.array(['you','good',
    'thank you','me','my',
    'hello'
])

no_sequences = 30
sequence_length = 30


# ============================================================
# CREATE FOLDERS
# ============================================================

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(
            os.path.join(DATA_PATH, action, str(sequence)),
            exist_ok=True
        )


# ============================================================
# EXTRACT 21 HAND LANDMARKS = 63 FEATURES
# ============================================================

def extract_keypoints(result):

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        base_x = hand[0].x
        base_y = hand[0].y
        base_z = hand[0].z

        pts = []

        for lm in hand:
            pts.append(lm.x - base_x)
            pts.append(lm.y - base_y)
            pts.append(lm.z - base_z)

        return np.array(pts)

    else:
        return np.zeros(21 * 3)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(image, result):

    if not result.hand_landmarks:
        return

    h, w, _ = image.shape

    for hand_landmarks in result.hand_landmarks:

        # ----------------------------------------
        # Draw connections
        # ----------------------------------------

        for start, end in HAND_CONNECTIONS:

            x1 = int(hand_landmarks[start].x * w)
            y1 = int(hand_landmarks[start].y * h)

            x2 = int(hand_landmarks[end].x * w)
            y2 = int(hand_landmarks[end].y * h)

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # ----------------------------------------
        # Draw landmark points
        # ----------------------------------------

        for lm in hand_landmarks:

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            cv2.circle(
                image,
                (cx, cy),
                5,
                (0, 255, 0),
                -1
            )


# ============================================================
# MEDIAPIPE TASKS API
# ============================================================

base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# DATA COLLECTION
# ============================================================

with vision.HandLandmarker.create_from_options(options) as landmarker:

    for action in actions:

        for sequence in range(no_sequences):

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                if not ret:
                    print("Camera frame nahi mila.")
                    break

                # BGR -> RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Detect
                detection_result = landmarker.detect(mp_image)

                image = frame.copy()

                # ==================================================
                # DRAW CONNECTED HAND LANDMARKS
                # ==================================================

                draw_hand_landmarks(
                    image,
                    detection_result
                )

                # ==================================================
                # TEXT
                # ==================================================

                cv2.putText(
                    image,
                    f'Action: {action}',
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    f'Video: {sequence + 1}/{no_sequences}',
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    f'Frame: {frame_num + 1}/{sequence_length}',
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ==================================================
                # STARTING COLLECTION
                # ==================================================

                if frame_num == 0:

                    cv2.putText(
                        image,
                        'STARTING COLLECTION',
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )

                    cv2.imshow(
                        'OpenCV Feed',
                        image
                    )

                    cv2.waitKey(2000)

                else:

                    cv2.imshow(
                        'OpenCV Feed',
                        image
                    )

                # ==================================================
                # SAVE 63 FEATURES
                # ==================================================

                keypoints = extract_keypoints(
                    detection_result
                )

                npy_path = os.path.join(
                    DATA_PATH,
                    action,
                    str(sequence),
                    str(frame_num)
                )

                np.save(
                    npy_path,
                    keypoints
                )

                # ESC = STOP
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()


cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")