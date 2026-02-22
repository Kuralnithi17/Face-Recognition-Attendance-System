import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime, date

IMAGES_PATH = "ImagesAttendance"
ATTENDANCE_FILE = "Attendance.csv"
WEBCAM_INDEX = 0


def face_confidence(distance, threshold=0.6):
    """
    Convert face distance (0 = perfect match) into a confidence percentage.
    threshold=0.6 is common for face_recognition.

    Note: This is an approximate "confidence-style" score for display.
    """
    if distance is None:
        return 0

    if distance > threshold:
        # For weak matches, keep it simple
        return int(max(0, (1.0 - distance) * 100))

    # For good matches, use a non-linear scaling so it looks nicer (higher %)
    range_val = (1.0 - threshold)
    linear_val = (1.0 - distance) / (range_val * 2.0)
    confidence = linear_val + ((1.0 - linear_val) * ((linear_val - 0.5) * 2) ** 0.2)
    return int(confidence * 100)


def load_known_faces(images_path: str):
    encodings = []
    names = []

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Folder not found: {images_path}")

    files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise ValueError(f"No images found in '{images_path}'. Add .jpg/.png face images.")

    for f in files:
        name = os.path.splitext(f)[0]
        path = os.path.join(images_path, f)

        img = face_recognition.load_image_file(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb)
        if len(locs) == 0:
            print(f"[SKIP] No face found in {f}")
            continue

        # Take the first face found
        enc = face_recognition.face_encodings(rgb, locs)[0]
        encodings.append(enc)
        names.append(name)
        print(f"[LOADED] {name}")

    if not encodings:
        raise ValueError("No valid faces loaded. Use clearer single-face images.")
    return encodings, names


def mark_attendance(name: str):
    today = str(date.today())
    now = datetime.now().strftime("%H:%M:%S")

    # Create CSV if not exists
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
            f.write("Name,Date,Time\n")

    # Read existing rows and prevent duplicate for same day
    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            n, d = parts[0], parts[1]
            if n == name and d == today:
                return

    with open(ATTENDANCE_FILE, "a", encoding="utf-8") as f:
        f.write(f"{name},{today},{now}\n")
    print(f"[MARKED] {name} {today} {now}")


def main():
    known_enc, known_names = load_known_faces(IMAGES_PATH)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Webcam not opening. Try setting WEBCAM_INDEX = 1.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Resize for speed
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)

        for (top, right, bottom, left), enc in zip(locs, encs):
            matches = face_recognition.compare_faces(known_enc, enc, tolerance=0.5)
            dists = face_recognition.face_distance(known_enc, enc)

            name = "Unknown"
            label = "Unknown"

            if len(dists) > 0:
                best = np.argmin(dists)
                best_dist = dists[best]

                if matches[best]:
                    name = known_names[best]
                    conf = face_confidence(best_dist)
                    label = f"{name} ({conf}%)"
                    mark_attendance(name)

            # Scale back up to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Green for known, Red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()