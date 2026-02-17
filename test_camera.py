# test_camera.py
import cv2

print("--- Starting Camera and Video File Test ---")

# --- Test 1: Try to open the video file ---
print("\nAttempting to open video file: 'test_video.mp4'...")
cap_file = cv2.VideoCapture('test_video.mp4')

if not cap_file.isOpened():
    print(">>> ERROR: Could not open the video file.")
else:
    print(">>> SUCCESS: Video file opened successfully.")
    ret, frame = cap_file.read()
    if ret:
        print(">>> SUCCESS: Successfully read a frame from the video.")
    else:
        print(">>> ERROR: Opened video file, but could not read a frame (file might be corrupted).")
    cap_file.release()

print("\n" + "="*40 + "\n")

# --- Test 2: Try to open the built-in webcam ---
print("Attempting to open built-in webcam (index 0)...")
cap_webcam = cv2.VideoCapture(0)

if not cap_webcam.isOpened():
    print(">>> ERROR: Could not open the built-in webcam (index 0).")
else:
    print(">>> SUCCESS: Built-in webcam (index 0) opened successfully.")
    cap_webcam.release()

print("\n--- Test Finished ---")