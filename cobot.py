import cv2
import time
import torch
import os
from ultralytics import YOLO
import socket

CAMERA_INDEX = 1
MODEL_PATH = "weights/yoloe-26l-seg.pt"
SAVE_DIR = "images"

CONF_THRES = 0.05    
EDGE_MARGIN = 15        # Pixel buffer from edge (excludes partial objects)
REQUIRED_STABILITY = 4    # Number of consistent frames required before capturing

try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '10.202.3.58'
    port = 5000
    server.connect((host, port))
except Exception as e: 
    server = None

def is_box_fully_in_frame(box, frame_shape, margin):
    """
    Returns True if the bounding box is completely inside the frame 
    (not touching edges within the margin).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    
    # Check bounds
    if x1 < margin: return False         # Touching Left
    if y1 < margin: return False         # Touching Top
    if x2 > (w - margin): return False   # Touching Right
    if y2 > (h - margin): return False   # Touching Bottom
    
    return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    start = time.perf_counter()
    model = YOLO(MODEL_PATH).to(device)
    end = time.perf_counter()
    
    print(f"loadin time {end - start}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    start = time.perf_counter()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    end = time.perf_counter()
    
    print(f"start time {end - start}")

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    # QUERY LOOP
    while True:
        try:
            target_class = input("\nEnter object class (or press q to exit): ").strip()
        except EOFError:
            break
            
        if not target_class or target_class.lower() == 'q':
            break
        
        if server:
            server.sendall(b"s")

        model.set_classes([target_class])
        print(f"Detecting class: {target_class}")

        detected = False
        stability_counter = 0  # Counter for temporal stability

        # INFERENCE LOOP 
        while not detected:
            start_loop = time.time()
            ret, frame = cap.read()
            
            if not ret:
                continue

            results = model.predict(
                frame,
                conf=CONF_THRES,
                task="segment",
                verbose=False
            )[0]

            #  Filter Detections
            valid_box = None
            
            if results.boxes is not None and len(results.boxes) > 0:
                # Find the best box that is fully in frame
                for i, box in enumerate(results.boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    if is_box_fully_in_frame(coords, frame.shape, EDGE_MARGIN):
                        valid_box = results[i] # This is a valid complete object
                        final_coords = coords  
                        break 

            # Stability Logic
            annotated = frame.copy()
            
            if valid_box:
                stability_counter += 1
    
                annotated = valid_box.plot()
                
                if stability_counter >= REQUIRED_STABILITY:
                    detected = True # Trigger exit
                
            else:
                # Object lost or touching edge -> Reset counter
                if stability_counter > 0:
                    print(f"Stability Lost (Count was {stability_counter})")
                stability_counter = 0
                cv2.putText(annotated, "Searching...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow("Live Feed", annotated)
            
            #  Save Logic (Only if detected flag is set by stability logic)
            if detected:
                timestamp = int(time.time())
                filename = f"{SAVE_DIR}/{target_class.replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Saved detection: {filename}")
                print("Coordinates : ", final_coords.tolist())
                cv2.waitKey(500) 

            latency_ms = (time.time() - start_loop) * 1000
            cv2.setWindowTitle(
                "Live Feed",
                f"Live Feed | Latency: {latency_ms:.1f} ms"
            )
            # Exit CURRENT inference only
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(" Stopping current inference.")
                break

        # destroy windows after each query 
        cv2.destroyAllWindows()

        # clear CUDA cache after each query 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if server:
            server.sendall(b'f')

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if server:
        server.close()

if __name__ == "__main__":
    main()
