import cv2
import time
import torch
import os
import socket
import re
from ultralytics import YOLO
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# CONFIGURATION
CAMERA_INDEX = 1
MODEL_PATH = "weights/yoloe-26l-seg.pt"
SAVE_DIR = "images"

# Detection Constants
CONF_THRES = 0.05    
EDGE_MARGIN = 15          # Pixel buffer from edge
REQUIRED_STABILITY = 4    # Frames required for stability

# LLM Constants
LLM_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
WEIGHTS_DIR = "weights"
LLM_LOCAL_PATH = os.path.join(WEIGHTS_DIR, LLM_FILE)

# SOCKET SETUP 
# try:
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     host = '10.202.3.58'
#     port = 5000
#     server.connect((host, port))
# except Exception as e: 
#     print(f"Socket connection failed (ignoring): {e}")
#     server = None


class CommandParser:
    def __init__(self):
        print(f"Loading Llama 3.2-3B-Instruct...")
        
        # Download if not exists
        if not os.path.exists(LLM_LOCAL_PATH):
            print(f"Downloading {LLM_FILE} to '{WEIGHTS_DIR}/'...")
            hf_hub_download(
                repo_id=LLM_REPO, 
                filename=LLM_FILE, 
                local_dir=WEIGHTS_DIR, 
                local_dir_use_symlinks=False 
            )
        else:
            print(f"Found local model at {LLM_LOCAL_PATH}")

        # Initialize LLM
        self.llm = Llama(
            model_path=LLM_LOCAL_PATH, 
            n_gpu_layers=-1, # Offload all to GPU if available
            n_ctx=2048,      
            verbose=False
        )

    def parse_command(self, user_text):
        """Extracts the target object from a text command."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robotic command parser.\n"
                    "Extract the TARGET OBJECT NAME from the user command.\n\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Output ONLY the object noun phrase. No verbs, no articles.\n"
                    "2. Preserve adjectives only if explicitly spoken.\n"
                    "3. Remove action words (pick up, grab, take, etc.).\n"
                    "4. Output must be lowercase.\n"
                    "5. If no object mentioned, output: NONE"
                )
            },
            {
                "role": "user",
                "content": f"User Command: {user_text}"
            }
        ]

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=60,      
            temperature=0.0     
        )
        
        parsed_object = output['choices'][0]['message']['content'].strip().lower()
        
        # Cleanup punctuation and specific phrases
        parsed_object = re.sub(r'[^\w\s]', '', parsed_object) 
        if "pick up" in parsed_object: parsed_object = parsed_object.replace("pick up", "").strip()
        
        if parsed_object == 'none' or not parsed_object:
            return None
        
        return parsed_object


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
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Initialize LLM Parser
    parser = CommandParser()

    start = time.perf_counter()
    model = YOLO(MODEL_PATH).to(device)
    end = time.perf_counter()
    print(f"YOLO loading time {end - start:.4f}s")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    
    # QUERY LOOP
    while True:
        try:
            # Get text input from the user
            user_input = input("\nEnter command or 'q' to exit: ").strip()
        except EOFError:
            break
            
        if not user_input or user_input.lower() == 'q':
            break
        
        # Parse the input using LLM
        print("Parsing command...")
        target_class = parser.parse_command(user_input)
        
        if not target_class:
            print("Could not understand object name. Try again.")
            continue
            
        print(f"parsed target: '{target_class}'")
        
        # if server:
        #     try:
        #         server.sendall(b"s")
        #     except:
        #         pass

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
            final_coords = None
            
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
                cv2.putText(annotated, f"Scanning for {target_class}...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow("Live Feed", annotated)
            
            #  Save Logic (Only if detected flag is set by stability logic)
            if detected:
                timestamp = int(time.time())
                filename = f"{SAVE_DIR}/{target_class.replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Saved detection: {filename}")
                if final_coords is not None:
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

        # if server:
        #     try:
        #         server.sendall(b'f')
        #     except:
        #         pass

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    # if server:
    #     server.close()

if __name__ == "__main__":
    main()
