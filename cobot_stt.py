import cv2
import time
import torch
import os
import socket
import numpy as np
import sounddevice as sd
import re
from ultralytics import YOLO
from faster_whisper import WhisperModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


CAMERA_INDEX = 1
MODEL_PATH = "weights/yoloe-26l-seg.pt"

# LLM CONFIGURATION (3B Model) 
LLM_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
WEIGHTS_DIR = "weights"
LLM_LOCAL_PATH = os.path.join(WEIGHTS_DIR, LLM_FILE)

SAVE_DIR = "output_images"
CONF_THRES = 0.04    
EDGE_MARGIN = 15        
REQUIRED_STABILITY = 4    

#  AUDIO CONFIG 
SAMPLE_RATE = 16000
DURATION = 3.0  
WHISPER_SIZE = "distil-large-v3" 

# SOCKET SETUP 
# try:
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     host = '10.202.3.58'
#     port = 5000
#     server.settimeout(2) 
#     server.connect((host, port))
# except Exception as e: 
#     server = None


class VoiceAgent:
    def __init__(self):
        
        # 1. Load Whisper (STT)
        print(f"1. Loading Whisper ({WHISPER_SIZE})")
        self.stt_model = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
        
        # 2. Load LLM (Parsing)
        print(f"2. Loading llama 3.2-3B-instruct")
        
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

        self.llm = Llama(
            model_path=LLM_LOCAL_PATH, 
            n_gpu_layers=-1, 
            n_ctx=2048,      
            verbose=False
        )

    def listen_and_parse(self):
        """Records audio, cleans it, transcribes, and parses."""
        # A. Record Audio
        print(f"LISTENING ({DURATION}s)... ")
        try:
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() 
            print("Processing...")
            
            # Normalize Audio (Volume Boost)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
                
        except Exception as e:
            print(f"Microphone Error: {e}")
            return None

        # B. Transcribe (STT)
        # Domain-biased transcription for robotic object commands
        # Initial prompt anchors command structure; hotwords stabilize object nouns
        segments, _ = self.stt_model.transcribe(
           audio_data[:, 0],
           beam_size=5,
           vad_filter=True,
           vad_parameters=dict(min_silence_duration_ms=500),
           initial_prompt=(
           "Pick up the matchbox. "
           "Pick up the doll. "
           "Pick up the spoon. "
           "Pick up the steel glass. "
           "Pick up the green mirror. "
           "Pick up the plastic bottle."
            ),
           hotwords=(
           "matchbox, doll, spoon, steel glass, "
           "green mirror, plastic bottle"
           )
        )

        text = "".join([segment.text for segment in segments]).strip()
        print(f"Heard: '{text}'")

        if not text:
            return None

        # C. Parse (LLM)
        messages = [
       {
        "role": "system",
        "content": (
            "You are a robotic command parser used in a real-world manipulation system.\n\n"
            "Your task is to extract the TARGET OBJECT NAME from a spoken user command.\n\n"
            "CRITICAL CONSTRAINTS (must follow exactly):\n"
            "1. Output ONLY the object noun phrase. No verbs. No articles. No explanations."
            "   - If a word can be both an action and a tangible object, treat it as an object ONLY when it clearly refers to a physical item. \n"
            "2. Do NOT invent, infer, or hallucinate attributes.\n"
            "   - If the user says \"doll\", output \"doll\".\n"
            "   - Do NOT add words like \"steel\", \"plastic\", \"small\", etc unless explicitly spoken.\n"
            "3. Preserve adjectives ONLY if they are explicitly present in the command.\n"
            "   - \"green mirror\" -> \"green mirror\"\n"
            "   - \"mirror\" -> \"mirror\"\n"
            "4. Correct common phonetic / ASR errors for physical objects.\n"
            "   Examples:\n"
            "   - \"matt's box\" -> \"matchbox\"\n"
            "   - \"maths box\" -> \"matchbox\"\n"
            "   - \"fabi call\" -> \"fevicol\"\n"
            "   - \"still glass\" -> \"steel glass\"\n"
            "5. If a concrete noun phrase referring to a physical object exists anywhere in the command, ALWAYS output that noun phrase, even if action words appear earlier or the grammar is malformed.\n"
            "6. Remove all action words, filler words, and function words that are NOT part of the object noun phrase.Examples to remove: pick, pick up, grab, take, give, bring, the, a, an, please."
            "7. Output must be a lowercase noun phrase suitable for direct use as a vision model class name.\n"
            "8. If no clear physical object is mentioned, output exactly: NONE\n\n"
            "OUTPUT FORMAT:\n"
            "- Plain text only\n"
            "- No punctuation\n"
            "- No quotes\n"
            "- No extra whitespace"
         )
       },
      {
        "role": "user",
        "content": f"User Command: {text}"
      }
    ]

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=60,      
            temperature=0.0     # ZERO temp = Maximum Logic/Strictness
        )
        
        parsed_object = output['choices'][0]['message']['content'].strip().lower()
        
        # Cleanup regex
        parsed_object = re.sub(r'[^\w\s]', '', parsed_object) # Remove punctuation
        if "pick up" in parsed_object: parsed_object = parsed_object.replace("pick up", "").strip()
        
        print(f"Parsed Target: '{parsed_object}'")
        
        if parsed_object == 'none' or not parsed_object:
            return None
        
        return parsed_object

def is_box_fully_in_frame(box, frame_shape, margin):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    if x1 < margin: return False        
    if y1 < margin: return False        
    if x2 > (w - margin): return False  
    if y2 > (h - margin): return False  
    return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True) 

    agent = VoiceAgent()

    start = time.perf_counter()
    model = YOLO(MODEL_PATH).to(device)
    end = time.perf_counter()
    print(f"YOLO load time {end - start:.4f}s")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    while True:
        try:
            print("\nPress 'ENTER' to speak a command, or 'q' to quit.")
            key_input = input() 
            if key_input.lower() == 'q':
                break
            
            target_class = agent.listen_and_parse()
            
            if not target_class:
                print("No valid object found. Please try again.")
                continue

        except KeyboardInterrupt:
            break
        
        # if server:
        #     try:
        #         server.sendall(b"s")
        #     except:
        #         pass

        model.set_classes([target_class])
        print(f"SEARCHING FOR: {target_class}")

        detected = False
        stability_counter = 0  

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

            valid_box = None
            final_coords = None
            
            if results.boxes is not None and len(results.boxes) > 0:
                for i, box in enumerate(results.boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    if is_box_fully_in_frame(coords, frame.shape, EDGE_MARGIN):
                        valid_box = results[i] 
                        final_coords = coords  
                        break 

            annotated = frame.copy()
            
            if valid_box:
                stability_counter += 1
                annotated = valid_box.plot()

                if stability_counter >= REQUIRED_STABILITY:
                    detected = True 
                
            else:
                if stability_counter > 0:
                    print(f"Stability Lost (Count was {stability_counter})")
                stability_counter = 0
                cv2.putText(annotated, f"Scanning for {target_class}...", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Live Feed", annotated)
            
            if detected:
                timestamp = int(time.time())
                filename = f"{SAVE_DIR}/{target_class.replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"CAPTURED: {filename}")
                
                if final_coords is not None:
                     print("Coordinates:", final_coords.tolist())
                
                cv2.waitKey(1000) 

            latency_ms = (time.time() - start_loop) * 1000
            cv2.setWindowTitle("Live Feed", f"Live Feed | Latency: {latency_ms:.1f} ms")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(" Stopping current search.")
                break
            
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # if server:
        #     try:
        #         server.sendall(b'f')
        #     except:
        #         pass

    cap.release()
    cv2.destroyAllWindows()
    # if server:
    #     server.close()

if __name__ == "__main__":
    main()
