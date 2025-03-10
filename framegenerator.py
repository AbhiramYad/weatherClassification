import cv2
import os
import numpy as np
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import time
from scipy.stats import mode
import math

# Paths
video_path = r"E:\accident\accident_vedios\accident_vedios\videoplayback_1.mp4"
frames_folder = r"E:\accident\framesgenerated"
processed_folder = r"E:\accident\weather_classified_frames"

# Create folders for different weather conditions
weather_classes = ["sunny", "rainy", "cloudy", "foggy", "night"]
for weather in weather_classes:
    os.makedirs(os.path.join(processed_folder, weather), exist_ok=True)

# Make sure frames folder exists
os.makedirs(frames_folder, exist_ok=True)

# Enhanced model loading with ensemble approach
def load_weather_model():
    try:
        # Create an ensemble of models for better prediction
        models_list = []
        
        # Model 1: ResNet50
        model1 = models.resnet50(pretrained=True)
        num_ftrs1 = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs1, len(weather_classes))
        models_list.append(('resnet50', model1))
        
        # Model 2: EfficientNet
        try:
            model2 = models.efficientnet_b0(pretrained=True)
            num_ftrs2 = model2.classifier[1].in_features
            model2.classifier[1] = nn.Linear(num_ftrs2, len(weather_classes))
            models_list.append(('efficientnet', model2))
        except:
            print("EfficientNet not available, skipping in ensemble")
        
        # Model 3: MobileNetV3
        try:
            model3 = models.mobilenet_v3_large(pretrained=True)
            num_ftrs3 = model3.classifier[3].in_features
            model3.classifier[3] = nn.Linear(num_ftrs3, len(weather_classes))
            models_list.append(('mobilenet', model3))
        except:
            print("MobileNetV3 not available, skipping in ensemble")
        
        # Load saved models if available
        model_path = r"E:\accident\weathermodel.pth"
        if os.path.exists(model_path):
            try:
                # Try to load for the first model
                models_list[0][1].load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded trained model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}, using untrained model: {e}")
        
        # Set all models to evaluation mode
        for _, model in models_list:
            model.eval()
        
        return models_list
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Advanced rain detection using frequency domain analysis
def detect_rain(frame, threshold=0.02):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate FFT
    f_transform = np.fft.fft2(blurred)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Analyze high frequency components (rain creates more high frequency components)
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Define high frequency region (outer region of the spectrum)
    mask = np.ones((h, w), np.uint8)
    center_radius = min(h, w) // 4
    cv2.circle(mask, (center_w, center_h), center_radius, 0, -1)
    
    # Calculate ratio of high frequency to total frequency energy
    high_freq_energy = np.sum(magnitude_spectrum * mask)
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # Also check for vertical lines (rain streaks)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
    
    vertical_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of the line
            if x2 - x1 != 0:  # Avoid division by zero
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                # Count near-vertical lines (rain streaks)
                if 70 < angle < 110:
                    vertical_lines += 1
    
    vertical_line_density = vertical_lines / (h * w) if h * w > 0 else 0
    
    # Combined rain detection
    return high_freq_ratio > threshold or vertical_line_density > 0.0001

# Improved night detection using histogram analysis
def is_night(frame, threshold=70):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Calculate percentage of dark pixels
    dark_pixels = np.sum(hist[:threshold]) / np.sum(hist) * 100
    
    # Check if there are light sources (potential street lamps or car headlights)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    light_sources = np.sum(binary > 0) / (gray.shape[0] * gray.shape[1]) * 100
    
    # Combined night detection
    return dark_pixels > 75 or (dark_pixels > 60 and light_sources < 5)

# Improved fog detection using depth discontinuity
def detect_fog(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate local contrast
    kernel_size = 15
    local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_var = cv2.GaussianBlur(gray * gray, (kernel_size, kernel_size), 0) - local_mean * local_mean
    
    # Low contrast is characteristic of fog
    low_contrast_ratio = np.sum(local_var < 100) / (gray.shape[0] * gray.shape[1])
    
    # Also check for edge visibility
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Check for whitish color in upper part of the image (sky region)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sky_region = hsv[0:int(hsv.shape[0]/3), :, :]
    sky_saturation = np.mean(sky_region[:,:,1])
    sky_brightness = np.mean(sky_region[:,:,2])
    
    # Combined fog detection
    return (low_contrast_ratio > 0.8 and edge_density < 0.05) or (sky_saturation < 30 and sky_brightness > 150)

# Comprehensive weather classification
def classify_weather_comprehensive(frame):
    # Check for night first
    if is_night(frame):
        return "night"
    
    # Check for rain
    if detect_rain(frame):
        return "rainy"
    
    # Check for fog
    if detect_fog(frame):
        return "foggy"
    
    # Convert to different color spaces for further analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Image statistics
    brightness = np.mean(hsv[:,:,2])
    saturation = np.mean(hsv[:,:,1])
    color_variance = np.var(lab[:,:,1]) + np.var(lab[:,:,2])  # a* and b* channels in LAB
    
    # Sky analysis
    sky_region = hsv[0:int(hsv.shape[0]/3), :, :]
    sky_brightness = np.mean(sky_region[:,:,2])
    sky_saturation = np.mean(sky_region[:,:,1])
    sky_hue = np.mean(sky_region[:,:,0])
    sky_hue_variance = np.var(sky_region[:,:,0])
    
    # Sunny detection
    is_sunny = (sky_brightness > 160 and sky_saturation < 50 and brightness > 150) or \
               (sky_hue > 20 and sky_hue < 40 and sky_brightness > 150)  # Yellowish sky
    
    # Cloudy detection
    is_cloudy = (sky_brightness > 100 and sky_saturation < 40 and sky_hue_variance < 100) or \
                (sky_brightness < 160 and sky_brightness > 100 and sky_saturation < 60)
    
    # Make decision based on combined factors
    if is_sunny:
        return "sunny"
    elif is_cloudy:
        return "cloudy"
    elif brightness > 140:  # Bright but not matching specific patterns
        return "sunny"
    else:
        return "cloudy"  # Default case

# Function to get model predictions
def get_model_predictions(models_list, frame, transform, device):
    if not models_list:
        return None
    
    # Convert OpenCV BGR to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Process for model
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Get predictions from each model
    predictions = []
    with torch.no_grad():
        for _, model in models_list:
            model.to(device)
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    
    # Return the most common prediction (majority voting)
    if predictions:
        return mode(predictions, keepdims=True).mode[0]

    else:
        return None

# Extract frames and organize them into class folders
def extract_and_classify_frames():
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened successfully. FPS: {fps}, Estimated frames: {frame_count_estimate}")
    
    # Try to load models for classification
    models_list = load_weather_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Extract frames
    frame_count = 0
    start_time = time.time()
    
    # We'll use a more sophisticated temporal smoothing with weighted history
    class_history = {cls: 0 for cls in weather_classes}
    decay_factor = 0.7  # Decay factor for history
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 5 == 0:  # Save every 5th frame
            # Save the original frame
            frame_filename = os.path.join(frames_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Classify the frame
            # Try model prediction first
            weather_class_idx = get_model_predictions(models_list, frame, transform, device) if models_list else None
            
            if weather_class_idx is not None:
                weather = weather_classes[weather_class_idx]
                confidence = 0.7  # Higher confidence for model prediction
            else:
                # Fall back to comprehensive classification
                weather = classify_weather_comprehensive(frame)
                confidence = 0.5  # Lower confidence for rule-based classification
            
            # Update class history with decay
            for cls in class_history:
                class_history[cls] *= decay_factor
            class_history[weather] += confidence
            
            # Get the most likely class from history
            weather = max(class_history, key=class_history.get)
            
            # Copy to appropriate weather folder
            destination = os.path.join(processed_folder, weather, f"frame_{frame_count:04d}.jpg")
            shutil.copy(frame_filename, destination)
            
            if frame_count % 50 == 0:
                elapsed_time = time.time() - start_time
                frames_processed = frame_count // 5 + 1
                fps_processing = frames_processed / elapsed_time if elapsed_time > 0 else 0
                print(f"Processed frame {frame_count} as {weather} ({fps_processing:.2f} fps)")
                print(f"Class probabilities: {class_history}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count//5} frames (every 5th frame) to {frames_folder}")
    print(f"Classified frames are organized in {processed_folder}")
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")

# Run the extraction and classification
if __name__ == "__main__":
    extract_and_classify_frames()