import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import csv

# Paths
frame_folder = "E:/accident2/testimagesweather"  # Ensure this is a directory
model_path = "E:/accident2/weathermodel.pth"
csv_output = "E:/accident2/weather_conditions1.csv"

# Weather categories - updated to only include the 4 required classes
weather_labels = ["cloudy", "foggy", "rainy", "sunny","snowy"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(weather_labels))

# Load the saved model
try:
    checkpoint = torch.load(model_path, map_location=device)
    # Check if the loaded file is a state_dict or a checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # If the checkpoint contains classes, verify they match
        if 'classes' in checkpoint:
            saved_classes = checkpoint['classes']
            if set(saved_classes) != set(weather_labels):
                print(f"Warning: Saved classes {saved_classes} don't match expected classes {weather_labels}")
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict weather conditions
def predict_weather(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        prediction_index = predicted.item()
        
        # Ensure the prediction index is valid
        if 0 <= prediction_index < len(weather_labels):
            return weather_labels[prediction_index]
        else:
            print(f"Warning: Invalid prediction index {prediction_index} for {image_path}")
            return "unknown"
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "error"

# Process frames and store predictions
try:
    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Path", "Weather Condition"])
        
        if os.path.isdir(frame_folder):  # If it's a directory, process all images
            image_count = 0
            for frame in sorted(os.listdir(frame_folder)):
                frame_path = os.path.join(frame_folder, frame)
                if os.path.isfile(frame_path) and frame.lower().endswith(('.png', '.jpg', '.jpeg')):
                    condition = predict_weather(frame_path)
                    writer.writerow([frame_path, condition])
                    image_count += 1
            
            print(f"Processed {image_count} images from directory {frame_folder}")
        
        elif os.path.isfile(frame_folder):  # If it's a single image file, process it
            condition = predict_weather(frame_folder)
            writer.writerow([frame_folder, condition])
            print(f"Processed single image {frame_folder}")
        
        else:
            print("Error: The specified frame path is neither a file nor a directory.")

    print(f"Predictions saved to {csv_output}")

except Exception as e:
    print(f"An error occurred: {str(e)}")