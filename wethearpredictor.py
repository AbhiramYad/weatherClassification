import cv2
import os
import csv
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image

# Paths
frame_folder = "E:/accident2/weather_data"
csv_output = "E:/accident2/weather_conditions.csv"
model_save_path = "E:/accident2/weathermodel.pth"

# Define your four weather classes
weather_classes = ["cloudy", "foggy", "rainy", "sunny"]

def create_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Expected structure: data_dir/class_name/image_files
    # Verify that each class folder exists
    for class_name in weather_classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Created missing class directory: {class_path}")
    
    # Check if the directory has images in the class subdirectories
    for class_name in weather_classes:
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"Warning: No images found in {class_path}")
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Verify that classes match
    if set(dataset.classes) != set(weather_classes):
        print(f"Warning: Dataset classes {dataset.classes} don't match expected classes {weather_classes}")
    
    return dataloader, dataset.classes

# Create a function to predict and save weather conditions
def predict_weather(model, frame_folder, csv_output, classes):
    # Set model to evaluation mode
    model.eval()
    
    # Setup transformation for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if frame_folder contains individual images or class subdirectories
    is_dataset_format = any(os.path.isdir(os.path.join(frame_folder, d)) for d in os.listdir(frame_folder))
    
    # Prepare CSV file
    with open(csv_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Weather Condition'])
        
        # If it's not in dataset format, predict on individual images
        if not is_dataset_format:
            # Process each image in the folder
            for filename in os.listdir(frame_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(frame_folder, filename)
                    
                    # Open and transform image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        _, predicted = torch.max(outputs, 1)
                        weather_condition = classes[predicted.item()]
                    
                    # Write to CSV
                    writer.writerow([filename, weather_condition])
        else:
            # Process images in class subdirectories
            for class_dir in os.listdir(frame_folder):
                class_path = os.path.join(frame_folder, class_dir)
                if os.path.isdir(class_path):
                    for filename in os.listdir(class_path):
                        if filename.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_path, filename)
                            
                            # Open and transform image
                            image = Image.open(img_path).convert('RGB')
                            image_tensor = transform(image).unsqueeze(0).to(device)
                            
                            # Make prediction
                            with torch.no_grad():
                                outputs = model(image_tensor)
                                _, predicted = torch.max(outputs, 1)
                                weather_condition = classes[predicted.item()]
                            
                            # Write to CSV
                            writer.writerow([f"{class_dir}/{filename}", weather_condition])
    
    print(f"Weather predictions saved to {csv_output}")

try:
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Creating dataloaders...")
    dataloader, dataset_classes = create_dataloaders(frame_folder)
    print(f"Found {len(dataset_classes)} classes: {dataset_classes}")
    
    # Define model
    print("Setting up the model...")
    model = models.mobilenet_v2(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(dataset_classes))
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 5
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': dataset_classes
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Use the model to predict weather conditions for frames
    print("Generating weather predictions...")
    predict_weather(model, frame_folder, csv_output, dataset_classes)
    
except Exception as e:
    print(f"An error occurred: {str(e)}")