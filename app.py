import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Allows cross-origin requests
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the CNN Model (should match training architecture)
class CNN(nn.Module):
    def __init__(self, input_dim=3, out_1=32, out_2=64, out_3=128, out_4=256, out_5=512):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, out_1, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.conv2 = nn.Conv2d(out_1, out_2, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.conv3 = nn.Conv2d(out_2, out_3, kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.conv4 = nn.Conv2d(out_3, out_4, kernel_size=3, padding=2)
        self.bn4 = nn.BatchNorm2d(out_4)
        self.conv5 = nn.Conv2d(out_4, out_5, kernel_size=3, padding=2)
        self.bn5 = nn.BatchNorm2d(out_5)

        self.fc1 = nn.Linear(512 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc5 = nn.Linear(4096, 2)

        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxPool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxPool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxPool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxPool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxPool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation, use softmax in inference

        return x

# Load the model
model_path = os.path.join(os.getcwd(), "model_fake.pth")  # Relative path
model = CNN()
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure this matches training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])
    return transform(image).unsqueeze(0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
        tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return jsonify({"prediction": predicted_class, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
