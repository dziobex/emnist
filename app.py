import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from models import CNNBlock, SimpleCNN, ResidualBlock, ResNet

class DrawingApp:
    def __init__(self, root, cnn_path, resnet_path, canvas_size=(280, 280), img_size=(28, 28)):
        self.root = root
        self.canvas_size = canvas_size
        self.img_size = img_size
        self.image = Image.new("L", self.canvas_size)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1], bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_image)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_position)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()

        self.result_label_cnn = tk.Label(self.root, text="CNN: ", font=("Helvetica", 18))
        self.result_label_cnn.pack()

        self.result_label_resnet = tk.Label(self.root, text="ResNet: ", font=("Helvetica", 18))
        self.result_label_resnet.pack()

        # init both models and load the state dictionaries
        self.model_cnn = SimpleCNN()
        self.model_cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
        self.model_cnn.eval()

        self.model_resnet = ResNet(ResidualBlock, layers=[1, 1, 1, 1], num_classes=47)
        self.model_resnet.load_state_dict(torch.load(resnet_path, weights_only=True))
        self.model_resnet.eval()

        # load the EMNIST mapping
        mapping = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None)
        self.mapping = {mapping.iloc[i, 0]: chr(mapping.iloc[i, 1]) for i in range(len(mapping))}
        
        self.softmax = nn.Softmax(dim=1)

        self.last_x, self.last_y = None, None

    def draw_image(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=10, capstyle=tk.ROUND,
                                    smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=10)
        self.last_x, self.last_y = x, y

    def reset_last_position(self, event):
        self.last_x, self.last_y = None, None

    def get_image_data(self):
        img_resized = self.image.resize(self.img_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_array = img_array.reshape(1, 1, 28, 28)
        return img_array

    def predict_digit(self):
        img_data = self.get_image_data()
        img_tensor = torch.tensor(img_data, dtype=torch.float32)

        with torch.no_grad():
            cnn_output = self.model_cnn(img_tensor)
            resnet_output = self.model_resnet(img_tensor)
            
            cnn_probs = self.softmax(cnn_output)
            resnet_probs = self.softmax(resnet_output)

            cnn_conf, cnn_pred = torch.max(cnn_probs, 1)
            resnet_conf, resnet_pred = torch.max(resnet_probs, 1)

        cnn_char = self.mapping[cnn_pred.item()]
        resnet_char = self.mapping[resnet_pred.item()]
        
        self.result_label_cnn.config(
            text=f"CNN: {cnn_char} ({cnn_conf.item():.2%})"
        )
        self.result_label_resnet.config(
            text=f"ResNet: {resnet_char} ({resnet_conf.item():.2%})"
        )

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", self.canvas_size)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label_cnn.config(text="CNN: ")
        self.result_label_resnet.config(text="ResNet: ")

root = tk.Tk()
root.title("Digit Recognizer with CNN and ResNet")

cnn_path = 'model.pth'
resnet_path = 'rn_model.pth'

app = DrawingApp(root, cnn_path=cnn_path, resnet_path=resnet_path)
root.mainloop()
