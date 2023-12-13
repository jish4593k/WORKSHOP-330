import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow import keras
import numpy as np

# Blueprint of the Person
class Person:
    def __init__(self, attractive):
        self.attractive = attractive

# Neural Network using PyTorch and Keras
class AttractivenessClassifier(nn.Module):
    def __init__(self):
        super(AttractivenessClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

p1 = Person(attractive=False)
p2 = Person(attractive=True)
p3 = Person(attractive=True)

queue = [p1, p2, p3]

X_train = torch.tensor([[int(p.attractive)] for p in queue], dtype=torch.float32)
y_train = torch.tensor([[1.0] if p.attractive else [0.0] for p in queue], dtype=torch.float32)

model = AttractivenessClassifier()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

keras_model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])
keras_model.compile(optimizer='adam', loss='binary_crossentropy')


class GUI(tk.Tk):
    def __init__(self):
        super(GUI, self).__init__()

        self.title("Attractiveness Classifier")
        self.geometry("300x150")

        self.label = tk.Label(self, text="Is the person attractive?")
        self.label.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_attractiveness)
        self.predict_button.pack(pady=10)

    def predict_attractiveness(self):
        user_input = messagebox.askyesno("Input", "Is the person attractive?")
        input_tensor = torch.tensor([[int(user_input)]], dtype=torch.float32)

        
        with torch.no_grad():
            pytorch_prediction = model(input_tensor).item()

        # Keras Model Prediction
        keras_prediction = keras_model.predict(np.array([[int(user_input)]]))[0][0]

    
        self.result_label.config(text=f"PyTorch Prediction: {pytorch_prediction:.4f}\nKeras Prediction: {keras_prediction:.4f}")

gui = GUI()
gui.mainloop()
