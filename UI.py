"""
@author: Tomás Obregón
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import pandas as pd
import numpy as np
import cv2

import torch 
import torch.nn as nn
import torch.nn.functional as F

#%%
class Model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = torch.nn.Conv2d(in_channels=1,        # 1 color channel (Grey)
                                     out_channels=3,       # How many outputs we're gonna have
                                     kernel_size=5, stride=1, padding=0, bias=True)
        
        self.conv2 = torch.nn.Conv2d(in_channels=3,        # 3 outputs from conv1
                                     out_channels=5,       # How many outputs we're gonna have
                                     kernel_size=5, stride=1, padding=0, bias=True)
        
        self.conv3 = torch.nn.Conv2d(in_channels=5,        # 5 outputs from conv2
                                     out_channels=10,      # How many outputs we're gonna have
                                     kernel_size=5, stride=1, padding=0, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Linear layers
        self.l1 = nn.Linear(10 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(0.2)  
        self.l2       = nn.Linear(256, 128)   
        self.dropout2 = nn.Dropout(0.2)  
        self.l3 = nn.Linear(128, 10)   
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        x = self.maxpool(self.conv3(x))   
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.l1(x))
        x = self.dropout1(x)  
        x = F.relu(self.l2(x))
        x = self.dropout2(x)  
        x = F.relu(self.l3(x))

        return torch.softmax(x, dim=1)

    
model_CNN = Model_CNN()
model_CNN.load_state_dict(torch.load('https://github.com/TomasManuelObregon/digit-recognizer-app/blob/375968b031d3c2301653a1ee4a12549a2cca3958/model_parameters_conv.pth'))

#%% 

st.title("Draw a number from 0 to 9")
col1, col2 = st.columns([.4, .6])

with col1:
    canvas_result = st_canvas(
        fill_color="black",  
        background_color="black",
        stroke_color="white", 
        stroke_width=20,      
        update_streamlit=True,
        
        height=280,          
        width=280,            
        drawing_mode="freedraw", 
        key="canvas",
    )


    if canvas_result.image_data is not None:
        image_data = np.array(canvas_result.image_data).astype(np.uint8)
        image_resized = cv2.resize(image_data, (28, 28), interpolation=cv2.INTER_AREA)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2GRAY)


        st.image(image_resized, caption="28x28 pixels image", use_column_width=True)


        output = torch.tensor(image_resized, dtype=torch.float32).reshape(-1, 1, 28, 28)/255

        model_CNN.eval()
        with torch.no_grad():
            prediction = model_CNN(output)
            
            inverted_prediction = torch.flip(prediction[0], dims=[0])
            max_value, max_index = torch.max(inverted_prediction, dim=0)


with col2:
    if max_value >= .85:
        st.markdown("<h2 style='text-align: center;'>Predict:</h>", unsafe_allow_html=True)
        
        source = pd.DataFrame({
                'Number': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                'Probability': inverted_prediction})

    
        with st.container(border=True):
            st.bar_chart(source, x="Number", y="Probability",
                          horizontal=True, height=490)
            
    else: 
        st.markdown("<h1 style='text-align: center;'>Sorry...</h1>", unsafe_allow_html=True)
        warning = "The Neural Network can't figure it out. Try again!"
        st.markdown(f"<h2 style='text-align: center;'>{warning}</h2>", unsafe_allow_html=True)
