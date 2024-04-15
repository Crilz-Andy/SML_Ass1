import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras import layers, models


file_path = 'domain1_train_data.json'
train_data_domain1 = pd.read_json(file_path, lines=True)
file_path = 'domain2_train_data.json'
train_data_domain2 = pd.read_json(file_path, lines=True)
file_path = 'test_data.json'
test_data = pd.read_json(file_path, lines=True)
# split into train and validation data
label_column = 'label'
train_data_domain1, validation_data_domain1 = train_test_split(train_data_domain1, test_size=0.1, random_state=42, stratify=train_data_domain1[label_column])
train_data_domain2, validation_data_domain2 = train_test_split(train_data_domain2, test_size=0.1, random_state=42, stratify=train_data_domain2[label_column])

train_domain1_y = np.array(train_data_domain1[label_column])

## extract train_domain1_x
vectorizer = CountVectorizer()

# Fit the vectorizer to the text data
vectorizer.fit(train_data_domain1['text'].apply(str))

# Transform the text data into a 2D array
train_domain1_x = vectorizer.transform(train_data_domain1['text'].apply(str)).toarray()

# Calculate the number of features
num_features = train_domain1_x.shape[1]

# use svm model to train
# Define your model using TensorFlow/Keras
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(num_features,)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_domain1_x, train_domain1_y, epochs=10, batch_size=32)
