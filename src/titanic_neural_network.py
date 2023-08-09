import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from titanic_preprocessing import preprocess_titanic_data

# Load and preprocess the data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
X_train, X_test, y_train, y_test = preprocess_titanic_data(url)

# Create a neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
predictions = model.predict_classes(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", accuracy)
