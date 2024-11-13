import numpy as np
import pandas as pd
import cv2
import pytesseract
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Load and preprocess dataset
df = pd.read_csv('mail_data.csv')
data = df.where((pd.notnull(df)), '')

# Encode 'spam' as 0 and 'ham' as 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X = data['Message']
Y = data['Category']

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction with TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Check accuracy on training and test data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data: ', accuracy_on_test_data)

# Function to extract text from an image using pytesseract
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img)
    
    return text

# Now you can pass in the image for classification
def classify_image(image_path):
    # Step 1: Extract text from the image
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text: ", extracted_text)
    
    # Step 2: Use the text for classification
    input_your_mail = [extracted_text]  # Prepare it as a list
    
    # Step 3: Transform the text into features using the vectorizer
    input_data_features = feature_extraction.transform(input_your_mail)
    
    # Step 4: Make a prediction
    prediction = model.predict(input_data_features)
    
    # Step 5: Output result
    if prediction[0] == 1:
        print('Ham mail')
    else:
        print('Spam mail')

# Example usage
image_path = '/image.png'  # Replace with your image path
classify_image(image_path)
