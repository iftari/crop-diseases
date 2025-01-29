import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from joblib import dump, load
import os

# Image Processing
class Transform:
    def __init__(self, dirs: list[str]) -> None:
        self.from_path = 'data'
        self.to_path = 'processed_data'
        self.dirs = dirs
        self.cls = {dirs[i]: i for i in range(len(dirs))}
        self.images_path = [
            (f'data\\{dir}\\{img}', f'processed_data\\{img.split(".")[0]}_{self.cls[dir]}.{img.split(".")[1]}') 
            for dir in self.dirs for img in os.listdir(f'data\\{dir}')
        ]
        print(f"Total images to process: {len(self.images_path)}")
    
    def transform(self):
        for img, pimg in self.images_path:
            try:
                # Read the image in grayscale
                image = io.imread(img, as_gray=True)

                # Resize the image to (28, 28)
                resized_img = transform.resize(image, (28, 28))

                # Convert floating-point array to uint8
                pimg_array = (resized_img * 255).astype(np.uint8)

                # Save the processed image
                io.imsave(pimg, pimg_array)

            except Exception as e:
                print(f"Error processing image {img}: {e}")

        return True

# Load and preprocess images
tf_image = Transform(['Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut'])
tf_image.transform()

# Load processed images and labels
images = []
labels = []
files = os.listdir(r'processed_data')
for pimg in files:
    if pimg.endswith(('.png', '.jpg', '.jpeg')):
        # Read the image in grayscale
        image = io.imread(f'processed_data\\{pimg}', as_gray=True)
        
        # Resize the image to (28, 28) and flatten it
        resized_image = transform.resize(image, (28, 28)).ravel() / 255
        
        # Append the preprocessed image and its label
        images.append(resized_image)
        label = int(pimg.split('.')[0][-1])
        labels.append(label)

# Convert to numpy arrays
Images = np.array(images)
Labels = np.array(labels)

# Verify the shape of the data
print(Images.shape)  # Should be (n_samples, 784)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Images, Labels, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluate model
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Save model
dump(svm, 'svmClassifier.joblib')

# Load model and predict
model = load('svmClassifier.joblib')
predictions = model.predict(Images)
print(predictions)