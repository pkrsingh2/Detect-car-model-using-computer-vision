# Detect-car-model-using-computer-vision
# Summary: 
 This project is aimed at creating multiclass object detection model for car
detection and classification. The Stanford car dataset is used for the model
building. Full lifecycle of model building including image sampling and
augmentation, model selection, training, testing and validation is performed by
using a preliminary model. We find that the preliminary model has much higher
training accuracy than testing and would need drastic performance
improvements. Spaces of improvement are identified for improving performance
of model with setting up of pipeline to test multiple algorithms efficiently. 

**Technologies Used:**
* Python
* Streamlit
* Docker

# Dataset:
The car detection model will be prepared using The Stanford Cars dataset, which is developed by Stanford University AI Lab specifically to create models for differentiating car types from each other.
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

**Data description:**
Train Images: Consists of real images of cars as per the make and year of the car.
Test Images: Consists of real images of cars as per the make and year of the car.
Train Annotation: Consists of bounding box region for training images.
Test Annotation: Consists of bounding box region for testing images.

**CAR MODEL CLASSIFIER APP**
https://carmodelclassifier.herokuapp.com/
