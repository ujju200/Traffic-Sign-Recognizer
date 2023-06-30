# Traffic-Sign-Recognizer
The Traffic Sign Recognizer project is a powerful application developed using convolutional neural networks (CNNs), Python, and TensorFlow. Its main objective is to detect and classify traffic signs from images accurately. The project utilizes a Kaggle dataset to train the CNN model, ensuring a robust and accurate recognition system.

# Tech Stack

1. **CNNs(Convolutional Neural Networks)** - The CNN architecture leverages the power of deep learning to automatically learn and extract relevant features from the input images, making it highly accurate in identifying traffic signs. Through extensive training, the model has acquired the ability to distinguish between signs, including speed limits, yield signs, and stop signs.
   
2. **Tkinter** - To enhance user experience and provide a user-friendly interface, the project also incorporates the Tkinter library to create a graphical user interface (GUI). This GUI allows users to interact with the application by selecting an image file and receiving the recognition results in a visually appealing manner.
   
3. **Tensorflow** - It is a Python library for making and training different types of neural networks like Convolutional Neural networks.

# Working

1. Dataset Acquisition: The project begins by obtaining a dataset of labeled traffic sign images from Kaggle or a similar source. This dataset contains a wide range of traffic signs, each associated with a specific class or label.

2. Data Preprocessing: The acquired dataset is preprocessed to ensure uniformity and consistency. This step may involve resizing the images to a consistent size, normalizing pixel values, and augmenting the dataset by applying rotation, flipping, and scaling transformations. Data preprocessing helps improve the performance and generalization of the model.

3. Model Training: A convolutional neural network (CNN) model is trained using the preprocessed dataset. The CNN architecture typically consists of multiple convolutional layers and pooling layers to extract relevant features from the input images. Additional fully connected layers and activation functions are incorporated to enable classification. The model is trained using a suitable optimization algorithm (e.g., stochastic gradient descent) and a loss function (e.g., categorical cross-entropy). The goal is to minimize the loss and optimize the model's ability to classify traffic signs accurately.

4. Model Evaluation: Once the model training is complete, it is evaluated using a separate validation dataset. The evaluation helps assess the model's performance metrics, such as accuracy, precision, recall, and F1 score. It ensures the trained model can accurately classify traffic signs and generalize well to unseen data.

5. Command Line App Development: The project is implemented using Python as a command line application. The application prompts the user to provide an image file containing a traffic sign for recognition. The user inputs the file path, and the application loads the image.

6. Image Processing: The loaded image is preprocessed to match the input format expected by the trained CNN model. This preprocessing may involve resizing the image to a specific size and normalizing the pixel values.

7. Traffic Sign Recognition: The preprocessed image is fed into the trained CNN model, which performs forward propagation and predicts the class or label of the traffic sign in the image. The model outputs a probability distribution over the different classes, and the class with the highest probability is selected as the predicted traffic sign.

8. Result Presentation: The application displays the predicted traffic sign class to the user through the GUI made by Tkinter, providing information about the recognized traffic sign. The user can then take appropriate actions based on the recognized sign, such as adjusting driving behavior or following traffic rules.


## Here is a Demo Video of its working

https://github.com/ujju200/Traffic-Sign-Recognizer/assets/129884469/58aa70b4-0d24-43f1-a471-54dba6996ff4





