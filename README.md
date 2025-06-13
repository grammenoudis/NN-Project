# Γεώργιος Γραμμενούδης 4632

# MNIST Digit Classification with TensorFlow

## Workflow

1. **Data Loading and Preprocessing:**
   - Loaded the MNIST dataset 
   - Visualized a random sample of 25 images to understand the data.
   - Flattened the 28x28 images into 784-dimensional vectors.
   - Converted labels to one-hot encoded vectors for multi-class classification.

2. **Model Building:**
   - Constructed a neural network with two hidden layers.
   - Added Dropout layers for regularization.
   - Used relu activation for hidden layers and softmax for output layer.

3. **Training:**
   - Compiled the model using SGD optimizer with a learning rate of 0.0009.
   - Trained the model for 15 epochs.
   - Monitored training and validation loss and accuracy over epochs.

4. **Evaluation and Visualization:**
   - Plotted the training and validation loss and accuracy curves to check model performance and to see if it is overfitting or not.

---

### Selected Hyperparameters
| Hyperparameter | Value           |
|----------------|-----------------|
| Optimizer      | SGD             |
| Learning Rate  | 0.0009          |
| Loss Function  | Categorical Crossentropy |
| Epochs         | 15              |
| Batch Size     | Default (32)    |
| Activation    | ReLU            |
| Dropout Rate   | 0.2 (Dense), 0.3 (CNN) |

---

### Improved *CNN*
  - Reshape input vectors back to 28x28 grayscale images with one channel.
  - Two convolutional layers with relu activation:
    - 1st Conv2D layer: 32 filters of size 3x3 + MaxPooling (2x2)
    - 2nd Conv2D layer: 64 filters of size 3x3 + MaxPooling (2x2)
  - Flatten feature maps before passing to dense layers.
  - Dense layer with 128 units and relu activation.
  - Dropout (0.3) for regularization.
  - Output softmax layer with 10 classes.

