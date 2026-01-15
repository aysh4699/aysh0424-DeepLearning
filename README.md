import tensorflow as tf #for building and training deep learning models
import os #Provides functions for interacting with the operating system (paths, directories).
import zipfile #Used to extract downloaded ZIP datasets.
import matplotlib.pyplot as plt #Used to plot training and validation accuracy graphs.
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Loads images from directories and applies preprocessing and augmentation.
from tensorflow.keras.applications import VGG16 #Imports the pre-trained VGG16 model.
from tensorflow.keras import layers, models, optimizers #Provides tools to build, compile, and optimize neural networks

# Step 1: Download and extract the dataset automatically
!wget --no-check-certificate https://storage.googleapis.com/download.tensorflow.org/data/rps.zip -O /tmp/rps.zip #download the rps training dataset
!wget --no-check-certificate https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip -O /tmp/rps-test-set.zip #download the rps test set

# Extract training set
with zipfile.ZipFile('/tmp/rps.zip', 'r') as zip_ref: #Extracts the training images into the /tmp/rps directory
    zip_ref.extractall('/tmp/')

# Extract test set
with zipfile.ZipFile('/tmp/rps-test-set.zip', 'r') as zip_ref: #Extracts the test images into the /tmp/rps directory
    zip_ref.extractall('/tmp/')

# Dataset paths (after extraction: /tmp/rps/rock, /tmp/rps/paper, etc.)
train_dir = '/tmp/rps' #Specifies the directory containing training images
test_dir = '/tmp/rps-test-set' #Specifies the directory containing test images

# Step 2: Data augmentation and generators
IMG_SIZE = (224, 224)  # VGG16 expects 224x224 and Resizes images to 224×224 pixels as required by VGG16.
BATCH_SIZE = 32 #Processes 32 images at a time during training

 #Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalizes pixel values to the range [0, 1].
    rotation_range=40, #Randomly rotates images to improve robustness
    width_shift_range=0.2,
    height_shift_range=0.2, # randomly shifts images horizontally and vertically.
    shear_range=0.2, #applies shearing transformations
    zoom_range=0.2, #randomly zoom images in and out
    horizontal_flip=True, #randomly flips image horizontally
    fill_mode='nearest', #fills empty pixels created by transformations
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow_from_directory( #loads training images from directory
    train_dir,
    target_size=IMG_SIZE, #resize all images 224x224
    batch_size=BATCH_SIZE, #loads images in batch of 32
    class_mode='categorical', #uses one hot encoding for class labels
    subset='training' #specifies training subset
)

val_generator = train_datagen.flow_from_directory( #loads validation images from the same directory
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' #specifies validation set
)

test_datagen = ImageDataGenerator(rescale=1./255) #normalizes test images without augmentation
test_generator = test_datagen.flow_from_directory( #loads test images
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False #disables shuffling to preserve label order during evaluation
)

# Load VGG16 pre-trained model (feature extraction)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base and Freezes all VGG16 layers (feature extraction).

# Build the model
model = models.Sequential([ #creates a sequential neural network
    base_model, #Adds the pre-trained VGG16 model
    layers.GlobalAveragePooling2D(), #reduces spatial dimensions and prevents overfitting
    layers.Dropout(0.5), #Randomly disables neurons to improve generalization
    layers.Dense(512, activation='relu'), #Adds a fully connected layer for learning task-specific features
    layers.Dropout(0.5),#further reduces overfitting
    layers.Dense(3, activation='softmax')  # 3 classes: rock, paper, scissors
])

model.summary()

# Compile
model.compile(optimizer='adam', #use adam optimizer for efficient learning
              loss='categorical_crossentropy', #use loss functionfor multiclass classification
              metrics=['accuracy'])

# Step 4: Train with feature extraction (10 epochs)
history_feat = model.fit( #starts training the model
    train_generator,
    epochs=10, #trains 10 epochs
    validation_data=val_generator #Validates model performance after each epoch
)

# Step 5: Fine-tuning (unfreeze top layers of VGG16)
base_model.trainable = True

# Freeze all layers except the last few (fine-tune only top layers)
for layer in base_model.layers[:-20]:  # Freezes lower layers and fine-tunes only the top 20 layers. (VGG16 model)
    layer.trainable = False

# Re-compile with lower learning rate
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), #uses a lower learning rate to avoid large weight updates
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train more (fine-tuning) (Fine tuning training)
history_fine = model.fit(
    train_generator,
    epochs=20,  # More epochs for fine-tuning
    validation_data=val_generator
)

# Step 6: Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator) #evaluates model performance on unseen test data
print(f"\nTest Accuracy: {test_acc * 100:.2f}%") #print the final test accuracy

# Optional: Plot training history
acc = history_feat.history['accuracy'] + history_fine.history['accuracy'] #combines accuracy from both training phases
val_acc = history_feat.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.plot(acc, label='Training Accuracy') #plots training accuracy
plt.plot(val_acc, label='Validation Accuracy') #plots validation accuracy
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
