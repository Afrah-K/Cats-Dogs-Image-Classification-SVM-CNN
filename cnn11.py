#import libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Recall, Precision
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

#Directory path to dataset

test_dir = r'dataset\test_set\test_set'
train_dir = r'dataset\training_set\training_set'
validation_dir = r'dataset\validation_set\validation_set'

#Load and preprocess the Dataset (Data Augmentation)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE, 
    class_mode = 'binary',
    shuffle = True,
    seed = 42)

test_dataset = test_datagen.flow_from_directory(
    test_dir,  
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    shuffle = True,
    seed = 42)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,  
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_names = ['cats', 'dogs'],
    shuffle = True,
    seed = 42)

#Create the CNN Architecture

model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 128,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1))

#Compile the model

model.compile(
    optimizer='adam',
    loss = tf.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy', Precision(), Recall()])

#Train the model
    
model.fit(train_dataset, validation_data = validation_dataset, epochs = 10)

#Evaluate the model

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)
print("OVERALL")
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('Precision:', test_precision)
print('Recall:', test_recall)
          
          