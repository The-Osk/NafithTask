import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class TrainModel:
    """Train the model"""
    def __init__(self, batch_size=32, img_height=256, img_width=256, dataset_dir = "dataset"):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        
        self.data_dir = dataset_dir
        
    def train_the_model(self):
    
        #training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        #validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        
        
        normalization_layer = layers.Rescaling(1./255)
        
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        
        
        data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(self.img_height,
                                        self.img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
        )
        
                
        # model = Sequential([
        #     data_augmentation,
        #     layers.Rescaling(1. / 255),
        #     layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(64, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(128, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Dropout(0.2),
        #     layers.Flatten(),
        #     layers.Dense(256, activation='relu'),
        #     layers.Dense(256, activation='relu'),
        #     layers.Dense(3)
        # ])
        
        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255),
        
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
        
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
        
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
        
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
        
            layers.Flatten(),
        
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3)
        ])
        
        
        
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        
        
        epochs = 50
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
    
        
        test_loss, test_acc = model.evaluate(val_ds, verbose=2)
        print(test_acc)
        
        model.save("model_1")
