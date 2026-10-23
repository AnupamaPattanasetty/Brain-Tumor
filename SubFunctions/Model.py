import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from termcolor import cprint
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from keras.applications.vgg19 import VGG19
import lightgbm as lgb
from keras.losses import categorical_crossentropy
import tensorflow as tf
from SubFunctions.Attention import SqueezeExcitation, CBAM, EfficientChannel, AFF
from SubFunctions.ContinualLearning import Task_Incremental_Learning

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))


class Network:

    def __init__(self, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8,
                 x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8,
                 y_train, y_test, epochs):
        # Constructor to initialize class attributes.
        self.x_train1 = x_train1  # Training data
        self.x_train2 = x_train2  # Training data
        self.x_train3 = x_train3  # Training data
        self.x_train4 = x_train4  # Training data
        self.x_train5 = x_train5  # Training data
        self.x_train6 = x_train6  # Training data
        self.x_train7 = x_train7  # Training data
        self.x_train8 = x_train8  # Training data


        self.x_test1 = x_test1  # Testing data
        self.x_test2 = x_test2  # Testing data
        self.x_test3 = x_test3  # Testing data
        self.x_test4 = x_test4  # Testing data
        self.x_test5 = x_test5  # Testing data
        self.x_test6 = x_test6  # Testing data
        self.x_test7 = x_test7  # Testing data
        self.x_test8 = x_test8  # Testing data

        self.y_train = y_train  # Training labels
        self.y_test = y_test  # Testing labels
        self.epochs = epochs  # Number of training epochs
        self.batch_size = 32  # Batch size
        self.lr = 0.001  # Learning rate




    @staticmethod
    def hybrid_loss(target, prediction):
        target = tf.cast(target, np.float32)
        prediction = tf.cast(prediction, np.float32)
        # Categorical cross entropy loss
        cce_loss = tf.keras.losses.categorical_crossentropy(target, prediction)

        # Hinge categorical loss
        mae_loss = tf.keras.losses.MAE(target, prediction)

        hybrid_loss = tf.math.reduce_mean([cce_loss, mae_loss])

        return hybrid_loss


    def HFACNet(self, epochs=None):
        cprint("")
        cprint("================================", color='magenta')
        cprint("[⚠️] Distributed attentional mixed learning enabled physics based DCNN ", color='grey', on_color='on_magenta')
        cprint("================================", color='magenta')

        if epochs is None:
            epochs = self.epochs

        x_train = self.x_train8
        x_test = self.x_test8


        y_train = to_categorical(self.y_train)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():

            chanDim = -1
            input_layer = Input(shape=x_train.shape[1:])

            # ---- Block 1 ----
            x = Conv2D(8, (3, 3), padding="same")(input_layer)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
            x = Dropout(0.25)(x)

            # ---- Block 2 ----
            x = Conv2D(16, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)

            x = Conv2D(16, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)

            x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3))(x)
            x = Dropout(0.25)(x)

            # ---- Block 3 ----
            x = Conv2D(32, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)

            x = Conv2D(32, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)

            x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)

            channels = x.shape[-1]
            x1 = SqueezeExcitation(num_channels=channels)(x)
            x2 = CBAM(channels=channels)(x)
            x3 = EfficientChannel(channels=channels, k_size=3)(x)

            x11 = AFF(channels=channels)(x1, x2)
            x = AFF(channels=channels)(x11, x3)

            x = Dropout(0.25)(x)

            # ---- Dense / Classifier ----
            x = Flatten()(x)
            x = Dense(128)(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(32)(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)

            output_layer = Dense(y_train.shape[1], activation="softmax")(x)

            # Build model
            model = Model(inputs=input_layer, outputs=output_layer)

            model.summary()

            # Compile the model with Adam optimizer, hybrid loss  and accuracy
            model.compile(loss=self.hybrid_loss, optimizer=Adam(learning_rate=self.lr),
                          metrics=['accuracy'])
            model.summary()
        # from keras.utils import plot_model
        # plot_model(model, to_file='Results\\Architecture.png', show_shapes=True, show_layer_names=True, show_layer_activations=True,
        #            show_dtype=True, dpi=1200)


        # Prepare data for continual learning
        train_sets = Task_Incremental_Learning(train_data=x_train, train_labels=self.y_train).increment_data()

        # Train the model incrementally
        for train_set in range(len(train_sets)):
            cprint(f"[⚠️] Increment is {train_set} ", color='grey', on_color='on_red')
            train_data = train_sets[train_set][0]
            train_labels = train_sets[train_set][1]
            train_labels = to_categorical(train_labels)
            model.fit(train_data, train_labels, batch_size=self.batch_size, epochs=epochs, shuffle=True, verbose=1)


        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)
        return yhat

