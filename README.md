This guide will walk you through the simple steps needed to build and train an Image Processing model using TensorFlow with GPU on Rising Cloud. 

1. Install the Rising Cloud Command Line Interface (CLI)
In order to run the Rising Cloud commands in this guide, you will need to install the Rising Cloud Command Line Interface. This program provides you with the utilities to setup your Rising Cloud Task or Web Service, upload your application to Rising Cloud, setup authentication, and more.

2. Login to Rising Cloud Using the CLI
Using a command line console (called terminal on Mac OS X and command prompt on Windows) run the Rising Cloud login command. The interface will request your Rising Cloud email address and password.

risingcloud login

3. Initialize Your Rising Cloud Task
Create a new directory on your workstation to place your project files in, then open this directory with your command line.

Using the command line in your project directory, run the following command replacing $GPU_TYPE with your preference of GPU server and $TASK_URL with your unique task name. Your unique task name must be at least 12 characters long and consist of only alphanumeric characters and hyphens (-). This task name is unique to all tasks on Rising Cloud. A unique URL will be provided to you for sending jobs to your task. If a task name is not available, the CLI will return with an error so you can try again.

risingcloud init -s --gpu $GPU_TYPE $TASK_URL

Use the following command to get a list of currently available GPU’s

risingcloud lsrr

This creates a risingcloud.yaml file in your project directory. This file will be used to configure your build script.

4. Create your Rising Cloud Task
Configuring your I/O

When a Rising Cloud Job is run, input is written to request.json in the top level of your project directory. Your application will need to read this to respond to it. When your application terminates, output is read from response.json, if it exists, and is stored in Rising Cloud’s Job Results database for retrieval.

Input to Rising Cloud Tasks has to come a JSON. If you are planning on using pdfs, images, or other non-JSONable data as input to your neural net, you will have to use the input JSON to give your application URLs to download the data from. Likewise, if the output of your application is an image, you will need your application to store the application in a database and return information on how to retrieve it in the output (such as a URL to an Amazon S3 object.) See our Statelessness guide to understand why this is, and for information about connecting to external data sources.

Create Your TensorFlow Program

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Grab GPUs that TF found
gpus = tf.config.list_physical_devices('GPU')

if len(gpus) < 1:
    print("WARNING: No GPUs found by Tensorflow")
else:
    print("Succesfully registered GPUs with Tensorflow")

# Download and prep data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
    label_mode='fine'
)
x_train = x_train / 255
x_test = x_test / 255
y_train = tf.keras.utils.to_categorical(
    y_train, num_classes=None, dtype='float32'
)
y_test = tf.keras.utils.to_categorical(
    y_test, num_classes=None, dtype='float32'
)

# Construct dummy model
input_tensor = tf.keras.layers.Input(shape=(32,32,3))
x = MobileNetV2(include_top=False,
                  weights=None,
                  classes=100)(input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(100, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_tensor], outputs=[preds])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Configure data generator and train model
epochs = 10
batch_size = 32

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=5. / 32,
    height_shift_range=5. / 32,
    horizontal_flip=True
)
generator.fit(x_train)
model.fit(generator.flow(x_train, y_train),
                             validation_data=(x_test, y_test),
                             steps_per_epoch=(len(x_train) // batch_size),
                             epochs=epochs, verbose=1,)

Configure your risingcloud.yaml

When you ran risingcloud init, a new risingcloud.yaml file should have generated in your project directory. Open that file now in your editor.  Change the from Base Image and Deps to the following:

from: tensorflow/tensorflow:latest-gpu
deps:
  - pip3 install -r requirements.txt
We need to tell Rising Cloud what to run when a new request comes in, and set a time to 10min (600,000 ms). Change run and timeout to:

run: python3 train.py
timeout: 600000

5. Build and Deploy your Rising Cloud Task
Use the push command to push your updated risingcloud.yaml to your Task on Rising Cloud.

risingcloud push
Use the build command to zip, upload, and build your app on Rising Cloud.

risingcloud build
Use the deploy command to deploy your app as soon as the build is complete. Change $TASK to your unique task name.

risingcloud deploy $TASK
Alternatively, you could also use a combination to push, build and deploy all at once.

risingcloud build -r -d
Rising Cloud will now build out the infrastructure necessary to run and scale your application including networking, load balancing and DNS. Allow DNS a few minutes to propogate and then your app will be ready and available to use!

6. Queue Jobs for your Rising Cloud Task
Make requests

Rising Cloud will take some time to build and deploy your Rising Cloud Task. Once it is done, you can make HTTPS POST requests with JSON bodies to https://{your project URL}.risingcloud.app/risingcloud/jobs to queue jobs for Rising Cloud Task. These requests will return JSON responses with a “jobId” field containing the ID of your job. Make an HTTP GET request to https://{your project URL}.risingcloud.app/risingcloud/jobs/{job ID} in order to check on the status of the job.

Because we are training a model, we will make a blank JSON request with no body:

{}

Congratulations, you’ve successfully trained an Image Classifier TensorFlow and GPU on Rising Cloud!
