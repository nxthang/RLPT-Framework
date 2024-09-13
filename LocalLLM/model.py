# Import libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_nlp
from keras import ops
import tensorflow as tf

import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import plotly.graph_objs as go
import plotly.express as px
import util 


keras.utils.set_random_seed(util.CFG.seed)

# Get devices default "gpu" or "tpu"
devices = keras.distribution.list_devices()
print("Device:", devices)

if len(devices) > 1:
    # Data parallelism
    data_parallel = keras.distribution.DataParallel(devices=devices)

    # Set the global distribution.
    keras.distribution.set_distribution(data_parallel)

keras.mixed_precision.set_global_policy("mixed_float16")

BASE_PATH = "/input/data"

# Train-Valid data
data = json.load(open(f"{BASE_PATH}/train.json"))

# Initialize empty arrays
words = np.empty(len(data), dtype=object)
labels = np.empty(len(data), dtype=object)

# Fill the arrays
for i, x in tqdm(enumerate(data), total=len(data)):
    words[i] = np.array(x["tokens"])
    labels[i] = np.array([util.CFG.label2id[label] for label in x["labels"]])

# Splitting the data into training and testing sets
train_words, valid_words, train_labels, valid_labels = train_test_split(
    words, labels, test_size=0.2, random_state=util.CFG.seed
)

# To convert string input or list of strings input to numerical tokens
tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
    util.CFG.preset,
)

# Preprocessing layer to add spetical tokens: [CLS], [SEP], [PAD]
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    sequence_length=10,
)

# Build Train & Valid Dataloader
train_ds = util.build_dataset(train_words, train_labels,  batch_size=util.CFG.train_batch_size,
                         seq_len=util.CFG.train_seq_len, shuffle=True)

valid_ds = util.build_dataset(valid_words, valid_labels, batch_size=util.CFG.train_batch_size, 
                         seq_len=util.CFG.train_seq_len, shuffle=False)


# Modeling
# Build Token Classification model
backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
    util.CFG.preset,
)
out = backbone.output
out = keras.layers.Dense(util.CFG.num_labels, name="logits")(out)
out = keras.layers.Activation("softmax", dtype="float32", name="prediction")(out)
model = keras.models.Model(backbone.input, out)

# Compile model for optimizer, loss and metric
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=util.CrossEntropy(),
    metrics=[util.FBetaScore()],
)

# Summary of the model architecture
model.summary()

