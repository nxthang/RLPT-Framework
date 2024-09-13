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

# Configuration
class CFG:
    seed = 42
    preset = "deberta_v3_large_en" # name of pretrained backbone
    train_seq_len = 1024 # max size of input sequence for training
    train_batch_size = 2 * 8 # size of the input batch in training, x 2 as two GPUs
    infer_seq_len = 2000 # max size of input sequence for inference
    infer_batch_size = 2 * 2 # size of the input batch in inference, x 2 as two GPUs
    epochs = 6 # number of epochs to train
    lr_mode = "exp" # lr scheduler mode from one of "cos", "step", "exp"
    
    labels = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
              "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
              "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
              "I-STREET_ADDRESS","I-URL_PERSONAL","O"]
    id2label = dict(enumerate(labels)) # integer label to BIO format label mapping
    label2id = {v:k for k,v in id2label.items()} # BIO format label to integer label mapping
    num_labels = len(labels) # number of PII (NER) tags
    
    train = True # whether to train or use already trained model

keras.utils.set_random_seed(CFG.seed)

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
    labels[i] = np.array([CFG.label2id[label] for label in x["labels"]])

# Splitting the data into training and testing sets
train_words, valid_words, train_labels, valid_labels = train_test_split(
    words, labels, test_size=0.2, random_state=CFG.seed
)

# To convert string input or list of strings input to numerical tokens
tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
    CFG.preset,
)

# Preprocessing layer to add spetical tokens: [CLS], [SEP], [PAD]
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    sequence_length=10,
)
