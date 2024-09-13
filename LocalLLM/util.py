# Data Processing
def get_tokens(words, seq_len, packer):
    # Tokenize input
    token_words = tf.expand_dims(
        tokenizer(words), axis=-1
    )  # ex: (words) ["It's", "a", "cat"] ->  (token_words) [[1, 2], [3], [4]]
    tokens = tf.reshape(
        token_words, [-1]
    )  # ex: (token_words) [[1, 2], [3], [4]] -> (tokens) [1, 2, 3, 4]
    # Pad tokens
    tokens = packer(tokens)[0][:seq_len]
    inputs = {"token_ids": tokens, "padding_mask": tokens != 0}
    return inputs, tokens, token_words


def get_token_ids(token_words):
    # Get word indices
    word_ids = tf.range(tf.shape(token_words)[0])
    # Get size of each word
    word_size = tf.reshape(tf.map_fn(lambda word: tf.shape(word)[0:1], token_words), [-1])
    # Repeat word_id with size of word to get token_id
    token_ids = tf.repeat(word_ids, word_size)
    return token_ids


def get_token_labels(word_labels, token_ids, seq_len):
    # Create token_labels from word_labels ->  alignment
    token_labels = tf.gather(word_labels, token_ids)
    # Only label the first token of a given word and assign -100 to others
    mask = tf.concat([[True], token_ids[1:] != token_ids[:-1]], axis=0)
    token_labels = tf.where(mask, token_labels, -100)
    # Truncate to max sequence length
    token_labels = token_labels[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_labels to align with tokens (use -100 to pad for loss/metric ignore)
    pad_start = 1  # for [CLS] token
    pad_end = seq_len - tf.shape(token_labels)[0] - 1  # for [SEP] and [PAD] tokens
    token_labels = tf.pad(token_labels, [[pad_start, pad_end]], constant_values=-100)
    return token_labels


def process_token_ids(token_ids, seq_len):
    # Truncate to max sequence length
    token_ids = token_ids[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_ids to align with tokens (use -1 to pad for later identification)
    pad_start = 1  # [CLS] token
    pad_end = seq_len - tf.shape(token_ids)[0] - 1  # [SEP] and [PAD] tokens
    token_ids = tf.pad(token_ids, [[pad_start, pad_end]], constant_values=-1)
    return token_ids


def process_data(seq_len=720, has_label=True, return_ids=False):
    # To add spetical tokens: [CLS], [SEP], [PAD]
    packer = keras_nlp.layers.MultiSegmentPacker(
        start_value=tokenizer.cls_token_id,
        end_value=tokenizer.sep_token_id,
        sequence_length=seq_len,
    )

    def process(x):
        # Generate inputs from tokens
        inputs, tokens, words_int = get_tokens(x["words"], seq_len, packer)
        # Generate token_ids for maping tokens to words
        token_ids = get_token_ids(words_int)
        if has_label:
            # Generate token_labels from word_labels
            token_labels = get_token_labels(x["labels"], token_ids, seq_len)
            return inputs, token_labels
        elif return_ids:
            # Pad token_ids to align with tokens
            token_ids = process_token_ids(token_ids, seq_len)
            return token_ids
        else:
            return inputs

    return process
