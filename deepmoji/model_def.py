""" Model definition functions and weight loading.
"""

from __future__ import print_function, division

from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2
from deepmoji.attlayer import AttentionWeightedAverage
from deepmoji.global_variables import NB_TOKENS, NB_EMOJI_CLASSES


def deepmoji_emojis(maxlen, weight_path, return_attention=False):
    """ Loads the pretrained DeepMoji model for extracting features
        from the penultimate feature layer. In this way, it transforms
        the text into its emotional encoding.

    # Arguments:
        maxlen: Maximum length of a sentence (given in tokens).
        weight_path: Path to model weights to be loaded.
        return_attention: If true, output will be weight of each input token
            used for the prediction

    # Returns:
        Pretrained model for encoding text into feature vectors.
    """

    model = deepmoji_architecture(nb_classes=NB_EMOJI_CLASSES,
                                  nb_tokens=NB_TOKENS, maxlen=maxlen,
                                  return_attention=return_attention)
    model.load_weights(weight_path, by_name=False)
    return model


def deepmoji_architecture(nb_classes, nb_tokens, maxlen, feature_output=False, embed_dropout_rate=0, final_dropout_rate=0, embed_l2=1E-6, return_attention=False):
    """
    Returns the DeepMoji architecture uninitialized and
    without using the pretrained model weights.

    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        feature_output: If True the model returns the penultimate
                        feature vector rather than Softmax probabilities
                        (defaults to False).
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.

    # Returns:
        Model with the given parameters.
    """
    # define embedding layer that turns word tokens into vectors
    # an activation function is used to bound the values of the embedding
    model_input = Input(shape=(maxlen,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    embed = Embedding(input_dim=nb_tokens,
                      output_dim=256,
                      mask_zero=True,
                      input_length=maxlen,
                      embeddings_regularizer=embed_reg,
                      name='embedding')
    x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])

    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
    weights = None
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
    if return_attention:
        x, weights = x

    if not feature_output:
        # output class probabilities
        if final_dropout_rate != 0:
            x = Dropout(final_dropout_rate)(x)

        if nb_classes > 2:
            outputs = [Dense(nb_classes, activation='softmax', name='softmax')(x)]
        else:
            outputs = [Dense(1, activation='sigmoid', name='softmax')(x)]
    else:
        # output penultimate feature vector
        outputs = [x]

    if return_attention:
        # add the attention weights to the outputs if required
        outputs.append(weights)

    return Model(inputs=[model_input], outputs=outputs, name="DeepMoji")




