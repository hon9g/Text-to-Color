'''
Provides functionality for converting a given list of tokens (words) into
numbers, according to the given vocabulary.
'''
from __future__ import print_function, division

import numpy as np
from deepmoji.word_generator import WordGenerator
from deepmoji.global_variables import SPECIAL_TOKENS
from copy import deepcopy


class SentenceTokenizer():
    """ Create numpy array of tokens corresponding to input sentences.
        The vocabulary can include Unicode tokens.
    """

    def __init__(self, vocabulary, fixed_length, custom_wordgen=None,
                 ignore_sentences_with_only_custom=False, masking_value=0,
                 unknown_value=1):
        """ Needs a dictionary as input for the vocabulary.
        """

        if len(vocabulary) > np.iinfo('uint16').max:
            raise ValueError('Dictionary is too big ({} tokens) for the numpy '
                             'datatypes used (max limit={}). Reduce vocabulary'
                             ' or adjust code accordingly!'
                             .format(len(vocabulary), np.iinfo('uint16').max))

        # Shouldn't be able to modify the given vocabulary
        self.vocabulary = deepcopy(vocabulary)
        self.fixed_length = fixed_length
        self.ignore_sentences_with_only_custom = ignore_sentences_with_only_custom
        self.masking_value = masking_value
        self.unknown_value = unknown_value

        # Initialized with an empty stream of sentences that must then be fed
        # to the generator at a later point for reusability.
        # A custom word generator can be used for domain-specific filtering etc
        if custom_wordgen is not None:
            assert custom_wordgen.stream is None
            self.wordgen = custom_wordgen
            self.uses_custom_wordgen = True
        else:
            self.wordgen = WordGenerator(None, allow_unicode_text=True,
                                         ignore_emojis=False,
                                         remove_variation_selectors=True,
                                         break_replacement=True)
            self.uses_custom_wordgen = False

    def tokenize_sentences(self, sentences, reset_stats=True, max_sentences=None):
        """ Converts a given list of sentences into a numpy array according to
            its vocabulary.

        # Arguments:
            sentences: List of sentences to be tokenized.
            reset_stats: Whether the word generator's stats should be reset.
            max_sentences: Maximum length of sentences. Must be set if the
                length cannot be inferred from the input.

        # Returns:
            Numpy array of the tokenization sentences with masking,
            infos,
            stats

        # Raises:
            ValueError: When maximum length is not set and cannot be inferred.
        """

        if max_sentences is None and not hasattr(sentences, '__len__'):
            raise ValueError('Either you must provide an array with a length'
                             'attribute (e.g. a list) or specify the maximum '
                             'length yourself using `max_sentences`!')
        n_sentences = (max_sentences if max_sentences is not None
                       else len(sentences))

        if self.masking_value == 0:
            tokens = np.zeros((n_sentences, self.fixed_length), dtype='uint16')
        else:
            tokens = (np.ones((n_sentences, self.fixed_length), dtype='uint16') *
                      self.masking_value)

        if reset_stats:
            self.wordgen.reset_stats()

        # With a custom word generator info can be extracted from each
        # sentence (e.g. labels)
        infos = []

        # Returns words as strings and then map them to vocabulary
        self.wordgen.stream = sentences
        next_insert = 0
        n_ignored_unknowns = 0
        for s_words, s_info in self.wordgen:
            s_tokens = self.find_tokens(s_words)

            if (self.ignore_sentences_with_only_custom and
                np.all([True if t < len(SPECIAL_TOKENS)
                        else False for t in s_tokens])):
                n_ignored_unknowns += 1
                continue
            if len(s_tokens) > self.fixed_length:
                s_tokens = s_tokens[:self.fixed_length]
            tokens[next_insert, :len(s_tokens)] = s_tokens
            infos.append(s_info)
            next_insert += 1

        # For standard word generators all sentences should be tokenized
        # this is not necessarily the case for custom wordgenerators as they
        # may filter the sentences etc.
        if not self.uses_custom_wordgen and not self.ignore_sentences_with_only_custom:
            assert len(sentences) == next_insert
        else:
            # adjust based on actual tokens received
            tokens = tokens[:next_insert]
            infos = infos[:next_insert]
        return tokens, infos, self.wordgen.stats

    def find_tokens(self, words):
        assert len(words) > 0
        tokens = []
        for w in words:
            try:
                tokens.append(self.vocabulary[w])
            except KeyError:
                tokens.append(self.unknown_value)
        return tokens