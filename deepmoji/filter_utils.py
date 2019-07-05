
from __future__ import print_function, division
import sys
import re
import string
import emoji
from deepmoji.tokenizer import RE_MENTION, RE_URL
from deepmoji.global_variables import SPECIAL_TOKENS
from itertools import groupby

AtMentionRegex = re.compile(RE_MENTION)
urlRegex = re.compile(RE_URL)

# from http://bit.ly/2rdjgjE (UTF-8 encodings and Unicode chars)
VARIATION_SELECTORS = [u'\ufe00',
                       u'\ufe01',
                       u'\ufe02',
                       u'\ufe03',
                       u'\ufe04',
                       u'\ufe05',
                       u'\ufe06',
                       u'\ufe07',
                       u'\ufe08',
                       u'\ufe09',
                       u'\ufe0a',
                       u'\ufe0b',
                       u'\ufe0c',
                       u'\ufe0d',
                       u'\ufe0e',
                       u'\ufe0f']

# from https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
ALL_CHARS = (chr(i) for i in range(sys.maxunicode))
CONTROL_CHARS = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
CONTROL_CHAR_REGEX = re.compile('[%s]' % re.escape(CONTROL_CHARS))


def is_special_token(word):
    equal = False
    for spec in SPECIAL_TOKENS:
        if word == spec:
            equal = True
            break
    return equal


def punct_word(word, punctuation=string.punctuation):
    return all([True if c in punctuation else False for c in word])


def separate_emojis_and_text(text):
    emoji_chars = []
    non_emoji_chars = []
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            emoji_chars.append(c)
        else:
            non_emoji_chars.append(c)
    return ''.join(emoji_chars), ''.join(non_emoji_chars)


def remove_variation_selectors(text):
    """ Remove styling glyph variants for Unicode characters.
        For instance, remove skin color from emojis.
    """
    for var in VARIATION_SELECTORS:
        text = text.replace(var, u'')
    return text


def shorten_word(word):
    """ Shorten groupings of 3+ identical consecutive chars to 2, e.g. '!!!!' --> '!!'
    """

    # only shorten ASCII words
    try:
        word.encode().decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        return word

    # must have at least 3 char to be shortened
    if len(word) < 3:
        return word

    # find groups of 3+ consecutive letters
    letter_groups = [list(g) for k, g in groupby(word)]
    triple_or_more = [''.join(g) for g in letter_groups if len(g) >= 3]
    if len(triple_or_more) == 0:
        return word

    # replace letters to find the short word
    short_word = word
    for trip in triple_or_more:
        short_word = short_word.replace(trip, trip[0] * 2)

    return short_word


def detect_special_tokens(word):
    try:
        int(word)
        word = SPECIAL_TOKENS[4]
    except ValueError:
        if AtMentionRegex.findall(word):
            word = SPECIAL_TOKENS[2]
        elif urlRegex.findall(word):
            word = SPECIAL_TOKENS[3]
    return word


def process_word(word):
    """ Shortening and converting the word to a special token if relevant.
    """
    word = shorten_word(word)
    word = detect_special_tokens(word)
    return word


def convert_linebreaks(text):
    # ugly hack handling non-breaking space no matter how badly it's been encoded in the input
    # space around to ensure proper tokenization
    for r in [u'\\\\n', u'\\n', u'\n', u'\\\\r', u'\\r', u'\r', '<br>']:
        text = text.replace(r, u' ' + SPECIAL_TOKENS[5] + u' ')
    return text
