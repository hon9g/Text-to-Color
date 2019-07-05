from __future__ import print_function, division
import codecs
from emoji import UNICODE_EMOJI


def read_english(path="english_words.txt", add_emojis=True):
    # read english words for filtering (includes emojis as part of set)
    english = set()
    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            line = line.strip().lower().replace('\n', '')
            if len(line):
                english.add(line)
    if add_emojis:
        for e in UNICODE_EMOJI:
            english.add(e)
    return english