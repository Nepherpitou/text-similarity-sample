import math
import string
from typing import List, Iterable

import nltk
import num2words
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

NGRAM_SIZE = 3


def highlight_punctuation(text: str) -> str:
    for char in string.punctuation:
        text = text.replace(char, f' {char} ')
    return text


def convert_numbers(words: List[str]) -> List[str]:
    result = []
    for word in words:
        if word.isnumeric():
            result.append(num2words.num2words(word))
        else:
            result.append(word)
    return result


def is_punctuation(word: str) -> bool:
    for char in string.punctuation:
        if word == char:
            return True
    return False


def cleanup_punctuation(words: List[str]) -> List[str]:
    return [w for w in words if w.isalpha()]


def cleanup_stopwords(words: List[str]) -> List[str]:
    return [w for w in words if w not in stopwords.words('english')]


def stem_words(words: List[str], lemmatizer: WordNetLemmatizer) -> List[str]:
    return [lemmatizer.lemmatize(w) for w in words]


def relative_distance(word_1: str, word_2: str) -> float:
    return nltk.masi_distance(set(word_1), set(word_2))


def best_distance(word: str, words: Iterable[str]) -> float:
    distances = [relative_distance(word, w) for w in words]
    return max(distances)


original = "TransPerfect improves chatbot performance up to 30% compared to a translation approach. We’re able to " \
           "remove bias and incorporate diversity and nuance for target markets. Let’s have a call with a Solution " \
           "Engineer, who can better walk through our process with you. Is Tuesday morning or afternoon better for " \
           "you? "
recognized = "Transparent effect improves childhood performance up to 30% compared to a translation approach. Were " \
             "able to remove barriers and incorporate diversity and nuance for target markets. Let's say we're " \
             "solution Engineer who can better walk through a process with you. Is choose a morning or afternoon. " \
             "Better for you. "

original = 'up to 3 hello my name hello world asdads a sf asfga sdafdfgas'
recognized = 'up to 3 up to 3'

original_cleaned = highlight_punctuation(original.lower())
recognized_cleaned = highlight_punctuation(recognized.lower())

lemmatizer = WordNetLemmatizer()

original_words = stem_words(
    words=cleanup_punctuation(convert_numbers(nltk.word_tokenize(original_cleaned))),
    lemmatizer=lemmatizer)
recognized_words = stem_words(
    words=cleanup_punctuation(convert_numbers(nltk.word_tokenize(recognized_cleaned))),
    lemmatizer=lemmatizer)

original_ngrams = list(nltk.ngrams(original_words, NGRAM_SIZE))
recognized_ngrams = list(nltk.ngrams(recognized_words, NGRAM_SIZE))

print(original_ngrams)
print(recognized_ngrams)


def similarity_local(xs: List[str], ys: List[str]) -> float:
    if len(xs) != len(ys):
        return 0
    distances = [1 - nltk.jaccard_distance(set(x), set(y)) for x, y in zip(xs, ys)]
    return np.average(distances)
    # , weights=[len(x) for x in xs]) TODO: Investigate why result aren't isomorphic with weights


matches = []
pool = recognized_ngrams.copy()
for i in range(len(original_ngrams)):
    orig = original_ngrams[i]
    similarities = [(similarity_local(orig, recog), recog) for recog in pool]
    if len(similarities) > 0:
        best_c, best_ng = max(similarities, key=lambda x: x[0])
        matches.append((best_c, orig, best_ng))
        pool.remove(best_ng)
        print(f'N-gram: {orig}; Similar: {(best_c, best_ng)}; Pool: {pool}')

# for ot in original_ngrams:
#     similarities = [(similarity_local(ot, rt), rt) for rt in recognized_ngrams]
#     best = max(similarities, key=lambda x: x[0])
#     print(f'N-gram: {ot}; Similar: {best}')
#     matches.append((best[0], ot, best[1]))

print('Matches:', matches)

CONTEXT_PRE = math.ceil(NGRAM_SIZE / 2)
CONTEXT_POST = NGRAM_SIZE - CONTEXT_PRE
coeffs = [(mc, mo) for mc, mo, *_ in matches]
(e_mc, e_mo) = coeffs[0]
extension = []
for i in range(CONTEXT_PRE):
    prepend = tuple('' for _ in range(CONTEXT_PRE - i))
    extension += [(e_mc, (prepend + e_mo)[:NGRAM_SIZE])]
coeffs = extension + coeffs

print('N-Gram Coefficients:', coeffs)

word_coeffs = []
for i in range(CONTEXT_PRE, len(coeffs) - (CONTEXT_POST - 1)):
    context = coeffs[i - CONTEXT_PRE:i + CONTEXT_POST]
    avg = np.max([c for c, *_ in context])
    word, *_ = coeffs[i][1]
    word_coeffs += [(word, avg)]

print('Word Coefficients:', word_coeffs)

print('By N-Grams:', np.average([m for m, *_ in matches]))
print('By Word:', np.average([c for _, c in word_coeffs]))
