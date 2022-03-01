import string
import time
from typing import List, Iterable, Tuple, Optional

import nltk
import num2words
import numpy as np
from nltk.metrics import distance

NGRAM_SIZE = 3


class Match(object):
    sample: Optional[List[str]]
    reference: Optional[List[str]]
    value: float

    def __init__(self, sample: Optional[List[str]], reference: Optional[List[str]], value: float):
        self.sample = sample
        self.reference = reference
        self.value = value

    def __unicode__(self):
        return f'Match{{{self.value}, sample={self.sample}, reference={self.reference})}}'

    def __repr__(self):
        return self.__unicode__()


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


def relative_distance(word_1: str, word_2: str) -> float:
    return nltk.masi_distance(set(word_1), set(word_2))


def best_distance(word: str, words: Iterable[str]) -> float:
    distances = [relative_distance(word, w) for w in words]
    return max(distances)


def similarity_local(xs: List[str], ys: List[str]) -> float:
    if len(xs) != len(ys):
        return 0
    # distances = [1 - float(nltk.edit_distance(x, y)) / max(len(x), len(y)) for x, y in zip(xs, ys)]
    # distances = [1 - nltk.jaccard_distance(set(x), set(y)) for x, y in zip(xs, ys)]
    distances = [1.0 if x == y else distance.jaro_similarity(x, y) for x, y in zip(xs, ys)]
    # print(distances, xs, ys)
    return np.average(distances)
    # , weights=[len(x) for x in xs]) TODO: Investigate why result aren't isomorphic with weights


def similarity_intermediate(xs: List[List[str]], pool: List[List[str]]) -> List[Tuple[float, List[str], List[str]]]:
    similarities_best: List[Tuple[float, List[str], List[str]]] = []
    for i in range(len(xs)):
        x = xs[i]
        similarities = [(similarity_local(x, y), y) for y in pool]
        if len(similarities) > 0:
            best_sim, best_ngram = max(similarities, key=lambda s: s[0])
            similarities_best.append((best_sim, x, best_ngram))
    return similarities_best


def similarity_reified(references: List[List[str]], samples: List[List[str]]) -> List[Match]:
    matches: List[Match] = []
    pool_original = references.copy()
    pool_recognized = samples.copy()
    while pool_recognized and pool_original:
        intermediate = similarity_intermediate(pool_original, pool_recognized)
        best_sim, original_ngram, recognized_ngram = max(intermediate, key=lambda x: x[0])
        match = Match(recognized_ngram, original_ngram, best_sim)
        matches.append(match)
        pool_recognized.remove(recognized_ngram)
        pool_original.remove(original_ngram)
    for x in pool_original:
        matches.append(Match(None, x, 0.0))
    for x in pool_recognized:
        matches.append(Match(x, None, 0.0))
    return matches


def prepare_words(text: str) -> list[str]:
    return cleanup_punctuation(
        convert_numbers(
            nltk.word_tokenize(
                highlight_punctuation(
                    text.lower()
                )
            )
        )
    )


if __name__ == '__main__':
    sample = "I am looking forward to working with you on your client's website. One thing I did want to bring up " \
             "is accessibility. I know you asked us to translate the website, but an accessible site is more " \
             "inclusive, ranks higher in search engines, and provides a better user experience to everyone, " \
             "regardless of disabilities. Do you have a plan in place to ensure the site is compliant with the " \
             "applicable guidelines and laws? Absolutely! I will run a quick automated assessment to see how things " \
             "are now.  So, can you get a few times that work for you and your end client later this week? "
    reference = "So can you will get a few time to speak with you and your client? Yes, lets you do this call. Ha-ha."
    start_time = time.time_ns()

    reference_words = prepare_words(reference)
    sample_words = prepare_words(sample)

    ngram_size = min(NGRAM_SIZE, len(reference_words), len(sample_words))
    reference_ngrams: List[List[str]] = [list(x) for x in nltk.ngrams(reference_words, ngram_size)]
    sample_ngrams: List[List[str]] = [list(x) for x in nltk.ngrams(sample_words, ngram_size)]

    matches = similarity_reified(reference_ngrams, sample_ngrams)

    print('Avg by ngrams', np.average([m.value for m in matches]))
    print(matches)

    print(f'Execution takes {(time.time_ns() - start_time) / 1000000} ms')
