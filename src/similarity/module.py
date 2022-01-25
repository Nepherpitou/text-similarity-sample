import math
import string
import time
from typing import List, Iterable, Tuple, Callable

import nltk
import num2words
import numpy as np

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


def relative_distance(word_1: str, word_2: str) -> float:
    return nltk.masi_distance(set(word_1), set(word_2))


def best_distance(word: str, words: Iterable[str]) -> float:
    distances = [relative_distance(word, w) for w in words]
    return max(distances)


def similarity_local(xs: List[str], ys: List[str]) -> float:
    if len(xs) != len(ys):
        return 0
    # distances = [1 - float(nltk.edit_distance(x, y)) / max(len(x), len(y)) for x, y in zip(xs, ys)]
    distances = [1 - nltk.jaccard_distance(set(x), set(y)) for x, y in zip(xs, ys)]
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


def similarity_reified(xs: List[List[str]], ys: List[List[str]]) -> List[Tuple[float, List[str], List[str]]]:
    similarity_matched: List[Tuple[float, List[str], List[str]]] = []
    pool_original = xs.copy()
    pool_recognized = ys.copy()
    while pool_recognized and pool_original:
        intermediate = similarity_intermediate(pool_original, pool_recognized)
        best_sim, original_ngram, recognized_ngram = max(intermediate, key=lambda x: x[0])
        similarity_matched.append((best_sim, original_ngram, recognized_ngram))
        pool_recognized.remove(recognized_ngram)
        pool_original.remove(original_ngram)
    return similarity_matched


def similarity_full(
        xs: List[List[str]],
        matched: List[Tuple[float, List[str], List[str]]]
) -> List[Tuple[float, List[str]]]:
    similarities: List[Tuple[float, List[str]]] = []
    for original_ngram in xs:
        similar = (0.0, original_ngram)
        for (best_sim, best_original_ngram, _) in matched:
            if original_ngram == best_original_ngram:
                similar = (best_sim, best_original_ngram)
                break
        similarities.append(similar)
    return similarities


def similarity_extend(similarities: list[tuple[float, list[str]]], ngram_size: int) -> list[tuple[float, list[str]]]:
    prepend_size = math.ceil(ngram_size / 2) if ngram_size > 1 else 0
    coeffs = [(mc, list(mo)) for mc, mo in similarities]
    (e_mc, e_mo) = coeffs[0]
    extension = []
    for i in range(prepend_size):
        prepend = ['' for _ in range(prepend_size - i)]
        extension += [(e_mc, (prepend + e_mo)[:ngram_size])]
    coeffs = extension + coeffs
    return coeffs


def similarity_by_word(
        extended: list[tuple[float, list[str]]],
        ngram_size: int
) -> list[tuple[str, float]]:
    prepend_size = math.ceil(ngram_size / 2) if ngram_size > 1 else 0
    append_size = ngram_size - prepend_size
    words_similarity: list[tuple[str, float]] = []
    for i in range(prepend_size, len(extended) - (append_size - 1)):
        context = extended[i - prepend_size:i + append_size]
        avg = np.max([c for c, *_ in context])
        word, *_ = extended[i][1]
        words_similarity += [(word, avg)]
    return words_similarity


def _compute_ngram_similarity(
        expected_words: list[str],
        tested_words: list[str],
        ngram_size: int,
        method: Callable[[List[List[str]], List[List[str]]], List[Tuple[float, List[str], List[str]]]]
) -> list[tuple[float, list[str]]]:
    ngrams_expected = list(nltk.ngrams(expected_words, ngram_size))
    ngrams_tested = list(nltk.ngrams(tested_words, ngram_size))
    matched = method(ngrams_expected, ngrams_tested)
    full = similarity_full(ngrams_expected, matched)
    return full


def compute_ngram_similarity_fast(
        expected_words: list[str],
        tested_words: list[str],
        ngram_size: int
) -> list[tuple[float, list[str]]]:
    return _compute_ngram_similarity(expected_words, tested_words, ngram_size, similarity_intermediate)


def compute_ngram_similarity_full(
        expected_words: list[str],
        tested_words: list[str],
        ngram_size: int
) -> list[tuple[float, list[str]]]:
    return _compute_ngram_similarity(expected_words, tested_words, ngram_size, similarity_reified)


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
    original = "TransPerfect improves chatbot performance up to 30% compared to a translation approach. We’re able to " \
               "remove bias and incorporate diversity and nuance for target markets. Let’s have a call with a Solution " \
               "Engineer, who can better walk through our process with you. Is Tuesday morning or afternoon better for " \
               "you?"
    recognized = "Transparent effect improves childhood performance up to 30% compared to a translation approach. Were " \
                 "able to remove barriers and incorporate diversity and nuance for target markets. Let's say we're " \
                 "solution Engineer who can better walk through a process with you. Is choose a morning or afternoon. " \
                 "Better for you. "

    # original = 'tes tes'
    # recognized = 'test test'

    start_time = time.time_ns()

    original_words = prepare_words(original)
    recognized_words = prepare_words(recognized)
    ngram_size = min(NGRAM_SIZE, len(original_words), len(recognized_words))
    ngram_similarity = compute_ngram_similarity_full(recognized_words, original_words, ngram_size)

    print('All ngrams:', ngram_similarity)

    similarity_extended = similarity_extend(ngram_similarity, ngram_size)
    similarity_words = similarity_by_word(similarity_extended, ngram_size)
    incorrect_words = [(w, c) for w, c in similarity_words if c < 0.8]

    print('N-Gram Coefficients:', similarity_extended)
    print('Word Coefficients:', similarity_words)

    print('Avg By N-Grams:', np.average([m for m, *_ in ngram_similarity]))
    print('Avg By Word:', np.average([c for _, c in similarity_words]))

    print('Missed words:', incorrect_words)

    print(f'Execution takes {(time.time_ns() - start_time) / 1000000} ms')
