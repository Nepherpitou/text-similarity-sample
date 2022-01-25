from typing import List, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.similarity import compute_ngram_similarity_full, similarity_extend, similarity_by_word, \
    compute_ngram_similarity_fast, prepare_words

NGRAM_SIZE = 3
app = FastAPI()


class Input(BaseModel):
    reference: str
    sample: str


class Response(object):
    similarity: float
    words: List[Tuple[str, float]]
    ngrams: List[Tuple[float, List[str]]]

    def __init__(self, similarity: float, words: List[Tuple[str, float]], ngrams: List[Tuple[float, List[str]]]):
        self.similarity = similarity
        self.words = words
        self.ngrams = ngrams

    def to_json(self):
        return {
            'similarity': self.similarity,
            'words': [{'word': w, 'similarity': c} for w, c in self.words],
            'ngrams': [{'ngram': ' '.join(ng), 'similarity': c} for c, ng in self.ngrams]
        }


def compute_response(ngram_similarity: list[tuple[float, list[str]]]) -> dict:
    similarity_extended = similarity_extend(ngram_similarity, NGRAM_SIZE)
    similarity_words = sorted(similarity_by_word(similarity_extended, NGRAM_SIZE), key=lambda wc: wc[1])

    resp = Response(
        similarity=np.average([m for m, *_ in ngram_similarity]),
        words=similarity_words,
        ngrams=ngram_similarity
    )

    return resp.to_json()


@app.post('/similar/fast')
def compute_fast(body: Input):
    words_reference = prepare_words(body.reference)
    words_sample = prepare_words(body.sample)
    ngram_size = min(NGRAM_SIZE, len(words_sample), len(words_reference))
    ngram_similarity = compute_ngram_similarity_fast(words_sample, words_reference, ngram_size)
    return compute_response(ngram_similarity)


@app.post('/similar/full')
def compute_full(body: Input):
    words_reference = prepare_words(body.reference)
    words_sample = prepare_words(body.sample)
    ngram_size = min(NGRAM_SIZE, len(words_sample), len(words_reference))
    ngram_similarity = compute_ngram_similarity_full(words_sample, words_reference, ngram_size)
    return compute_response(ngram_similarity)
