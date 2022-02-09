from typing import List

import nltk
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.similarity import prepare_words, similarity_reified, Match

NGRAM_SIZE = 3
app = FastAPI()


class Input(BaseModel):
    reference: str
    sample: str


class Response(object):
    similarity: float
    matches: List[Match]

    def __init__(self, similarity: float, matches: List[Match]):
        self.similarity = similarity
        self.matches = matches

    def to_json(self):
        return {
            'similarity': self.similarity,
            'matches': [{'value': m.value, 'reference': m.reference, 'sample': m.sample} for m in self.matches],
        }


def compute_response(matches: list[Match]) -> dict:
    resp = Response(
        similarity=np.average([m.value for m in matches]),
        matches=matches
    )
    return resp.to_json()


@app.post('/similar/full')
def compute_full(body: Input):
    words_reference = prepare_words(body.reference)
    words_sample = prepare_words(body.sample)
    ngram_size = min(NGRAM_SIZE, len(words_sample), len(words_reference))
    reference_ngrams: List[List[str]] = [list(x) for x in nltk.ngrams(words_reference, ngram_size)]
    sample_ngrams: List[List[str]] = [list(x) for x in nltk.ngrams(words_sample, ngram_size)]

    matches = similarity_reified(reference_ngrams, sample_ngrams)

    return compute_response(matches)
