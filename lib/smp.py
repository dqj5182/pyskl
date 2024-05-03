# flake8: noqa: F401, F403
import numpy as np
import requests


def intop(pred, label, n):
    pred = [np.argsort(x)[-n:] for x in pred]
    hit = [(l in p) for l, p in zip(label, pred)]
    return hit

def comb(scores, coeffs):
    ret = [x * coeffs[0] for x in scores[0]]
    for i in range(1, len(scores)):
        ret = [x + y for x, y in zip(ret, [x * coeffs[i] for x in scores[i]])]
    return ret

def auto_mix2(scores):
    assert len(scores) == 2
    return {'1:1': comb(scores, [1, 1]), '2:1': comb(scores, [2, 1]), '1:2': comb(scores, [1, 2])}

def topk(score, label, k=1):
    return np.mean(intop(score, label, k)) if isinstance(k, int) else [topk(score, label, kk) for kk in k]

def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)