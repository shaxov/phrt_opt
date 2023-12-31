import numpy as np


def quality(x, y, axis=0):
    q = np.sum(x.conj() * y, axis=axis)
    q = np.abs(q)
    q = np.square(q)

    d = np.sum(np.abs(x) * np.abs(y), axis=axis)
    d = np.square(d)
    q /= d

    if q.size == 1:
        q = np.squeeze(q)[()]
    return 1 - q


def quality_norm(x, y, axis=0):
    n = np.shape(x)[axis]
    n2 = n * n
    x = x / np.abs(x)
    y = y / np.abs(y)
    return quality(x, y, axis)


def projection(x, y, axis=0):
    d = np.sum(np.conj(x) * y, axis=axis, keepdims=True)
    d = np.angle(d)
    d = x - np.exp(-1j * d) * y
    d = np.linalg.norm(d, axis=axis)

    if d.size == 1:
        d = np.squeeze(d)[()]
    return d


def projection_norm(x, y):
    return projection(x, y) / np.max([np.linalg.norm(x), np.linalg.norm(x)])


def get(name):
    return {
        "quality": quality,
        "quality_norm": quality_norm,
        "projection": projection,
        "projection_norm": projection_norm,
    }[name]
