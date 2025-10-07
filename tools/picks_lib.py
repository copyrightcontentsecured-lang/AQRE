import numpy as np
LABELS = ["A","D","H"]

def decide_pick(probs, min_conf, draw_margin):
    probs = np.asarray(probs, dtype=float)
    idx = np.argsort(probs)[::-1]
    top, second = probs[idx[0]], probs[idx[1]]
    if float(top) < float(min_conf):
        return "D"
    if (top - second) < float(draw_margin):
        return "D"
    return LABELS[int(idx[0])]
