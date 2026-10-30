"""Transparent error and repetition metrics for local speech evaluations.

SPDX-License-Identifier: Apache-2.0
"""

import re
import unicodedata


def words(text):
    clean = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return re.findall(r"[^\W_]+", clean, flags=re.UNICODE)


def edit_counts(reference, hypothesis):
    """Return substitutions, deletions, insertions, and reference length."""
    a, b = list(reference), list(hypothesis)
    previous = [(i, 0, i, 0) for i in range(len(b) + 1)]
    for token in a:
        current = [(previous[0][0] + 1, previous[0][1], previous[0][2] + 1, previous[0][3])]
        for j, other in enumerate(b, 1):
            same = token == other
            sub = previous[j-1] if same else (previous[j-1][0] + 1, previous[j-1][1] + 1,
                                                 previous[j-1][2], previous[j-1][3])
            delete = (previous[j][0] + 1, previous[j][1], previous[j][2] + 1, previous[j][3])
            insert = (current[j-1][0] + 1, current[j-1][1], current[j-1][2], current[j-1][3] + 1)
            current.append(min(sub, delete, insert))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return {"substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "reference_length": len(a)}


def error_rates(reference, hypothesis):
    expected, actual = words(reference), words(hypothesis)
    ref_chars, hyp_chars = list(" ".join(expected)), list(" ".join(actual))
    word_counts = edit_counts(expected, actual)
    char_counts = edit_counts(ref_chars, hyp_chars)
    return {"wer": round(sum(word_counts[k] for k in ("substitutions", "deletions", "insertions")) /
                         max(1, len(expected)), 4),
            "cer": round(sum(char_counts[k] for k in ("substitutions", "deletions", "insertions")) /
                         max(1, len(ref_chars)), 4),
            "word_errors": word_counts, "char_errors": char_counts}


def adjacent_repetitions(text):
    """Find immediately repeated 1–4 word spans; diagnostic, not an ASR ground truth."""
    tokens = words(text)
    found = []
    i = 0
    while i < len(tokens):
        for size in range(min(4, (len(tokens)-i)//2), 0, -1):
            if tokens[i:i+size] == tokens[i+size:i+2*size]:
                found.append({"start_word": i, "phrase": " ".join(tokens[i:i+size]), "words": size})
                i += size * 2
                break
        else:
            i += 1
    return found
