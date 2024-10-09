# Based on seminar materials

# Don't forget to support cases when target_text == ''

import numpy as np

def levenshtein_distance(target_text, predict_text) -> float:
    target_size = len(target_text)
    predict_size = len(predict_text)

    dist_matrix = np.zeros((target_size + 1, predict_size + 1))

    dist_matrix[:, 0] = np.arange(target_size + 1)
    dist_matrix[0, :] = np.arange(predict_size + 1)

    for i in range(1, target_size + 1):
        for j in range(1, predict_size + 1):
            if target_text[i - 1] == predict_text[j - 1]:
                penalty = 0
            else:
                penalty = 1

            substitution_move = dist_matrix[i - 1, j - 1] + penalty
            deletion_move = dist_matrix[i - 1, j] + 1
            insertion_move = dist_matrix[i, j - 1] + 1
            
            dist_matrix[i, j] = min(
                substitution_move, 
                deletion_move, 
                insertion_move
            )

    return dist_matrix[target_size, predict_size]


def calc_cer(target_text, predicted_text) -> float:
    print(predicted_text)

    target_chars = list(target_text)
    predicted_chars = list(predicted_text)

    if len(target_chars) == 0:
        return 1

    return levenshtein_distance(
        target_chars, 
        predicted_chars
    ) / len(target_chars)


def calc_wer(target_text, predicted_text) -> float:
    # print('-' * 100)
    # print(predicted_text)
    # print('----')
    # print(target_text)
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if len(target_words) == 0:
        return 1

    return levenshtein_distance(
        target_words, 
        predicted_words
    ) / len(target_words)
