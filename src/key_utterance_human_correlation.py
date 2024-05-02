import sys

import jsonlines
import numpy as np
from scipy.stats import spearmanr

sys.path.append("../")
from utils.fleiss_kappa import fleiss_kappa


def read_data():
    with jsonlines.open(
        "../data/openai_log_key_utterance.jsonl", "r"
    ) as reader:
        prompting_dataset = list(reader)

    with jsonlines.open("../data/human_log_attribution.jsonl", "r") as reader:
        human_dataset = list(reader)
    return prompting_dataset, human_dataset


def hard_code_key(attributed_utterances):
    new_attributed_utterances = {}
    for key in attributed_utterances:
        utterance_num = int(key.split(" ")[1])
        new_utterance_num = utterance_num + 1
        new_key = key.replace(str(utterance_num), str(new_utterance_num))
        new_attributed_utterances[new_key] = attributed_utterances[key]
    return new_attributed_utterances


def build_paired_scores(
    human_attributed_utterances, prompt_attributed_utterances
):
    paired_scores = []
    for key in human_attributed_utterances:
        human_scores = human_attributed_utterances[key][-1]
        prompt_score = prompt_attributed_utterances[key][-1]
        if isinstance(human_scores, dict) and prompt_score != -1:
            human_score = 0
            for key, score in human_scores.items():
                human_score += score
            human_score /= len(human_scores)
            paired_scores.append((human_score, prompt_score))
    return paired_scores


if __name__ == "__main__":
    prompting_dataset, human_dataset = read_data()
    paired_scores_dataset = []
    for human_data in human_dataset:
        for prompt_data in prompting_dataset:
            if (
                human_data["episode_id"] == prompt_data["episode_id"]
                and human_data["agent"] == prompt_data["agent"]
            ):
                human_attributed_utterances = human_data[
                    "attributed_utterances"
                ]
                prompt_attributed_utterances = prompt_data[
                    "key_utterance_judgement"
                ]
                paired_scores = build_paired_scores(
                    human_attributed_utterances, prompt_attributed_utterances
                )
                paired_scores_dataset += paired_scores
                break
    prompt_scores = []
    for score in paired_scores_dataset:
        if score[0] == 3:
            prompt_scores.append(1)
        else:
            prompt_scores.append(0)

    human_scores = []
    for score in paired_scores_dataset:
        if score[1] == "YES":
            human_scores.append(1)
        else:
            human_scores.append(0)

    spearman_corr, _ = spearmanr(human_scores, prompt_scores)

    # calculate the Recall / Precision / F1
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy = accuracy_score(human_scores, prompt_scores)
    precision = precision_score(human_scores, prompt_scores)
    recall = recall_score(human_scores, prompt_scores)
    f1 = f1_score(human_scores, prompt_scores)
    print("accuracy: {}".format(accuracy))
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))

    kappa_input = np.array([prompt_scores, human_scores]).T
    fleiss_kappa_score = fleiss_kappa(kappa_input)
    print("Fleiss' Kappa: {}".format(fleiss_kappa_score))
