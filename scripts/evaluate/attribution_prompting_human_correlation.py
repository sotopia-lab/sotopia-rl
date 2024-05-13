import argparse
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
)
import os

import jsonlines
from scipy.stats import spearmanr


def read_data(data_dir: str) -> Tuple[Iterable[Any], Iterable[Any]]:
    with jsonlines.open(os.path.join(data_dir, "openai_log_attribution.jsonl"), "r") as reader:
        prompting_dataset = list(reader)

    with jsonlines.open(os.path.join(data_dir, "human_log_attribution.jsonl"), "r") as reader:
        human_dataset = list(reader)
    return prompting_dataset, human_dataset


def hard_code_key(attributed_utterances: Any) -> Any:
    new_attributed_utterances = {}
    for key in attributed_utterances:
        utterance_num = int(key.split(" ")[1])
        new_utterance_num = utterance_num + 1
        new_key = key.replace(str(utterance_num), str(new_utterance_num))
        new_attributed_utterances[new_key] = attributed_utterances[key]
    return new_attributed_utterances


def build_paired_scores(
    human_attributed_utterances: Any, prompt_attributed_utterances: Any, average: bool = False, annotator: int = 0
) -> List[Tuple[int, int]]:
    paired_scores = []
    for key in human_attributed_utterances:
        human_scores = human_attributed_utterances[key][-2]
        prompt_score = prompt_attributed_utterances[key][-1]
        if isinstance(human_scores, dict) and prompt_score != -1:
            sorted_human_scores = sorted(
                human_scores.items(), key=lambda x: x[0]
            )
            ann0, ann1 = sorted_human_scores[0][1], sorted_human_scores[1][1]
            if average:
                human_score = (ann0 + ann1) / 2
            else:
                human_score = ann0 if annotator == 0 else ann1
            paired_scores.append((human_score, prompt_score))
    return paired_scores


def main(data_dir: str, average: bool, annotator: int) -> None:
    prompting_dataset, human_dataset = read_data(data_dir)
    paired_scores_dataset = []
    paired_convs = []
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
                    "attributed_utterances"
                ]
                paired_scores = build_paired_scores(
                    human_attributed_utterances, prompt_attributed_utterances, average=average, annotator=annotator
                )
                paired_scores_dataset += paired_scores
                paired_convs.append(paired_scores)
                break
    human_scores = [score[0] for score in paired_scores_dataset]
    prompt_scores = [score[1] for score in paired_scores_dataset]
    spearman_corr, _ = spearmanr(human_scores, prompt_scores)
    agreement_rate = len(
        [1 for score in paired_scores_dataset if score[0] == score[1]]
    ) / len(paired_scores_dataset)
    avg_diff = sum(
        [abs(score[0] - score[1]) for score in paired_scores_dataset]
    ) / len(paired_scores_dataset)
    print("average difference: {}".format(avg_diff))
    print("spearman correlation: {}".format(spearman_corr))
    print("exact match: {}".format(agreement_rate))
    
    agreement_3_list = []
    agreement_3_linient_list = []
    for i, conv in enumerate(paired_convs):
        human_scores = [score[0] for score in conv]
        prompt_scores = [score[1] for score in conv]
        human_key_utterance = human_scores.index(3) if 3 in human_scores else -1
        prompt_key_utterance = prompt_scores.index(3) if 3 in prompt_scores else -1
        if human_key_utterance == prompt_key_utterance:
            agreement_3_list.append(1)
        else:
            agreement_3_list.append(0)
        if human_key_utterance != -1 and prompt_key_utterance != -1:
            agreement_3_linient_list.append(1)
        else:
            agreement_3_linient_list.append(0)
    print("agreement rate: {}".format(sum(agreement_3_list) / len(agreement_3_list)))
    print("agreement linient rate: {}".format(sum(agreement_3_linient_list) / len(agreement_3_linient_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--average', action='store_true', help='Whether to average the human scores')
    parser.add_argument('--annotator', type=int, required=False, help='Which human annotator to use')
    
    args = parser.parse_args()
    print(args.data_dir, args.average, args.annotator)
    main(args.data_dir, args.average, args.annotator)