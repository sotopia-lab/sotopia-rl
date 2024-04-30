import jsonlines
from scipy.stats import spearmanr


def read_data():
    with jsonlines.open("../data/openai_log_attribution.jsonl", "r") as reader:
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
    # import pdb; pdb.set_trace()
    paired_scores = []
    seen_3 = False
    for key in human_attributed_utterances:
        human_scores = human_attributed_utterances[key][-1]
        prompt_score = prompt_attributed_utterances[key][-1]
        if isinstance(human_scores, dict) and prompt_score != -1:
            # human_score = 0
            # for key, score in human_scores.items():
            #     human_score += score
            # human_score /= len(human_scores)
            sorted_human_scores = sorted(human_scores.items(), key=lambda x: x[0])
            # human_score = sorted_human_scores[0][1]
            ann0, ann1 = sorted_human_scores[0][1], sorted_human_scores[1][1]
            if seen_3:
                ann1 = 0
            if ann1 == 3 and not seen_3:
                seen_3 = True
            human_score = ann1
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
                    "attributed_utterances"
                ]
                paired_scores = build_paired_scores(
                    human_attributed_utterances, prompt_attributed_utterances
                )
                paired_scores_dataset += paired_scores
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
    print("agreeement rate: {}".format(agreement_rate))
    print("average difference: {}".format(avg_diff))
