import json
from datetime import datetime
from typing import Tuple, Dict, Any

from scipy.stats import spearmanr

with open("../../data/human_log_attribution.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

def convert_to_unix_timestamp_milliseconds(date_time: str) -> int:
    timestamp_dt = datetime.fromisoformat(date_time.rstrip("Z"))
    unix_timestamp_milliseconds = int(timestamp_dt.timestamp() * 1000)
    return unix_timestamp_milliseconds

def preprocessing(data: list[Dict[str, Any]]) -> Tuple[list[Any], list[Any], list[Any]]:
    count_single = 0
    count_double = 0
    result = []
    convs = []
    key_uttrs = []
    for item in data:
        conv = []
        key_uttr = []
        for key in item["attributed_utterances"]:
            if (
                len(item["attributed_utterances"][key]) > 2
                and len(item["attributed_utterances"][key][2]) > 0
            ):
                count_single += 1
                if len(item["attributed_utterances"][key][2]) > 1:
                    count_double += 1
                    result.append(item["attributed_utterances"][key])
                    conv.append(item["attributed_utterances"][key][2])
                    key_uttr.append(item["attributed_utterances"][key][3])
        if len(conv) > 1:
            convs.append(conv)
            key_uttrs.append(key_uttr)
    print(f"Count of single annotated: {count_single}")
    print(f"Count of double annotated: {count_double}")
    return result, convs, key_uttrs

def calc_correlation(result: list[Any]) -> None:
    ann0 = []
    ann1 = []
    for res in result:
        lis = []
        for key, item in res[2].items():
            time_int = convert_to_unix_timestamp_milliseconds(key)
            lis.append((time_int, item))
        lis.sort(key=lambda x: x[0])
        ann0.append(lis[0][1])
        ann1.append(lis[1][1])
        
    # calculate cohens kappa
    corr: Tuple[float, float] = spearmanr(ann0, ann1)
    print(f"spearmanr: {corr}")
    
    # count exact match
    count_exact_match = 0
    for i in range(len(ann0)):
        if ann0[i] == ann1[i]:
            count_exact_match += 1
            
    average_difference = sum([abs(a - b) for a, b in zip(ann0, ann1)]) / len(ann0)
    print(f"Count of exact match: {count_exact_match}")
    print(f"Percentage of exact match: {count_exact_match / len(ann0)}")
    print(f"Average difference: {average_difference}")

def calc_3_agreement(convs: list[Any]) -> None:
    agreement_list = []
    agreement_linient_list = []
    for conv in convs:
        ann0, ann1 = [], []
        for pair in conv:
            lis = []
            for key, item in pair.items():
                time_int = convert_to_unix_timestamp_milliseconds(key)
                lis.append((time_int, item))
            lis.sort(key=lambda x: x[0])
            ann0.append(lis[0][1])
            ann1.append(lis[1][1])
        key_utter_0 = ann0.index(3) if 3 in ann0 else -1
        key_utter_1 = ann1.index(3) if 3 in ann1 else -1
        agreement_list.append(key_utter_0 == key_utter_1)
        agreement_linient_list.append(
            abs(key_utter_0 - key_utter_1) <= 1)
        
    print("3 Rating Agreement Rate: ", sum(agreement_list) / len(agreement_list))
    print("3 Rating Agreement Linient Rate: ", sum(agreement_linient_list) / len(agreement_linient_list))

def calc_key_uttr_agreement(convs: list[Any]) -> None:
    agreement_list = []
    agreement_linient_list = []
    ann0, ann1 = [], []
    for conv in convs:
        # import pdb; pdb.set_trace()
        for pair in conv:
            # import pdb; pdb.set_trace()
            lis = []
            for key, item in pair.items():
                time_int = convert_to_unix_timestamp_milliseconds(key)
                lis.append((time_int, item))
            lis.sort(key=lambda x: x[0])
            if len(lis) < 2:
                continue
            assert lis[0][0] != lis[1][0]
            # import pdb; pdb.set_trace()
            ann0.append(1 if lis[0][1] == "Yes" else 0)
            ann1.append(1 if lis[1][1] == "Yes" else 0)
        # import pdb; pdb.set_trace()
        # count the number of exact match
    print(len(ann0))
    for i in range(len(ann0)):
        if ann0[i] == ann1[i]:
            agreement_list.append(1)
        else:
            agreement_list.append(0)
    # count the number of linient match
    for i in range(len(ann0)):
        if abs(ann0[i] - ann1[i]) <= 1:
            agreement_linient_list.append(1)
        else:
            agreement_linient_list.append(0)
        
    print("Key Utterance Agreement Rate: ", sum(agreement_list) / len(agreement_list))
    print("Key Utterance Agreement Linient Rate: ", sum(agreement_linient_list) / len(agreement_linient_list))
    print("Confusion Matrix with Annotator 0 as Ground Truth:")
    # calculate confusion matrix with annotator 0 as ground truth
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(ann0)):
        if ann0[i] == 1:
            if ann1[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if ann1[i] == 1:
                fp += 1
            else:
                tn += 1
    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"Specificity: {tn / (tn + fp)}")
    print(f"F1 Score: {2 * tp / (2 * tp + fp + fn)}")

if __name__ == "__main__":
    result, convs, key_uttrs = preprocessing(data)
    calc_correlation(result)
    calc_3_agreement(convs)
    calc_key_uttr_agreement(key_uttrs)