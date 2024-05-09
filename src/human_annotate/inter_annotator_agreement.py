import json
from datetime import datetime

# from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

with open("../data/human_log_attribution.jsonl", "r") as f:
    data = [json.loads(line) for line in f]


def convert_to_unix_timestamp_milliseconds(date_time: str) -> int:
    timestamp_dt = datetime.fromisoformat(date_time.rstrip("Z"))
    unix_timestamp_milliseconds = int(timestamp_dt.timestamp() * 1000)
    return unix_timestamp_milliseconds

count_single = 0
count_double = 0
result = []
convs = []
for item in data:
    # import pdb; pdb.set_trace()
    conv = []
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
    if len(conv) > 1:
        convs.append(conv)
ann0 = []
ann1 = []
for res in result:
    lis = []
    for key, item in res[2].items():
        time_int = convert_to_unix_timestamp_milliseconds(key)
        lis.append((time_int, item))
    lis.sort(key=lambda x: x[0])
    # print("Utterance: ", res[0])
    # print("Scores: ", lis)
    ann0.append(lis[0][1])
    ann1.append(lis[1][1])

# calculate cohens kappa
spearmanr = spearmanr(ann0, ann1)

print(f"spearmanr: {spearmanr}")
print(f"Count of single annotated: {count_single}")
print(f"Count of double annotated: {count_double}")

# count exact match
count_exact_match = 0
for i in range(len(ann0)):
    if ann0[i] == ann1[i]:
        count_exact_match += 1

average_difference = sum([abs(a - b) for a, b in zip(ann0, ann1)]) / len(ann0)
print(f"Count of exact match: {count_exact_match}")
print(f"Percentage of exact match: {count_exact_match / len(ann0)}")
print(f"Average difference: {average_difference}")

agreement_list = []
agreement_linient_list = []
for conv in convs:
    # import pdb; pdb.set_trace()
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

print("Agreement rate: ", sum(agreement_list) / len(agreement_list))
print("Agreement linient rate: ", sum(agreement_linient_list) / len(agreement_linient_list))