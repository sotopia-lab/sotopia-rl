import json
from datetime import datetime

with open("../data/human_log_attribution.jsonl", "r") as f:
    data = [json.loads(line) for line in f]


def convert_to_unix_timestamp_milliseconds(date_time):
    timestamp_dt = datetime.fromisoformat(date_time.rstrip("Z"))
    unix_timestamp_milliseconds = int(timestamp_dt.timestamp() * 1000)
    return unix_timestamp_milliseconds


count_single = 0
count_double = 0
result = []
for item in data:
    for key in item["attributed_utterances"]:
        if (
            len(item["attributed_utterances"][key]) > 2
            and len(item["attributed_utterances"][key][2]) > 0
        ):
            count_single += 1
            if len(item["attributed_utterances"][key][2]) > 1:
                count_double += 1
                result.append(item["attributed_utterances"][key])

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
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(ann0, ann1)

print(f"Cohen's Kappa: {kappa}")
print(f"Count of single annotated: {count_single}")
print(f"Count of double annotated: {count_double}")

# count exact match
count_exact_match = 0
for i in range(len(ann0)):
    if ann0[i] == ann1[i]:
        count_exact_match += 1
print(f"Count of exact match: {count_exact_match}")
print(f"Percentage of exact match: {count_exact_match / len(ann0)}")
