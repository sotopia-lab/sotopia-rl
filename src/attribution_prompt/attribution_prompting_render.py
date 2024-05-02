import json
from collections import OrderedDict
from pprint import pprint

import rich

with open("../data/openai_log_attribution.jsonl", "r") as f:
    logs = [json.loads(line, object_hook=OrderedDict) for line in f]


def render_log(log):
    print(log["scenario"] + "\n")
    print("Agent: " + log["agent"] + "\n")
    print("Goal: " + log["goal"] + "\n")

    for key, val in log["attributed_utterances"].items():
        print(key)
        print("Utterance: ", val[0])
        if val[1] == -1:
            print("Attribution: ", "mask")
        else:
            print("Attribution: ", val[1])
            print("Attributed reward: ", val[1] / 3 * log["goal_score"])

        print()
    print("goal achieving score: ", log["goal_score"])


render_log(logs[11])
