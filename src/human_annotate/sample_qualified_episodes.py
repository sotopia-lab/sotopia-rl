import json
import re
from collections import defaultdict

from tqdm import tqdm

from .get_form_responses import get_form

with open("sotopia_episodes_v1.jsonl", "r") as f:
    episodes = [json.loads(line) for line in f]

print(episodes[0].keys())

num_turns = []
len_episodes = []
for episode in episodes:
    num_turns.append(episode["social_interactions"].count("said:"))
    len_episodes.append(len(episode["social_interactions"]))

# # plot distribution of number of turns
# import matplotlib.pyplot as plt
# import numpy as np

# plt.hist(num_turns, bins=np.arange(0, 50, 1))
# plt.xlabel("Number of Utterances")
# plt.ylabel("Frequency")
# plt.title("Distribution of Number of Turns in Episodes")
# plt.savefig("num_turns.png")

# # plot distribution of length of episodes
# plt.hist(len_episodes, bins=np.arange(0, 200, 1))
# plt.xlabel("Length of Episode")
# plt.ylabel("Frequency")
# plt.title("Distribution of Length of Episodes")
# plt.savefig("len_episodes.png")

# exit()

qualifying_episodes = []
for episode in episodes:
    if (
        (
            episode["rewards"][0]["goal"] >= 8
            or episode["rewards"][0]["goal"] <= 2
        )
        and (
            episode["rewards"][1]["goal"] >= 8
            or episode["rewards"][1]["goal"] <= 2
        )
        and (episode["social_interactions"].count("said:") > 2)
        and (
            episode["experiment_model_name_pairs"][1] == "gpt-4"
            or episode["experiment_model_name_pairs"][1] == "gpt-3.5-turbo"
        )
        and (
            episode["experiment_model_name_pairs"][2] == "gpt-4"
            or episode["experiment_model_name_pairs"][2] == "gpt-3.5-turbo"
        )
    ):
        qualifying_episodes.append(episode)

print(len(qualifying_episodes))

codename_to_episode = defaultdict(list)
for episode in qualifying_episodes:
    codename_to_episode[episode["codename"]].append(episode)

with open("forms_left.txt", "r") as f:
    forms_left = f.readlines()
print("forms_left: " + str(len(forms_left)))

forms_left = [form.strip() for form in forms_left]


def extract_scenario(text: str) -> str:
    match = re.search(r"Scenario:\n(.*?)(?=\n\n|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No scenario found")


left_forms = []
for formId in tqdm(forms_left):
    left_forms.append(get_form(formId))

example_episodes = []
visited_codename = set()
for episode in qualifying_episodes:
    if episode["codename"] in visited_codename:
        continue

    is_valid = True
    for form in left_forms:
        scenario = extract_scenario(form["items"][0]["description"])
        if scenario in episode["scenario"]:
            is_valid = False
            break
    if is_valid:
        example_episodes.append(episode)
        visited_codename.add(episode["codename"])

    if len(example_episodes) == 20:
        break

with open("example_episodes.jsonl", "w") as f:
    for episode in example_episodes:
        f.write(json.dumps(episode) + "\n")

# example_episodes = []
# visited_codename = set()
# count = 0
# for episode in episodes:
#     if episode['codename'] not in visited_codename \
#             and episode['rewards'][0]['goal'] > 7 and episode['rewards'][1]['goal'] > 7:
#                 count += 1
#                 example_episodes.append(episode)
#                 visited_codename.add(episode['codename'])
#     if count == 10:
#         break

# count = 0
# for episode in episodes:
#     if episode['codename'] not in visited_codename \
#             and abs( episode['rewards'][0]['goal'] - episode['rewards'][1]['goal'] ) > 4:
#                 count += 1
#                 example_episodes.append(episode)
#                 visited_codename.add(episode['codename'])
#     if count == 10:
#         break

# count = 0
# for episode in episodes:
#     if episode['codename'] not in visited_codename \
#             and abs( episode['rewards'][0]['goal'] < 2 - episode['rewards'][1]['goal'] ) < 2:
#                 count += 1
#                 example_episodes.append(episode)
#                 visited_codename.add(episode['codename'])
#     if count == 10:
#         break

# with open("example_episodes.jsonl", 'w') as f:
#     for episode in example_episodes:
#         f.write(json.dumps(episode) + "\n")
