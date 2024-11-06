import json
import statistics as stats

with open("../../data/gpt35_gpt4_prompt_response_pairs.json", "r") as f:
    data = json.load(f)

prompt_lens = []
response_lens_35 = []
response_lens_4 = []
for d in data:
    prompt_lens.append(len(d["prompt"]))
    response_lens_35.append(len(d["gpt-3.5-turbo"]))
    response_lens_4.append(len(d["gpt-4-turbo"]))

print(stats.mean(prompt_lens))
print(stats.mean(response_lens_35))
print(stats.mean(response_lens_4))
