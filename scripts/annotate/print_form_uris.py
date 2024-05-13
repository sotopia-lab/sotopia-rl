import json

with open("../../data/form_uris.jsonl", "r") as f:
    form_uris = [json.loads(line) for line in f]
for item in form_uris:
    print(item['formId'])