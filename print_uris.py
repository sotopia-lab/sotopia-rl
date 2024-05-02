import json

with open('form_uris.jsonl') as f:
    uris = [json.loads(line) for line in f]

for uri in uris[0:28]:
    print(uri['responderUri'])