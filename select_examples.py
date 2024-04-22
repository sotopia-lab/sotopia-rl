import json

with open("episodes.jsonl", 'r') as f:
    episodes = [json.loads(line) for line in f]
    
example_episodes = []
visited_codename = set()
count = 0
for episode in episodes:
    if episode['codename'] not in visited_codename \
            and episode['rewards'][0]['goal'] > 7 and episode['rewards'][1]['goal'] > 7:
                count += 1
                example_episodes.append(episode)
                visited_codename.add(episode['codename'])
    if count == 10:
        break

count = 0
for episode in episodes:
    if episode['codename'] not in visited_codename \
            and abs( episode['rewards'][0]['goal'] - episode['rewards'][1]['goal'] ) > 4:
                count += 1
                example_episodes.append(episode)
                visited_codename.add(episode['codename'])
    if count == 10:
        break

count = 0
for episode in episodes:
    if episode['codename'] not in visited_codename \
            and abs( episode['rewards'][0]['goal'] < 2 - episode['rewards'][1]['goal'] ) < 2:
                count += 1
                example_episodes.append(episode)
                visited_codename.add(episode['codename'])
    if count == 10:
        break

with open("example_episodes.jsonl", 'w') as f:
    for episode in example_episodes:
        f.write(json.dumps(episode) + "\n")