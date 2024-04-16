import json
import os
from openai import OpenAI
from pprint import pprint
from tqdm import tqdm

# set environment variables for OpenAI API
with open("openai_api.key", 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.readline().strip()

client = OpenAI()

def openai_call(model="gpt-3.5-turbo", prompt=""):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response


with open("experiment_episodes.jsonl", 'r') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]
    
# print(data[0].keys())
# print(data[0]['social_interactions'])
# ['Giselle Rousseau said: "Hi Ethan, I know we have different perspectives on how to care for our pets. However, I\'m hoping we can find a compromise that allows our pets to have a bit of freedom while ensuring they are safe and cared for. Are you open to discussing this further?"\n\nEthan Johnson said: "Absolutely, Giselle. I agree with you that freedom is important for our pets, but I also believe in their safety. I\'m interested in hearing your ideas on how we could create a shared space that will allow our pets that freedom while also ensuring they are safe and protected. Let\'s find a middle ground."\n\nGiselle Rousseau said: "I\'m glad we can discuss this, Ethan. One possible way to ensure the freedom and safety of our pets is investing in a big enough enclosure that would mimic a natural environment. It could have areas for play, rest, and exploring. Thus, they can express their natural behaviors while being under our watch. Also, there could be safety measures in place, like supervising them while they are in the enclosure. What do you think about this idea?"\n\nEthan Johnson said: "I appreciate your thoughts, Giselle. An enclosure that mimics a natural environment sounds like a good compromise that could grant our pets freedom while still being safe. However, continuous supervision might not always be plausible due to our respective schedules. I propose that we could use cages or pens during times when we might be unable to supervise them. That way, they are guaranteed safety, especially from potential dangers like traffic and predators. How does that sound to you?"\n\nGiselle Rousseau said: "Ethan, I understand your concern about unsupervised periods. Safety is indeed our priority. But pens or cages, they confine their movement and freedom to a significant extent which can lead to behavioural issues like aggression or anxiety. Perhaps, we could consider a secure outdoor pet playpen or an enclosed pet run instead of cages? We could also look into smart pet monitoring systems that allow us to keep a check on them remotely, keeping them safe while providing a bit more freedom. How about that?"\n\nEthan Johnson said: "Giselle, your suggestion of an outdoor pet playpen or an enclosed pet run, combined with a smart pet monitoring system, seems a practical solution. I appreciate your concern for our pets\' psychological well-being. Rest assured, the pens or cages I referred to could be spacious enough to avoid confinement or limit movement, ensuring our pets\' safety and health, but your idea might serve both freedom and safety. I\'m open to exploring these options further."\n\nGiselle Rousseau said: "I\'m delighted to hear your openness toward these suggestions, Ethan. The spacious cages you mentioned also seem like a viable option during instances of extreme weather or emergency situations. It\'s crucial to strike a balance between our pets\' safety and their need for freedom to ensure their overall well-being. Let\'s research more on this topic and reconvene to discuss the best options for our pets. How does that sound?"\n\nEthan Johnson said: "You\'re absolutely right, Giselle. We should prioritize the well-being of our pets and strive to strike the balance between their need for freedom and safety. I\'m more than willing to research more and figure out the best, feasible solution that will cater to both these needs. Great discussion today, looking forward to convening again with more solid ideas."\n\nGiselle Rousseau said: "I agree, Ethan. Together we can definitely create a space where our pets can roam free while also keeping them safe. Let\'s catch up soon after doing some research. Have a wonderful day!"\n\nEthan Johnson said: "Definitely, Giselle. This has been a great discussion, and I\'m looking forward to seeing what we can come up with. I\'ll start researching more on the options mentioned and hopefully, we can create a safe and stimulating environment for our pets. Have a wonderful day too!"\n\nGiselle Rousseau: left the conversation']

# print(data[0]['social_goals'])
# [{'Giselle Rousseau': 'Persuade Agent2 to consider the benefits of allowing pets more freedom and the potential positive impact on their well-being (<extra_info>Extra information: Agent1 values the natural instincts and happiness of animals and believes confinement can lead to behavioral issues.</extra_info>)', 'Ethan Johnson': 'Convince Agent1 of the safety and health advantages of using cages or pens for pets, especially when owners are not around to supervise them (<extra_info>Extra information: Agent2 is concerned about the dangers of the outside world, such as traffic and predators, and believes in preventive measures to protect their pets.</extra_info>)'}]

def processing_history_and_goals(episode):
    conversation = episode['social_interactions'][0].split("\n\n")
    goals = episode['social_goals'][0]
    agent1, agent2 = goals.keys()
    result = []
    for i, utterance in enumerate(conversation):
        if utterance.startswith("Giselle Rousseau said:"):
            speaker = "Giselle Rousseau"
        elif utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            print("Unknown speaker: {}".format(utterance))
            raise ValueError
        result.append({
            "speaker": speaker,
            "utterance": utterance,
            "history": "".join(conversation[:i]),
            "goal": goals[speaker]
        })
    return result

HEADER_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the most recent statement made by that agent. Your objective is to assess how much of the social goal have been achieved on a scale from 0 to 10. 
"""
EPILOGUE_INSTRUCTIONS = """
How much do you think {speaker} effectively achieved their social goal so far? Please start your answer with a number between 0 and 10, followed by any additional comments or explanations you may have. 
"""

def format_prompt_openai(speaker, utterance, history, goal):
    return HEADER_INSTRUCTIONS + f"{history}\n\n{speaker}'s utterance: \"{utterance}\"\n\n{speaker}'s social goal: {goal}\n\n" + EPILOGUE_INSTRUCTIONS.format(speaker=speaker)

for episode in data[0:5]:
    processed = processing_history_and_goals(episode)
    log = {"episode_id": episode['episode_id'], "log": []}
    for turn in tqdm(processed):
        prompt = format_prompt_openai(turn['speaker'], turn['utterance'], turn['history'], turn['goal'])
        response = openai_call(prompt=prompt)
        # import pdb; pdb.set_trace()
        log["log"].append({
            "prompt": prompt,
            "response": response.choices[0].message.content
        })
    with open("openai_log_scalar.jsonl", 'a') as f:
        f.write(json.dumps(log, indent=4) + "\n")