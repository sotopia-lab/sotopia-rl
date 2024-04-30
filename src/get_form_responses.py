import json
import pickle
from tqdm import tqdm

from google_form_api_wrapper import get_form, get_form_responses

def add_responses_to_sheet(log, form_mapping):
    """Add responses to rewared attribution log."""
    for i in range(len(log)):
        print(f"Log: {i}")
        # form_id = form_ids[i]["formId"]
        # form_schema = get_form(form_id)
        # responses = get_form_responses(form_id)
        form_schema, responses = form_mapping[(log[i]['episode_id'], log[i]['agent'])]
        # import pdb; pdb.set_trace()
        # print(f"Log: {log[i]}")
        print(f"Form ID: {form_schema['formId']}")
        # print(f"Form schema: {form_schema}")
        print(f"Responses: {responses}")
        for key in log[i]['attributed_utterances']:
            print(f"  Key: {key}")
            for item in form_schema['items']:
                if item['title'].split(":")[0] == key:
                    break
            if 'questionItem' in item:
                question_id = item['questionItem']['question']['questionId']
            else:
                print("  No question item found")
                continue
            for response in responses:
                response_id = response['responseId']
                for _, response_item in response['answers'].items():
                    response_question_id = response_item['questionId']
                    if response_question_id == question_id:
                        print(f"  Response ID: {response_id}")
                        if len(log[i]['attributed_utterances'][key]) == 2:
                            log[i]['attributed_utterances'][key].append({})
                        log[i]['attributed_utterances'][key][2].update({response['lastSubmittedTime']: int(response_item['textAnswers']['answers'][0]['value'])})
                        break
        print(f"Updated log")
    return log

def get_episode_agent_to_form_mapping(form_ids):
    """Get episode-agent pair to response mapping."""
    episode_agent_to_response_mapping = {}
    for form_id in tqdm(form_ids):
        form = get_form(form_id)
        episode_id = form['info']['title'].split(" ")[-1]
        agent = "{} {}".format(form['info']['title'].split(" ")[-3], form['info']['title'].split(" ")[-2])
        responses = get_form_responses(form_id)
        episode_agent_to_response_mapping[(episode_id, agent)] = (form, responses)
    return episode_agent_to_response_mapping

if __name__ == "__main__":
    with open("../data/openai_log_attribution.jsonl", "r") as f:
        log = [json.loads(line) for line in f]
    
    with open("../data/form_ids.txt", "r") as f:
        form_ids = f.read().splitlines()
    
    # form_mapping = get_episode_agent_to_form_mapping(form_ids)
    # assert len(form_mapping) == len(form_ids)
    # with open("../data/form_mapping.pkl", "wb") as f:
    #     pickle.dump(form_mapping, f)
    with open("../data/form_mapping.pkl", "rb") as f:
        form_mapping = pickle.load(f)
    log = add_responses_to_sheet(log, form_mapping)
    
    with open("../data/human_log_attribution.jsonl", "w") as f:
        for item in log:
            f.write(json.dumps(item))
            f.write("\n")