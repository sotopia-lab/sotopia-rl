import json
import os
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Tuple
from sotopia_rl.prompter.attribution_methods import ATTRIBUTION_METHOD_DICT
import jsonlines
from openai import OpenAI
import asyncio
import aiofiles
from collections import Counter
from tqdm.asyncio import tqdm_asyncio  # tqdm helper for async iterators

client = OpenAI()

def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content


def parse_conversation(
    episode: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Extract and parse conversation and goals from the episode."""
    conversation = episode["social_interactions"].split("\n\n")
    goals = episode["social_goals"]
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for utterance in conversation:
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append(
            (speaker, utterance[len(speaker) + 1 :].strip())
        )  # Strip the speaker from the utterance
    return parsed_conversation, goals


def get_key_utterance_dict(conversation: List[Tuple[str, str]]) -> Dict[str, List[Any]]:
    key_utterance_dict: Dict[str, List[Any]] = OrderedDict()
    for i, (speaker, utterance) in enumerate(conversation):
        key_utterance_dict[f"Utterance {i//2} by {speaker}"] = [
            utterance,
            0,
        ]
    return key_utterance_dict


def generate_reward_attribution(
    data_dir: str,
    llm_name: str = "gpt-3.5-turbo",
    input_file: str = "example_episodes_with_scores.jsonl",
    output_file: str = "openai_log_attribution.jsonl",
    attribution_method_name: str = "direct",
    attribution_instruction_name: str = "default",
) -> None:
    with jsonlines.open(os.path.join(data_dir, input_file), "r") as reader:
        data = list(reader)

    if os.path.exists(os.path.join(data_dir, output_file)):
        with jsonlines.open(os.path.join(data_dir, output_file), "r") as reader:
            finished_episodes = list(reader)
    else:
        finished_episodes = []

    finished_episode_ids = Counter(
        [episode["episode_id"] for episode in finished_episodes]
    )
    print(f"Number of episodes in total: ", len(data))
    print(f"Number of episodes finished: ", len(finished_episodes))

    results = finished_episodes
    get_attribution_single_conv = ATTRIBUTION_METHOD_DICT[attribution_method_name]
    for episode in tqdm(data):
        if (
            episode["episode_id"] in finished_episode_ids
            and finished_episode_ids[episode["episode_id"]] > 1
        ):
            print(f"finished episode {episode['episode_id']}")
            continue
        elif (
            episode["episode_id"] in finished_episode_ids
            and finished_episode_ids[episode["episode_id"]] == 1
        ):
            results.pop()  # rerun the unfinished episode pair
            finished_episode_ids[episode["episode_id"]] -= 1
            print(f"Incomplete episode. Rerun episode {episode['episode_id']}")

        # starting from here
        conversation, goals = parse_conversation(episode)
        agents = list(goals.keys())
        for agent in agents:
            key_utterance_dict = get_key_utterance_dict(conversation)
            attribution_rewards = get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name)

            for key in key_utterance_dict:
                if agent in key and key in attribution_rewards:
                    key_utterance_dict[key][1] = attribution_rewards[key]
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "scenario": episode["scenario"],
                    "agent": agent,
                    "goal": goals[agent],
                    "attributed_utterances": key_utterance_dict,
                    "is_first_speaker": agent == agents[0],
                    "goal_score": episode["scores"][agent],
                }
            )
            with open(os.path.join(data_dir, output_file), "a") as f:
                f.write(json.dumps(results[-1]) + "\n")

async def process_episode(
    episode,
    get_attribution_single_conv,
    llm_name,
    attribution_instruction_name,
    data_dir,
    output_file,
    file_lock
):
    # Parse conversation and goals for the current episode.
    conversation, goals = parse_conversation(episode)
    agents = list(goals.keys())
    
    # Create a list to store all agent results for batch writing
    agent_results = []
    
    # Process each agent in the episode.
    for agent in agents:
        key_utterance_dict = get_key_utterance_dict(conversation)
        # Run the (potentially blocking) attribution function in a separate thread.
        attribution_rewards = await asyncio.to_thread(
            get_attribution_single_conv,
            conversation,
            agent,
            goals,
            episode,
            llm_name,
            attribution_instruction_name
        )
        # Update the key_utterance_dict with the attribution rewards.
        for key in key_utterance_dict:
            if agent in key and key in attribution_rewards:
                key_utterance_dict[key][1] = attribution_rewards[key]
        
        # Prepare the result entry.
        result_entry = {
            "episode_id": episode["episode_id"],
            "scenario": episode["scenario"],
            "agent": agent,
            "goal": goals[agent],
            "attributed_utterances": key_utterance_dict,
            "is_first_speaker": agent == agents[0],
            "goal_score": episode["scores"][agent],
        }
        
        agent_results.append(result_entry)
    
    # Write all agent results at once to reduce I/O operations
    async with file_lock:
        async with aiofiles.open(os.path.join(data_dir, output_file), "a") as f:
            for result in agent_results:
                await f.write(json.dumps(result) + "\n")

def parallel_generate_reward_attribution(
    data_dir: str,
    llm_name: str = "gpt-3.5-turbo",
    input_file: str = "example_episodes_with_scores.jsonl",
    output_file: str = "openai_log_attribution.jsonl",
    attribution_method_name: str = "direct",
    attribution_instruction_name: str = "default",
    max_concurrency: int = 10  # Increased to 10 as requested
) -> None:
    # Load the episodes from the input file.
    input_path = os.path.join(data_dir, input_file)
    with jsonlines.open(input_path, "r") as reader:
        data = list(reader)
    
    # Load finished episodes if the output file exists.
    output_path = os.path.join(data_dir, output_file)
    if os.path.exists(output_path):
        # Use a more efficient approach to identify finished episodes
        finished_episode_ids = set()
        with jsonlines.open(output_path, "r") as reader:
            for episode in reader:
                finished_episode_ids.add(episode["episode_id"])
    else:
        finished_episode_ids = set()
    
    # Count the number of unique episodes
    total_episodes = len(data)
    finished_count = len(finished_episode_ids)
    
    print(f"Number of episodes in total: {total_episodes}")
    print(f"Number of episodes finished: {finished_count}")
    
    # Get the attribution function based on the method name.
    get_attribution_single_conv = ATTRIBUTION_METHOD_DICT[attribution_method_name]
    
    # Prepare a lock for asynchronous file writes.
    file_lock = asyncio.Lock()
    
    # Create a semaphore to limit concurrency.
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Create a task queue for better management of concurrent tasks
    task_queue = []
    
    async def sem_task(episode):
        # Ensure that only a limited number of episodes are processed concurrently.
        async with semaphore:
            await process_episode(
                episode,
                get_attribution_single_conv,
                llm_name,
                attribution_instruction_name,
                data_dir,
                output_file,
                file_lock
            )
    
    # Filter out episodes that are already finished
    episodes_to_process = [episode for episode in data if episode["episode_id"] not in finished_episode_ids]
    
    # Create tasks for episodes that need processing
    for episode in episodes_to_process:
        task_queue.append(sem_task(episode))
    
    # Process all episodes concurrently with a progress bar.
    if task_queue:
        print(f"Processing {len(task_queue)} episodes with {max_concurrency} concurrent tasks...")
        # await tqdm_asyncio.gather(*task_queue)
        asyncio.run(tqdm_asyncio.gather(*task_queue))
    else:
        print("No episodes to process.")