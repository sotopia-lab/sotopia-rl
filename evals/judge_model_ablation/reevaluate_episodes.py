import argparse
import asyncio
import json
from copy import deepcopy

from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    SotopiaDimensions,
)
from sotopia.envs.parallel import get_bio
from sotopia.messages.message_classes import ScriptBackground
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

POSSIBLE_MODELS = [
    "o3-mini",
    "claude/claude-3-5-haiku-20241022",
    "claude/claude-3-5-sonnet-20241022",
    "together_ai/deepseek-ai/DeepSeek-R1",
    "claude/claude-3-7-sonnet-20250219",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    "together_ai/deepseek-ai/DeepSeek-V3",
    "gpt-4o-mini",
    "gpt-4",
    "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "together_ai/deepseek-ai/DeepSeek-R1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    ]

SOTOPIA_HARD_ENVS = ["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]

def get_script_background(episode: EpisodeLog) -> ScriptBackground:
    environment = EnvironmentProfile.get(pk=episode.environment)
    relationship_type: RelationshipType = environment.relationship
    agent_id1 = episode.agents[0]
    agent_id2 = episode.agents[1]
    agent1 = AgentProfile.get(pk=agent_id1)
    agent2 = AgentProfile.get(pk=agent_id2)
    agent1_name = f"{agent1.first_name} {agent1.last_name}"
    agent2_name = f"{agent2.first_name} {agent2.last_name}"
    agent1_background = get_bio(relationship_type, agent1, 0)
    agent2_background = get_bio(relationship_type, agent2, 1)
    script_background = ScriptBackground(
        scenario=environment.scenario,
        p1_name=agent1_name,
        p2_name=agent2_name,
        p1_background=agent1_background,
        p2_background=agent2_background,
        p1_goal=environment.agent_goals[0],
        p2_goal=environment.agent_goals[1],
    )
    return script_background

def get_history(episode: EpisodeLog, script_background: ScriptBackground) -> str:
    messages = episode.render_for_humans()[1]
    selected_messages = messages[1:-2]
    first_message = messages[0].split("\n\n")[-1]
    selected_messages = [first_message] + selected_messages

    history = script_background.to_natural_language()
    for i, message in enumerate(selected_messages):
        history += f"\nTurn #{i}\n" + message
    return history

BANNED_EPI_IDS = [
    "01JQ91S0W3P9GGAW4MDZ8F31GC",
]

# def filter_episodes(episodes):
#     print(len(episodes))
#     env_agents_to_ep_dict = defaultdict(list)
#     for episode in episodes:
#         env_agents_to_ep_dict[(episode.environment, episode.agents[0], episode.agents[1])].append(episode)
#         env_agents_to_ep_dict[(episode.environment, episode.agents[1], episode.agents[0])].append(episode)

#     filtered_episodes = []
#     # visited_envs = set()
#     for episode in episodes:
#         if episode.environment not in SOTOPIA_HARD_ENVS:
#             continue
#         # if episode.environment in visited_envs:
#         #     continue
#         for banned_epi_id in BANNED_EPI_IDS:
#             if episode.pk == banned_epi_id:
#                 continue
#         if len(env_agents_to_ep_dict[(episode.environment, episode.agents[0], episode.agents[1])]) == 2:
#             filtered_episodes.extend(env_agents_to_ep_dict[(episode.environment, episode.agents[0], episode.agents[1])])
#             # visited_envs.add(episode.environment)
#     return filtered_episodes

def filter_episodes(episodes):
    return episodes

async def evaluate_episode(
    episode,
    model_name: str,
    evaluator,
    semaphore: asyncio.Semaphore,
    max_retry: int = 3,
):
    """Evaluate a single episode, retrying on empty result."""
    async with semaphore:                         # ← concurrency guard
        print(f"Processing episode {episode.pk} with model {model_name}")

        script_background = get_script_background(episode)
        history = get_history(episode, script_background)

        retries_left = max_retry
        while True:
            result = await evaluator.__acall__(0, None, history=history, temperature=0.0)
            if result:
                break

            retries_left -= 1
            if retries_left == 0:
                print(f"Failed to evaluate episode {episode.pk} after {max_retry} retries")
                result = {}
                break
            print(f"Retrying episode {episode.pk}, remaining retries: {retries_left}")

        models = deepcopy(episode.models)
        models[0] = model_name
        return episode.pk, {
            "episode_id": episode.pk,
            "models": models,
            "rating": result,
        }

async def run_eval_async(tag: str, model_name: str, max_concurrent: int = 10):
    """Async entry-point that evaluates all episodes for the given tag."""
    all_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
    print(f"Total episodes found: {len(all_episodes)}")

    filtered_episodes = filter_episodes(all_episodes)
    print(f"Filtered episodes: {len(filtered_episodes)}")

    evaluator = ReachGoalLLMEvaluator(
        model_name,
        EvaluationForTwoAgents[SotopiaDimensions],
    )

    sem = asyncio.Semaphore(max_concurrent)

    # Launch all evaluations under semaphore control
    tasks = [
        evaluate_episode(ep, model_name, evaluator, sem)
        for ep in filtered_episodes
    ]

    tqdm.write(f"Evaluating {len(tasks)} episodes with model {model_name}")
    results_list = await tqdm_asyncio.gather(
        *tasks, total=len(tasks), desc="Processing episodes"
    )

    results = {pk: data for pk, data in results_list}

    cache_path = f".cache/{tag}_eval_{model_name.replace('/', '_')}_results.json"
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {cache_path}")

# ----- optional synchronous wrapper -------------

def run_eval(tag: str, model_name: str, max_concurrent: int = 10):
    """Convenience sync wrapper (keeps notebooks / scripts simple)."""
    asyncio.run(run_eval_async(tag, model_name, max_concurrent))

TAG = "grpo_rm_goal_0511_step_2200_vs_sft_0510_epoch_500_step_200-0512"

def main(tag, model_name):
    run_eval(tag, model_name, 50)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Re-evaluate episodes with a specific model.")
    args.add_argument("--model_name", type=str, default="", help="Index of the model to use for re-evaluation.")
    args.add_argument("--tag", type=str, default=TAG, help="Tag of the episodes to re-evaluate.")
    args = args.parse_args()
    assert args.model_name in POSSIBLE_MODELS, f"Model name {args.model_name} is not in the list of possible models: {POSSIBLE_MODELS}"
    main(args.tag, args.model_name)
