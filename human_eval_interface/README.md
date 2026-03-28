# Sotopia Annotation Interface

A lightweight web UI for annotating Sotopia simulation conversation episodes with goal-achieving scores and utterance-level reward attribution.

## Setup

```bash
cd human_eval_interface
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5050 in your browser.

## Usage

1. Enter your annotator name in the sidebar.
2. Select a split (`split_1` or `split_2`).
3. Click an episode to view the scenario, agent backgrounds, goals, and conversation.
4. Annotate two sections:
   - **Goal Achieving Score (0-10)** — per agent, how well they achieved their stated goal.
   - **Reward Attribution (0-10)** — per utterance, how much it contributed to the agent's goal achievement.
5. Click **Save Annotation**. Use Previous/Next to navigate between episodes.

Green dots in the sidebar indicate completed annotations. The progress bar tracks your overall completion.

## Annotation Output

Annotations are saved as JSON files under:

```
human_eval_interface/annotations/{annotator}/{split}/{episode_id}.json
```

Example:

```json
{
  "episode_id": "01JTYE0E5PGKE232AZ2VBQH7GA",
  "split": "split_1",
  "annotator": "alice",
  "goal_scores": {
    "Mia Davis": 3,
    "Benjamin Jackson": 8
  },
  "reward_attribution": {
    "agent_1_turn_0": 5,
    "agent_1_turn_1": 2,
    "agent_2_turn_0": 7,
    "agent_2_turn_1": 3
  }
}
```

## Data

Episode markdown files are read from `human_eval/split_1/` and `human_eval/split_2/`. The interface itself lives in `human_eval_interface/` and does not modify the source data.
