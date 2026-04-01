# Sotopia Annotation Interface

A lightweight web UI for annotating Sotopia simulation conversation episodes across 7 evaluation dimensions, with per-utterance reward attribution and optional AI-assisted suggestions.

## Setup

```bash
cd human_eval_interface
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5050 in your browser.

### AI Suggestions (optional)

To enable AI-assisted scoring, add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=o4-mini   # optional, defaults to o4-mini
```

When enabled, an "Enable AI suggestions" toggle appears in the sidebar. Clicking "Ask AI" on an episode sends the conversation to the model and displays suggested scores with reasoning inline next to each slider. You can then "Apply AI Scores" to prefill all sliders, or ignore them entirely.

## Usage

1. Enter your annotator name in the sidebar.
2. Select a split (`split_1` or `split_2`).
3. Click an episode to view the codename, scenario, agent backgrounds, goals, and conversation.
4. For each of the 7 dimensions, annotate:
   - **Dimension Score** — per agent, within the dimension's range.
   - **Reward Attribution (0-3)** — per utterance, how much it contributed to the dimension outcome.
5. Click **Save Annotation**. Use Previous/Next to navigate between episodes.

Green dots in the sidebar indicate completed annotations. The progress bar tracks overall completion.

## Dimensions

| Dimension | Range | Description |
|-----------|-------|-------------|
| Goal Achieving (GOAL) | 0 to 10 | How well the agent achieved their stated social goal |
| Believability (BEL) | 0 to 10 | How natural, realistic, and character-consistent the behavior is |
| Knowledge (KNO) | 0 to 10 | How much new, relevant information the agent gained |
| Secret Keeping (SEC) | -10 to 0 | How well the agent kept their secret (0 = fully kept) |
| Relationship (REL) | -5 to 5 | Change in relationship quality (positive = improved) |
| Social Rules (SOC) | -10 to 0 | Adherence to social norms and legal rules (0 = no violations) |
| Financial & Material (FIN) | -5 to 5 | Net financial/material outcome (positive = gain) |

Each dimension has a foldable scoring guide and attribution guide in the UI.

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
  "agent_1_name": "Mia Davis",
  "agent_2_name": "Benjamin Jackson",
  "dimension_scores": {
    "goal": { "Mia Davis": 2, "Benjamin Jackson": 10 },
    "believability": { "Mia Davis": 9, "Benjamin Jackson": 10 },
    "knowledge": { "Mia Davis": 0, "Benjamin Jackson": 0 },
    "secret": { "Mia Davis": 0, "Benjamin Jackson": 0 },
    "relationship": { "Mia Davis": -2, "Benjamin Jackson": -4 },
    "social_rules": { "Mia Davis": 0, "Benjamin Jackson": 0 },
    "financial": { "Mia Davis": 0, "Benjamin Jackson": 0 }
  },
  "dimension_attributions": {
    "goal": {
      "agent_1_turn_0": 1, "agent_1_turn_1": 3, "agent_1_turn_2": 0,
      "agent_2_turn_0": 3, "agent_2_turn_1": 1
    },
    "believability": { "..." : "..." },
    "knowledge": { "..." : "..." },
    "secret": { "..." : "..." },
    "relationship": { "..." : "..." },
    "social_rules": { "..." : "..." },
    "financial": { "..." : "..." }
  }
}
```

- `agent_1` = first speaker, `agent_2` = second speaker.
- Attribution keys follow the pattern `agent_{1,2}_turn_{n}` where `n` is the agent's utterance index (0-based, counting only that agent's turns).

## Data

Episode markdown files are read from `human_eval/split_1/` and `human_eval/split_2/`. The interface does not modify the source data.
