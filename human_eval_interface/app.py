import json
import os
import re
import glob
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "human_eval"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

AI_ENABLED = bool(os.environ.get("OPENAI_API_KEY"))
AI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")

DIMENSION_META = {
    "goal": {"range": "[0-10]", "desc": "Goal achievement — how well the agent achieved their stated social goal. 0 = nothing achieved, 10 = all goals fully achieved."},
    "believability": {"range": "[0-10]", "desc": "Believability — how natural, realistic, and character-consistent the agent's behavior is. 0 = robotic/incoherent, 10 = perfectly human-like."},
    "knowledge": {"range": "[0-10]", "desc": "Knowledge — how much new, relevant information the agent gained. 0 = nothing new, 10 = exceptional new knowledge."},
    "secret": {"range": "[-10 to 0]", "desc": "Secret keeping — how well the agent kept their secret. 0 = fully kept, -10 = completely revealed."},
    "relationship": {"range": "[-5 to 5]", "desc": "Relationship — change in relationship quality. -5 = severely damaged, 0 = unchanged, 5 = greatly improved."},
    "social_rules": {"range": "[-10 to 0]", "desc": "Social rules — adherence to social norms and legal rules. 0 = no violations, -10 = extreme violations."},
    "financial": {"range": "[-5 to 5]", "desc": "Financial/material benefits — net financial outcome. -5 = severe loss, 0 = neutral, 5 = major gain."},
}


def build_ai_prompt(episode: dict, agents: list[str], agent_turns: dict) -> str:
    """Build a prompt asking GPT to score all dimensions and attributions."""
    lines = []
    lines.append("You are an expert annotator for social simulation conversations.")
    lines.append("Given the following conversation episode, provide scores and reasoning for each dimension and each agent.")
    lines.append("")
    lines.append(f"## Scenario: {episode['scenario']}")
    lines.append(f"## Codename: {episode['codename']}")
    lines.append("")

    for a in episode["agents"]:
        lines.append(f"### {a['name']}")
        lines.append(a["background"])
        lines.append(f"Goal: {episode['goals'].get(a['name'], 'N/A')}")
        lines.append("")

    lines.append("## Conversation")
    for i, t in enumerate(episode["turns"]):
        prefix = "[action]" if t["is_action"] else ""
        lines.append(f"Turn {i} ({t['speaker']}){prefix}: {t['utterance']}")
    lines.append("")

    lines.append("## Dimensions to evaluate")
    for key, meta in DIMENSION_META.items():
        lines.append(f"- **{key}** {meta['range']}: {meta['desc']}")
    lines.append("")

    lines.append("## Instructions")
    lines.append(f"Agent 1 (first speaker) = {agents[0]}")
    lines.append(f"Agent 2 (second speaker) = {agents[1]}")
    lines.append("")
    lines.append("For EACH dimension, provide:")
    lines.append("1. A per-agent score within the stated range, with 1-2 sentence reasoning that explains why this score was chosen AND why adjacent/alternative scores were not appropriate.")
    lines.append("2. A per-utterance attribution score (0-3) for each agent's utterances only, with 1-2 sentence reasoning per utterance that explains why this score was chosen AND why a higher or lower score would not fit.")
    lines.append("   - 0 = no impact, 1 = minor, 2 = significant, 3 = critical (assign 3 to at most one utterance per agent unless equally critical).")
    lines.append("")

    agent_turn_info = {}
    for idx, a in enumerate(agents):
        turns = agent_turns.get(a, [])
        agent_turn_info[a] = [(t["localIdx"], t["globalIdx"]) for t in turns]

    lines.append("## Agent utterance mapping (for attribution keys)")
    for idx, a in enumerate(agents):
        for local_idx, global_idx in agent_turn_info[a]:
            lines.append(f"  agent_{idx+1}_turn_{local_idx} = Turn {global_idx} by {a}")
    lines.append("")

    lines.append('Respond with ONLY a JSON object (no markdown fences) in this exact structure:')
    lines.append('{')
    lines.append('  "dimension_scores": {')
    lines.append('    "<dim_key>": {')
    lines.append(f'      "{agents[0]}": {{"score": <int>, "reasoning": "<1-2 sentences: why this score, and why not higher/lower>"}},')
    lines.append(f'      "{agents[1]}": {{"score": <int>, "reasoning": "<1-2 sentences: why this score, and why not higher/lower>"}}')
    lines.append('    }, ...')
    lines.append('  },')
    lines.append('  "dimension_attributions": {')
    lines.append('    "<dim_key>": {')
    lines.append('      "agent_1_turn_0": {"score": <int 0-3>, "reasoning": "<1-2 sentences: why this score, and why not higher/lower>"},')
    lines.append('      "agent_2_turn_0": {"score": <int 0-3>, "reasoning": "<1-2 sentences: why this score, and why not higher/lower>"}, ...')
    lines.append('    }, ...')
    lines.append('  }')
    lines.append('}')

    return "\n".join(lines)


def parse_episode(md_text: str) -> dict:
    """Parse a markdown episode file into structured data."""
    episode: dict = {"scenario": "", "codename": "", "agents": [], "goals": {}, "turns": []}

    lines = md_text.strip().split("\n")

    scenario_match = re.search(r"^# Scenario\s*\n+(.+)", md_text, re.MULTILINE)
    if scenario_match:
        episode["scenario"] = scenario_match.group(1).strip()

    codename_match = re.search(r"\*\*Codename:\*\*\s*(.+)", md_text)
    if codename_match:
        episode["codename"] = codename_match.group(1).strip()

    agent_blocks = re.findall(
        r"\*\*([A-Z][a-z]+ [A-Z][a-zA-Z'-]+)\*\*\n(.+?)(?=\n\*\*[A-Z]|\n## |\Z)",
        md_text,
        re.DOTALL,
    )
    for name, bio in agent_blocks:
        if "Codename" not in name:
            episode["agents"].append({"name": name, "background": bio.strip()})

    goal_matches = re.findall(
        r"- \*\*(.+?)\*\*:\s*(.+?)(?=\n- \*\*|\n\n|\n##|\Z)", md_text, re.DOTALL
    )
    for name, goal in goal_matches:
        episode["goals"][name] = goal.strip()

    for line in lines:
        line = line.strip()
        if not line.startswith(">"):
            continue
        content = line[1:].strip()
        said_match = re.match(r"(.+?)\s+said:\s*(.+)", content)
        action_match = re.match(r"(.+?):\s*(left the conversation.*)", content)
        if said_match:
            speaker = said_match.group(1).strip()
            utt = said_match.group(2).strip().strip('"').strip('\u201c').strip('\u201d')
            episode["turns"].append({"speaker": speaker, "utterance": utt, "is_action": False})
        elif action_match:
            speaker = action_match.group(1).strip()
            utt = action_match.group(2).strip()
            episode["turns"].append({"speaker": speaker, "utterance": utt, "is_action": True})

    return episode


def get_splits() -> list[str]:
    splits = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("split_"):
            splits.append(d.name)
    return splits


def get_episode_ids(split: str) -> list[str]:
    split_dir = DATA_DIR / split
    ids = []
    for f in sorted(split_dir.glob("*.md")):
        ids.append(f.stem)
    return ids


def load_annotation(split: str, episode_id: str, annotator: str) -> dict | None:
    path = ANNOTATIONS_DIR / annotator / split / f"{episode_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_annotation(split: str, episode_id: str, annotator: str, data: dict):
    out_dir = ANNOTATIONS_DIR / annotator / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{episode_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/splits")
def api_splits():
    return jsonify(get_splits())


@app.route("/api/episodes/<split>")
def api_episodes(split: str):
    annotator = request.args.get("annotator", "")
    ids = get_episode_ids(split)
    result = []
    for eid in ids:
        ann = load_annotation(split, eid, annotator) if annotator else None
        result.append({"id": eid, "annotated": ann is not None})
    return jsonify(result)


@app.route("/api/episode/<split>/<episode_id>")
def api_episode(split: str, episode_id: str):
    md_path = DATA_DIR / split / f"{episode_id}.md"
    if not md_path.exists():
        return jsonify({"error": "not found"}), 404
    with open(md_path) as f:
        md_text = f.read()
    episode = parse_episode(md_text)
    episode["id"] = episode_id
    episode["split"] = split
    return jsonify(episode)


@app.route("/api/annotation/<split>/<episode_id>", methods=["GET"])
def api_get_annotation(split: str, episode_id: str):
    annotator = request.args.get("annotator", "default")
    ann = load_annotation(split, episode_id, annotator)
    if ann is None:
        return jsonify(None)
    return jsonify(ann)


@app.route("/api/annotation/<split>/<episode_id>", methods=["POST"])
def api_save_annotation(split: str, episode_id: str):
    annotator = request.json.get("annotator", "default")

    md_path = DATA_DIR / split / f"{episode_id}.md"
    agent_1_name, agent_2_name = "", ""
    if md_path.exists():
        with open(md_path) as f:
            ep = parse_episode(f.read())
        seen = []
        for t in ep["turns"]:
            if t["speaker"] not in seen:
                seen.append(t["speaker"])
            if len(seen) == 2:
                break
        agent_1_name = seen[0] if len(seen) > 0 else ""
        agent_2_name = seen[1] if len(seen) > 1 else ""

    data = {
        "episode_id": episode_id,
        "split": split,
        "annotator": annotator,
        "agent_1_name": agent_1_name,
        "agent_2_name": agent_2_name,
        "dimension_scores": request.json.get("dimension_scores", {}),
        "dimension_attributions": request.json.get("dimension_attributions", {}),
    }
    save_annotation(split, episode_id, annotator, data)
    return jsonify({"status": "ok"})


@app.route("/api/ai-enabled")
def api_ai_enabled():
    return jsonify({"enabled": AI_ENABLED})


@app.route("/api/ai-suggest/<split>/<episode_id>", methods=["POST"])
def api_ai_suggest(split: str, episode_id: str):
    if not AI_ENABLED:
        return jsonify({"error": "AI suggestions not enabled (no OPENAI_API_KEY)"}), 400

    from openai import OpenAI
    client = OpenAI()

    md_path = DATA_DIR / split / f"{episode_id}.md"
    if not md_path.exists():
        return jsonify({"error": "episode not found"}), 404

    with open(md_path) as f:
        episode = parse_episode(f.read())

    seen = []
    for t in episode["turns"]:
        if t["speaker"] not in seen:
            seen.append(t["speaker"])
        if len(seen) == 2:
            break
    agents = seen if len(seen) == 2 else [a["name"] for a in episode["agents"]]

    agent_turns: dict[str, list] = {a: [] for a in agents}
    counters = {a: 0 for a in agents}
    for i, t in enumerate(episode["turns"]):
        if t["speaker"] in agent_turns:
            agent_turns[t["speaker"]].append({
                "globalIdx": i, "localIdx": counters[t["speaker"]], **t
            })
            counters[t["speaker"]] += 1

    prompt = build_ai_prompt(episode, agents, agent_turns)

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/api/progress/<split>")
def api_progress(split: str):
    annotator = request.args.get("annotator", "default")
    ids = get_episode_ids(split)
    annotated = sum(1 for eid in ids if load_annotation(split, eid, annotator) is not None)
    return jsonify({"total": len(ids), "annotated": annotated})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
