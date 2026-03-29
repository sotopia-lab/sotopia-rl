import json
import os
import re
import glob
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "human_eval"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)


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


@app.route("/api/progress/<split>")
def api_progress(split: str):
    annotator = request.args.get("annotator", "default")
    ids = get_episode_ids(split)
    annotated = sum(1 for eid in ids if load_annotation(split, eid, annotator) is not None)
    return jsonify({"total": len(ids), "annotated": annotated})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
