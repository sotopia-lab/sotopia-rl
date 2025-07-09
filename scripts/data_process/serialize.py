from sotopia.database import EpisodeLog, episodes_to_jsonl

episodes: list[EpisodeLog] = EpisodeLog.find(EpisodeLog.tag=="Qwen2.5-7b-Instruct_vs_Qwen2.5-7b-Instruct-0510")

episodes_to_jsonl(episodes, "Qwen2.5-7b-Instruct_vs_Qwen2.5-7b-Instruct-0510.jsonl")
