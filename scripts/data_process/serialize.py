from sotopia.database import episodes_to_jsonl, EpisodeLog
 
episodes: list[EpisodeLog] = EpisodeLog.find(EpisodeLog.tag=="Qwen2.5-7b-Instruct_vs_Qwen2.5-7b-Instruct-0510")
 
episodes_to_jsonl(episodes, "Qwen2.5-7b-Instruct_vs_Qwen2.5-7b-Instruct-0510.jsonl")