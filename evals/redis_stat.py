from sotopia import EpisodeLog

# find episode log by tag
Episodes = EpisodeLog.find(EpisodeLog.tag == "aug20_gpt4_llama-2-70b-chat_zqi2").all()
len(Episodes)  ## Episode Log