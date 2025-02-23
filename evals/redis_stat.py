from sotopia.database.logs import EpisodeLog

# find episode log by tag
#Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_rejection-sampling-rm-baseline-and-sft_vs_sotopia_gemma-2-2b-it-sft-1108_sample_5").all()
Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1204_sample_40").all()
#Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_rejection-sampling-rm-key-utterance-and-sft_vs_sotopia_gemma-2-2b-it-sft-1109_sample_1").all()

#Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_gemma-2-2b-it-sft-ppo-rm-key_vs_sotopia_gemma-2-2b-it-sft-1010_v2").all()
#Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_gemma-2-2b-it-sft-ppo-rm-direct_vs_sotopia_gemma-2-2b-it-sft-1010").all()
#Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_gemma-2-2b-it-sft-ppo-rm-key_vs_sotopia_gemma-2-2b-it-sft-1010").all()

print(len(Episodes))  ## Episode Log

'''
for episode in Episodes:
    conversation = episode.messages[1:]
    rewards = episode.rewards
    if rewards == [0, 0]:
        EpisodeLog.delete(episode.pk)
    print(len(conversation))
    #if len(conversation) <=2 :
    #    EpisodeLog.delete(episode.pk)
'''
import pdb; pdb.set_trace()

tot_rewards1 = {'believability': 0.0, 'relationship': 0.0, 'knowledge': 0.0, 'secret': 0.0, 'social_rules': 0.0, 'financial_and_material_benefits': 0.0, 'goal': 0.0, 'overall_score': 0.0}
tot_rewards2 = {'believability': 0.0, 'relationship': 0.0, 'knowledge': 0.0, 'secret': 0.0, 'social_rules': 0.0, 'financial_and_material_benefits': 0.0, 'goal': 0.0, 'overall_score': 0.0}
episode_num = 0
Episodes = Episodes[-20:]
print(len(Episodes))
for episode in Episodes:
    rewards = episode.rewards
    print(rewards)
    conversation = episode.messages[1:]
    print(len(conversation))
    #if len(conversation) < 5:
    #    import pdb; pdb.set_trace()
    try:
        reward1, reward2 = rewards[0][-1], rewards[1][-1]
        for key, value in reward1.items():
            tot_rewards1[key] += value
        for key, value in reward2.items():
            tot_rewards2[key] += value
        episode_num += 1
    except Exception as e:
        print(e, episode.pk)
        pass

for key in tot_rewards1:
    tot_rewards1[key] /= episode_num

for key in tot_rewards2:
    tot_rewards2[key] /= episode_num

print(tot_rewards1)
print(tot_rewards2)
