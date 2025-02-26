import pickle

intention_config_dir = '../PIEPredict-master/data/pie/intention/context_loc_pretrained/configs.pkl'
with open(intention_config_dir, "rb") as f:
    intention_configs = pickle.load(f)

print(intention_configs)

pedestrian_intents = '../PIEPredict-master/data/pie/intention/context_loc_pretrained/ped_intents.pkl'

with open(pedestrian_intents, "rb") as f:
    ped_intent = pickle.load(f)


# print(ped_intent.keys())
# for k in ped_intent.keys():
#     print(ped_intent[k])
#     print("-"*100)