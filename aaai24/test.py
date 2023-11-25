import math
from transformers import LlamaModel, AutoTokenizer
import torch

# import debugpy
# debugpy.connect(('192.168.1.50', 6789))
# debugpy.wait_for_client()
# debugpy.breakpoint()

import pydevd_pycharm

from utils.modeling_reward_model import RewardModel
pydevd_pycharm.settrace('192.168.1.50', port=6789, stdoutToServer=True, stderrToServer=True)


base_model = LlamaModel.from_pretrained("xverse/XVERSE-7B-Chat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-7B-Chat", fast_tokenizer=True)

tokenizer.pad_token = tokenizer.eos_token
base_model.config.end_token_id = tokenizer.eos_token_id
base_model.config.pad_token_id = base_model.config.eos_token_id

base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

reward_model.to(torch.device("cuda:0"))

state_dict = torch.load("./data/model/xverse_base_reward_model/pytorch_model.bin", map_location=torch.device('cpu'))

reward_model.load_state_dict()

print("hello")
