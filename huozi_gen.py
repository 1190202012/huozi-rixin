import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

zh_prompt = "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\n{user_input}<|endofutterance|>\n<|beginofutterance|>助手\n"

examples = ["杭州亚运会中国金牌数是多少？",'北京的首都是哪里','when was the last time anyone was on the moon']

generate_kwargs = {
    "max_new_tokens": 200,
    "do_sample": True,
    # "repetition_penalty": 1.03,
    "top_k": 30,
    "top_p": 0.9,
}

example_list = [zh_prompt.replace("{user_input}", q) for q in examples]

model = AutoModelForCausalLM.from_pretrained("HIT-SCIR/huozi-7b-rlhf")
tokenizer = AutoTokenizer.from_pretrained("HIT-SCIR/huozi-7b-rlhf")

device = torch.device("cuda:0")
tokenizer.padding_side = "left"
tokens = tokenizer(example_list, padding="longest", return_tensors="pt")
tokens = tokens.to(device)
model.to(device)
outputs = model.generate(**tokens, **generate_kwargs)

raw_result = tokenizer.batch_decode(outputs)

print(raw_result)

result = []
for r in raw_result:
    temp = r.split("<|beginofutterance|>助手\n")[1].split("<|endofutterance|>")[0]
    result.append(temp)

print(result)

