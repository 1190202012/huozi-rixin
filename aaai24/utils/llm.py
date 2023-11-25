from typing import List, Optional
import torch
import math
from tqdm import tqdm
import transformers
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader
from .openai_tools import prompt_openai_api, OPENAI_MODEL_LIST
from .modeling_reward_model import RewardModel
from torch.nn.parallel import DistributedDataParallel as DDP


class LLM:
    """"
    除非使用OPENAI API，否则lm_{generate, encode, reward}均需要传入gpu，没有默认。
    DataParallel这种一个进程多个GPU的，慢且复杂，建议DistributeDataParallel、torch.multiprocessing或Deepspeed
    """
    llm_config_dict = {}
    llms = {}
    openai_usage_log = None
    gpu_ids = []
    ddp = False

    @classmethod
    def get_llm_config(cls, _config):
        for value in _config.values():
            cls.llm_config_dict[value["model_name"]] = value

    @classmethod
    def initial_all(cls, gpu, llm_names):
        device = torch.device(f"cuda:{gpu.replace('gpu', '')}")
        for llm_name in llm_names.replace(" ", "").split(","):
            llm_config = cls.llm_config_dict[llm_name]

            if "reward_model" not in llm_name:
                model_class = getattr(transformers, llm_config["model_class"])
                model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True,
                                                    torch_dtype=torch.float16 if llm_config["fp16"] else torch.float32,
                                                    device_map=gpu.replace("gpu", "cuda:"))
                if cls.ddp:
                    model = DDP(model, device_ids=[int(gpu.replace("gpu", ""))])

                model.eval()

                tokenizer_class = getattr(transformers, llm_config["tokenizer_class"])
                tokenizer_path = llm_config.get("tokenizer_path", llm_config["model_path"])
                tokenizer = tokenizer_class.from_pretrained(tokenizer_path, trust_remote_code=True)

                if (
                        "baichuan" in tokenizer.name_or_path or "llama" in tokenizer.name_or_path or "Llama" in tokenizer.name_or_path) and tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.unk_token

                if "padding_side" in llm_config:
                    tokenizer.padding_side = llm_config["padding_side"]

                cls.llms[gpu + ":" + llm_config["model_name"]] = (model, tokenizer)

                print(
                    f"Successfully initial {gpu + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")
            else:
                if "llama2" in llm_config["model_name"]:
                    base_model = LlamaModel.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
                    tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", fast_tokenizer=True)

                    tokenizer.pad_token = tokenizer.eos_token
                    base_model.config.end_token_id = tokenizer.eos_token_id
                    base_model.config.pad_token_id = tokenizer.pad_token_id

                    base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

                    reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

                    reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')),
                                                 strict=False)

                    if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                        reward_model.half()

                    reward_model.to(device)

                    if cls.ddp:
                        reward_model = DDP(reward_model, device_ids=[int(gpu.replace("gpu", ""))])

                    reward_model.eval()

                    cls.llms[gpu + ":" + llm_config["model_name"]] = (reward_model, tokenizer)

                    print(
                        f"Successfully initial {gpu + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")
                elif "xverse" in llm_config["model_name"]:
                    base_model = LlamaModel.from_pretrained("xverse/XVERSE-7B-Chat", trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-7B-Chat", fast_tokenizer=True)

                    tokenizer.pad_token = tokenizer.eos_token
                    base_model.config.end_token_id = tokenizer.eos_token_id
                    base_model.config.pad_token_id = base_model.config.eos_token_id

                    base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

                    reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

                    reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')),
                                                 strict=False)

                    if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                        reward_model.half()

                    reward_model.to(device)

                    if cls.ddp:
                        reward_model = DDP(reward_model, device_ids=[int(gpu.replace("gpu", ""))])

                    reward_model.eval()

                    cls.llms[gpu + ":" + llm_config["model_name"]] = (reward_model, tokenizer)

                    print(
                        f"Successfully initial {gpu + ':' + llm_config['model_name']}. {len(cls.llms)}/{len(llm_names.split(','))}")

    @classmethod
    def initial_lm(cls, model_name, gpu):
        if gpu + ":" + model_name in cls.llms.keys():
            return gpu + ":" + model_name

        llm_config = cls.llm_config_dict[model_name]

        model_class = getattr(transformers, llm_config["model_class"])
        model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True,
                                            torch_dtype=torch.float16 if llm_config["fp16"] else torch.float32,
                                            device_map=gpu.replace("gpu", "cuda:"))
        if cls.ddp:
            model = DDP(model, device_ids=[int(gpu.replace("gpu", ""))])

        model.eval()

        tokenizer_class = getattr(transformers, llm_config["tokenizer_class"])
        tokenizer_path = llm_config.get("tokenizer_path", llm_config["model_path"])
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, trust_remote_code=True)

        if (
                "baichuan" in tokenizer.name_or_path or "llama" in tokenizer.name_or_path or "Llama" in tokenizer.name_or_path) and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        if "padding_side" in llm_config:
            tokenizer.padding_side = llm_config["padding_side"]

        cls.llms[gpu + ":" + model_name] = (model, tokenizer)

        print(f"Successfully initial {gpu + ':' + model_name}")

        return gpu + ":" + model_name

    @classmethod
    def initial_rm(cls, model_name, gpu):
        if gpu + ":" + model_name in cls.llms.keys():
            return gpu + ":" + model_name

        device = torch.device(f"cuda:{gpu.replace('gpu', '')}")
        llm_config = cls.llm_config_dict[model_name]

        if "llama2" in model_name:
            base_model = LlamaModel.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
            tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", fast_tokenizer=True)

            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.end_token_id = tokenizer.eos_token_id
            base_model.config.pad_token_id = tokenizer.pad_token_id

            base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

            reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

            reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')),
                                         strict=False)

            if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                reward_model.half()

            reward_model.to(device)

            if cls.ddp:
                reward_model = DDP(reward_model, device_ids=[int(gpu.replace("gpu", ""))])

            reward_model.eval()

            cls.llms[gpu + ":" + model_name] = (reward_model, tokenizer)
        elif "xverse" in model_name:
            base_model = LlamaModel.from_pretrained("xverse/XVERSE-7B-Chat", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-7B-Chat", fast_tokenizer=True)

            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.end_token_id = tokenizer.eos_token_id
            base_model.config.pad_token_id = base_model.config.eos_token_id

            base_model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

            reward_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0)

            reward_model.load_state_dict(torch.load(llm_config["model_path"], map_location=torch.device('cpu')),
                                         strict=False)

            if llm_config["fp16"] and reward_model.rwtranrsformer.dtype == torch.float32:
                reward_model.half()

            reward_model.to(device)

            if cls.ddp:
                reward_model = DDP(reward_model, device_ids=[int(gpu.replace("gpu", ""))])

            reward_model.eval()

            cls.llms[gpu + ":" + model_name] = (reward_model, tokenizer)

        print(f"Successfully initial {gpu + ':' + model_name}")

        return gpu + ":" + model_name

    @classmethod
    def release_one(cls, model_name):
        del cls.llms[model_name][0]
        del cls.llms[model_name][1]

    @classmethod
    def release_all(cls):
        for llm in cls.llms.values():
            model, tokenizer = llm
            del model
            del tokenizer

        cls.llms = {}

        if cls.openai_usage_log is not None:
            cls.openai_usage_log.close()

    @classmethod
    def lm_generate(cls, **kwargs):
        if kwargs["model_name"] in OPENAI_MODEL_LIST:
            if cls.openai_usage_log is None:
                cls.openai_usage_log = open("openai_usage.jsonl", "a", encoding="UTF-8")

            kwargs["usage_log"] = cls.openai_usage_log
            kwargs["messages"] = kwargs["prompts"]
            if "max_new_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs["max_new_tokens"]
            generated_sequences, _ = prompt_openai_api(**kwargs)
            return generated_sequences
        else:
            model_name = cls.initial_lm(kwargs["model_name"], kwargs["gpu"])
            model, tokenizer = cls.llms[model_name]
            generated_sequences = cls._frozen_lm_generate(model, tokenizer, kwargs["prompts"],
                                                          kwargs["tokenize_kwargs"], kwargs["generate_kwargs"])
            return generated_sequences

    @classmethod
    def _frozen_lm_generate(cls, model, tokenizer, prompts, tokenize_kwargs, generate_kwargs) -> List[str]:
        if type(prompts) is str:
            prompts = [prompts]

        module = model.module if LLM.ddp else model
        max_sequence_length = max(getattr(module.config, "max_position_embeddings", 0),
                                  getattr(module.config, "n_positions", 0), getattr(module.config, "seq_length", 0))
        
        if max_sequence_length == 0:
            max_sequence_length = 2048
        
        max_prompt_length = max_sequence_length - generate_kwargs["max_new_tokens"]

        tokenize_kwargs["max_length"] = max_prompt_length
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["return_tensors"] = "pt"

        tokenize_prompt = tokenizer(prompts, **tokenize_kwargs)

        batch_size = generate_kwargs.pop("batch_size", 10)

        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids = tokenize_prompt["input_ids"].to(module.device)
                attention_mask = tokenize_prompt["attention_mask"].to(module.device)
                output_sequences = module.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                   **generate_kwargs)

                if module.config._name_or_path in ["HIT-SCIR/huozi-7b-rlhf", "HIT-SCIR/huozi-7b-sft"]:
                    raw_output_str = tokenizer.batch_decode(output_sequences)
                    output_str = []
                    for r in raw_output_str:
                        temp = r.split("<|beginofutterance|>助手\n")[1].split("<|endofutterance|>")[0]
                        output_str.append(temp)
                else:
                    output_seq = output_sequences[:, input_ids.shape[1]:]
                    output_str = tokenizer.batch_decode(output_seq, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)

                return output_str
            else:
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,
                                                                pad_to_multiple_of=8 if module.dtype == torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokenize_prompt.data), batch_size=batch_size,
                                        collate_fn=data_collator)
                generated_sequences = []

                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.device)
                    attention_mask = batch["attention_mask"].to(module.device)
                    output_sequences = module.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                       **generate_kwargs)

                    if module.config._name_or_path in ["HIT-SCIR/huozi-7b-rlhf", "HIT-SCIR/huozi-7b-sft"]:
                        raw_output_str = tokenizer.batch_decode(output_sequences)
                        output_str = []
                        for r in raw_output_str:
                            a = r.split("<|beginofutterance|>助手\n")
                            if len(a) <= 1:
                                print(r)
                                temp = ""
                            else:
                                b = a[1]
                                c = b.split("<|endofutterance|>")
                                if len(c) == 0:
                                    print(r)
                                    temp = "I don't know"
                                else:
                                    temp = c[0]
                            output_str.append(temp)
                            
                            # temp = r.split("<|beginofutterance|>助手\n")[1].split("<|endofutterance|>")[0]
                            # output_str.append(temp)
                    else:
                        output_seq = output_sequences[:, input_ids.shape[1]:]
                        output_str = tokenizer.batch_decode(output_seq, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=True)
                    generated_sequences.extend(output_str)
                return generated_sequences

    @classmethod
    def lm_encode(cls, model_name, prompts, gpu, tokenize_kwargs, encode_kwargs):
        if type(prompts) is str:
            prompts = [prompts]

        model_name = cls.initial_lm(model_name, gpu)

        model, tokenizer = cls.llms[model_name]

        tokenize_kwargs["return_tensors"] = "pt"
        tokens = tokenizer(prompts, **tokenize_kwargs)

        batch_size = encode_kwargs.get("batch_size", 10)
        pooling_method = encode_kwargs.get("pooling_method", None)

        assert pooling_method in ["mean", "sum", "max", "cls"] if pooling_method is not None else True

        module = model.module if cls.ddp else model

        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids, attention_mask = tokens["input_ids"].to(module.device), tokens["attention_mask"].to(
                    module.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                output = LLM.pooling(output, pooling_method, attention_mask).cpu()
            else:
                outputs = []
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,
                                                                pad_to_multiple_of=8 if module.dtype == torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokens.data), batch_size=batch_size, collate_fn=data_collator)
                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.device)
                    attention_mask = batch["attention_mask"].to(module.device)
                    output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    output = LLM.pooling(output, pooling_method, attention_mask).cpu()
                    outputs.append(output)
                output = torch.cat(outputs)
        return output

    @staticmethod
    def pooling(output, pooling_method, attention_mask=None):
        if pooling_method is None:
            return output
        elif pooling_method == "cls":
            return output[:, 0, :]

        assert attention_mask is not None, "For pooling_method in [mean, sum, max], attention_mask is needed."

        no_padding_output = output.masked_fill(~attention_mask[..., None].bool(), 0.)

        if pooling_method == "mean":
            output = no_padding_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif pooling_method == "sum":
            output = no_padding_output.sum(dim=1)
        elif pooling_method == "max":
            output = no_padding_output.max(dim=1)

        return output

    @classmethod
    def lm_reward(cls, model_name, prompts, gpu, tokenize_kwargs, reward_kwargs):
        if type(prompts) is str:
            prompts = [prompts]

        model_name = cls.initial_rm(model_name, gpu)

        model, tokenizer = cls.llms[model_name]

        tokenize_kwargs["return_tensors"] = "pt"
        tokens = tokenizer(prompts, **tokenize_kwargs)

        batch_size = reward_kwargs.get("batch_size", 10)

        module = model.module if cls.ddp else model

        with torch.no_grad():
            if len(prompts) <= batch_size:
                input_ids, attention_mask = tokens["input_ids"].to(module.rwtranrsformer.device), tokens[
                    "attention_mask"].to(module.rwtranrsformer.device)
                score_list = module.forward_value(input_ids=input_ids, attention_mask=attention_mask, prompt_length=2)
            else:
                score_list = []
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,
                                                                pad_to_multiple_of=8 if module.rwtranrsformer.dtype == torch.float16 else None)
                dataloader = DataLoader(Dataset.from_dict(tokens.data), batch_size=batch_size, collate_fn=data_collator)
                for batch in tqdm(dataloader):
                    input_ids = batch["input_ids"].to(module.rwtranrsformer.device)
                    attention_mask = batch["attention_mask"].to(module.rwtranrsformer.device)
                    score_list.extend(
                        module.forward_value(input_ids=input_ids, attention_mask=attention_mask, prompt_length=2))

        return score_list
