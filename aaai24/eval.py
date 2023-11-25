import json
import math
import os
import torch.cuda
import torch.distributed as dist
import time
from loguru import logger
import emoji
import traceback

from aaai24.data import load_example
from aaai24.retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from aaai24.knowledge import SummarizeConstructor, ContriverConstructor
from aaai24.response import StandardGenerator
from aaai24.voting import StandardVoter
from aaai24.scorer import RewardScorer

from .utils import truncate_en_doc, truncate_zh_doc, LLM


def try_wrapped_run_eval(rank, config, debug=False):
    # llm
    LLM.get_llm_config(config["LLMConfig"])
    LLM.gpu_ids = config["gpu"]
    LLM.ddp = config["ddp"]
    
    world_size = len(config["gpu"])
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # load data
    begin_time = time.time()
    dataset = load_example(config["test_file"]) # 如果你在这里加上索引，那么他的len函数就返回属性数量而非样本数量了，那么下一行就会报错

    part_num = math.ceil(len(dataset)/world_size)
    begin, end = part_num * rank, part_num * (rank + 1)
    
    queries = dataset["query"][begin:end]
    language_list = dataset["language"][begin:end]
    truthful_answer_list = dataset["truthful answer"][begin:end]
    
    load_time_usage = time.time() - begin_time
    
    # replace rank placeholder in save_path
    if config["result_path"] is not None and "$rank$" in config["result_path"]:
        config["result_path"] = config["result_path"].replace("$rank$", str(rank))
    
    if config["result_info_path"] is not None and "$rank$" in config["result_info_path"]:
        config["result_info_path"] = config["result_info_path"].replace("$rank$", str(rank))
    
    for name, module in config["ModuleConfig"].items():
        if module["ban"]:
            continue
        
        if module["save_result_path"] is not None and "$rank$" in module["save_result_path"]:
            module["save_result_path"] = module["save_result_path"].replace("$rank$", str(rank))
            
        if module["save_info_path"] is not None and "$rank$" in module["save_info_path"]:
            module["save_info_path"] = module["save_info_path"].replace("$rank$", str(rank))
    
    # retrieval
    begin_time = time.time()
    web_retriever = WebRetriever(config["ModuleConfig"]["Web"])
    web_docs_list = web_retriever.batch_retrieve(queries, language_list)
    web_retriever.save_if_set()
    
    wiki_retriever = WikiRetriever(config["ModuleConfig"]["Wiki"])
    wiki_docs_list = wiki_retriever.batch_retrieve(queries, language_list)
    wiki_retriever.save_if_set()
    
    gen_retriever = GenerationRetriever(config["ModuleConfig"]["Gendoc"])
    gen_doc_list = gen_retriever.batch_retrieve(queries, language_list, f"gpu{rank}")
    gen_retriever.save_if_set()
    
    retrieval_time_usage = time.time() - begin_time
    
    docs_list = []
    for query_index in range(len(queries)):
        web_docs = web_docs_list[query_index]
        wiki_docs = wiki_docs_list[query_index]
        truncate_doc = truncate_en_doc if language_list[query_index] == "en" else truncate_zh_doc
        
        docs = {
            "google good": truncate_doc(web_docs[0], 1500, 0) if len(web_docs) > 0 else "",
            "google plus wiki": "\n".join(
                [truncate_doc(web_docs[i], 1200 // (min(len(web_docs), 3))) for i in range(min(len(web_docs), 3))]
                + [truncate_doc(wiki_docs[0], 150), truncate_doc(wiki_docs[1], 150)]),
            "google no truncate merged": "\n\n".join(web_docs),
            "gen doc": gen_doc_list[query_index],
        }

        docs_list.append(docs)
    
    for docs in docs_list:
        for name, value in docs.items():
            docs[name] = emoji.replace_emoji(value, replace='')
     
    # knowledge
    begin_time = time.time()
    
    summarizer = SummarizeConstructor(config["ModuleConfig"]["Summarizer"])
    docs_list_1 = [docs["google good"] for docs in docs_list]
    docs_list_2 = [docs["google plus wiki"] for docs in docs_list]
    knowledge_list_1 = summarizer.batch_construct(queries * 2, language_list * 2, docs_list_1 + docs_list_2, f"gpu{rank}")
    summarizer.save_if_set()
    
    contriver = ContriverConstructor(config["ModuleConfig"]["Contriver"])
    docs_list_3 = [docs["google no truncate merged"] for docs in docs_list]
    knowledge_list_2 = contriver.batch_construct(queries, language_list, docs_list_3, f"gpu{rank}")
    contriver.save_if_set()
    
    knowledge_list = []
    for query_index in range(len(queries)):
        knowledge_list.append({
            "summarize google good": knowledge_list_1[query_index],
            "summarize google plus wiki": knowledge_list_1[query_index + len(queries)],
            "contrive google no truncate merged": knowledge_list_2[query_index],
            "gen doc": docs_list[query_index]["gen doc"], 
        })
    
    knowledge_time_usage = time.time() - begin_time
    
    for knowledge in knowledge_list:
        for name, value in knowledge.items():
            knowledge[name] = emoji.replace_emoji(value, replace='')
    
    # response
    begin_time = time.time()
    
    generator = StandardGenerator(config["ModuleConfig"]["Generator"])
    
    _query_list, _language_list, _knowledge_list, _name_list = [], [], [], []
    for q, l, ks in zip(queries, language_list, knowledge_list):
        for n, k in ks.items():
            _query_list.append(q)
            _language_list.append(l)
            _knowledge_list.append(k)
            _name_list.append(n)
    _responses_list = generator.batch_response(_query_list, _language_list, _knowledge_list, f"gpu{rank}")
    generator.save_if_set()
    
    responses_list = []
    for query_index in range(len(queries)):
        responses_list.append({_name_list[query_index * 4 + i]: _responses_list[query_index * 4 + i] for i in range(4)})
    
    for responses in responses_list:
        for name, value in responses.items():
            responses[name] = emoji.replace_emoji(value, replace='')
    
    response_time_usage = time.time() - begin_time
    
    LLM.release_all()
    
    # vote
    begin_time = time.time()
    voter = StandardVoter(config["ModuleConfig"]["Voter"])
    name_list = ["summarize google good", "summarize google plus wiki", "contrive google no truncate merged", "gen doc"]
    _responses_list = [[responses_list[i][j] for j in name_list] for i in range(len(responses_list))]
    _vote_scores_list = voter.batch_voting(queries, language_list, _responses_list, f"gpu{rank}")
    voter.save_if_set()
    
    vote_scores_list = [{name:vote_score for name, vote_score in zip(name_list, _vote_scores_list[i])} for i in range(len(queries))]    
    vote_time_usage = time.time() - begin_time
    
    # score
    begin_time = time.time()
    select_query_list, select_response_list, select_language_list, select_name_list = [], [], [], []
    select_index = [0]
    threshold = 0.8
    answer_list = [None for _ in range(len(queries))]
    answer_name_list = ["" for _ in range(len(queries))]
    
    for query_index in range(len(queries)):
        vote_scores = vote_scores_list[query_index]
        filter_names = [name for name, score in vote_scores.items() if score >= threshold]
        if len(filter_names) == 1:            
            answer_list[query_index] = responses_list[query_index][filter_names[0]]
            answer_name_list[query_index] = filter_names[0]
            select_index.append(select_index[-1])
            continue       
        
        if len(filter_names) == 0:
            filter_names = list(vote_scores.keys())

        for name in filter_names:
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][name])
            select_name_list.append(name)
        
        select_index.append(select_index[-1] + len(filter_names))
    
    scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
    scores_list = scorer.batch_score(select_query_list, select_language_list, select_response_list, f"gpu{rank}")
    scorer.save_if_set()
    score_time_usage = time.time() - begin_time
    
    LLM.release_all()
    
    # answer
    for query_index in range(len(queries)):
        select_responses = select_response_list[select_index[query_index]:select_index[query_index+1]]
        select_scores = scores_list[select_index[query_index]:select_index[query_index+1]]
        select_queries = select_query_list[select_index[query_index]:select_index[query_index+1]]
        select_names = select_name_list[select_index[query_index]:select_index[query_index+1]]
        
        if len(select_queries) == 0:
            assert answer_list[query_index] is not None
            continue
            
        for i in range(1, len(select_queries)):
            assert select_queries[i] == select_queries[0] 

        answer_list[query_index] = select_responses[select_scores.index(max(select_scores))]
        answer_name_list[query_index] = select_names[select_scores.index(max(select_scores))]
    
    time_usage_dict = {
        "load_time_usage": load_time_usage,
        "retrieval_time_usage": retrieval_time_usage,
        "knowledge_time_usage": knowledge_time_usage,
        "response_time_usage": response_time_usage,
        "vote_time_usage": vote_time_usage,
        "score_time_usage": score_time_usage
    }
    
    logger.info(time_usage_dict.__str__())
    if config["result_info_path"] is not None:
        json.dump(time_usage_dict, open(config["result_info_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent =4)

    # no doc for performance comparison
    generator.config["en_template_id"] = 4
    generator.zh_prompt_template, generator.en_prompt_template = generator.build_prompt_template(generator.config)
    # 默认不使用cache
    no_knowledge_response_list =  generator._batch_response(queries, language_list, [""]*len(queries), f"gpu{rank}")
    
    for index, response in enumerate(no_knowledge_response_list):
        no_knowledge_response_list[index] = emoji.replace_emoji(response, replace='')
    
    result = {}
    for query_index, query in enumerate(queries):
        result[query] = {
            "ground_truth_answer": truthful_answer_list[query_index], 
            "system_answer": answer_list[query_index],
            "system_answer_name": answer_name_list[query_index]
        }
        
        for name in name_list:
            temp = {
                "knowledge": knowledge_list[query_index][name], 
                "response": responses_list[query_index][name],
                "vote_score": vote_scores_list[query_index][name],
                # 因为奖励模型的评分不是每个回复都有，所以用0.0代替即可
                "reward_score": 0.0,
                "final_score": 0.0,
            }
            result[query][name] = temp

        result[query]["no knowledge response"] = {
            "knowledge": None,
            "response": no_knowledge_response_list[query_index],
            "vote_score": 0.0,
            "reward_score": 0.0,
            "final_score": 0.0
        }
        
    if config["result_path"] is not None:
        json.dump(result, open(config["result_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
    
    if rank == 0:
        for i in range(world_size):
            while not os.path.exists(config["output_dir"].replace("$rank$", str(i)) + "result.json"):
                continue
        
        # 文件创建后还要一段时间写入内容
        time.sleep(5)
        
        for file in os.listdir(config["output_dir"].replace("$rank$", "0")):
            result = {}
            for i in range(world_size):        
                part_result = json.load(open(config["output_dir"].replace("$rank$", str(i)) + file, "r", encoding="UTF-8"))
                
                for key, value in part_result.items():
                    result[key] = value        
            
            json.dump(result, open(config["output_dir"].replace("$rank$/", "") + file , "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
       
def run_eval(rank, config, debug=False):
    try:
        try_wrapped_run_eval(rank, config, debug=False)
    except:
        print(f"{rank}报错")
        print(traceback.format_exc())
        exit(-1)
