import json
import torch.cuda
import time
from loguru import logger

from aaai24.data import load_example
from aaai24.retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from aaai24.knowledge import SummarizeConstructor, ContriverConstructor
from aaai24.response import StandardGenerator
from aaai24.voting import StandardVoter
from aaai24.scorer import RewardScorer

from .utils import truncate_en_doc, truncate_zh_doc, LLM


def run_example(config, debug=False):
    # llm
    LLM.get_llm_config(config["LLMConfig"])
    LLM.gpu_ids = config["gpu"]
    LLM.ddp = config["ddp"]
    
    # load data
    begin_time = time.time()
    dataset = load_example(config["test_file"])
    queries = dataset["query"][:5]
    language_list = dataset["language"][:5]
    load_time_usage = time.time() - begin_time
    
    # retrieval
    begin_time = time.time()
    web_retriever = WebRetriever(config["ModuleConfig"]["Web"])
    web_docs_list = web_retriever.batch_retrieve(queries, language_list)
    web_retriever.save_if_set()
    
    wiki_retriever = WikiRetriever(config["ModuleConfig"]["Wiki"])
    wiki_docs_list = wiki_retriever.batch_retrieve(queries, language_list)
    wiki_retriever.save_if_set()
    
    gen_retriever = GenerationRetriever(config["ModuleConfig"]["Gendoc"])
    gen_doc_list = gen_retriever.batch_retrieve(queries, language_list, "gpu0")
    gen_retriever.save_if_set()
    
    retrieval_time_usage = time.time() - begin_time

    docs_list = []
    for query_index in range(len(queries)):
        web_docs = web_docs_list[query_index]
        wiki_docs = wiki_docs_list[query_index]
        truncate_doc = truncate_en_doc if language_list[query_index] == "en" else truncate_zh_doc
        
        docs = {
            "google good": truncate_doc(web_docs[0], 1500, 0),
            "google plus wiki": "\n".join(
                [truncate_doc(web_docs[i], 1200 // (min(len(web_docs), 3))) for i in range(min(len(web_docs), 3))]
                + [truncate_doc(wiki_docs[0], 150), truncate_doc(wiki_docs[1], 150)]),
            "google no truncate merged": "\n\n".join(web_docs),
            "gen doc": gen_doc_list[query_index],
        }

        docs_list.append(docs)
    
    # knowledge
    begin_time = time.time()
    
    summarizer = SummarizeConstructor(config["ModuleConfig"]["Summarizer"])
    docs_list_1 = [docs["google good"] for docs in docs_list]
    docs_list_2 = [docs["google plus wiki"] for docs in docs_list]
    knowledge_list_1 = summarizer.batch_construct(queries * 2, language_list * 2, docs_list_1 + docs_list_2, "gpu0")
    summarizer.save_if_set()
    
    contriver = ContriverConstructor(config["ModuleConfig"]["Contriver"])
    docs_list_3 = [docs["google no truncate merged"] for docs in docs_list]
    knowledge_list_2 = contriver.batch_construct(queries, language_list, docs_list_3, "gpu0")
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
    _responses_list = generator.batch_response(_query_list, _language_list, _knowledge_list, "gpu0")
    generator.save_if_set()
    
    responses_list = []
    for query_index in range(len(queries)):
        responses_list.append({_name_list[query_index * 4 + i]: _responses_list[query_index * 4 + i] for i in range(4)})
    
    response_time_usage = time.time() - begin_time
    
    LLM.release_all()
    
    # vote
    begin_time = time.time()
    voter = StandardVoter(config["ModuleConfig"]["Voter"])
    name_list = ["summarize google good", "summarize google plus wiki", "contrive google no truncate merged", "gen doc"]
    _responses_list = [[responses_list[i][j] for j in name_list] for i in range(len(responses_list))]
    _vote_scores_list = voter.batch_voting(queries, language_list, _responses_list, "gpu0")
    voter.save_if_set()
    
    vote_scores_list = [{name:vote_score for name, vote_score in zip(name_list, _vote_scores_list[i])} for i in range(len(queries))]    
    vote_time_usage = time.time() - begin_time
    
    # score
    begin_time = time.time()
    select_query_list, select_response_list, select_language_list = [], [], []
    select_index = [0]
    threshold = 0.85
    answer_list = [None for _ in range(len(queries))]
    
    for query_index in range(len(queries)):
        vote_scores = vote_scores_list[query_index]
        filter_names = [name for name, score in vote_scores.items() if score >= threshold]
        if len(filter_names) == 1:            
            answer_list[query_index] = responses_list[query_index][filter_names[0]]
            select_index.append(select_index[-1])
            continue       
        
        if len(filter_names) == 0:
            filter_names = list(vote_scores.keys())

        for name in filter_names:
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][name])
        
        select_index.append(select_index[-1] + len(filter_names))
    
    scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
    scores_list = scorer.batch_score(select_query_list, select_language_list, select_response_list, "gpu0")
    scorer.save_if_set()
    score_time_usage = time.time() - begin_time
    
    LLM.release_all()
    
    # answer
    for query_index in range(len(queries)):
        select_responses = select_response_list[select_index[query_index]:select_index[query_index+1]]
        select_scores = scores_list[select_index[query_index]:select_index[query_index+1]]
        select_queries = select_query_list[select_index[query_index]:select_index[query_index+1]]

        if len(select_queries) == 0:
            assert answer_list[query_index] is not None
            continue
            
        for i in range(1, len(select_queries)):
            assert select_queries[i] == select_queries[0] 

        answer_list[query_index] = select_responses[select_scores.index(max(select_scores))]
    
    if config["result_path"] is not None:
        json.dump({query:answer for query, answer in zip(queries, answer_list)}, open(config["result_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent =4)
    
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

def run_one_example(config, debug=False):
    # llm
    LLM.get_llm_config(config["LLMConfig"])
    LLM.gpu_ids = config["gpu"]
    LLM.ddp = config["ddp"]

    # load data
    begin_time = time.time()
    dataset = load_example(config["test_file"])
    query = dataset["query"][0]
    language = dataset["language"][0]
    load_time_usage = time.time() - begin_time
    
    # retrieval
    begin_time = time.time()
    web_retriever = WebRetriever(config["ModuleConfig"]["Web"])
    web_docs = web_retriever.retrieve(query, language)
    web_retriever.save_if_set()
    
    wiki_retriever = WikiRetriever(config["ModuleConfig"]["Wiki"])
    wiki_docs = wiki_retriever.retrieve(query, language)
    wiki_retriever.save_if_set()
    
    gen_retriever = GenerationRetriever(config["ModuleConfig"]["Gendoc"])
    gen_doc = gen_retriever.retrieve(query, language, "gpu0")
    gen_retriever.save_if_set()
    
    retrieval_time_usage = time.time() - begin_time

    truncate_doc = truncate_en_doc if language == "en" else truncate_zh_doc
    docs = {
        "google good": truncate_doc(web_docs[0], 1500, 0) if len(web_docs) > 0 else "",
        "google plus wiki": "\n".join(
            [truncate_doc(web_docs[i], 1200 // (min(len(web_docs), 3))) for i in range(min(len(web_docs), 3))]
            + [truncate_doc(wiki_docs[0], 150), truncate_doc(wiki_docs[1], 150)]),
        "google no truncate merged": "\n\n".join(web_docs),
        "gen doc": gen_doc,
    }

    # knowledge
    begin_time = time.time()
    
    summarizer = SummarizeConstructor(config["ModuleConfig"]["Summarizer"])
    knowledge1 = summarizer.construct(query, language, docs["google good"], "gpu0")
    knowledge2 = summarizer.construct(query, language, docs["google plus wiki"], "gpu0")
    summarizer.save_if_set()
    
    contriver = ContriverConstructor(config["ModuleConfig"]["Contriver"])
    knowledge3 = contriver.construct(query, language, docs["google no truncate merged"], "gpu0")
    contriver.save_if_set()
    
    knowledge = {
        "summarize google good": knowledge1,
        "summarize google plus wiki": knowledge2,
        "contrive google no truncate merged": knowledge3,
        "gen doc": docs["gen doc"], 
    }
    
    knowledge_time_usage = time.time() - begin_time
    
    # response
    begin_time = time.time()
    
    generator = StandardGenerator(config["ModuleConfig"]["Generator"])
    
    responses = {}
    for _name, _knowledge in knowledge.items():
        responses[_name] = generator.response(query, language, _knowledge, "gpu0")
                
    generator.save_if_set()
    
    response_time_usage = time.time() - begin_time
    
    LLM.release_all()
    
    # vote
    begin_time = time.time()
    
    voter = StandardVoter(config["ModuleConfig"]["Voter"])
    
    name_list = ["summarize google good", "summarize google plus wiki", "contrive google no truncate merged", "gen doc"]
    response_list = [responses[name_list[i]] for i in range(len(name_list))]
    
    vote_score_list = voter.voting(query, language, response_list, "gpu0")
    voter.save_if_set()
    
    vote_scores = {name: vote_score for name, vote_score in zip(name_list, vote_score_list)}    
    
    vote_time_usage = time.time() - begin_time
    
    # score & answer
    begin_time = time.time()
    threshold = 0.85
    
    filter_names = [name for name, score in vote_scores.items() if score >= threshold]
    
    if len(filter_names) == 1:            
        answer = responses[filter_names[0]]
    else:
        if len(filter_names) == 0:
            filter_names = list(vote_scores.keys())

        scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
        best_response, best_score = None, None
        for _name in filter_names:
            score = scorer.score(query, language, responses[_name], "gpu0")
            if best_score is None or score > best_score:
                best_response = responses[_name]

        scorer.save_if_set()
        score_time_usage = time.time() - begin_time
        answer = best_response
    
    LLM.release_all()

    time_usage_dict = {
        "load_time_usage": load_time_usage,
        "retrieval_time_usage": retrieval_time_usage,
        "knowledge_time_usage": knowledge_time_usage,
        "response_time_usage": response_time_usage,
        "vote_time_usage": vote_time_usage,
        "score_time_usage": score_time_usage
    }
    
    logger.info(time_usage_dict.__str__())
    logger.info(f"query: {query}\nanswer: {answer}\n")