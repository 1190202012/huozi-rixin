import json

from loguru import logger


class BaseGenerator:
    def __init__(self, config):
        self.result_dict = {}
        self.info_dict = {}
        
        if config["ban"]:
            logger.warning("[Generate response: false]")
            self.save_result, self.save_info = False, False
            self.response = lambda query, language, knowledge, gpu="gpu0": None
            self.batch_response = lambda queries, language_list, knowledge_list, query_only, gpu="gpu0": [None] * len(queries)
            return
        
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]

        if config["load_result_path"] is not None:
            logger.warning(f"[Load docs from {config['load_result_path']}]")
            self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None:
            logger.warning(f"[Load info from {config['load_info_path']}]")
            self.load_info(config["load_info_path"])
   
    def response(self, query, language, knowledge, gpu="gpu0"):
        return self.result_dict[self.key(query, knowledge)] if self.key(query, knowledge) in self.result_dict.keys() else self._response(query, language, knowledge, gpu)
    
    def batch_response(self, queries, language_list, knowledge_list, query_only, gpu="gpu0"):
        return self._batch_response(queries, language_list, knowledge_list, query_only, gpu) if len(self.result_dict) == 0 else self._batch_response_use_cache(queries, language_list, knowledge_list, query_only, gpu)

    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))

    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))

    @staticmethod
    def key(query, knowledge):
        return f"query: {query}\nknowledge: {knowledge}\n"

    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    def _response(self, query, language, knowledge, gpu="gpu0"):
        pass

    def _batch_response(self, queries, language_list, knowledge_list, query_only, gpu="gpu0"):
        pass

    def _batch_response_use_cache(self, queries, language_list, knowledge_list, query_only, gpu="gpu0"):
        cache = set(self.result_dict.keys())
        no_cache_queries = []
        no_cache_language_list = []
        no_cache_knowledge_list = []
        for query, language, knowledge in zip(queries, language_list, knowledge_list):
            if self.key(query, knowledge) not in cache:
                no_cache_queries.append(query)
                no_cache_language_list.append(language)
                no_cache_knowledge_list.append(knowledge)
        if len(no_cache_queries) > 0:
            self._batch_response(no_cache_queries, no_cache_language_list, no_cache_knowledge_list, query_only, gpu)
        return [self.result_dict[self.key(query, knowledge)] for query, knowledge in zip(queries, knowledge_list)]