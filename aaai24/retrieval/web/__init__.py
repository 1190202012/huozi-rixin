import json
import math
import time

from tqdm import tqdm

# import sys,os
# sys.path.append("/home/xyli/CodingFile/HuoziRixin/aaai24/retrieval")
# from base import BaseRetriever
# from search import serp_api, filter_urls, serper_api
# from fetch import fetch
# from extract import SyntaxExtractor

from ..base import BaseRetriever
from .search import serp_api, filter_urls, serper_api
from .fetch import fetch
from .extract import SyntaxExtractor

from loguru import logger


class WebRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

        if not config["ban"]:
            self.ssl_verify = config["ssl_verify"]
            self.extractor = SyntaxExtractor(config)

    def _retrieve(self, query, language, gpu="gpu0"):
        if self.log_detail:
            logger.info(f"begin to search '{query}'.")
            t = time.time()

        url_cache = self.info_dict.get(query, None)
        if url_cache is None:
            self.info_dict[query] = {"search": None, "fetch": None}
            _urls, info = serper_api(query, language)
            self.info_dict[query]["search"] = info
        else:
            _urls = [item["link"] for item in url_cache["search"]["organic"]]

        urls = filter_urls(_urls)

        if self.log_detail:
            if len(urls) > 0:
                logger.info(f"search {len(urls)} urls. Time cost: {time.time() - t}")
            else:
                logger.warning("return no available url.")
                return []

        if self.log_detail:
            logger.info("begin to fetch.")
            t = time.time()

        fetch_pages = fetch(urls)

        if self.log_detail:
            if len(fetch_pages) > 0:
                logger.info(f"fetch {len(fetch_pages)} pages. Time cost: {time.time() - t}")
            else:
                logger.warning("fetch no available page.")
                return []

        if self.log_detail:
            logger.info("begin to extract.")
            t = time.time()

        docs = []
        self.info_dict[query]["fetch"] = {}
        for url, text in fetch_pages.items():
            doc = self.extractor.extract(language,  text)
            if doc is not None:
                docs.append(doc)
                self.info_dict[query]["fetch"][url] = doc

        if len(docs) == 0:
            for url, text in fetch_pages.items():
                doc = self.extractor.extract(language,  text, loosen=True)
                if doc is not None:
                    docs.append(doc)
                    self.info_dict[query]["fetch"][url] = doc
        
        if self.log_detail:
            if len(docs) > 0:
                logger.info(f"extract {len(docs)} docs. Time cost: {time.time() - t}")
            else:
                logger.warning("extract no available doc.")

        self.result_dict[query] = docs

        return docs

    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        query_urls_dict = {}
        
        if self.log_detail:
            logger.info(f"begin to search.")
        
        t = time.time()
        for query, language in tqdm(zip(queries, language_list)):
            url_cache = self.info_dict.get(query, None)
            if url_cache is None:
                self.info_dict[query] = {"search": None, "fetch": None}
                _urls, info = serper_api(query, language)
                self.info_dict[query]["search"] = info
            else:
                _urls = [item["link"] for item in url_cache["search"]["organic"]]

            urls = filter_urls(_urls)
            
            query_urls_dict[query] = urls

        total_num = sum([len(urls) for urls in query_urls_dict.values()])

        if self.log_detail:
            if total_num > 0:
                logger.info(f"search {total_num} urls. Time cost: {time.time() - t}")
            else:
                logger.warning("return no available url.")

        if self.log_detail:
            logger.info("begin to fetch.")
        
        t = time.time()

        flat_urls_list = [u for urls in query_urls_dict.values() for u in urls]
        flat_fetch_pages = {}
        
        fetch_part_num = 20
        _num = math.ceil(len(flat_urls_list)/fetch_part_num)
        for i in tqdm(range(_num)):
            flat_fetch_pages.update(fetch(flat_urls_list[i*fetch_part_num:(i+1)*fetch_part_num]))
            time.sleep(2)

        if self.log_detail:   
            if len(flat_fetch_pages) > 0:
                logger.info(f"fetch {len(flat_fetch_pages)} pages. Time cost: {time.time() - t}")
            else:
                logger.warning("fetch no available page.")
                return [[] for i in range(len(queries))]

        docs_list = []
        
        if self.log_detail:
            logger.info("begin to extract.")
        
        t = time.time()
        for query, language in tqdm(zip(queries, language_list)):
            docs = []
            self.info_dict[query]["fetch"] = {}
            fetch_pages = {url: flat_fetch_pages[url] for url in query_urls_dict[query]}
            
            for url, text in fetch_pages.items():
                doc = self.extractor.extract(language,  text)
                if doc is not None:
                    docs.append(doc)
                    self.info_dict[query]["fetch"][url] = doc

            if len(docs) == 0:
                for url, text in fetch_pages.items():
                    doc = self.extractor.extract(language,  text, loosen=True)
                    if doc is not None:
                        docs.append(doc)
                        self.info_dict[query]["fetch"][url] = doc

            self.result_dict[query] = docs
            docs_list.append(docs)

        total_num = sum([len(docs) for docs in docs_list])

        if self.log_detail:
            if total_num > 0:
                logger.info(f"extract {total_num} docs. Time cost: {time.time() - t}")
            else:
                logger.warning("extract no available doc.")
        
        return docs_list

if __name__ == "__main__":
    _config = {
        "ban": False,
        "load_result_path": None,
        "save_result_path":None,
        "load_info_path": None,
        "save_info_path": None,
        "log_detail": True,
        "ssl_verify": False,
        "min_doc_len": 50,
        "max_doc_len": 1000
    }
    
    retriever = WebRetriever(_config)

    docs = retriever.retrieve("请问2023年爆发的巴以冲突局势如何？", "zh")
    
    print(docs)