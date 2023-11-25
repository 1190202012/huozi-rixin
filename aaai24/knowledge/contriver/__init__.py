import re

from ..base import BaseConstructor
from ...utils import truncate_en_doc, truncate_zh_doc, LLM


class ContriverConstructor(BaseConstructor):
    def __init__(self, config):
        super().__init__(config)

        if not config["ban"]:
            self.config = config
            self.min_knowledge_len = config["min_knowledge_len"]

    def construct(self, query, language, doc, gpu="gpu0", top_k=10):
        return self._construct(query, language, doc, gpu="gpu0", top_k=10) 

    def _construct(self, query, language, doc, gpu="gpu0", top_k=10):
        if type(doc) is list:
            assert False, "If you have many docs for a query, use batch_construct."

        paragraphs = []
        for item in doc.split("\n"):
            item = item.strip()
            if not item:
                continue
            paragraphs.append(item)

        temp = []
        for para in paragraphs:
            para = para.strip()
            para = re.sub(r"\[\d+\]", "", para)
            if language == "en":
                para = truncate_en_doc(para, 100, 10)
            else:
                para = truncate_zh_doc(para, 100, 10)
            if para is None:
                continue
            temp.append(para)
        paragraphs = temp
        
        kwargs = self.config
        
        if language == "en":
            query_model_name = "en_query_encoder"
            paragraph_model_name = "en_paragraph_encoder"
            instruction = ""
        elif language == "zh":
            query_model_name = "zh_encoder"
            paragraph_model_name = "zh_encoder"
            instruction = "为这个句子生成表示以用于检索相关文章："
            kwargs["encode_kwargs"]["pooling_method"] = "cls"     
           
        query_embedding = LLM.lm_encode(query_model_name, [instruction + query], gpu, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])[0]
        paragraph_embeddings = LLM.lm_encode(paragraph_model_name, paragraphs, gpu ,kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])

        scores = query_embedding @ paragraph_embeddings.t()

        sorted_paragraphs = sorted(paragraphs, key=lambda p: scores[paragraphs.index(p)], reverse=True)

        knowledge = "\n".join(sorted_paragraphs[:top_k])

        if len(knowledge.split(" ")) * 2 < self.min_knowledge_len and len(sorted_paragraphs) >= top_k+1:
            knowledge += "\n" + sorted_paragraphs[top_k]

        # knowledge = truncate_en_doc(knowledge, self.max_doc_len, self.min_doc_len)

        self.result_dict[self.key(query, doc)] = knowledge
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"doc": doc, "result": knowledge})

        return knowledge

    def _batch_construct(self, queries, language_list, doc_list, gpu="gpu0", top_k=10):
        paragraphs_list = []
        paragraphs_index = [0]
        
        for language, doc in zip(language_list, doc_list):
            paragraphs = []
            for item in doc.split("\n"):
                item = item.strip()
                if not item:
                    continue
                paragraphs.append(item)

            temp = []
            for para in paragraphs:
                para = para.strip()
                para = re.sub(r"\[\d+\]", "", para)
                if language == "en":
                    para = truncate_en_doc(para, 100, 10)
                else:
                    para = truncate_zh_doc(para, 100, 10)
                if para is None:
                    continue
                temp.append(para)
            paragraphs = temp
            
            paragraphs_list.append(paragraphs)
            paragraphs_index.append(paragraphs_index[-1] + len(paragraphs))
        
        kwargs = self.config
        
        if language == "en":
            query_model_name = "en_query_encoder"
            paragraph_model_name = "en_paragraph_encoder"
            instruction = ""
        elif language == "zh":
            query_model_name = "zh_encoder"
            paragraph_model_name = "zh_encoder"
            instruction = "为这个句子生成表示以用于检索相关文章："
            kwargs["encode_kwargs"]["pooling_method"] = "cls" 
        
        queries_embedding = LLM.lm_encode(query_model_name, [instruction + q for q in queries], gpu, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
        paragraphs_embeddings = LLM.lm_encode(paragraph_model_name, [p for ps in paragraphs_list for p in ps], gpu, kwargs["tokenize_kwargs"], kwargs["encode_kwargs"])
        
        paragraphs_embedding_list = []
        for i in range(len(queries)):
            paragraphs_embedding_list.append(paragraphs_embeddings[paragraphs_index[i]: paragraphs_index[i+1]])

        knowledge_list = []
        for query_embedding, paragraphs_embedding, query, doc, paragraphs in zip(queries_embedding, paragraphs_embedding_list, queries, doc_list, paragraphs_list):     
            scores = query_embedding @ paragraphs_embedding.t()
            sorted_paragraphs = sorted(paragraphs, key=lambda p: scores[paragraphs.index(p)], reverse=True)

            knowledge = "\n".join(sorted_paragraphs[:top_k])

            if len(knowledge.split(" ")) * 2 < self.min_knowledge_len and len(sorted_paragraphs) >= top_k+1:
                knowledge += "\n" + sorted_paragraphs[top_k]

            # knowledge = truncate_en_doc(knowledge, self.max_knowledge_len, self.min_knowledge_len)

            knowledge_list.append(knowledge)
            
            self.result_dict[self.key(query, doc)] = knowledge
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"doc": doc, "result": knowledge})

        return knowledge_list