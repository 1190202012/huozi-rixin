from ..base import BaseConstructor
from ...utils import LLM, PromptTemplate


zh_summarization_template = {
    1: "请用少于200字如实的总结以下文档：\n文档：###\n{doc}\n###",
    2: "请如实总结以下文档,总结中应该包含用于回答'{query}'这一问题的最主要的信息且字数在50字以内：\n文档：###\n{doc}\n###",
    3: "请如实总结以下文档，总结中应该包含用于回答'{query}'这一问题的最主要的信息且字数在200字以内：\n文档：###\n{doc}\n###",
    4: "请如实总结以下文档，总结中应该包含用于回答问题的最主要的信息且字数在200字以内：\n\n问题: {query}\n\n文档：{doc}\n\n总结: ",
    5: "你是一个非常有帮助的文档总结助手，你可以从文档中提取出最有价值的信息并避免表达你自己的观点。\n\n请如实总结下列文档，总结中应该包含用于回答'{query}'最主要的信息且字数在200字以内：\n\n文档：###\n{doc}\n###"
}

# Template 4 is the best after test.
en_summarization_template = {
    1: "Please truthfully summarize the document below in less than 200 words:\nDocument:###\n{doc}\n###",
    2: "Please truthfully summarize the document below, the summary should contain the most important information relevant to answering the question '{query}' and be within 50 words:\nDocument: ###\n{doc}\n###",
    3: "Please truthfully summarize the document below, the summary should contain the most important information relevant to answering the question '{query}' and be within 200 words:\nDocument: ###\n{doc}\n###",
    # 4: "Please truthfully summarize the document below, the summary should contain the most important information relevant to answering the query and be within 200 words:\n\nquery: {query}\n\ndocument: {doc}\n\nsummary: ",
    4: "<s>[INST] Please truthfully summarize the document below, the summary should contain the most important information relevant to answer the query and be within 200 words:\n\nquery: {query}\n\ndocument: {doc}\n\nsummary: [/INST]",
    5: "[INST] <<SYS>>\nYou are an incredibly helpful document summarization assistant capable of extracting the most valuable information from documents and escape to express your own view.\n<</SYS>>\n\nPlease faithfully summarize the document below, the summary should contain the most important information relevant to answering the question '{query}' and be within 200 words:\n\nDocument: \n{doc}\n\n [/INST]",
    6: "[INST] <<SYS>>\nYou are an incredibly helpful document summarization assistant capable of extracting the most valuable information from documents and escape to express your own view.\n<</SYS>>\n\nPlease faithfully summarize the document below, the summary should contain the most important information relevant to answering the query and be within 200 words:\n\nquery: {query}\n\ndocument: {doc}\n\nsummary: [/INST]",}

zh_knowledge_length = {
    1: 200,
    2: 50,
    3: 200,
    4: 200,
}

en_knowledge_length = {
    1: 200,
    2: 50,
    3: 200,
    4: 200,
    5: 200,
    6: 200
}

zh_system_message = "你是一个非常有帮助的文档总结助手，你可以从文档中提取出最有价值的信息。"
en_system_message = "You are an incredibly helpful document summarization assistant capable of extracting the most valuable information from documents."


class SummarizeConstructor(BaseConstructor):
    def __init__(self, config):
        super().__init__(config)

        if not config["ban"]:
            self.config = config
            self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
            self.zh_knowledge_length = zh_knowledge_length[config["zh_template_id"]]
            self.en_knowledge_length = en_knowledge_length[config["en_template_id"]]

    def _construct(self, query, language, doc, gpu="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_knowledge_length if language == "zh" else self.en_knowledge_length
        
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["gpu"] = gpu
        
        knowledge = LLM.lm_generate(**kwargs)[0]
        if type(knowledge) is list:
            knowledge = knowledge[0]

        r = knowledge
        while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
            if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                r = r[1:]
            if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                r = r[3:]
        knowledge = r

        self.result_dict[self.key(query, doc)] = knowledge
        
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"doc": doc, "fill": fill_info, "knowledge": knowledge, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"]})

        return knowledge

    def _batch_construct(self, query_list, language_list, doc_list, gpu="gpu0"):
        prompts = []
        infos = []
        for query, language, doc in zip(query_list, language_list, doc_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})

            prompts.append(prompt)
            infos.append(fill_info)

        kwargs = self.config
        kwargs["prompts"] = prompts
        
        if "zh" in language_list and "en" in language_list:
            assert False, "it's recommended to divide chinese and english queries into two individual parts."
        elif "zh" in language_list and "en" not in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_knowledge_length
        elif "zh" not in language_list and "en" in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.en_knowledge_length       
        
        if "zh" in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["gpu"] = gpu
        
        knowledge_list = LLM.lm_generate(**kwargs)            
        
        for index, r in enumerate(knowledge_list):
            while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
                if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                    r = r[1:]
                if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                    r = r[3:]
            knowledge_list[index] = r
        
        for query, language, doc, fill_info, knowledge in zip(query_list, language_list, doc_list, infos, knowledge_list):
            self.result_dict[self.key(query, doc)] = knowledge
            
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"doc": doc, "fill": fill_info, "knowledge": knowledge, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"]})

        return knowledge_list

    @staticmethod
    def build_prompt_template(config):
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_summarization_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="summarize",
                                            template_id=config["zh_template_id"])
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_summarization_template[config["en_template_id"]],
                                            system_message=en_system_message, task_name="summarize",
                                            template_id=config["en_template_id"])
        return zh_prompt_template, en_prompt_template
