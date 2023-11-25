from aaai24.retrieval.base import BaseRetriever
from aaai24.utils import LLM, PromptTemplate

zh_generation_template = {
    1: "请给出回答'{query}'所必须的信息。\n\n",
    2: "你是一个非常有帮助的助手，你可以给出回答问题所需要的信息，从而帮助他人给出回答。\n\n请避开答案并给出回答'{query}'所必须的信息。\n\n"
}

en_generation_template = {
    1: "Please escape final answer and give supportive information which is needed to answer '{query}'.\n\n",
    # 所有开源的LLM，即使有chat_completion，也都可以查阅其具体实现，使用text_completion的方式进行替代。经过实验，不加前面的system information答案更加干净，少了很多"sure","great","I am happy to help"等句子。
    2: "[INST] <<SYS>>\nYou are an incredibly helpful information generation assistant. You can give the supportive information needed to answer the question.\n<</SYS>>\n\n Please escape final answer and give supportive information which is needed to answer '{query}'. \n\n [/INST]",
    3: "[INST] <<SYS>>\nYou are an incredibly helpful information generation assistant. You can give the supportive information needed to answer the question.\n<</SYS>>\n\n Please give supportive information which is needed to answer: {query} \n\n [/INST]",
    4: "<s>[INST] Generate a background document to answer the given question.\n{query} [/INST]",
}

zh_doc_length = {
    1: 300,
    2: 300,
}

en_doc_length = {
    1: 300, 
    2: 300,
    3: 300,
    4: 300,
}

zh_system_message = "你是一个非常有帮助的助手，你可以给出回答问题所需要的信息，从而帮助他人给出回答。"
en_system_message = "You are an incredibly helpful information generation assistant. You can give the information needed to answer the question, thereby helping others to give an answer."


class GenerationRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

        if not config["ban"]:
            self.config = config
            self.zh_prompt_template, self.en_prompt_template = self.build_prompt_template(config)
            self.zh_doc_length = zh_doc_length[config["zh_template_id"]]
            self.en_doc_length = en_doc_length[config["en_template_id"]]

    def _retrieve(self, query, language, gpu="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_doc_length if language == "zh" else self.en_doc_length
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["gpu"] = gpu
        
        doc = LLM.lm_generate(**kwargs)[0]

        self.result_dict[query] = doc
        self.info_dict[query] = {"fill_info": fill_info, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"], "gen doc": doc}

        return doc

    def _batch_retrieve(self, queries, language_list, gpu="gpu0"):
        prompts = []
        infos = []
        for query, language in zip(queries, language_list):
            if language == "zh":
                prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query})
            else:
                prompt, fill_info = self.en_prompt_template.build_prompt({"query": query})

            prompts.append(prompt)
            infos.append(fill_info)

        kwargs = self.config
        kwargs["prompts"] = prompts
        if "zh" in language_list and "en" in language_list:
            # kwargs["max_tokens"] = min(self.zh_doc_length, self.en_doc_length)
            assert False, "it's recommended to divide chinese and english queries into two individual parts."
        elif "zh" in language_list and "en" not in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.zh_doc_length
        elif "zh" not in language_list and "en" in language_list:
            kwargs["generate_kwargs"]["max_new_tokens"] = self.en_doc_length

        if "zh" in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["gpu"] = gpu
        
        doc_list = LLM.lm_generate(**kwargs)

        for query, language,fill_info, doc in zip(queries, language_list, infos, doc_list):
            if self.save_result:
                self.result_dict[query] = doc
            if self.save_info:
                self.info_dict[query] = {"fill_info": fill_info, "model": self.config["zh_model_name"] if language=="zh" else self.config["en_model_name"], "gen doc": doc}

        return doc_list

    @staticmethod
    def build_prompt_template(config):
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_generation_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="gendoc",
                                            template_id=config["zh_template_id"])
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_generation_template[config["en_template_id"]],
                                            system_message=en_system_message, task_name="gendoc",
                                            template_id=config["en_template_id"])
        return zh_prompt_template, en_prompt_template
