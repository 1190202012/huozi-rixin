from ..base import BaseGenerator
from ...utils import PromptTemplate, LLM

zh_qa_template = {
    1: "{query}",
    # "假设如下事实是正确的：\n\n{knowledge}\n\n请根据上述文档如实的回答下列问题：{query}",
    # "你是一个高度智能的问答机器人, 如果你被问到了有事实依据的问题，你可以给我一个**简洁、短并且精确的答案**，如果你被问到了难以回答的问题，你可以回答不知道。\n\n假设以下事实是正确的：\n\n{knowledge}\n\n请回答以下问题：{query}\n\n",
    2: "假设如下事实是正确的：\n\n{knowledge}\n\n请根据上述事实简洁、明确地回答下列问题：{query}\n\n",
    3: "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\n请用简要的根据以下文档回答问题。\n\n文档：{knowledge}\n\n问题：{query}<|endofutterance|>\n<|beginofutterance|>助手\n",
    4: "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\n请简要回答问题：\n{query}<|endofutterance|>\n<|beginofutterance|>助手\n"
}

en_qa_template = {
    1: "{query}",
    # "Assuming the following paragraphs are true:\n\n{knowledge}\n\nPlease answer the following questions: {query} within 20 tokens",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me the answer. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease answer the following questions: {query}\n\n [/INST]",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a brief, short, accurate answer within 20 tokens. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease answer the following questions: {query} within 20 tokens\n\n [/INST]",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer within 10 words **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following questions: {query} within 10 words.\n\n [/INST]",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer with one or few words **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following questions: {query} with one or few words.\n\n [/INST]",
    # "reference:\n\n{knowledge}\n\nPlease carefully read the reference provided and answer the following question with a short answer:\nQuestion:{query}\nAnswer:", 
    # "Please answer question based on provided document within 20 tokens. Your answer should be brief, short, accurate.\n\nQuestion: {query}\n\nDocument: {knowledge}\n\nAnswer:",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer**. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'. Not say 'of course', 'brief answer' \n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease straightforward answer the following questions: {query}\n\n [/INST]",
    # "Answer the question '{query}' with one or few words: ",
    # "Briefly answer the question with one or few words.\nQuestion: {query}\nAnswer: ",
    # "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer within 10 words **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nPlease directly answer the following questions: {query} within 10 words.\n\n [/INST]",
    2: "[INST] <<SYS>>\nYou are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'.\n<</SYS>>\n\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following questions: {query}.\n\n [/INST]",
    3: "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\nAssuming the following paragraphs are true:\n\n{knowledge}\n\nPlease directly answer the following question with one or few words:\n{query}<|endofutterance|>\n<|beginofutterance|>助手\n",
    4: "<|beginofutterance|>系统\n你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。\n你的名字是“活字”。\n你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。\n你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。\n你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值<|endofutterance|>\n<|beginofutterance|>用户\nPlease directly answer the following question with one or few words:\n{query}<|endofutterance|>\n<|beginofutterance|>助手\n",
    5: "Please directly answer the following question with one or few words:\n{query}"
}

zh_system_message = "你是一个非常智能的问答机器人。有过你被问了有有事实依据的问题，你可以给出你的答案，如果你被问了难以回答的问题，你可以回答你不知道。"
en_system_message = "You are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me the answer. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'"


class StandardGenerator(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        
        if not config["ban"]:
            self.config = config
            self.zh_prompt_template, self.en_prompt_template, self.zh_prompt_query_only_template, self.en_prompt_query_only_template = self.build_prompt_template(config)

    @staticmethod
    def build_prompt_template(config):
        zh_prompt_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_qa_template[config["zh_template_id"]],
                                            system_message=zh_system_message, task_name="qa",
                                            template_id=config["zh_template_id"])
        en_prompt_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_qa_template[config["en_template_id"]],
                                            system_message=en_system_message, task_name="qa",
                                            template_id=config["en_template_id"])

        zh_prompt_query_only_template = PromptTemplate(language="zh", model_name=config["zh_model_name"],
                                            template=zh_qa_template[config["zh_query_only_template_id"]],
                                            system_message=zh_system_message, task_name="qa",
                                            template_id=config["zh_query_only_template_id"])
        en_prompt_query_only_template = PromptTemplate(language="en", model_name=config["en_model_name"],
                                            template=en_qa_template[config["en_query_only_template_id"]],
                                            system_message=en_system_message, task_name="qa",
                                            template_id=config["en_query_only_template_id"])
        return zh_prompt_template, en_prompt_template, zh_prompt_query_only_template, en_prompt_query_only_template

    def _response(self, query, language, knowledge, gpu="gpu0"):
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "knowledge": knowledge})
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "knowledge": knowledge})

        kwargs = self.config
        kwargs["prompts"] = [prompt]
        
        if language == "zh":
            kwargs["model_name"] = kwargs["zh_model_name"]
        else:
            kwargs["model_name"] = kwargs["en_model_name"]
        
        kwargs["gpu"] = gpu
        
        response = LLM.lm_generate(**kwargs)[0]
        
        r = response
        while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
            if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                r = r[1:]
            if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                r = r[3:]
        response = r
        
        self.result_dict[self.key(query, knowledge)] = response
        
        if query not in self.info_dict.keys():
            self.info_dict[query] = []
        self.info_dict[query].append({"knowledge": knowledge, "fill": fill_info, "response": response})

        return response

    def _batch_response(self, query_list, language_list, knowledge_list, query_only, gpu="gpu0"):
        prompts = []
        infos = []
        if query_only:
            for query, language, knowledge in zip(query_list, language_list, knowledge_list):
                if language == "zh":
                    prompt, fill_info = self.zh_prompt_query_only_template.build_prompt({"query": query})
                else:
                    prompt, fill_info = self.en_prompt_query_only_template.build_prompt({"query": query})

                prompts.append(prompt)
                infos.append(fill_info)
        else:
            for query, language, knowledge in zip(query_list, language_list, knowledge_list):
                if language == "zh":
                    prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "knowledge": knowledge})
                else:
                    prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "knowledge": knowledge})

                prompts.append(prompt)
                infos.append(fill_info)

        kwargs = self.config
        kwargs["prompts"] = prompts
        
        if "zh" in language_list and "en" not in language_list:
            kwargs["model_name"] = kwargs["zh_model_name"]
        elif "zh" not in language_list and "en" in language_list:
            kwargs["model_name"] = kwargs["en_model_name"]
        else:
            assert False, "Now not support mix 'en' and 'zh'."
        
        kwargs["gpu"] = gpu
        
        response_list = LLM.lm_generate(**kwargs)
        
        for index, r in enumerate(response_list):
            while (len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]) or (len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789"):
                if len(r) >= 1 and r[0] in [" ", "\n", ",", ".", ";", "?",  "，", "。",  "；", "？"]:
                    r = r[1:]
                if len(r) >= 3 and r[0] == "[" and r[2] == "]" and r[1] in "0123456789":
                    r = r[3:]
            response_list[index] = r

        for query, knowledge, fill_info, response in zip(query_list, knowledge_list, infos, response_list):
            self.result_dict[self.key(query, knowledge)] = response
                
            if query not in self.info_dict.keys():
                self.info_dict[query] = []
            self.info_dict[query].append({"knowledge": knowledge, "fill": fill_info, "response": response})

        return response_list
