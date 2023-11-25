import string

from .openai_tools import CHAT_MODEL_LIST


class PromptTemplate:
    def __init__(self, language: str, model_name: str, template: str, system_message: str = None,
                 template_id: int = None, task_name: str = None):
        """
        language: str - specific the language of instruction, support 'en' and 'zh'
        model_name: str - model_name help to determine whether to use chat_completion format for turbo/gpt-4
        or completion for normal causal LM
        template: str - template str with placeholder around with curly bracket
        system_message: str - while model_name is turbo/gpt-4, system message is need to use chat_completions
        return_fill_info: bool - return fill stage info contains template, text, template id, task name
        template_id: str - if return fill info, template id will also be contained
        task_name: str - if return info, task name will also be contained
        """
        assert language in ["zh", "en"], "please use 'zh' for chinese, 'en' for english."
        self.language = language
        self.template = template
        self.placeholders = self.parse_template_placeholder(template)
        self.template_id = template_id
        self.task_name = task_name

        if model_name in CHAT_MODEL_LIST:
            self.chat = True
            if system_message is None:
                assert False, f"{model_name} need system_message."
        else:
            self.chat = False
        self.system_message = system_message

    def build_chat_message(self, prompt):
        """
        change input format from completion to chat
        parameters:
        prompt: str
        """
        system_message = {"role": "system", "content": self.system_message}
        query_message = {"role": "user", "content": prompt}
        return [system_message, query_message]

    def build_prompt(self, text):
        """
        fill in the templates
        parameters:
        text: Dict{str: str} - key is the name of placeholder in the template, value is the str to be fill

        return:
        prompt: List[str] or List[List[dict]] - list of prompts generated. If completion, return List[str],
        if chat completion, List[List[dict]]
        info: fill info
        """
        # TODO add clean function to process query and doc in text
        prompt, info = self.fill(text)

        if self.chat:
            prompt = self.build_chat_message(prompt)
            info["system_message"] = self.system_message

        return prompt, info

    def fill(self, text):
        """
        fill text in prompt placeholders.
        :param text: dict - key is placeholder name, value is the str to be fill in
        :return:
        instruction: prompt with placeholders filled
        """
        assert set(self.placeholders).issubset(
            set(text.keys())), f"{set(self.placeholders) - set(text.keys())} should be given"

        prompt = self.template.format(**text)

        info = {
            "template": self.template,
            "text": text,
            "result": prompt,
            "template_id": self.template_id,
            "task_name": self.task_name
        }

        return prompt, info

    @staticmethod
    def parse_template_placeholder(template):
        """
        return the name of all placeholders in the template

        Parameters:
        template: str - normally there should be placeholder(s) around with curly bracket inside.

        Return:
        placeholders: list -  list of all placeholder name in the template

        examples:
        instructions = "请用不超过50字的摘要如实的总结下述文档,摘要应该包含用于回答'{query}'这一问题的最主要的信息：{doc}\n"
        parse_template_placeholder(instructions)

        -> ['query', 'doc']
        """
        placeholders = []

        for parse_result in string.Formatter().parse(template):
            name = parse_result[1]
            if name is None:
                continue
            elif name == "":
                raise TypeError(f"no placeholder name is given in '{template}', check curly bracket")
            else:
                placeholders.append(name)

        return placeholders
