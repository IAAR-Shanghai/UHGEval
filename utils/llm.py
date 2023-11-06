import copy
import datetime
import requests
import json
import re
from abc import ABC, abstractmethod

import openai
from loguru import logger

from configs import real_config as conf


class LanguageModel(ABC):
    def __init__(
            self, 
            model_name: str = None, 
            temperature: float = 1.0, 
            max_new_tokens: int = 1024, 
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    @abstractmethod  # 由各个语言模型子类具体实现
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """所有的请求调用可以走这个安全调用，防止实验时突入停止"""
        try:
            response = self.request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    def continue_writing(self, obj:dict) -> str:
        """续写"""
        template = '''你是一名新华社新闻工作者。我希望你能辅助我完成一篇新闻的撰写。

请你根据我已经写好的文本为我续写一段话。下面是一个例子：

已经写好的文本：

《（文物考古）洛阳首现唐代梨园弟子墓志》
2016-10-27 15:14:41
    新华社郑州10月27日专电（记者桂娟）两方唐代梨园弟子墓志日前现身洛阳师范学院，专家初步推测墓主人为唐代粟特乐人曹乾琳夫妇。这是洛阳首次发现唐代梨园弟子墓志，为古代丝路文化交流研究再添宝贵资料。

续写的文本：

<response>
\n  正在洛阳师范学院河洛文化国际研究中心文物陈列馆展出的这两方墓志，出土于洛阳市龙门园区张沟社区。其中，曹乾琳墓志长宽各47厘米，盖文篆书“大唐故曹府君墓志铭”，墓志文字为楷书，字迹清晰可见。
</response>

现在我已经写好的文本是：

{}

请你完成要续写的文本（续写的文本写在<response></response>之间）：
'''
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"][:10]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]

    @staticmethod  # 部分子类的 continue_writing 会直接调用该静态方法
    def _continue_writing_without_instruction(self, obj:dict) -> str:
        """续写，无指令版本
        部分模型可能由于指令微调不充分，指令跟随效果不好，因此不增加指令，直接输入文本。
        """
        template = "{}"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split(query)[-1] if query in res else res
        real_res = real_res.replace('<s>', '').replace('</s>', '').strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]

    def extract_kws(self, sentence:str) -> list[str]:  # TODO: 更换非政治类型
        """提取关键词"""
        template = '''你是一名新华社新闻工作者。
        
我需要你帮我从一句话中筛选出重要的词组或句子。不需要使用项目列表，每行一个关键词即可。下面是一个例子：

句子：

在她公开表示不支持将半自动步枪的购买年龄限制提高至21岁以及反对禁售“撞火枪托”的同时，这一立场与时事的发展和一些政治观点产生了冲突。

关键词：

<keywords>
公开表示
不支持
半自动步枪
购买年龄限制
提高
21岁
反对
禁售
撞火枪托
立场
政治观点
冲突
</keywords>

现在我的句子是：

{}

请给出提取出来的关键词（写在<keywords></keywords>之间）：
'''
        query = template.format(sentence)
        res = self.safe_request(query)
        kws = res.split('<keywords>')[-1].split('</keywords>')[0].split('\n')
        filtered = [
            s.strip() 
            for s in kws 
            if s.strip() and s.strip() in sentence
            ]  # 去除空字符串以及不在原sentence中的
        return filtered

    def is_kw_hallucinated(self, kw:str, obj:dict, with_reason: bool = False) -> int | tuple[int, str]:
        """判断一个关键词是否存在幻觉
        
        Returns
            int: 0 或 1（不包含幻觉或包含幻觉）；返回-1如果存在错误
            [str: 模型输出的原因]
        """
        template = '''你是一名新华社新闻工作者。你需要判断新闻续写中的某个关键词是否符合现实。

新闻的开头："""《（体育）欧洲篮球联赛第二阶段小组赛阿尔巴出线形势不乐观》\n2015-03-05 06:01:47\n     新华社柏林3月4日电（记者王东）2014-2015赛季欧洲篮球联赛第二阶段小组赛第九轮今晚开始。在E组的比赛中，柏林阿尔巴队主场以80：72力挫立陶宛考纳斯萨拉基利斯队，9轮过后4胜5负，稳居小组第五名，但是出线形势仍不乐观。"""

续写："""柏林阿尔巴队主教练约翰·卡明斯表示，虽然目前的排名并不理想，但他们仍然会全力以赴，争取每个比赛的胜利。"""

判断：续写中的"""约翰·卡明斯"""不符合现实，因为不存在此人。

新闻的开头："""《（社会广角）两岁幼童病急转院争时间 山西交警接力救助获点赞》\n2015-05-13 18:48:50\n    新华社太原5月13日专电（记者梁晓飞）“有一小孩玻璃球卡到喉咙，从平遥到太原的路上，路遇晋KT3666，请大家避让！”5月12日，这样一条信息在微信朋友圈中快速传播。"""

续写："""\n据了解，事发当时，一名两岁幼童在家中玩耍时不慎将玻璃球吞下，导致喉咙被卡住，孩子的父母立即驾车带着他前往医院。"""

判断：续写中的"""喉咙"""不符合现实，因为应该是肺气管。

新闻的开头："""《（赛车）红牛飘移大赛追逐赛上演师徒大战》\n2015-03-22 21:27:18\n  新华社广州3月22日体育专电 红牛飘移大赛追逐赛22日在珠海赛车场轰鸣上演。日本车手末永直登在决赛中击败澳大利亚人琼斯，赢得冠军，中国香港车手邓智仑排名第三。"""

续写："""\n  作为红牛飘移大赛经典保留赛事，追逐赛每次都是最受车迷欢迎的项目。"""

判断：续写中的"""车迷"""符合现实。

新闻的开头："""《{headLine}》\n{broadcastDate}\n{newsBeginning}"""

续写："""{continuation}"""

判断：续写中的"""{keyword}"""'''
        
        query = template.format(
            headLine=obj['headLine'],
            broadcastDate=obj['broadcastDate'],
            newsBeginning=obj['newsBeginning'],
            continuation=obj['hallucinatedContinuation'],
            keyword=kw
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # 去除复述
        if real_res.startswith('不符合现实'):
            answer = 1
        elif real_res.startswith('符合现实'):
            answer = 0
        else:
            answer = -1
        return (answer, real_res.split('。')[0]) if with_reason else answer

    def compare_two_continuation(self, contn1: str, contn2: str, obj: dict) -> int:
        """比较续写1和续写2哪个更好

        Returns:
            int: 1 或 2（即续写1或续写2）；返回-1如果存在错误
        """
        template = '''现在，你我希望你作为一个续写评测器，你需要选择出更符合现实，更准确的续写。

新闻的开头："""《（体育）欧洲篮球联赛第二阶段小组赛阿尔巴出线形势不乐观》\n2015-03-05 06:01:47\n     新华社柏林3月4日电（记者王东）2014-2015赛季欧洲篮球联赛第二阶段小组赛第九轮今晚开始。在E组的比赛中，柏林阿尔巴队主场以80：72力挫立陶宛考纳斯萨拉基利斯队，9轮过后4胜5负，稳居小组第五名，但是出线形势仍不乐观。"""

A："""柏林阿尔巴队主教练约翰·卡明斯表示，虽然目前的排名并不理想，但他们仍然会全力以赴，争取每个比赛的胜利。"""

B："""\n    欧洲篮球联赛一般安排在每周四和周五进行，但是阿尔巴和考纳斯的这场小组赛提前一天进行。"""

哪个续写更符合现实，更准确：B更符合现实，更准确

新闻的开头："""《（赛车）红牛飘移大赛追逐赛上演师徒大战》\n2015-03-22 21:27:18\n  新华社广州3月22日体育专电 红牛飘移大赛追逐赛22日在珠海赛车场轰鸣上演。日本车手末永直登在决赛中击败澳大利亚人琼斯，赢得冠军，中国香港车手邓智仑排名第三。"""

A："""\n  作为红牛飘移大赛经典保留赛事，追逐赛每次都是最受车迷欢迎的项目。"""

B：""" 末永直登与邓智仑师徒大战备受关注，双方在比赛中表现出色，为观众带来了一场精彩绝伦的飘移盛宴。"""

哪个续写更符合现实，更准确：A更符合现实，更准确

新闻的开头："""《（文物考古）洛阳首现唐代梨园弟子墓志》\n2016-10-27 15:14:41\n    新华社郑州10月27日专电（记者桂娟）两方唐代梨园弟子墓志日前现身洛阳师范学院，专家初步推测墓主人为唐代粟特乐人曹乾琳夫妇。这是洛阳首次发现唐代梨园弟子墓志，为古代丝路文化交流研究再添宝贵资料。"""

A："""\n  正在洛阳师范学院河洛文化国际研究中心文物陈列馆展出的这两方墓志，出土于洛阳市龙门园区张沟社区。"""

B："""\n洛阳师范学院历史系副教授李涛在接受新闻记者采访时表示，这两方墓志的发现对于研究唐代音乐史具有非常重要的价值。"""

哪个续写更符合现实，更准确：A更符合现实，更准确

新闻的开头："""《{headLine}》\n{broadcastDate}\n{newsBeginning}"""

A："""{contn1}"""

B："""{contn2}"""

哪个续写更符合现实，更准确：'''
        query = template.format(
            headLine = obj['headLine'],
            broadcastDate = obj['broadcastDate'],
            newsBeginning = obj['newsBeginning'],
            contn1 = contn1,
            contn2 = contn2,
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # 去除复述
        real_res = real_res.split('更符合现实，更准确')[0].strip()  # 提取答案
        if real_res == 'A':
            answer = 1
        elif real_res == 'B':
            answer = 2
        else:
            answer = -1
        return answer

    def is_continuation_hallucinated(self, continuation:str, obj:dict, with_reason: bool = False) -> int | tuple[int, str]:
        """判断一个续写是否包含幻觉

        Returns:
            int: 0 或 1（不包含幻觉或包含幻觉）；返回-1如果存在错误
            [str: 模型输出的原因]
        """
        template = '''你是一名新华社新闻工作者。你需要判断新闻续写是否符合现实。

新闻的开头："""《（体育）欧洲篮球联赛第二阶段小组赛阿尔巴出线形势不乐观》\n2015-03-05 06:01:47\n     新华社柏林3月4日电（记者王东）2014-2015赛季欧洲篮球联赛第二阶段小组赛第九轮今晚开始。在E组的比赛中，柏林阿尔巴队主场以80：72力挫立陶宛考纳斯萨拉基利斯队，9轮过后4胜5负，稳居小组第五名，但是出线形势仍不乐观。"""

续写："""柏林阿尔巴队主教练约翰·卡明斯表示，虽然目前的排名并不理想，但他们仍然会全力以赴，争取每个比赛的胜利。"""

判断：续写不符合现实，因为并无“约翰·卡明斯”这个人。

新闻的开头："""《（赛车）红牛飘移大赛追逐赛上演师徒大战》\n2015-03-22 21:27:18\n  新华社广州3月22日体育专电 红牛飘移大赛追逐赛22日在珠海赛车场轰鸣上演。日本车手末永直登在决赛中击败澳大利亚人琼斯，赢得冠军，中国香港车手邓智仑排名第三。"""

续写："""\n  作为红牛飘移大赛经典保留赛事，追逐赛每次都是最受车迷欢迎的项目。"""

判断：续写符合现实。

新闻的开头："""《（社会广角）两岁幼童病急转院争时间 山西交警接力救助获点赞》\n2015-05-13 18:48:50\n    新华社太原5月13日专电（记者梁晓飞）“有一小孩玻璃球卡到喉咙，从平遥到太原的路上，路遇晋KT3666，请大家避让！”5月12日，这样一条信息在微信朋友圈中快速传播。"""

续写："""\n据了解，事发当时，一名两岁幼童在家中玩耍时不慎将玻璃球吞下，导致喉咙被卡住，孩子的父母立即驾车带着他前往医院。"""

判断：续写不符合现实，因为孩子吞下的是散碎的花生米，而非玻璃球；且异物卡在的是孩子的肺气管，而非喉咙。

新闻的开头："""《{headLine}》\n{broadcastDate}\n{newsBeginning}"""

续写："""{continuation}"""

判断：'''

        query = template.format(
            headLine = obj['headLine'],
            broadcastDate = obj['broadcastDate'],
            newsBeginning = obj['newsBeginning'],
            continuation = continuation
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # 去除复述
        if real_res.startswith('续写不符合现实'):
            answer = 1
        elif real_res.startswith('续写符合现实'):
            answer = 0
        else:
            answer = -1
        return (answer, real_res.split('。')[0]) if with_reason else answer


class Aquila_34B_Chat(LanguageModel):
    def request(self, query) -> str:
        url = conf.Aquila_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": float(self.params['temperature']),
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Aquila_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices']
        return res

    def continue_writing(self, obj: dict) -> str:
        """续写"""
        return super()._continue_writing_without_instruction(self, obj)


class Baichuan2_13B_Chat(LanguageModel):
    def request(self, query) -> str:
        url = conf.Baichuan2_13B_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Baichuan2_13B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class Baichuan2_53B_Chat(LanguageModel):
    def request(self, query) -> str:
        import time
        url = conf.Baichuan2_53B_url
        api_key = conf.Baichuan2_53B_api_key
        secret_key = conf.Baichuan2_53B_secret_key
        time_stamp = int(time.time())

        json_data = json.dumps({
            "model": "Baichuan2-53B",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "parameters": {
                "temperature": self.params['temperature'],
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        def _calculate_md5(input_string):
            import hashlib
            md5 = hashlib.md5()
            md5.update(input_string.encode('utf-8'))
            encrypted = md5.hexdigest()
            return encrypted
        signature = _calculate_md5(secret_key + json_data + str(time_stamp))
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        res = requests.post(url, data=json_data, headers=headers)
        res = res.json()['data']['messages'][0]['content']
        return res


class ChatGLM2_6B_Chat(LanguageModel):
    """只适合简单的问答，续写效果不稳定，指令跟随效果极差"""
    def request(self, query) -> str:
        url = conf.ChatGLM2_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.ChatGLM2_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class GPT(LanguageModel):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.token_consumed = 0
        self.system_message = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: {datetime.datetime.now().date()}"""  # Useful trick

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        res = openai.ChatCompletion.create(
            model = self.params['model_name'],
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user","content": query},
                ],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res["choices"][0]["message"]["content"]

        self.token_consumed += res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {self.token_consumed}') if self.report else ()
        return real_res


class GPT_transit(LanguageModel):
    """部署在中转服务器上的"""
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.token_consumed = 0
        self.system_message = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: {datetime.datetime.now().date()}"""  # Useful trick

    def request(self, query: str) -> str:
        url = conf.GPT_transit_url
        payload = json.dumps({
            "model": self.params['model_name'],
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query}
            ],
            "temperature": self.params['temperature'],
            'max_tokens': self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
        })
        headers = {
            'token': conf.GPT_transit_token,
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]

        self.token_consumed += res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {self.token_consumed}') if self.report else ()
        return real_res


class InternLM_20B_Chat(LanguageModel):
    def request(self, query) -> str:
        url = conf.InternLM_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.InternLM_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class Qwen_14B_Chat(LanguageModel):
    def request(self, query) -> str:
        url = conf.Qwen_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Qwen_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res

    def continue_writing(self, obj: dict) -> str:
        """续写"""
        return super()._continue_writing_without_instruction(self, obj)


class Xinyu_7B_Chat(LanguageModel):
    """自研的仅用于新闻生产领域的大模型，续写稳定性高"""
    def request(self, query) -> str:
        url = conf.Xinyu_7B_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Xinyu_7B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res

    def continue_writing(self, obj:dict) -> str:
        """续写"""
        template = "Human: 【生成任务：文本续写】我要你担任新闻编辑。我将为您提供与新闻相关的故事或主题，您将续写一篇评论文章，对已有文本进行符合逻辑的续写。您应该利用自己的经验，深思熟虑地解释为什么某事很重要，用事实支持主张，并补充已有故事中可能缺少的逻辑段落。\n请对以下文本进行续写。\n {} Assistant:"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('Assistant:')[-1].split('</s>')[0].strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]


class Xinyu_70B_Chat(LanguageModel):
    """自研的仅用于新闻生产领域的大模型，续写稳定性高"""
    def request(self, query) -> str:
        url = conf.Xinyu_70B_url
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        })
        headers = {
        'token': conf.Xinyu_70B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['text'][0]
        return res

    def continue_writing(self, obj:dict) -> str:
        """续写"""
        template = "Human: 【生成任务：文本续写】我要你担任新闻编辑。我将为您提供与新闻相关的故事或主题，您将续写一篇评论文章，对已有文本进行符合逻辑的续写。您应该利用自己的经验，深思熟虑地解释为什么某事很重要，用事实支持主张，并补充已有故事中可能缺少的逻辑段落。\n请对以下文本进行续写。\n {} Assistant:"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('Assistant:')[-1].split('</s>')[0].strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]
