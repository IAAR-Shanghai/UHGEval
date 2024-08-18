import re

from ... import metrics
from .eval_base import BaseUHGEvaluator

PROMPT_TEMPLATE_FOR_CONTINUATION = '''你是一名新华社新闻工作者。我希望你能辅助我完成一篇新闻的撰写。

请你根据我已经写好的文本为我续写一段话。下面是一个例子：

已经写好的文本：

《（文物考古）洛阳首现唐代梨园弟子墓志》
2016-10-27 15:14:41
    新华社郑州10月27日专电（记者桂娟）两方唐代梨园弟子墓志日前现身洛阳师范学院，专家初步推测墓主人为唐代粟特乐人曹乾琳夫妇。这是洛阳首次发现唐代梨园弟子墓志，为古代丝路文化交流研究再添宝贵资料。

续写的文本：

\n  正在洛阳师范学院河洛文化国际研究中心文物陈列馆展出的这两方墓志，出土于洛阳市龙门园区张沟社区。其中，曹乾琳墓志长宽各47厘米，盖文篆书“大唐故曹府君墓志铭”，墓志文字为楷书，字迹清晰可见。

已经写好的文本：

{}

续写的文本：
'''

PROMPT_TEMPLATE_FOR_EXTRACTION = '''你是一名新华社新闻工作者。
        
我需要你帮我从一句话中筛选出重要的词组或句子。不需要使用项目列表，每行一个关键词即可。下面是一个例子：

句子：

与去年同期相比，基金发行数量和份额今年以来均明显缩水。Wind数据显示，截至《经济参考报》记者发稿，年内发行基金数量共计1028只，合并发行份额为8719.89亿份。

关键词：

<keywords>
去年同期相比
基金发行数量
份额
今年以来
明显缩水
Wind数据
经济参考报
记者
发稿
年内
发行基金数量
1028只
合并发行份额
8719.89亿份
</keywords>

现在我的句子是：

{}

请给出提取出来的关键词（写在<keywords></keywords>之间）：
'''

class UHGGenerativeEvaluator(BaseUHGEvaluator):

    def set_generation_configs(self):
        new_configs = {
            "temperature": 0.1,
            "max_new_tokens": 128,
            "top_p": 0.9,
            "top_k": 5,
        }
        self.model.update_generation_configs(new_configs)    

    def continue_writing(self, data_point:dict) -> str:
        """Given a data point, continue writing the news."""
        news_lead = f'《{data_point["headLine"]}》\n{data_point["broadcastDate"][:10]}\n{data_point["newsBeginning"]}'
        query = PROMPT_TEMPLATE_FOR_CONTINUATION.format(news_lead)
        response = self.model.safe_request(query)
        continuation = re.split(r'(?<=[。；？！])', response)[0]
        return continuation

    def extract_kws(self, text: str) -> list[str]:
        """Extract keywords from the given text."""
        query = PROMPT_TEMPLATE_FOR_EXTRACTION.format(text)
        response = self.model.safe_request(query)
        kws = response.split('<keywords>')[-1].split('</keywords>')[0].split('\n')
        filtered = [
            s.strip() 
            for s in kws 
            if s.strip() and s.strip() in text
        ]
        return filtered

    def scoring(self, data_point: dict) -> dict:
        continuation = self.continue_writing(data_point)
        reference = data_point['newsRemainder']
        keywords = self.extract_kws(continuation)
        return {
            'metrics': {
                'bleu_4': metrics.bleu_4(continuation, reference),
                'rouge_l': metrics.rouge_l(continuation, reference),
                'kw_prec': metrics.keyword_precision(keywords, reference),
                'bert_score': metrics.bert_score(continuation, reference),
                'length': len(continuation)
            },
            'log': {
                'continuation': continuation,
                'keywords': keywords,
            },
            'valid': len(continuation) != 0
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'bleu_4': 0, 'rouge_l': 0, 'kw_prec': 0, 'bert_score': 0, 'length': 0}
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        overall = {f'avg_{key}': value / len(results) for key, value in overall.items()}
        overall['num'] = len(results)
        return overall
