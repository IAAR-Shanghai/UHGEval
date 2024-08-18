from .eval_base import BaseUHGEvaluator

PROMPT_TEMPLATE = '''现在，你我希望你作为一个续写评测器，你需要选择出更符合现实，更准确的续写。

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


class UHGSelectiveEvaluator(BaseUHGEvaluator):

    def which_is_true(self, contn1: str, contn2: str, obj: dict) -> int:
        """Given two continuations, determine which one is more accurate.

        Returns:
            int: 1 if contn1 is more accurate, 2 if contn2 is more accurate,
            -1 if an error occurs.
        """
        query = PROMPT_TEMPLATE.format(
            headLine=obj["headLine"],
            broadcastDate=obj["broadcastDate"],
            newsBeginning=obj["newsBeginning"],
            contn1=contn1,
            contn2=contn2,
        )
        response = self.model.safe_request(query)

        alphabet_answer = response.split("更符合现实，更准确")[0].strip()
        if alphabet_answer == "A":
            answer = 1
        elif alphabet_answer == "B":
            answer = 2
        else:
            answer = -1
        return answer

    def scoring(self, data_point: dict) -> dict:
        hallu = data_point["hallucinatedContinuation"]
        unhallu = data_point["realContinuation"]

        # Randomly choose the order of the two continuations
        if first_hallu_then_unhallu := len(hallu) % 2 == 0:
            answer = self.which_is_true(hallu, unhallu, data_point)
            correct = answer == 2
        else:
            answer = self.which_is_true(unhallu, hallu, data_point)
            correct = answer == 1
        return {
            "metrics": {"correct": correct},
            "log": {"first_hallu_then_unhallu": first_hallu_then_unhallu},
            "valid": answer in [1, 2],
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            "accuracy": sum([result["metrics"]["correct"] for result in results])
            / len(results),
            "num": len(results),
        }
