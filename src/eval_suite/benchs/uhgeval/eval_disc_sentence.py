from .eval_base import BaseUHGEvaluator

PROMPT_TEMPLATE = '''你是一名新华社新闻工作者。你需要判断新闻续写是否符合现实。

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


class UHGDiscSentenceEvaluator(BaseUHGEvaluator):

    def is_cont_hallu(self, continuation: str, data_point: dict) -> tuple[int, str]:
        """Determine whether a continuation is hallucinated.

        Returns:
            tuple: (answer, reason). `answer` is 1 if hallucinated, 0 if not, and -1 if error.
        """
        query = PROMPT_TEMPLATE.format(
            headLine=data_point["headLine"],
            broadcastDate=data_point["broadcastDate"],
            newsBeginning=data_point["newsBeginning"],
            continuation=continuation,
        )
        response = self.model.safe_request(query)
        response = response.strip()

        if response.startswith("续写不符合现实"):
            answer = 1
        elif response.startswith("续写符合现实"):
            answer = 0
        else:
            answer = -1
        justification = response.split("。")[0]
        return (answer, justification)

    def scoring(self, data_point: dict) -> dict:
        hallu_pred, hallu_justification = self.is_cont_hallu(
            data_point["hallucinatedContinuation"], data_point
        )
        unhallu_pred, unhallu_justification = self.is_cont_hallu(
            data_point["realContinuation"], data_point
        )

        return {
            "metrics": {"accuracy": ((hallu_pred == 1) + (unhallu_pred == 0)) / 2.0},
            "log": {
                "response_to_hallu": hallu_justification,
                "response_to_unhallu": unhallu_justification,
            },
            "valid": hallu_pred in {0, 1} and unhallu_pred in {0, 1},
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            "avg_accuracy": sum([result["metrics"]["accuracy"] for result in results])
            / len(results),
            "num": len(results),
        }
