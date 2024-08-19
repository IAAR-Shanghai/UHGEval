from .eval_base import BaseUHGEvaluator

PROMPT_TEMPLATE = '''你是一名新华社新闻工作者。你需要判断新闻续写中的某个关键词是否符合现实。请直接输出符合现实或不符合现实，同时给出原因。

新闻的开头："""《（体育）欧洲篮球联赛第二阶段小组赛阿尔巴出线形势不乐观》\n2015-03-05 06:01:47\n     新华社柏林3月4日电（记者王东）2014-2015赛季欧洲篮球联赛第二阶段小组赛第九轮今晚开始。在E组的比赛中，柏林阿尔巴队主场以80：72力挫立陶宛考纳斯萨拉基利斯队，9轮过后4胜5负，稳居小组第五名，但是出线形势仍不乐观。"""

续写："""柏林阿尔巴队主教练约翰·卡明斯表示，虽然目前的排名并不理想，但他们仍然会全力以赴，争取每个比赛的胜利。"""

是否符合现实：续写中的"""约翰·卡明斯"""不符合现实，因为不存在此人。

新闻的开头："""《（社会广角）两岁幼童病急转院争时间 山西交警接力救助获点赞》\n2015-05-13 18:48:50\n    新华社太原5月13日专电（记者梁晓飞）“有一小孩玻璃球卡到喉咙，从平遥到太原的路上，路遇晋KT3666，请大家避让！”5月12日，这样一条信息在微信朋友圈中快速传播。"""

续写："""\n据了解，事发当时，一名两岁幼童在家中玩耍时不慎将玻璃球吞下，导致喉咙被卡住，孩子的父母立即驾车带着他前往医院。"""

是否符合现实：续写中的"""喉咙"""不符合现实，因为应该是肺气管。

新闻的开头："""《（赛车）红牛飘移大赛追逐赛上演师徒大战》\n2015-03-22 21:27:18\n  新华社广州3月22日体育专电 红牛飘移大赛追逐赛22日在珠海赛车场轰鸣上演。日本车手末永直登在决赛中击败澳大利亚人琼斯，赢得冠军，中国香港车手邓智仑排名第三。"""

续写："""\n  作为红牛飘移大赛经典保留赛事，追逐赛每次都是最受车迷欢迎的项目。"""

是否符合现实：续写中的"""车迷"""符合现实。

新闻的开头："""《{headLine}》\n{broadcastDate}\n{newsBeginning}"""

续写："""{continuation}"""

是否符合现实：续写中的"""{keyword}"""'''


class UHGDiscKeywordEvaluator(BaseUHGEvaluator):
    """Evaluate the keyword hallucination detection ability of the model."""

    def is_kw_hallucinated(self, kw: str, data_point: dict) -> tuple[int, str]:
        """Determine whether a keyword is hallucinated in the continuation.

        Returns:
            tuple: (answer, reason). `answer` is 1 if hallucinated, 0 if not, and -1 if error.
        """
        query = PROMPT_TEMPLATE.format(
            headLine=data_point["headLine"],
            broadcastDate=data_point["broadcastDate"],
            newsBeginning=data_point["newsBeginning"],
            continuation=data_point["hallucinatedContinuation"],
            keyword=kw,
        )
        response = self.model.safe_request(query)

        if response.startswith("不符合现实"):
            hallucinated = 1
        elif response.startswith("符合现实"):
            hallucinated = 0
        else:
            hallucinated = -1
        justification = response.split("。")[0]
        return hallucinated, justification

    def scoring(self, data_point: dict) -> dict:
        """
        Note:
            True and False are used to represent whether the keyword is hallucinated.
            Positive and negative are used to represent the prediction results.
        """

        true, false = [], []
        for item in data_point["annotations"]:
            kw, justification = item.split("<sep>")
            if justification.startswith("不合理"):
                true.append(kw)
            else:
                false.append(kw)

        # Balancing data
        num_samples = min(len(true), len(false))
        selected_kws = true[:num_samples] + false[:num_samples]
        ground_truth = [1] * num_samples + [0] * num_samples

        # Prediction
        predictions = dict()
        for idx, kw in enumerate(selected_kws):
            label, justification = self.is_kw_hallucinated(kw, data_point)
            predictions[kw] = (ground_truth[idx], label, justification)
        predicted_labels = [item[1] for item in predictions.values()]

        # Accuracy calculation
        accuracy = (
            sum([a == b for a, b in zip(ground_truth, predicted_labels)])
            / (2 * num_samples)
            if num_samples > 0
            else 0
        )

        return {
            "metrics": {"accuracy": accuracy, "num_kws": num_samples * 2},
            "log": {"predictions": predictions},
            "valid": not any([answer == -1 for answer in predicted_labels]),
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {"accuracy": 0, "num_kws": 0}
        for result in results:
            overall = {
                key: overall[key] + result["metrics"][key] for key in overall.keys()
            }
        overall = {f"avg_{key}": value / len(results) for key, value in overall.items()}
        overall["num"] = len(results)
        return overall
