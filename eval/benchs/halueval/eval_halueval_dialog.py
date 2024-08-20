import random

from .eval_base import BaseHaluEvalEvaluator

INSTRUCTION = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a highly similar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""


class HaluEvalDialogEvaluator(BaseHaluEvalEvaluator):

    def scoring(self, data_point: dict) -> dict:
        if random.random() > 0.5:
            response_under_evaluation = data_point["hallucinated_response"]
            ground_truth = "Yes"
        else:
            response_under_evaluation = data_point["right_response"]
            ground_truth = "No"

        query = (
            INSTRUCTION
            + "\n\n#Dialogue History#: "
            + data_point["dialogue_history"]
            + "\n#Response#: "
            + response_under_evaluation
            + "\n#Your Judgement#: "
        )
        response = self.model.safe_request(query)

        answer = response.strip().split()
        # Extract the first word, such as "Yes", "No", "#Yes", "No."
        # Note: "".strip() returns [] instead of [""]
        answer = answer[0] if answer else ""
        # Remove the leading "#", ".", ","
        answer = answer.strip("#").strip(".").strip(",")

        return {
            "metrics": {
                "correct": ground_truth.lower() == answer.lower(),
            },
            "log": {
                "ground_truth": ground_truth,
                "answer": answer,
                "response": response,
            },
            "valid": answer.lower() in ["yes", "no"],
        }
