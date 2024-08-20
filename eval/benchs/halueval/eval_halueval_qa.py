import random

from .eval_base import BaseHaluEvalEvaluator

INSTRUCTION = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question context and intention.
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""


class HaluEvalQAEvaluator(BaseHaluEvalEvaluator):

    def scoring(self, data_point: dict) -> dict:
        if random.random() > 0.5:
            answer_under_evaluation = data_point["hallucinated_answer"]
            ground_truth = "Yes"
        else:
            answer_under_evaluation = data_point["right_answer"]
            ground_truth = "No"

        query = (
            INSTRUCTION
            + "\n\n#Question#: "
            + data_point["question"]
            + "\n#Answer#: "
            + answer_under_evaluation
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
