import random

from .eval_base import BaseHaluEvalEvaluator

INSTRUCTION = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.
#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.
#Your Judgement#: Yes

You are trying to determine if there exists some non-factual and incorrect information in the summary.  
#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the summary and the document.
#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.
#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.
#Your Judgement#: Yes

You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""


class HaluEvalSummaEvaluator(BaseHaluEvalEvaluator):

    def scoring(self, data_point: dict) -> dict:
        if random.random() > 0.5:
            summary_under_evaluation = data_point["hallucinated_summary"]
            ground_truth = "Yes"
        else:
            summary_under_evaluation = data_point["right_summary"]
            ground_truth = "No"

        query = (
            INSTRUCTION
            + "\n\n#Document#: "
            + data_point["document"]
            + "\n#Summary#: "
            + summary_under_evaluation
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
