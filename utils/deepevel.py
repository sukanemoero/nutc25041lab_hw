from typing import List, Optional
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)
from deepeval import evaluate


class RAGMetrics(BaseModel):
    Faithfulness: float
    Answer_Relevancy: float
    Contextual_Recall: float
    Contextual_Precision: float
    Contextual_Relevancy: float


class LangChainModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return '\n'.join(res.content) if isinstance(res.content, list) else res.content

    def get_model_name(self):
        return getattr(self.model, "model_name", "Custom LangChain Model")


class LocalDeepeval:
    def __init__(self, model, threshold: float = 0.7):
        self.model_wrapper = LangChainModel(model)

        self.metrics = [
            FaithfulnessMetric(threshold=threshold, model=self.model_wrapper),
            AnswerRelevancyMetric(threshold=threshold, model=self.model_wrapper),
            ContextualRecallMetric(threshold=threshold, model=self.model_wrapper),
            ContextualPrecisionMetric(threshold=threshold, model=self.model_wrapper),
            ContextualRelevancyMetric(threshold=threshold, model=self.model_wrapper),
        ]

    async def get_test_case(
        self,
        query: str,
        response: str,
        context: List[str],
        expected_output: Optional[str] = None,
    ):
        if expected_output is None:
            expected_output = await self.model_wrapper.a_generate(
                f"Please answer this question comprehensively: {query}"
            )

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context,
            expected_output=expected_output,
        )
        return test_case

    def evaluate_response(
        self,
        test_cases,
    ) -> list[RAGMetrics]:

        eval_result = evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
        )

        first_test_result = eval_result.test_results[0]
        metrics_data_list = first_test_result.metrics_data

        scores = []
        if metrics_data_list:
            for m in metrics_data_list:
                temp = {}
                key = m.name.replace(" ", "_")
                temp[key] = m.score
                score =RAGMetrics(
                    Faithfulness=temp.get("Faithfulness", -1),
                    Answer_Relevancy=temp.get("Answer_Relevancy", -1),
                    Contextual_Recall=temp.get("Contextual_Recall", -1),
                    Contextual_Precision=temp.get("Contextual_Precision", -1),
                    Contextual_Relevancy=temp.get("Contextual_Relevancy", -1),
                )
                scores += [score]

        return scores
