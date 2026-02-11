from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from openai import BaseModel
from utils.logger import logger


async def structured_invoke(
    llm: BaseChatModel,
    structured: dict[str, any] | type,
    prompts: list[BaseMessage] = [],
    max_count: int = 3,
):
    result_dumped = None
    structured_llm = llm.with_structured_output(structured, method="json_mode")
    error = Exception()
    success = False
    for i in range(max_count):
        try:
            results = await structured_llm.ainvoke(prompts)
            if isinstance(results, BaseModel):
                result_dumped = results.model_dump()
            else:
                result_dumped = results
            success = True
            break
        except OutputParserException as e:
            error = e
            logger.debug("Structured output has OutputParserException.")
            prompts += [
                HumanMessage(
                    content=f"[!ERROR] There are unexpected problems with the format you generated. Please check the output format and generate again. ({i + 1} tried)",
                    name="Error",
                )
            ]
        except ValueError as e:
            error = e
            logger.warning(f"Structured output has a value error: {e}")

    if success:
        return result_dumped
    else:
        raise OutputParserException(error)
