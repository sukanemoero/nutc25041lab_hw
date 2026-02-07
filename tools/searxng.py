from langchain.tools import tool
from httpx import AsyncClient, HTTPStatusError, RequestError, TimeoutException
from json import JSONDecodeError
from utils.logger import logger
from typing import Optional


def get_searxng_tool(
    xngconfig: Optional[dict[str, any]] = {},
    _locale: str = "zh-tw",
    limit: int = 3,
    httpx_client: AsyncClient = AsyncClient(),
):

    searxng_url = xngconfig.get("url", "")
    if not searxng_url:
        raise ValueError("SearXNG url is empty")

    @tool
    async def search_searxng(
        query: str, time_range: str = "all", locale: str = _locale
    ) -> str:
        """
        Search the web using SearXNG to find real-time information and URLs.

        Args:
            query: The search query string.
            time_range: Filter results by 'day', 'month', 'year', or 'all'.
            locale: The language code for the search results.
        """
        params = {"q": query, "format": "json", "language": locale}
        if time_range and time_range != "all":
            params["time_range"] = time_range
        result = []
        try:
            response = await httpx_client.get(searxng_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            valid_results = [r for r in results if "url" in r]

            result = valid_results[:limit]
        except HTTPStatusError as e:
            logger.error(
                f"Error response {e.response.status_code} while requesting {e.request.url!r}."
            )
        except TimeoutException:
            logger.error("The request timed out.")
        except RequestError as e:
            logger.error(f"An error occurred while requesting {e.request.url!r}.")
        except JSONDecodeError:
            logger.exception("JSON decode has a error.")
        except Exception:
            logger.exception("Except error.")
        finally:
            return result

    return search_searxng
