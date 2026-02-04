from httpx import AsyncClient
from time import sleep
from requests.exceptions import ReadTimeout
from typing import Union
from pathlib import PosixPath

BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
WAV_PATH = "./wav/Podcast_EP14.wav"
auth = ("nutc2504", "nutc2504")


async def get_srt(path: Union[str, PosixPath]) -> str:
    async def post_audio():
        async with AsyncClient() as client:
            with open(path, "rb") as f:
                return await client.post(
                    CREATE_URL, files={"audio": f}, timeout=60, auth=auth
                )

    r = await post_audio()
    r.raise_for_status()
    task_id = r.json()["id"]
    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    async def wait_download(url: str, max_tries=600):
        async with AsyncClient() as client:
            for _ in range(max_tries):
                try:
                    resp = await client.get(url, timeout=(5, 60), auth=auth)
                    if resp.status_code == 200:
                        return resp.text
                except ReadTimeout:
                    pass
                sleep(2)
        return None

    srt_text = await wait_download(srt_url, max_tries=600)
    return srt_text
