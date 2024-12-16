# import openai
from openai import OpenAI
client = OpenAI(
api_key="DK3i6fFhq6e0fpmT7kCtSdVDLQ9IKpzk",
base_url="https://azure-openai-api.shenmishajing.workers.dev/v1/")

url = "https://restapi.amap.com/v3/weather/weatherInfo"
key = '5fd8caec787c3e290942156fbe96f95d'

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # print("completion_with_backoff kwargs:", kwargs)
    # input()
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)