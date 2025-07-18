import os

from openai import OpenAI
from rich import print as rprint

from utility.utils import *

os.environ["OPENAI_API_KEY"] = "xxxxx"
os.environ["OPENAI_BASE_URL"] = "xxxxx"


class chatLLM:
    def __init__(self, prompt_system) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )

        self.prompt_system = prompt_system

    def chat(self, prompt_user, model="llama3.1-70b-instruct", save_path=None):
        self.prompt_user = prompt_user
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": self.prompt_user}
            ]
        )
        answer = completion.choices[0].message.content
        if save_path is not None:
            save_lines(save_path, [answer])
            rprint(f"Successfully query {model} and save to {save_path}")
        return answer
