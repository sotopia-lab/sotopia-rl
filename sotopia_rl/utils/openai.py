import os

from openai import OpenAI


def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "o4-mini"]:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    elif model.startswith("together_ai"):
        client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
        )
        together_model = "/".join(model.split("/")[1:])
        response = client.chat.completions.create(
            model=together_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text
    elif model.startswith("claude"):
        client = OpenAI(
        api_key=os.environ.get("CLAUDE_API_KEY"),
        base_url="https://api.anthropic.com/v1",
        )
        claude_model = "/".join(model.split("/")[1:])
        breakpoint()
        response = client.chat.completions.create(
            model=claude_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text
    else:
        raise ValueError(f"Model {model} not supported.")
