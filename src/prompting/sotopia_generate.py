import re
import os
from typing import TypeVar, Tuple
from functools import cache
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import BaseOutputParser, OutputParserException

from message_classes import ActionType, AgentAction
from utils import format_docstring
from langchain_callback_handler import LoggingCallbackHandler

HF_TOKEN_KEY_FILE="./hf_token.key"
if os.path.exists(HF_TOKEN_KEY_FILE):
    with open(HF_TOKEN_KEY_FILE, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()

OutputType = TypeVar("OutputType", bound=object)
log = logging.getLogger("generate")
logging_handler = LoggingCallbackHandler("langchain")

def generate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    temperature: float = 0.7,
) -> Tuple[str, AgentAction]:
    """
    Using langchain to generate an example episode
    """
    # try:
        # Normal case, model as agent
    template = """
        Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
        You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
        Note that {agent}'s goal is only visible to you.
        You should try your best to achieve {agent}'s goal in a way that align with their character traits.
        Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n
        {history}.
        You are at Turn #{turn_number}. Your available action types are
        {action_list}.
        Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

        Please only generate a JSON string including the action type and the argument.
        Your action should follow the given format:
        {format_instructions}
    """
    return generate(
        model_name=model_name,
        template=template,
        input_values=dict(
            agent=agent,
            turn_number=str(turn_number),
            history=history,
            action_list=" ".join(action_types),
        ),
        output_parser=PydanticOutputParser(pydantic_object=AgentAction),
        temperature=temperature,
    )

@cache
def prepare_model(model_name):
    compute_type = torch.float16
    
    if model_name == 'cmu-lti/sotopia-pi-mistral-7b-BC_SR':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", model_max_length=4096)
        model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="./.cache",
        device_map='cuda'
        )
        model = PeftModel.from_pretrained(model, model_name).to("cuda")
        
    elif model_name == 'cmu-lti/sotopia-pi-mistral-7b-BC_SR_4bit':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", model_max_length=4096)
        model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="./.cache",
        device_map='cuda',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_type,
            )
        )
        model = PeftModel.from_pretrained(model, model_name[0:-5]).to("cuda")
    
    elif model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", model_max_length=4096)
        tokenizer.model_max_length = 4096
        model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="./.cache",
        # device_map='cuda'
        )
        
    else:
        raise RuntimeError(f"Model {model_name} not supported")
    
    return model, tokenizer

def obtain_chain_hf(
    model_name: str,
    template: str,
    input_variables: list[str],
    temperature: float = 0.7,
    max_retries: int = 6,
    max_tokens: int = 2700
) -> LLMChain:
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="[INST] " + template + " [/INST]", input_variables=input_variables)
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    model, tokenizer = prepare_model(model_name)
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_new_tokens=100, 
                    temperature=temperature, 
                    return_full_text=False, 
                    do_sample=True,
                    num_beams=3,
                    )
    hf = HuggingFacePipeline(pipeline=pipe)
    chain = LLMChain(llm=hf, prompt=chat_prompt_template)
    return chain

def generate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: BaseOutputParser[OutputType],
    temperature: float = 0.7,
) -> Tuple[str, OutputType]:
    input_variables = re.findall(r"{(.*?)}", template)
    assert (
        set(input_variables) == set(list(input_values.keys()) + ["format_instructions"])
        or set(input_variables) == set(list(input_values.keys()))
    ), f"The variables in the template must match input_values except for format_instructions. Got {sorted(input_values.keys())}, expect {sorted(input_variables)}"
    # process template
    template = format_docstring(template)
    chain = obtain_chain(model_name, template, input_variables, temperature)
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()
    result = chain.predict([logging_handler], **input_values)
    prompt = logging_handler.retrive_prompt()
    print(f"Prompt:\n {prompt}")
    print(f"Result:\n {result}")
    try:
        parsed_result = output_parser.parse(result)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        log.debug(
            f"[red] Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        reformat_parsed_result = format_bad_output(
            result, format_instructions=output_parser.get_format_instructions()
        )
        print(f"Reformatted result:\n {reformat_parsed_result}")
        parsed_result = output_parser.parse(reformat_parsed_result)
    log.info(f"Generated result: {parsed_result}")
    return prompt, parsed_result

def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str = "gpt-3.5-turbo",
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """
    chain = obtain_chain(
        model_name=model_name,
        template=template,
        input_variables=re.findall(r"{(.*?)}", template),
    )
    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    reformat = chain.predict([logging_handler], **input_values)
    log.info(f"Reformated output: {reformat}")
    return reformat

def obtain_chain(
    model_name: str,
    template: str,
    input_variables: list[str],
    temperature: float = 0.7,
    max_retries: int = 6,
) -> LLMChain:
    """
    Using langchain to sample profiles for participants
    """
    if model_name in ["cmu-lti/sotopia-pi-mistral-7b-BC_SR", "cmu-lti/sotopia-pi-mistral-7b-BC_SR_4bit", "mistralai/Mistral-7B-Instruct-v0.1"]:
        return obtain_chain_hf(
            model_name=model_name,
            template=template,
            input_variables=input_variables,
            temperature=temperature,
            max_retries=max_retries,
        )
    
    model_name = _return_fixed_model_version(model_name)
    chat = ChatLiteLLM(
        model=model_name,
        temperature=temperature,
        max_tokens=2700,  # tweak as needed
        max_retries=max_retries,
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=template, input_variables=input_variables)
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    return chain

def _return_fixed_model_version(model_name: str) -> str:
    model_version_map = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-4": "gpt-4-0613",
        "gpt-4-turbo": "gpt-4-1106-preview",
    }
    return model_version_map[model_name] if model_name in model_version_map else model_name
