import http.client
import os
from pathlib import Path
from typing import Dict, List, Type

import google.generativeai as gemini
import tiktoken
import torch
import transformers
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from together import Together
from transformers import AutoModelForCausalLM, AutoTokenizer    

from agent.dataclasses.agent_spec import DeploymentName, NonHELMMapping
from agent.models.model_provider import ModelProvider
from benchmark.dataclasses.task_run import ModelInput, ModelResponse


class OpenAIModels(ModelProvider):
    def __init__(self):
        self.client = self.create_client()

    def create_client(self) -> OpenAI:
        return OpenAI(api_key=self._api_key())

    def query_model_provider(
        self,
        model_name: str,
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": input_message}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_response = response.choices[0].message.content or ""
        model_input = ModelInput(
            value=input_message,
            num_tokens=response.usage.prompt_tokens,
        )
        model_response = ModelResponse(
            value=model_response,
            full_response=model_response,
            num_tokens=response.usage.completion_tokens,
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str) -> List[int]:
        encoding = tiktoken.encoding_for_model(model_name)
        return encoding.encode(input)

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        encoding = tiktoken.encoding_for_model(model_name)
        return encoding.decode(input_tokens)

    def get_num_tokens(self, model_name: str, input: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(input))


class AzureOpenAIModels(OpenAIModels):
    def __init__(self):
        self.client = self.create_client()

    def _endpoint(self) -> str:
        env_var = "AZURE_OPENAI_ENDPOINT"
        current_dir = Path(__file__).resolve().parent.parent
        root_dir = current_dir.parent
        env_path = root_dir / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
        endpoint = os.getenv(env_var)
        if not endpoint:
            if env_path.is_file():
                raise ValueError(
                    f"{env_var} is not set in the .env file or environment variables"
                )
            else:
                raise ValueError(
                    f"{env_var} is not set in environment variables and .env file not found at {env_path}"
                )
        return endpoint

    def create_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_key=self._api_key(),
            azure_endpoint=self._endpoint(),
            api_version="2024-06-01",
        )


class AnthropicModels(ModelProvider):
    def __init__(self):
        self.client = self.create_client()

    def create_client(self) -> Anthropic:
        return Anthropic(api_key=self._api_key())

    def query_model_provider(
        self,
        model_name: str,
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ):
        return self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": input_message}],
            stop_sequences=stop_sequences,
        )

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_response = response.content[0].text  # type: ignore
        model_input = ModelInput(
            value=input_message, num_tokens=response.usage.input_tokens
        )
        model_response = ModelResponse(
            value=model_response,
            full_response=model_response,
            num_tokens=response.usage.output_tokens,
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str):
        raise NotImplementedError("Tokenization is not supported for Anthropic models")

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        raise NotImplementedError(
            "Decoding tokens is not supported for Anthropic models"
        )

    def get_num_tokens(self, model_name: str, input: str) -> int:
        client = self.create_client()
        token_count = client.count_tokens(
            input
        )  # Note: This isn't accurate for newer Anthropic models (i.e. Claude 3 or later) and can only be obtained in model response usage property.
        return token_count


class GoogleModels(ModelProvider):
    def __init__(self):
        self.client = None  # We'll initialize this later with the specific model

    def create_client(self, model_name: str) -> gemini.GenerativeModel:
        gemini.configure(api_key=self._api_key())
        return gemini.GenerativeModel(model_name)

    def query_model_provider(
        self,
        model_name: str,
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ) -> gemini.types.GenerateContentResponse:
        if self.client is None or self.client.model_name != model_name:
            self.client = self.create_client(model_name)
        return self.client.generate_content(
            contents=input_message,
            generation_config=gemini.types.GenerationConfig(
                temperature=temperature,
                stop_sequences=stop_sequences,
                max_output_tokens=max_tokens,
            ),
        )

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_input = ModelInput(
            value=input_message,
            num_tokens=self.client.count_tokens(input_message).total_tokens,
        )
        model_response = ModelResponse(
            value=response.text,
            full_response=response.text,
            num_tokens=response.usage_metadata.candidates_token_count,
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str):
        raise NotImplementedError("Tokenization is not supported for Gemini models")

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        raise NotImplementedError("Decoding tokens is not supported for Gemini models")

    def get_num_tokens(self, model_name: str, input: str) -> int:
        if self.client is None or self.client.model_name != model_name:
            self.client = self.create_client(model_name)
        return self.client.count_tokens(input).total_tokens


class TogetherModels(ModelProvider):
    def __init__(self):
        self.client = self.create_client()

    def create_client(self) -> Together:
        return Together(api_key=self._api_key())

    def query_model_provider(
        self,
        model_name: str,
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ):
        return self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": input_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_response = response.choices[0].message.content or ""
        model_input = ModelInput(
            value=input_message,
            num_tokens=response.usage.prompt_tokens,  # type: ignore
        )
        model_response = ModelResponse(
            value=model_response,
            full_response=model_response,
            num_tokens=response.usage.completion_tokens,  # type: ignore
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str):
        raise NotImplementedError("Tokenization is not supported for Together models")

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        raise NotImplementedError(
            "Decoding tokens is not supported for Together models"
        )

    def get_num_tokens(self, model_name: str, input: str) -> int:
        raise NotImplementedError(
            "Getting number of tokens is not supported for Together models"
        )
    
class QwenModels(ModelProvider):
    def __init__(self):
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def create_client(self):
        return None

    def load_model(self, model_name: str):
        if self.model_name == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            self.model_name = model_name
            self.model.eval()

    def query_model_provider(
        self,
        model_name: str,  # ignored for Qwen, since it's local
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ):
        self.load_model(model_name)
        # Wrap message for chat format
        messages = [{"role": "user", "content": input_message}]
        chat_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_data = self.tokenizer(chat_input, return_tensors='pt', add_special_tokens=False)

        input_ids = input_data["input_ids"]
        attention_mask = input_data["attention_mask"]

        # print(f"Input IDs: {input_ids}")
        # print(f"Decoded {self.tokenizer.decode(input_ids[0])}")

        # print(f"Attention Mask: {attention_mask}")

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=stop_sequences,
            )

        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # print(f"output_text {output_text}")

        output_text = output_text.replace(input_message, "")

        # Mimic OpenAI-style response
        return type("ChatCompletion", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": output_text})
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(output_ids[0]) - len(input_ids[0]),
            })()
        })

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_response = response.choices[0].message.content or ""
        model_input = ModelInput(
            value=input_message,
            num_tokens=response.usage.prompt_tokens,
        )
        model_response = ModelResponse(
            value=model_response,
            full_response=model_response,
            num_tokens=response.usage.completion_tokens,
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str) -> List[int]:
        self.load_model(model_name)
        return self.tokenizer.encode(input, add_special_tokens=False)

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        self.load_model(model_name)
        return self.tokenizer.decode(input_tokens)

    def get_num_tokens(self, model_name: str, input: str) -> int:
        self.load_model(model_name)
        return len(self.tokenize(model_name, input))
    
class QwenTrainModels(ModelProvider):
    # A version of QwenModels that doesn't load the model (to save time)
    # Used when doing local training where the model is queried in verl
    def __init__(self):
        self.model_name = None
        self.tokenizer = None
        self.device = "cpu"

    def create_client(self):
        return None

    def load_model(self, model_name: str):
        if self.model_name == None:
            self.model_name = model_name.replace("Qwen-train/", "Qwen/")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True) 

    def query_model_provider(
        self,
        model_name: str,  
        input_message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ):
        # This method is a no-op for QwenTrainModels, as it doesn't load the model
        self.load_model(model_name)
        # Wrap message for chat format
        messages = [{"role": "user", "content": input_message}]
        chat_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_data = self.tokenizer(chat_input, return_tensors='pt', add_special_tokens=False)

        input_ids = input_data["input_ids"]
        attention_mask = input_data["attention_mask"]

        # Mimic OpenAI-style response
        return type("ChatCompletion", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": ""})
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": 0,
            })()
        })

    def parse_model_response(
        self, response, input_message: str, time_taken_in_ms: float
    ) -> tuple[ModelInput, ModelResponse]:
        model_response = response.choices[0].message.content or ""
        model_input = ModelInput(
            value=input_message,
            num_tokens=response.usage.prompt_tokens,
        )
        model_response = ModelResponse(
            value=model_response,
            full_response=model_response,
            num_tokens=response.usage.completion_tokens,
            time_taken_in_ms=time_taken_in_ms,
        )
        return model_input, model_response

    def tokenize(self, model_name: str, input: str) -> List[int]:
        self.load_model(model_name)
        return self.tokenizer.encode(input, add_special_tokens=False)

    def decode_tokens(self, model_name: str, input_tokens: List[int]) -> str:
        self.load_model(model_name)
        return self.tokenizer.decode(input_tokens)

    def get_num_tokens(self, model_name: str, input: str) -> int:
        self.load_model(model_name)
        return len(self.tokenize(model_name, input))


PROVIDERS: Dict[str, Type[ModelProvider]] = {
    "openai": OpenAIModels,
    "azure": AzureOpenAIModels,
    "anthropic": AnthropicModels,
    "google": GoogleModels,
    "together": TogetherModels,
    "Qwen": QwenModels,
    "Qwen-train": QwenTrainModels,
}


def get_provider_instance(
    model_name: str, azure: bool = False
) -> tuple[ModelProvider, str]:
    try:
        deployment_name = DeploymentName.from_string(model_name)
    except ValueError as e:
        raise ValueError(f"Invalid model name: {model_name}. Error: {e}")

    non_helm_model = NonHELMMapping.mapping.get(deployment_name)
    if not non_helm_model:
        raise ValueError(f"Model {model_name} not supported in NonHELMMapping")

    provider_name = str(deployment_name).split("/")[0]
    if provider_name == "openai" and azure:
        provider_name = "azure"
    provider_class = PROVIDERS.get(provider_name)
    if not provider_class:
        raise ValueError(f"Provider {provider_name} not supported")

    return provider_class(), non_helm_model


@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(
        exception_types=(
            HTTPError,
            ConnectionError,
            Timeout,
            http.client.RemoteDisconnected,
        )
    ),
)
def get_num_tokens(model_name: str, input_message: str) -> int:
    """
    Get the number of tokens for non-HELM requests.
    Defaults to GPT-4 via tiktoken if there isn't a native implementation
    """
    provider_instance, non_helm_model = get_provider_instance(model_name)
    try:
        return provider_instance.get_num_tokens(
            model_name=non_helm_model, input=input_message
        )
    except (NotImplementedError, KeyError):
        # Fallback to tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(input_message))


def tokenize(model_name: str, input_message: str) -> List[int]:
    """
    Get the number of tokens for non-HELM requests.
    Defaults to GPT-4 via tiktoken if there isn't a native implementation
    """
    provider_instance, non_helm_model = get_provider_instance(model_name)
    try:
        return provider_instance.tokenize(
            model_name=non_helm_model, input=input_message
        )  # type: ignore
    except (NotImplementedError, KeyError):
        encoding = tiktoken.encoding_for_model("gpt-4")
        return encoding.encode(input_message)


def decode(model_name: str, input_tokens: List[int]) -> str:
    """
    Get the number of tokens for non-HELM requests.
    Defaults to GPT-4 via tiktoken if there isn't a native implementation
    """
    provider_instance, non_helm_model = get_provider_instance(model_name)
    try:
        return provider_instance.decode_tokens(
            model_name=non_helm_model, input_tokens=input_tokens
        )
    except (NotImplementedError, KeyError):
        encoding = tiktoken.encoding_for_model("gpt-4")
        return encoding.decode(input_tokens)


def non_helm_request(
    model_name: str,
    input_message: str,
    temperature: float,
    max_tokens: int,
    stop_sequences: List[str],
    azure: bool,
) -> tuple[ModelInput, ModelResponse]:
    provider_instance, non_helm_model = get_provider_instance(model_name, azure=azure)
    return provider_instance.query(
        model_name=non_helm_model,
        input_message=input_message,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
    )
