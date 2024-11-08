import enum
import json

import jinja2
from llama_cpp import Llama


class Result(enum.Enum):
    POSITIVE = 0
    NEGATIVE = 1
    ERROR = 2


def _parse_response_int(response: dict, debug) -> Result:
    """Parser which expects only one symbol - 0 or 1"""
    if "choices" not in response:
        return Result.ERROR
    if "text" not in response["choices"][0]:
        return Result.ERROR
    text = response["choices"][0]["text"]
    if debug:
        print(f"Response: <{text}>")
    if "0" in text:
        return Result.POSITIVE
    if "1" in text:
        return Result.NEGATIVE
    print(f"Incorrect output <{text}>")
    return Result.ERROR


def _parse_response_label(response: dict, debug) -> Result:
    """Parser which expects only one word - POSITIVE or NEGATIVE"""
    if "choices" not in response:
        return Result.ERROR
    if "text" not in response["choices"][0]:
        return Result.ERROR
    text = response["choices"][0]["text"]
    if debug:
        print(f"Response: <{text}>")
    if "POSITIVE" in text:
        return Result.POSITIVE
    if "NEGATIVE" in text:
        return Result.NEGATIVE
    print(f"Incorrect output <{text}>")
    return Result.ERROR


def _parse_response_json(response: dict, debug) -> Result:
    """Parser which expects json with field 'label'"""
    if "choices" not in response:
        return Result.ERROR
    if "text" not in response["choices"][0]:
        return Result.ERROR
    text = response["choices"][0]["text"]
    if debug:
        print(f"Response: <{text}>")
    try:
        dict_ = json.loads(text)
        assert "label" in dict_
    except Exception:
        print(f"Incorrect json <{text}>")
        return Result.ERROR
    if str(dict_["label"]) == "0":
        return Result.POSITIVE
    if str(dict_["label"]) == "1":
        return Result.NEGATIVE
    print(f"Incorrect output <{text}>")
    return Result.ERROR


def _parse_response_lines(response: dict, debug) -> Result:
    """Parser which expects several lines of response with one line `Label: ...`"""
    if "choices" not in response:
        return Result.ERROR
    if "text" not in response["choices"][0]:
        return Result.ERROR
    text = response["choices"][0]["text"]
    if debug:
        print(f"Response: <{text}>")
    for line in text.splitlines():
        if "Label:" in line:
            if "POSITIVE" in line:
                return Result.POSITIVE
            if "NEGATIVE" in line:
                return Result.NEGATIVE
    print(f"Incorrect output <{text}>")
    return Result.ERROR


PARSERS = {"int": _parse_response_int,
           "label": _parse_response_label,
           "json": _parse_response_json,
           "lines": _parse_response_lines}


def _generate_example(model, template: jinja2.Template, example: dict, config: dict, debug, parse: str):
    prompt = template.render(review=example["review"])
    if debug:
        print(f"Prompt:\n{prompt}\n***")
    output = model(
        prompt=prompt,
        echo=False,
        **config
    )
    return PARSERS[parse](output, debug)


def evaluate(model_path: str, template: str, dataset, config: dict = None, parse: str = "int",
             debug=False, n_ctx: int = 2048) -> dict:
    """Entry point to the evaluation framework.
    Make detection for each example from the dataset and returns dict with statistics.
    Args:
        model_path: path to .gguf file
        template: jinja2 template of the prompt
        dataset: Huggingface dataset with 'review' and 'label'
        config: Dict with generation parameters like temperature
        parse: Parser name. Possible values are `int`, `json`, `label` and `lines`.
        debug: If True, additional prints are activated
        n_ctx: Context length
    """
    config = {} if config is None else config

    llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    template_jinja = jinja2.Template(template)
    statistics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "error": 0}
    for example in dataset:
        result = _generate_example(llm, template_jinja, example, config, debug, parse)
        if result is Result.ERROR:
            statistics["error"] += 1
        elif result is Result.POSITIVE:
            if example["label"] == 0:
                statistics["tp"] += 1
            else:
                statistics["fp"] += 1
        elif result is Result.NEGATIVE:
            if example["label"] == 0:
                statistics["fn"] += 1
            else:
                statistics["tn"] += 1

    # Here we consider errors as incorrect classifications
    statistics["accuracy"] = (statistics["tp"] + statistics["tn"]) / len(dataset)
    statistics["precision"] = statistics["tp"] / (statistics["tp"] + statistics["fp"] + statistics["error"])
    statistics["recall"] = statistics["tp"] / (statistics["tp"] + statistics["fn"] + statistics["error"])
    return statistics
