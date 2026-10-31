"""Author: Joe Ricks
File: src/chat/jinja2_cases.py
Description: Renders real chat templates with jinja2 to make test cases for dovata-chat.
"""
import json
import os
import struct
import sys
import urllib.request
from datetime import datetime, timezone

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

USAGE = """Render real chat templates with jinja2, set up as Hugging Face's transformers sets it up,
and write the results as cases for dovata-chat's `matches_jinja2_cases` test.

Why: dovata-chat renders templates with minijinja, and models were trained on what
transformers renders with jinja2. This is the comparison that says the two agree. Run it
again after bumping minijinja or minijinja-contrib, or to check a new model's template.

    pip install jinja2
    python3 src/chat/jinja2_cases.py /tmp/cases.json
    DOVATA_CHAT_CASES=/tmp/cases.json cargo test -p dovata-chat --features preserve_order \
        matches_jinja2_cases -- --nocapture

The templates are fetched from the public tokenizer_config.json of each model, so no
third-party template is kept in the repository. If DOVATA_TEST_HEADER names a GGUF
header, its template is added too, as `gguf`.

The environment mirrors transformers' `_compile_jinja_template`: an
ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks and loopcontrols, its
tojson override, raise_exception and strftime_now. The render call mirrors
`render_jinja_template`: tools and documents always passed, the special tokens the
tokenizer has passed as keyword arguments. strftime_now is pinned to a UTC time, which
the test pins too.
"""

PINNED = 1721984707  # 2024-07-26 09:05:07 UTC; the test pins the same instant.

# name: (repo whose tokenizer_config.json holds the template, tool-call arguments as a
# JSON string rather than an object). DeepSeek V3 concatenates the arguments as a string.
MODELS = {
    "qwen3": ("Qwen/Qwen3-0.6B", False),
    "llama3": ("unsloth/Llama-3.2-1B-Instruct", False),
    "deepseekv3": ("deepseek-ai/DeepSeek-V3", True),
}


def make_env():
    """Purpose: This function builds a jinja2 environment set up the way Hugging Face's
    transformers sets one up.

    Inputs: none.

    Outputs: the jinja2 environment.
    """
    def raise_exception(message):
        """Purpose: This function lets a template refuse a conversation with a message.

        Inputs:
        - `message`: the template's message.

        Outputs: none. It always raises a `TemplateError`.
        """
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        """Purpose: This function prints a value as JSON, the way transformers' `tojson`
        does.

        Inputs:
        - `x`: the value to print.
        - `ensure_ascii`, `indent`, `separators`, `sort_keys`: the `json.dumps`
          options.

        Outputs: the JSON text.
        """
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        """Purpose: This function formats the pinned test time with a `strftime` format.

        Inputs:
        - `format`: the format string.

        Outputs: the formatted time.
        """
        return datetime.fromtimestamp(PINNED, timezone.utc).strftime(format)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render(template, messages, agp, extra, bos, eos):
    """Purpose: This function renders one test case with jinja2.

    Inputs:
    - `template`: the chat template.
    - `messages`: the conversation.
    - `agp`: whether to add the generation prompt.
    - `extra`: extra template values, such as `tools`.
    - `bos`: the BOS string, or `None`.
    - `eos`: the EOS string, or `None`.

    Outputs: `{"ok": text}` with the prompt, or `{"raised": message}` if the template
    refused.
    """
    kwargs = {}
    if bos is not None:
        kwargs["bos_token"] = bos
    if eos is not None:
        kwargs["eos_token"] = eos
    kwargs.update(extra)
    tools = kwargs.pop("tools", None)
    documents = kwargs.pop("documents", None)
    try:
        out = make_env().from_string(template).render(
            messages=messages, tools=tools, documents=documents, add_generation_prompt=agp, **kwargs
        )
        return {"ok": out}
    except jinja2.exceptions.TemplateError as e:
        return {"raised": str(e)}


def token_string(value):
    """Purpose: This function gets a special token's text from a `tokenizer_config.json`
    entry.

    Inputs:
    - `value`: the entry, a string or an AddedToken dict.

    Outputs: the token's text.
    """
    if isinstance(value, dict):
        return value.get("content")
    return value


def fetch_template(repo):
    """Purpose: This function downloads a model's chat template and special tokens from
    the Hugging Face Hub.

    Inputs:
    - `repo`: the model's repo id.

    Outputs: the template, the BOS string and the EOS string.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer_config.json"
    with urllib.request.urlopen(url) as resp:
        config = json.load(resp)
    template = config["chat_template"]
    if isinstance(template, list):  # named templates; take the default one
        template = next(t["template"] for t in template if t["name"] == "default")
    return template, token_string(config.get("bos_token")), token_string(config.get("eos_token"))


def gguf_template(path):
    """Purpose: This function reads the chat template from a GGUF file's header.

    Inputs:
    - `path`: the GGUF file.

    Outputs: the template text.
    """
    data = open(path, "rb").read(64 * 1024 * 1024)
    p = data.index(b"tokenizer.chat_template\x08\x00\x00\x00") + len(b"tokenizer.chat_template") + 4
    (n,) = struct.unpack_from("<Q", data, p)
    return data[p + 8 : p + 8 + n].decode("utf-8")


def msg(role, content, **extra):
    """Purpose: This function makes one chat message dict.

    Inputs:
    - `role`: the role, such as `user`.
    - `content`: the text, or `None`.
    - `extra`: any other keys, such as `tool_calls`.

    Outputs: the message dict.
    """
    return {"role": role, "content": content, **extra}


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}


def tool_call(arguments):
    """Purpose: This function makes one call to the test weather tool.

    Inputs:
    - `arguments`: the call's arguments, as a dict or a JSON string.

    Outputs: the tool call dict.
    """
    return {"type": "function", "function": {"name": "get_weather", "arguments": arguments}}


def conversations(args_as_string):
    """Purpose: This function builds the test conversations every template is rendered
    with.

    Inputs:
    - `args_as_string`: whether tool-call arguments are a JSON string instead of a
      dict.

    Outputs: a list of `(name, messages, add_generation_prompt, extra)` tuples.
    """
    args = '{"city": "Paris"}' if args_as_string else {"city": "Paris", "unit": "c"}
    sys_user = [msg("system", "You are terse."), msg("user", "Hi there")]
    multi = sys_user + [msg("assistant", "Hello."), msg("user", "Weather?")]
    calls = [
        msg("user", "Weather in Paris?"),
        msg("assistant", None, tool_calls=[tool_call(args)]),
        msg("tool", "18C and sunny"),
    ]
    return [
        ("sys_user_gen", sys_user, True, {}),
        ("sys_user_nogen", sys_user, False, {}),
        ("multi_gen", multi, True, {}),
        ("multi_nogen", multi, False, {}),
        ("user_only", [msg("user", "Hi")], True, {}),
        ("tools", sys_user, True, {"tools": [WEATHER]}),
        ("tools_no_system", [msg("user", "Hi")], True, {"tools": [WEATHER]}),
        ("tool_round_trip", calls, True, {"tools": [WEATHER]}),
        ("tool_round_trip_then_answer", calls + [msg("assistant", "It is 18C."), msg("user", "Thanks")], True, {"tools": [WEATHER]}),
        ("unicode", [msg("user", "café 你好 <b>&</b>")], True, {}),
    ]


EXTRA_CASES = {
    "qwen3": [
        ("thinking_history", [msg("user", "2+2?"), msg("assistant", "<think>\nadd\n</think>\n\n4"), msg("user", "3+3?")], True, {}),
        ("thinking_last_turn", [msg("user", "2+2?"), msg("assistant", "<think>\nadd\n</think>\n\n4")], False, {}),
        ("no_thinking", [msg("user", "Hi")], True, {"enable_thinking": False}),
        ("reasoning_content", [msg("user", "Q"), msg("assistant", "A", reasoning_content="R")], False, {}),
    ],
    "llama3": [
        ("date_override", [msg("user", "Hi")], True, {"date_string": "01 Jan 2025"}),
        ("tools_in_system", [msg("system", "S"), msg("user", "Hi")], True, {"tools": [WEATHER], "tools_in_user_message": False}),
        ("raise_no_user", [msg("system", "S")], True, {"tools": [WEATHER]}),
        ("raise_two_calls", [msg("user", "Hi"), msg("assistant", None, tool_calls=[tool_call({"city": "A"}), tool_call({"city": "B"})])], True, {}),
    ],
    "deepseekv3": [
        ("two_systems", [msg("system", "A"), msg("system", "B"), msg("user", "Hi")], True, {}),
        ("think_stripped", [msg("user", "Q"), msg("assistant", "<think>x</think>A"), msg("user", "Q2")], True, {}),
    ],
}


def main():
    """Purpose: This function renders every case and writes them to the JSON file named
    on the command line.

    Inputs: none. It reads the output path from `sys.argv`.

    Outputs: none.
    """
    if len(sys.argv) != 2:
        sys.exit(USAGE)
    sources = []
    for name, (repo, args_as_string) in MODELS.items():
        template, bos, eos = fetch_template(repo)
        sources.append((name, template, bos, eos, args_as_string))
    header = os.environ.get("DOVATA_TEST_HEADER")
    if header:
        sources.append(("gguf", gguf_template(header), None, None, False))

    cases = []
    for name, template, bos, eos, args_as_string in sources:
        for case, messages, agp, extra in conversations(args_as_string) + EXTRA_CASES.get(name, []):
            cases.append({
                "name": f"{name}/{case}",
                "template": template,
                "bos": bos,
                "eos": eos,
                "messages": messages,
                "add_generation_prompt": agp,
                "extra": extra,
                "expected": render(template, messages, agp, extra, bos, eos),
            })
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(cases)} cases from {len(sources)} templates, jinja2 {jinja2.__version__}", file=sys.stderr)


main()
