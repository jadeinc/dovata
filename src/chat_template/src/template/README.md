# src/chat_template/src/template: the Jinja environment chat templates render in

This folder builds the `minijinja` environment that `Chat` renders a model's chat
template in. It is set up to behave like the `jinja2` environment Hugging Face's
`transformers` uses, so a prompt rendered here is the same, byte for byte, as the prompt
`transformers` renders from the same template and conversation.

## Why it matches transformers

Model authors write their chat templates for `transformers` and test them there. A GGUF's
`tokenizer.chat_template` is copied from the model's `tokenizer_config.json`. A renderer
that differs in its whitespace handling, or in what a filter prints, gives a prompt the
model was never trained on, and nothing reports it. The model only answers worse.

`transformers` builds its environment in `_compile_jinja_template`, in
`src/transformers/utils/chat_template_utils.py`, and every setting there is mirrored
here:

- `trim_blocks` and `lstrip_blocks` are on, so block tags on their own lines leave no
  blank lines or indentation behind.
- The `loopcontrols` extension, `{% break %}` and `{% continue %}`, is on. In minijinja
  it is the `loop_controls` cargo feature.
- `tojson` is Python's `json.dumps` with its keyword arguments, from `pyjson.rs`, in
  place of Jinja's own `tojson`, which escapes HTML.
- The globals `raise_exception(message)` and `strftime_now(format)` are defined.

Under `transformers`, Python's string and dict methods, such as `strip`, `split`,
`startswith` and `items`, are Python itself. Here they come from minijinja-contrib's
`pycompat` callback. Auto-escaping is off for every template name, since a prompt is not
HTML.

## Where minijinja differs from jinja2, and how it is corrected

In jinja2, `none is iterable` is false, because Python cannot iterate `None`. In
minijinja it is true, because minijinja iterates none as empty. Qwen3-Coder's template
guards with `tools is iterable and tools | length > 0`, and `tools` is none whenever no
tools are passed, so minijinja's answer reaches `none | length` and fails. The `iterable`
test is replaced with jinja2's: a value is iterable when Python's `iter(value)` would
succeed. Strings, sequences and maps are iterable, and so is undefined, since jinja2's
`Undefined` iterates as empty. None is not.

## Known differences that are not corrected

- `strftime_now` formats in UTC, while `transformers` uses the machine's local time, so a
  prompt rendered near midnight can carry a different date. The next section says why.
- `transformers` also registers an `AssistantTracker` extension for `{% generation %}`
  blocks, which mark assistant tokens for training. No inference template needs them, and
  a template that uses one fails to compile here.

## raise_exception

Templates call `raise_exception(message)` to refuse a conversation they cannot render,
such as one whose roles do not alternate, or one with more tool calls than the model
takes. The function returns a minijinja error whose source is a `Raised` value holding
the message. `ChatError::from_render` in `src/chat_template/src/error.rs` walks the error's
sources looking for that value, so the program using dovata-chat-template gets `ChatError::Raised`
with the template's own words, kept apart from `ChatError::Template`, which means a
template bug.

## strftime_now, and why it is UTC

`transformers` calls Python's `datetime.now().strftime(format)`, which is local time.
Local time needs the machine's time zone database. Rust's standard library does not read
it, and a crate that does would be a dependency for one line of one template. A prompt is
also built at compile time here, on a build machine whose zone says nothing about where
the program will run, so local time would not be more right, only different. UTC is the
same everywhere, and `Chat::with_time` pins the time outright so two builds match.
`set_clock` is how it does so: it registers `strftime_now` again with the fixed time.

`clock.rs` implements the conversions of C's `strftime` in the C locale. Llama 3.1 and
later ask for `%d %b %Y`; the rest are there because they are cheap and a template could
plausibly use them. glibc's `-` flag drops the zero padding, so `%-d` prints `5` and not
`05`. An unknown conversion is printed as written, which is what glibc does. The date
comes from Howard Hinnant's `civil_from_days` algorithm, which turns a day count into a
proleptic Gregorian date with no table and no loop, and is exact over the whole range of
days that matters here.

## tojson, and why it is not minijinja's own

Jinja's standard `tojson` is for embedding JSON in HTML. It escapes `<`, `>`, `&` and
`'`, and minijinja's version prints compact JSON with no spaces. `transformers` replaces
it with `json.dumps(x, ensure_ascii=False, indent=indent, separators=separators,
sort_keys=sort_keys)`, and models are trained on that output: a tool schema with a space
after every comma and colon, and `<` left as it is. `pyjson.rs` prints the same bytes.

It takes the four keyword arguments `transformers` passes through: `indent`, as a number
of spaces or a string; `separators`, as an item separator and a key separator;
`sort_keys`; and `ensure_ascii`. Positional arguments are refused, because `transformers`
would read the first one as `ensure_ascii`, which no template means.

Map keys print in the order the map holds them. With minijinja's `preserve_order`
feature, which this package turns on, that is the order they were serialized in, which is
what a Python dict does. Keys that are numbers, booleans or none print as their JSON
spelling, as Python's encoder writes them.

Floats print as Python's `repr` prints them. Python and Rust both print the shortest
string that reads back as the same float, and both switch to an exponent below `1e-4` and
from `1e16` up. They differ only in how they write the exponent: Rust writes `1e16` and
`1e-7`, and Python writes `1e+16` and `1e-07`. `python_float` rewrites Rust's form into
Python's.

Strings are escaped as Python escapes them. With `ensure_ascii` off, which is the
`transformers` default, only the quote, the backslash and control characters are escaped,
so a tool description in Chinese or Japanese stays readable. With it on, everything
outside printable ASCII becomes `\uXXXX`, written as UTF-16 surrogate pairs past the
Basic Multilingual Plane, as Python does.

## Files

- `mod.rs`: `environment`, which builds the environment and registers one template;
  `set_clock`, which registers `strftime_now`; jinja2's `iterable` test; and
  `raise_exception` with its `Raised` error value.
- `clock.rs`: `strftime`, the C conversions in UTC, and the calendar arithmetic behind
  it.
- `pyjson.rs`: the `tojson` filter, and `dumps`, an entry point used only by the tests,
  which prints a value without going through a template.

## Who uses it

Only the rest of dovata-chat-template. `Chat::new` in `src/chat_template/src/lib.rs` calls `environment`,
`Chat::with_time` calls `set_clock`, and `ChatError::from_render` looks for `Raised`. The
tests in `src/chat_template/src/tests/template.rs` call `clock::strftime` and `pyjson::dumps`
directly, and compare their output with what CPython prints.
