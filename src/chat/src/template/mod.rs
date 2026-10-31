//! The Jinja environment chat templates render in, set up to behave like the one Hugging
//! Face's `transformers` uses.
//!
//! # Why match transformers
//!
//! Model authors write their templates for `transformers` and test them there; a GGUF's
//! `tokenizer.chat_template` is copied from the model's `tokenizer_config.json`. A
//! renderer that differs in whitespace handling or a filter's output gives a prompt the
//! model was never trained on, and nothing reports it. `transformers` builds its
//! environment in `_compile_jinja_template`, in `src/transformers/utils/chat_template_utils.py`,
//! and every setting there is mirrored here:
//!
//! * `trim_blocks` and `lstrip_blocks` on, so block tags on their own lines leave no
//!   blank lines or indentation behind.
//! * The `loopcontrols` extension: `{% break %}` and `{% continue %}`, which is the
//!   `loop_controls` feature of minijinja.
//! * A `tojson` filter that is Python's `json.dumps` with its keyword arguments, in
//!   [`pyjson`], instead of Jinja's HTML-escaping one.
//! * The globals `raise_exception(message)` and `strftime_now(format)`.
//!
//! Python's string and dict methods, such as `strip`, `split`, `startswith` and
//! `items`, are Python itself under `transformers`. Here they come from
//! minijinja-contrib's `pycompat` callback.
//!
//! # Where minijinja differs from jinja2, and is corrected
//!
//! * `none is iterable` is false in jinja2, because Python cannot iterate `None`, and
//!   true in minijinja, which iterates none as empty. Qwen3-Coder's template guards
//!   with `tools is iterable and tools | length > 0`, and `tools` is none whenever a
//!   caller passes no tools, so minijinja's answer reaches `none | length` and fails.
//!   The `iterable` test is replaced with jinja2's.
//!
//! # Known differences
//!
//! * `strftime_now` formats in UTC; `transformers` uses the machine's local time. A
//!   prompt rendered near midnight can carry a different date. See [`clock`].
//! * `transformers` also registers an `AssistantTracker` extension for
//!   `{% generation %}` blocks, which mark assistant tokens for training. No inference
//!   template needs them, and a template that uses one fails to compile here.

use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::{AutoEscape, Environment, Error, ErrorKind, Value};

pub(crate) mod clock;
pub(crate) mod pyjson;

/// Build the environment with `template` registered under [`crate::TEMPLATE_NAME`].
/// `now` pins `strftime_now`; `None` reads the system clock at each call.
pub(crate) fn environment(template: String, now: Option<SystemTime>) -> Result<Environment<'static>, Error> {
    let mut env = Environment::new();
    env.set_syntax(SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?);
    // A prompt is not HTML. The name has no extension, so escaping would be off anyway;
    // this says so where a reader looks.
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", pyjson::tojson);
    env.add_test("iterable", is_iterable);
    env.add_function("raise_exception", raise_exception);
    set_clock(&mut env, now);
    env.add_template_owned(crate::TEMPLATE_NAME, template)?;
    Ok(env)
}

/// Register `strftime_now`, reading `now` if set and the system clock if not.
pub(crate) fn set_clock(env: &mut Environment<'static>, now: Option<SystemTime>) {
    env.add_function("strftime_now", move |format: &str| -> String {
        clock::strftime(format, unix_seconds(now.unwrap_or_else(SystemTime::now)))
    });
}

fn unix_seconds(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(before) => -i64::try_from(before.duration().as_secs()).unwrap_or(i64::MAX),
    }
}

/// jinja2's `iterable` test: whether Python's `iter(value)` would succeed. Strings,
/// sequences and maps are iterable, and so is undefined, since jinja2's `Undefined`
/// iterates as empty. None is not.
fn is_iterable(value: &Value) -> bool {
    !value.is_none() && value.try_iter().is_ok()
}

/// A template's own refusal, carried as the source of the minijinja error so
/// [`ChatError::Raised`](crate::ChatError::Raised) can tell it from a template bug.
#[derive(Debug)]
pub(crate) struct Raised(pub(crate) String);

impl std::fmt::Display for Raised {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Raised {}

/// `raise_exception(message)`: templates call it to refuse a conversation they cannot
/// render, such as roles that do not alternate, or more tool calls than the model takes.
fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message.clone()).with_source(Raised(message)))
}
