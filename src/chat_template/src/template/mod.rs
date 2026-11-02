//! Author: Joe Ricks
//! File: src/chat_template/src/template/mod.rs
//! Description: Builds the Jinja environment that chat templates render in.

use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::{AutoEscape, Environment, Error, ErrorKind, Value};

pub(crate) mod clock;
pub(crate) mod pyjson;

/// Purpose: This function builds the template environment and adds one chat template to
/// it.
///
/// Inputs:
/// - `template`: the chat template source.
/// - `now`: the time `strftime_now` reports, or `None` for the system clock.
///
/// Outputs: the environment, or an error if the template does not compile.
pub(crate) fn environment(
    template: String,
    now: Option<SystemTime>,
) -> Result<Environment<'static>, Error>
{
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

/// Purpose: This function adds the `strftime_now` function to a template environment.
///
/// Inputs:
/// - `env`: the environment.
/// - `now`: the time to report, or `None` for the system clock.
///
/// Outputs: none.
pub(crate) fn set_clock(
    env: &mut Environment<'static>,
    now: Option<SystemTime>,
)
{
    env.add_function("strftime_now", move |format: &str| -> String {
        clock::strftime(format, unix_seconds(now.unwrap_or_else(SystemTime::now)))
    });
}

/// Purpose: This function converts a time to seconds since the Unix epoch.
///
/// Inputs:
/// - `t`: the time.
///
/// Outputs: the seconds, negative before 1970.
fn unix_seconds(
    t: SystemTime,
) -> i64
{
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(before) => -i64::try_from(before.duration().as_secs()).unwrap_or(i64::MAX),
    }
}

/// Purpose: This function tells whether a value is iterable, the way jinja2 decides it.
///
/// Inputs:
/// - `value`: the value to test.
///
/// Outputs: `true` if iterable, and `false` for none.
fn is_iterable(
    value: &Value,
) -> bool
{
    !value.is_none() && value.try_iter().is_ok()
}

/// Purpose: This struct holds the message a template passed to `raise_exception`.
#[derive(Debug)]
pub(crate) struct Raised(pub(crate) String);

impl std::fmt::Display for Raised {
    /// Purpose: This function prints a template's `raise_exception` message.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result
    {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Raised {}

/// Purpose: This function lets a template refuse a conversation with a message.
///
/// Inputs:
/// - `message`: the template's message.
///
/// Outputs: always an error that carries the message.
fn raise_exception(
    message: String,
) -> Result<String, Error>
{
    Err(Error::new(ErrorKind::InvalidOperation, message.clone()).with_source(Raised(message)))
}
