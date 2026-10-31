//! `tojson` as `transformers` defines it: Python's `json.dumps`, keyword arguments and
//! all.
//!
//! # Why not minijinja's own
//!
//! Jinja's built-in `tojson` is for embedding JSON in HTML. It escapes `<`, `>`, `&` and
//! `'`, and minijinja's prints compact JSON with no spaces. `transformers` replaces it
//! with `json.dumps(x, ensure_ascii=False, indent=indent, separators=separators,
//! sort_keys=sort_keys)`, and models are trained on that output: a tool schema with a
//! space after every `,` and `:`, and `<` left as it is. This module prints the same
//! bytes.
//!
//! Keyword arguments are the four `transformers` passes through: `indent` (spaces, or a
//! string), `separators` (an item and a key separator), `sort_keys` and `ensure_ascii`.
//! Positional arguments are refused; `transformers` would read the first one as
//! `ensure_ascii`, which no template means.
//!
//! Map keys print in the order the map holds them. With minijinja's `preserve_order`
//! feature that is the order they were serialized in, which is what Python's dict does.
//! Floats print as Python's `repr` does, which agrees with Rust's shortest round-trip
//! form except for how an exponent is written.

use std::fmt::Write;

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Error, ErrorKind, Value};

struct Options {
    /// `None` is compact, on one line. `Some` puts each item on its own line, indented
    /// by this string once per level; an empty string still breaks lines.
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
    ensure_ascii: bool,
}

/// The `tojson` filter.
pub(crate) fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent = match kwargs.get::<Option<Value>>("indent")? {
        None => None,
        Some(v) if v.kind() == ValueKind::String => Some(v.as_str().unwrap_or_default().to_owned()),
        // Python multiplies a space by the number, so a negative indent is no indent
        // at all, but still breaks lines.
        Some(v) => match v.as_i64() {
            Some(n) => Some(" ".repeat(usize::try_from(n).unwrap_or(0))),
            None => return Err(invalid(format!("tojson: indent must be a number or a string, not {}", v.kind()))),
        },
    };
    let (item_sep, key_sep) = match kwargs.get::<Option<Value>>("separators")? {
        Some(v) => {
            let parts: Vec<Value> = v.try_iter()?.collect();
            match parts.as_slice() {
                [item, key] if item.as_str().is_some() && key.as_str().is_some() => {
                    (item.as_str().unwrap_or_default().to_owned(), key.as_str().unwrap_or_default().to_owned())
                }
                _ => return Err(invalid("tojson: separators must be two strings".into())),
            }
        }
        // json.dumps' defaults: a space after the comma only when on one line.
        None if indent.is_some() => (",".into(), ": ".into()),
        None => (", ".into(), ": ".into()),
    };
    let opts = Options {
        indent,
        item_sep,
        key_sep,
        sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
        ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
    };
    kwargs.assert_all_used()?;

    let mut out = String::new();
    write_value(&mut out, value, &opts, 0)?;
    Ok(Value::from_safe_string(out))
}

/// Python's `json.dumps(value, ...)` with the given options.
#[cfg(test)]
pub(crate) fn dumps(value: &Value, indent: Option<&str>, sort_keys: bool, ensure_ascii: bool) -> String {
    let opts = Options {
        indent: indent.map(str::to_owned),
        item_sep: if indent.is_some() { ",".into() } else { ", ".into() },
        key_sep: ": ".into(),
        sort_keys,
        ensure_ascii,
    };
    let mut out = String::new();
    write_value(&mut out, value, &opts, 0).expect("serializable");
    out
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidOperation, msg)
}

fn write_value(out: &mut String, value: &Value, opts: &Options, depth: usize) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => write_number(out, value)?,
        ValueKind::String => write_string(out, value.as_str().unwrap_or_default(), opts.ensure_ascii),
        ValueKind::Seq => {
            let items: Vec<Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(out, opts, depth + 1);
                write_value(out, item, opts, depth + 1)?;
            }
            newline(out, opts, depth);
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<(String, Value)> =
                value.try_iter()?.map(|k| key_string(&k).map(|s| (s, k))).collect::<Result<_, _>>()?;
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            if opts.sort_keys {
                keys.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (name, key)) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(out, opts, depth + 1);
                write_string(out, name, opts.ensure_ascii);
                out.push_str(&opts.key_sep);
                write_value(out, &value.get_item(key)?, opts, depth + 1)?;
            }
            newline(out, opts, depth);
            out.push('}');
        }
        // Python cannot serialize these either: jinja2's Undefined, a generator, or an
        // arbitrary object all raise TypeError.
        other => return Err(invalid(format!("tojson: a {other} value is not JSON serializable"))),
    }
    Ok(())
}

fn newline(out: &mut String, opts: &Options, depth: usize) {
    if let Some(indent) = &opts.indent {
        out.push('\n');
        for _ in 0..depth {
            out.push_str(indent);
        }
    }
}

/// A map key as Python's encoder writes it: strings as they are, and numbers, booleans
/// and None as their JSON spelling.
fn key_string(key: &Value) -> Result<String, Error> {
    match key.kind() {
        ValueKind::String => Ok(key.as_str().unwrap_or_default().to_owned()),
        ValueKind::None => Ok("null".into()),
        ValueKind::Bool => Ok(if key.is_true() { "true" } else { "false" }.into()),
        ValueKind::Number => {
            let mut s = String::new();
            write_number(&mut s, key)?;
            Ok(s)
        }
        other => Err(invalid(format!("tojson: a {other} cannot be a JSON object key"))),
    }
}

fn write_number(out: &mut String, value: &Value) -> Result<(), Error> {
    if value.is_integer() {
        let _ = write!(out, "{value}");
        return Ok(());
    }
    let f = f64::try_from(value.clone())?;
    out.push_str(&python_float(f));
    Ok(())
}

/// Python's `float.__repr__`, which `json.dumps` uses. Both Python and Rust print the
/// shortest string that reads back as the same float, and both switch to an exponent
/// below 1e-4 and from 1e16 up; they differ only in spelling it. Rust writes `1e16`
/// and `1e-7`, Python `1e+16` and `1e-07`.
fn python_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let s = format!("{f:?}");
    let Some((mantissa, exp)) = s.split_once('e') else {
        return s;
    };
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(d) => ('-', d),
        None => ('+', exp),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

/// A JSON string the way Python's encoder escapes it. With `ensure_ascii` off, only the
/// quote, the backslash and control characters are escaped, so a CJK tool description
/// stays readable. With it on, everything outside printable ASCII becomes `\uXXXX`, in
/// UTF-16 surrogate pairs past the BMP, as Python does.
fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
