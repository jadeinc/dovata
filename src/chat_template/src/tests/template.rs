//! Author: Joe Ricks
//! File: src/chat_template/src/tests/template.rs
//! Description: Tests the template environment, `tojson` and `strftime_now` against
//! Python's output.

use std::time::{Duration, UNIX_EPOCH};

use minijinja::Value;
use serde::Serialize;
use serde_json::json;

use super::fixture;
use crate::template::{clock, pyjson};
use crate::{Chat, ChatError, Message};

/// Purpose: This function builds a test `Chat` with a given template.
///
/// Inputs:
/// - `template`: the template source.
///
/// Outputs: the `Chat`.
fn with(
    template: &str,
) -> Chat
{
    Chat::new(template.into(), fixture::tokenizer(), None, None).unwrap()
}

/// Purpose: This function renders a template with no messages.
///
/// Inputs:
/// - `template`: the template source.
///
/// Outputs: the rendered text.
fn render(
    template: &str,
) -> String
{
    with(template).render(&[], false).unwrap()
}

#[test]
fn pycompat_strip_works()
{
    // The calls Qwen3's template makes on assistant content.
    assert_eq!(render(r#"[{{ "  hi \n".strip() }}]"#), "[hi]");
    assert_eq!(render(r#"[{{ "\n\nhi\n".lstrip("\n") }}]"#), "[hi\n]");
    assert_eq!(render(r#"[{{ "hi\n\n".rstrip("\n") }}]"#), "[hi]");
    assert_eq!(render(r#"[{{ "xxhixx".strip("x") }}]"#), "[hi]");
}

#[test]
fn pycompat_string_and_dict_methods()
{
    let t = r#"{% set c = "<think>\nplan\n</think>\n\nanswer" %}{{ c.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') }}|{{ c.split('</think>')[-1].lstrip('\n') }}|{{ c.startswith('<think>') }}|{{ c.endswith('answer') }}|{% for k, v in {"a": 1}.items() %}{{ k }}={{ v }}{% endfor %}"#;
    assert_eq!(render(t), "plan|answer|True|True|a=1");
}

#[test]
fn loop_controls_work()
{
    let t = "{% for x in [1, 2, 3, 4, 5] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}";
    assert_eq!(render(t), "13");
}

#[test]
fn raise_exception_surfaces_as_raised()
{
    let t = "{% if messages[0].role != 'user' %}{{ raise_exception('Conversation must start with a user turn') }}{% endif %}ok";
    let err = with(t).render(&[Message::assistant("hi")], false).unwrap_err();
    assert!(matches!(&err, ChatError::Raised(m) if m == "Conversation must start with a user turn"), "{err:?}");
    assert_eq!(with(t).render(&[Message::user("hi")], false).unwrap(), "ok");
}

#[test]
fn strftime_now_uses_pinned_time()
{
    // 2024-07-26 09:05:07 UTC.
    let pinned = UNIX_EPOCH + Duration::from_secs(1_721_984_707);
    let chat = with(r#"{{ strftime_now("%d %b %Y") }}"#).with_time(pinned);
    assert_eq!(chat.render(&[], false).unwrap(), "26 Jul 2024");
    // Unpinned, it reads the clock, which is at least past the pinned date.
    let now = render(r#"{{ strftime_now("%Y") }}"#);
    assert!(now.parse::<i32>().unwrap() >= 2024, "{now}");
}

#[test]
fn strftime_conversions_match_python()
{
    let t = 1_721_984_707;
    assert_eq!(clock::strftime("%d %b %Y", t), "26 Jul 2024");
    assert_eq!(clock::strftime("%A %B %e %Y", t), "Friday July 26 2024");
    assert_eq!(clock::strftime("%Y-%m-%d %H:%M:%S", t), "2024-07-26 09:05:07");
    assert_eq!(clock::strftime("%j %y %I%p %a %%", t), "208 24 09AM Fri %");
    assert_eq!(clock::strftime("%F %T %u %w", t), "2024-07-26 09:05:07 5 5");
    // A leap day, the last second of it, and a time before the epoch.
    assert_eq!(clock::strftime("%j %d %b %Y %H:%M:%S %a", 951_868_799), "060 29 Feb 2000 23:59:59 Tue");
    assert_eq!(clock::strftime("%Y-%m-%d %a %j", -86_400), "1969-12-31 Wed 365");
    // glibc's no-padding flag, and an unknown conversion left as written.
    assert_eq!(clock::strftime("%-d/%-m %Q", 951_868_799 - 28 * 86_400), "1/2 %Q");
}

#[test]
fn trim_and_lstrip_blocks_are_on()
{
    // With both on, block tags on their own indented lines leave nothing behind.
    let t = "{% for m in messages %}\n    {% if m.role == 'user' %}\n[{{ m.content }}]\n    {% endif %}\n{% endfor %}\n";
    assert_eq!(with(t).render(&[Message::user("a"), Message::user("b")], false).unwrap(), "[a]\n[b]\n");
}

#[test]
fn tools_and_documents_default_to_none()
{
    let t = "{{ tools is defined }} {{ tools is none }} {{ documents is none }}";
    assert_eq!(render(t), "True True True");
    let chat = with(t);
    assert_eq!(chat.render_with(&[], false, &json!({ "tools": [] })).unwrap(), "True False True");
}

#[test]
fn none_is_not_iterable_as_in_python()
{
    // Qwen3-Coder's guard, with no tools passed. Plain minijinja says none is
    // iterable and then fails on `none | length`.
    let t = "{% if tools is iterable and tools | length > 0 %}tools{% else %}none{% endif %}";
    assert_eq!(render(t), "none");
    assert_eq!(render("{{ none is iterable }} {{ nothing is iterable }} {{ 'ab' is iterable }} {{ {} is iterable }} {{ 3 is iterable }}"), "False True True True False");
}

#[test]
fn template_syntax_error_is_reported_at_construction()
{
    let err = Chat::new("{% for %}".into(), fixture::tokenizer(), None, None).unwrap_err();
    assert!(matches!(err, ChatError::Template(_)), "{err:?}");
    assert!(err.to_string().contains("chat template"), "{err}");
}

/// Purpose: This struct holds a test tool definition, with its fields in a fixed order.
#[derive(Serialize)]
struct Tool {
    r#type: &'static str,
    function: Function,
}

/// Purpose: This struct holds the function part of a test tool definition.
#[derive(Serialize)]
struct Function {
    name: &'static str,
    description: &'static str,
    parameters: serde_json::Value,
}

/// Purpose: This function builds the test weather tool definition.
///
/// Inputs: none.
///
/// Outputs: the tool.
fn weather_tool() -> Tool
{
    Tool {
        r#type: "function",
        function: Function {
            name: "get_weather",
            description: "Weather <now> & \"later\"\n",
            parameters: json!({
                "type": "object",
                "properties": { "city": { "type": "string", "enum": [] } },
                "required": ["city"]
            }),
        },
    }
}

/// The weather tool as a template literal, so every map keeps its written order, the
/// way a Python dict literal does.
const TOOL_LITERAL: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather <now> & \"later\"\n", "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": []}}, "required": ["city"]}}}"#;

#[test]
fn tojson_matches_python_compact()
{
    assert_eq!(
        render(&format!("{{{{ {TOOL_LITERAL} | tojson }}}}")),
        r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather <now> & \"later\"\n", "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": []}}, "required": ["city"]}}}"#
    );
}

#[test]
fn tojson_keeps_serialized_key_order()
{
    /// Purpose: This struct holds the extra template values for one test.
    #[derive(Serialize)]
    struct Extra {
        tools: Vec<Tool>,
    }
    let chat = with("{{ tools[0] | tojson }}");

    // A struct keeps its field order whatever the features.
    let from_struct = chat.render_with(&[], false, &Extra { tools: vec![weather_tool()] }).unwrap();
    assert!(from_struct.starts_with(r#"{"type": "function", "function": {"name": "get_weather", "description""#), "{from_struct}");

    // A serde_json::Value sorts its keys, unless the `preserve_order` feature is on.
    let from_json = chat.render_with(&[], false, &json!({ "tools": [weather_tool()] })).unwrap();
    if cfg!(feature = "preserve_order") {
        assert_eq!(from_json, from_struct);
        assert!(from_struct.ends_with(r#""parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": []}}, "required": ["city"]}}}"#), "{from_struct}");
    } else {
        assert!(from_json.starts_with(r#"{"function": {"description": "#), "{from_json}");
        assert!(from_struct.ends_with(r#""parameters": {"properties": {"city": {"enum": [], "type": "string"}}, "required": ["city"], "type": "object"}}}"#), "{from_struct}");
    }
}

#[test]
fn tojson_indent_matches_python()
{
    assert_eq!(
        render(&format!("{{{{ {TOOL_LITERAL} | tojson(indent=4) }}}}")),
        "{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_weather\",\n        \"description\": \"Weather <now> & \\\"later\\\"\\n\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"city\": {\n                    \"type\": \"string\",\n                    \"enum\": []\n                }\n            },\n            \"required\": [\n                \"city\"\n            ]\n        }\n    }\n}"
    );
    assert_eq!(render(r#"{{ [1, {}] | tojson(indent="\t") }}"#), "[\n\t1,\n\t{}\n]");
    assert_eq!(render("{{ [1, [2]] | tojson(indent=0) }}"), "[\n1,\n[\n2\n]\n]");
}

#[test]
fn tojson_separators_sort_keys_and_ensure_ascii()
{
    assert_eq!(render(r#"{{ {"a": [1, 2]} | tojson(separators=(",", ":")) }}"#), r#"{"a":[1,2]}"#);
    assert_eq!(render(r#"{{ {"b": 1, "a": {"d": 2, "c": 3}} | tojson(sort_keys=true) }}"#), r#"{"a": {"c": 3, "d": 2}, "b": 1}"#);
    let s = Value::from("caf\u{e9} \u{4f60}\u{597d} \u{1F600} \u{7f} \u{1}");
    assert_eq!(pyjson::dumps(&s, None, false, false), "\"caf\u{e9} \u{4f60}\u{597d} \u{1F600} \u{7f} \\u0001\"");
    assert_eq!(pyjson::dumps(&s, None, false, true), r#""caf\u00e9 \u4f60\u597d \ud83d\ude00 \u007f \u0001""#);
    assert_eq!(render("{{ [none, true, false, 0, -3, 2.5] | tojson }}"), "[null, true, false, 0, -3, 2.5]");
}

#[test]
fn tojson_does_not_escape_html()
{
    // Jinja's built-in tojson would print \u003c and \u0026 here.
    assert_eq!(render(r#"{{ "<a href='x'>&</a>" | tojson }}"#), r#""<a href='x'>&</a>""#);
}

#[test]
fn tojson_floats_print_as_python_repr()
{
    let cases = [
        (1.0, "1.0"),
        (0.1, "0.1"),
        (1e16, "1e+16"),
        (1e15, "1000000000000000.0"),
        (1.5e-7, "1.5e-07"),
        (0.0001, "0.0001"),
        (123.456, "123.456"),
        (-2.5e300, "-2.5e+300"),
        (1e22, "1e+22"),
    ];
    for (f, python) in cases {
        assert_eq!(pyjson::dumps(&Value::from(f), None, false, false), python, "{f:?}");
    }
}

#[test]
fn tojson_refuses_undefined_and_unknown_arguments()
{
    let err = with("{{ nothing | tojson }}").render(&[], false).unwrap_err();
    assert!(err.to_string().contains("not JSON serializable"), "{err}");
    let err = with("{{ 1 | tojson(width=3) }}").render(&[], false).unwrap_err();
    assert!(matches!(err, ChatError::Template(_)), "{err:?}");
    assert_eq!(render("{{ [1, 2] | tojson(indent=none) }}"), "[1, 2]");
}
