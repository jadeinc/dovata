//! Author: Joe Ricks
//! File: src/layout/expr.rs
//! Description: Parses and evaluates the expressions layout files use for shapes, op
//! parameters and conditions.

use std::fmt;

/// Purpose: This enum holds a parsed layout expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Int(i64),
    Float(f64),
    /// A symbol, `L`, a loop variable, or `src.neN` / `src.nbN`.
    Name(String),
    /// `has(<tensor name template>)`. The name is kept raw, braces and all.
    Has(String),
    Call(Func, Box<Expr>),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// Purpose: This enum lists the functions an expression may call: `sqrt` and `row_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Func {
    Sqrt,
    RowSize,
}

/// Purpose: This enum lists the operators that take two values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Purpose: This enum holds an expression's value: an integer, a float or a truth value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl fmt::Display for Value {
    /// Purpose: This function formats an expression's value as text.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
        }
    }
}

/// Purpose: This trait gives an expression the values of the names it reads.
pub(crate) trait Scope {
    /// Purpose: This function returns the value of a name an expression reads.
    ///
    /// Inputs:
    /// - `name`: a symbol, `L`, a loop variable, or `src.*`.
    ///
    /// Outputs: the value, or an error when the name has none here.
    fn name(&self, name: &str) -> Result<Value, String>;
    /// Purpose: This function tells whether the model file holds a tensor.
    ///
    /// Inputs:
    /// - `template`: the tensor name, with its placeholders still in it.
    ///
    /// Outputs: `true` when the file holds the tensor, or an error.
    fn has(&self, template: &str) -> Result<bool, String>;
    /// Purpose: This function returns the bytes `n` elements of the op's first input type
    /// take.
    ///
    /// Inputs:
    /// - `n`: the element count.
    ///
    /// Outputs: the size in bytes, or an error.
    fn row_size(&self, n: i64) -> Result<i64, String>;
}

impl Expr {
    /// Purpose: This function parses one expression's text into a tree.
    ///
    /// Inputs:
    /// - `src`: the expression's text.
    ///
    /// Outputs: the parsed expression, or an error saying where the text went wrong.
    pub(crate) fn parse(src: &str) -> Result<Expr, String> {
        let tokens = lex(src)?;
        let mut p = Parser { tokens, pos: 0 };
        let e = p.expr()?;
        if p.pos != p.tokens.len() {
            return Err(format!("unexpected `{}` in `{src}`", p.tokens[p.pos]));
        }
        Ok(e)
    }

    /// Purpose: This function adds every name the expression reads to `out`.
    ///
    /// Inputs:
    /// - `out`: the list to add to.
    ///
    /// Outputs: none.
    pub(crate) fn names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Expr::Int(_) | Expr::Float(_) | Expr::Has(_) => {}
            Expr::Name(n) => out.push(n),
            Expr::Call(_, a) | Expr::Not(a) | Expr::Neg(a) => a.names(out),
            Expr::Binary(_, a, b) => {
                a.names(out);
                b.names(out);
            }
            Expr::Cond(c, a, b) => {
                c.names(out);
                a.names(out);
                b.names(out);
            }
        }
    }

    /// Purpose: This function adds the tensor name of every `has()` in the expression to
    /// `out`.
    ///
    /// Inputs:
    /// - `out`: the list to add to.
    ///
    /// Outputs: none.
    pub(crate) fn has_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Expr::Int(_) | Expr::Float(_) | Expr::Name(_) => {}
            Expr::Has(n) => out.push(n),
            Expr::Call(_, a) | Expr::Not(a) | Expr::Neg(a) => a.has_names(out),
            Expr::Binary(_, a, b) => {
                a.has_names(out);
                b.has_names(out);
            }
            Expr::Cond(c, a, b) => {
                c.has_names(out);
                a.has_names(out);
                b.has_names(out);
            }
        }
    }

    /// Purpose: This function tells whether the expression calls `row_size()`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` when it does.
    pub(crate) fn uses_row_size(&self) -> bool {
        match self {
            Expr::Int(_) | Expr::Float(_) | Expr::Name(_) | Expr::Has(_) => false,
            Expr::Call(f, a) => *f == Func::RowSize || a.uses_row_size(),
            Expr::Not(a) | Expr::Neg(a) => a.uses_row_size(),
            Expr::Binary(_, a, b) => a.uses_row_size() || b.uses_row_size(),
            Expr::Cond(c, a, b) => c.uses_row_size() || a.uses_row_size() || b.uses_row_size(),
        }
    }

    /// Purpose: This function computes the expression's value.
    ///
    /// Inputs:
    /// - `s`: the scope that gives names their values.
    ///
    /// Outputs: an integer, a float or a truth value, or an error.
    pub(crate) fn eval(&self, s: &dyn Scope) -> Result<Value, String> {
        Ok(match self {
            Expr::Int(v) => Value::Int(*v),
            Expr::Float(v) => Value::Float(*v),
            Expr::Name(n) => s.name(n)?,
            Expr::Has(t) => Value::Bool(s.has(t)?),
            Expr::Call(Func::Sqrt, a) => Value::Float(num(a.eval(s)?, "sqrt")?.sqrt()),
            Expr::Call(Func::RowSize, a) => match a.eval(s)? {
                Value::Int(n) => Value::Int(s.row_size(n)?),
                other => return Err(format!("row_size needs an integer element count, got {other}")),
            },
            Expr::Not(a) => Value::Bool(!truth(a.eval(s)?, "!")?),
            Expr::Neg(a) => match a.eval(s)? {
                Value::Int(v) => Value::Int(v.checked_neg().ok_or("integer overflow in negation")?),
                Value::Float(v) => Value::Float(-v),
                Value::Bool(_) => return Err("cannot negate a condition; use !".into()),
            },
            Expr::Cond(c, a, b) => {
                if truth(c.eval(s)?, "?:")? {
                    a.eval(s)?
                } else {
                    b.eval(s)?
                }
            }
            Expr::Binary(BinOp::And, a, b) => Value::Bool(truth(a.eval(s)?, "&&")? && truth(b.eval(s)?, "&&")?),
            Expr::Binary(BinOp::Or, a, b) => Value::Bool(truth(a.eval(s)?, "||")? || truth(b.eval(s)?, "||")?),
            Expr::Binary(op, a, b) => binary(*op, a.eval(s)?, b.eval(s)?)?,
        })
    }

    /// Purpose: This function computes the expression and requires an integer.
    ///
    /// Inputs:
    /// - `s`: the scope that gives names their values.
    ///
    /// Outputs: the integer, or an error for any other kind of value.
    pub(crate) fn eval_int(&self, s: &dyn Scope) -> Result<i64, String> {
        match self.eval(s)? {
            Value::Int(v) => Ok(v),
            other => Err(format!("expected an integer, got {other}")),
        }
    }

    /// Purpose: This function computes the expression and requires an integer that is not
    /// negative.
    ///
    /// Inputs:
    /// - `s`: the scope that gives names their values.
    ///
    /// Outputs: the value as a `u64`, or an error.
    pub(crate) fn eval_u64(&self, s: &dyn Scope) -> Result<u64, String> {
        let v = self.eval_int(s)?;
        u64::try_from(v).map_err(|_| format!("expected a size, got {v}"))
    }

    /// Purpose: This function computes the expression and requires a number.
    ///
    /// Inputs:
    /// - `s`: the scope that gives names their values.
    ///
    /// Outputs: the value as an `f64`, or an error for a truth value.
    pub(crate) fn eval_f64(&self, s: &dyn Scope) -> Result<f64, String> {
        num(self.eval(s)?, "a float parameter")
    }

    /// Purpose: This function computes the expression and requires a truth value.
    ///
    /// Inputs:
    /// - `s`: the scope that gives names their values.
    ///
    /// Outputs: the truth value, or an error for a number.
    pub(crate) fn eval_bool(&self, s: &dyn Scope) -> Result<bool, String> {
        truth(self.eval(s)?, "a condition")
    }
}

/// Purpose: This function reads a value as a float.
///
/// Inputs:
/// - `v`: the value.
/// - `what`: what needs the number, for the error.
///
/// Outputs: the float, or an error for a truth value.
fn num(v: Value, what: &str) -> Result<f64, String> {
    match v {
        #[allow(clippy::cast_precision_loss)]
        Value::Int(i) => Ok(i as f64),
        Value::Float(f) => Ok(f),
        Value::Bool(_) => Err(format!("{what} needs a number, got a condition")),
    }
}

/// Purpose: This function reads a value as a truth value.
///
/// Inputs:
/// - `v`: the value.
/// - `what`: what needs the truth value, for the error.
///
/// Outputs: the truth value, or an error for a number.
fn truth(v: Value, what: &str) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(b),
        other => Err(format!("{what} needs a condition, got the number {other}; compare it, as in `x != 0`")),
    }
}

/// Purpose: This function applies an arithmetic or comparison operator to two values.
///
/// Inputs:
/// - `op`: the operator.
/// - `a`: the left value.
/// - `b`: the right value.
///
/// Outputs: the result, or an error for overflow, division by zero, or a value of the
/// wrong kind.
fn binary(op: BinOp, a: Value, b: Value) -> Result<Value, String> {
    use BinOp::*;
    if let (Value::Bool(x), Value::Bool(y)) = (a, b) {
        return match op {
            Eq => Ok(Value::Bool(x == y)),
            Ne => Ok(Value::Bool(x != y)),
            _ => Err("arithmetic on a condition".into()),
        };
    }
    if let (Value::Int(x), Value::Int(y)) = (a, b) {
        let overflow = || "integer overflow".to_string();
        return Ok(match op {
            Add => Value::Int(x.checked_add(y).ok_or_else(overflow)?),
            Sub => Value::Int(x.checked_sub(y).ok_or_else(overflow)?),
            Mul => Value::Int(x.checked_mul(y).ok_or_else(overflow)?),
            Div if y == 0 => return Err("division by zero".into()),
            Div => Value::Int(x / y),
            Rem if y == 0 => return Err("remainder by zero".into()),
            Rem => Value::Int(x % y),
            Eq => Value::Bool(x == y),
            Ne => Value::Bool(x != y),
            Lt => Value::Bool(x < y),
            Le => Value::Bool(x <= y),
            Gt => Value::Bool(x > y),
            Ge => Value::Bool(x >= y),
            And | Or => unreachable!("handled before"),
        });
    }
    let (x, y) = (num(a, "arithmetic")?, num(b, "arithmetic")?);
    Ok(match op {
        Add => Value::Float(x + y),
        Sub => Value::Float(x - y),
        Mul => Value::Float(x * y),
        Div => Value::Float(x / y),
        Rem => return Err("% needs integers".into()),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or => unreachable!("handled before"),
    })
}

/// Purpose: This enum holds one token of an expression's text.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    Float(f64),
    Name(String),
    /// The raw text inside `has(...)`.
    Has(String),
    Sym(&'static str),
}

impl fmt::Display for Tok {
    /// Purpose: This function formats an expression token as it appeared in the text.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Int(v) => write!(f, "{v}"),
            Tok::Float(v) => write!(f, "{v}"),
            Tok::Name(n) => write!(f, "{n}"),
            Tok::Has(n) => write!(f, "has({n})"),
            Tok::Sym(s) => write!(f, "{s}"),
        }
    }
}

/// Purpose: This function splits an expression's text into tokens.
///
/// Inputs:
/// - `src`: the expression's text.
///
/// Outputs: the tokens, or an error for text the language does not allow.
fn lex(src: &str) -> Result<Vec<Tok>, String> {
    const SYMS: [&str; 20] = [
        "==", "!=", "<=", ">=", "&&", "||", "<", ">", "+", "-", "*", "/", "%", "(", ")", "?", ":", "!", ",", ".",
    ];
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.' || b[i] == b'e' || b[i] == b'E') {
                // An exponent may carry a sign: 1e-6.
                if (b[i] == b'e' || b[i] == b'E') && i + 1 < b.len() && (b[i + 1] == b'-' || b[i + 1] == b'+') {
                    i += 1;
                }
                i += 1;
            }
            let text = &src[start..i];
            if text.contains(['.', 'e', 'E']) {
                out.push(Tok::Float(text.parse().map_err(|_| format!("bad number `{text}` in `{src}`"))?));
            } else {
                out.push(Tok::Int(text.parse().map_err(|_| format!("bad number `{text}` in `{src}`"))?));
            }
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            // `src.ne0` is one name: a dot followed by a letter continues it.
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || (b[i] == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_alphabetic()))
            {
                i += 1;
            }
            let name = &src[start..i];
            if name == "has" && b.get(i) == Some(&b'(') {
                let close = src[i..].find(')').ok_or_else(|| format!("unclosed has( in `{src}`"))? + i;
                let inner = src[i + 1..close].trim();
                if inner.is_empty() {
                    return Err(format!("has() needs a tensor name in `{src}`"));
                }
                out.push(Tok::Has(inner.to_owned()));
                i = close + 1;
            } else {
                out.push(Tok::Name(name.to_owned()));
            }
        } else {
            let rest = &src[i..];
            let sym = SYMS.iter().find(|s| rest.starts_with(**s)).ok_or_else(|| format!("unexpected `{}` in `{src}`", c as char))?;
            out.push(Tok::Sym(sym));
            i += sym.len();
        }
    }
    Ok(out)
}

/// Purpose: This struct holds an expression's tokens and the parser's place in them.
struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    /// Purpose: This function tells whether the next token is a given symbol.
    ///
    /// Inputs:
    /// - `s`: the symbol.
    ///
    /// Outputs: `true` when it is.
    fn peek_sym(&self, s: &str) -> bool {
        matches!(self.tokens.get(self.pos), Some(Tok::Sym(t)) if *t == s)
    }

    /// Purpose: This function consumes the next token when it is a given symbol.
    ///
    /// Inputs:
    /// - `s`: the symbol.
    ///
    /// Outputs: `true` when the symbol was consumed.
    fn eat(&mut self, s: &str) -> bool {
        if self.peek_sym(s) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Purpose: This function consumes the next token, which must be a given symbol.
    ///
    /// Inputs:
    /// - `s`: the symbol that must come next.
    ///
    /// Outputs: `Ok(())`, or an error naming what was found instead.
    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(match self.tokens.get(self.pos) {
                Some(t) => format!("expected `{s}`, found `{t}`"),
                None => format!("expected `{s}` at the end"),
            })
        }
    }

    /// Purpose: This function parses a whole expression, including `c ? a : b`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn expr(&mut self) -> Result<Expr, String> {
        let c = self.or()?;
        if self.eat("?") {
            let a = self.expr()?;
            self.expect(":")?;
            let b = self.expr()?;
            return Ok(Expr::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    /// Purpose: This function parses conditions joined by `||`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn or(&mut self) -> Result<Expr, String> {
        let mut e = self.and()?;
        while self.eat("||") {
            e = Expr::Binary(BinOp::Or, Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    /// Purpose: This function parses conditions joined by `&&`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn and(&mut self) -> Result<Expr, String> {
        let mut e = self.cmp()?;
        while self.eat("&&") {
            e = Expr::Binary(BinOp::And, Box::new(e), Box::new(self.cmp()?));
        }
        Ok(e)
    }

    /// Purpose: This function parses at most one comparison, such as `L == n_layer - 1`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn cmp(&mut self) -> Result<Expr, String> {
        let e = self.sum()?;
        for (s, op) in [("==", BinOp::Eq), ("!=", BinOp::Ne), ("<=", BinOp::Le), (">=", BinOp::Ge), ("<", BinOp::Lt), (">", BinOp::Gt)] {
            if self.eat(s) {
                return Ok(Expr::Binary(op, Box::new(e), Box::new(self.sum()?)));
            }
        }
        Ok(e)
    }

    /// Purpose: This function parses additions and subtractions.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn sum(&mut self) -> Result<Expr, String> {
        let mut e = self.product()?;
        loop {
            let op = if self.eat("+") {
                BinOp::Add
            } else if self.eat("-") {
                BinOp::Sub
            } else {
                return Ok(e);
            };
            e = Expr::Binary(op, Box::new(e), Box::new(self.product()?));
        }
    }

    /// Purpose: This function parses multiplications, divisions and remainders.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn product(&mut self) -> Result<Expr, String> {
        let mut e = self.unary()?;
        loop {
            let op = if self.eat("*") {
                BinOp::Mul
            } else if self.eat("/") {
                BinOp::Div
            } else if self.eat("%") {
                BinOp::Rem
            } else {
                return Ok(e);
            };
            e = Expr::Binary(op, Box::new(e), Box::new(self.unary()?));
        }
    }

    /// Purpose: This function parses a leading `!` or `-` and what follows it.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat("!") {
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.primary()
    }

    /// Purpose: This function parses a number, a name, a function call, `has()`, or an
    /// expression in parentheses.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed expression, or an error.
    fn primary(&mut self) -> Result<Expr, String> {
        let t = self.tokens.get(self.pos).cloned().ok_or("expression ends too soon")?;
        self.pos += 1;
        match t {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::Float(v) => Ok(Expr::Float(v)),
            Tok::Has(n) => Ok(Expr::Has(n)),
            Tok::Name(n) => {
                if self.eat("(") {
                    let f = match n.as_str() {
                        "sqrt" => Func::Sqrt,
                        "row_size" => Func::RowSize,
                        other => return Err(format!("unknown function `{other}`; the functions are sqrt, row_size and has")),
                    };
                    let a = self.expr()?;
                    self.expect(")")?;
                    return Ok(Expr::Call(f, Box::new(a)));
                }
                Ok(Expr::Name(n))
            }
            Tok::Sym("(") => {
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            Tok::Sym(s) => Err(format!("unexpected `{s}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Purpose: This struct holds the fixed names and values the expression tests use.
    struct Vars(HashMap<&'static str, Value>);

    impl Scope for Vars {
        /// Purpose: This function returns a test variable's value.
        ///
        /// Inputs:
        /// - `n`: the name.
        ///
        /// Outputs: the value, or an error for a name the tests did not set.
        fn name(&self, n: &str) -> Result<Value, String> {
            self.0.get(n).copied().ok_or_else(|| format!("unknown {n}"))
        }
        /// Purpose: This function answers `has()` in the tests, where only
        /// `blk.{L}.attn_q.bias` exists.
        ///
        /// Inputs:
        /// - `t`: the tensor name.
        ///
        /// Outputs: `true` for that one name.
        fn has(&self, t: &str) -> Result<bool, String> {
            Ok(t == "blk.{L}.attn_q.bias")
        }
        /// Purpose: This function answers `row_size()` in the tests with 18 bytes per 32
        /// elements.
        ///
        /// Inputs:
        /// - `n`: the element count.
        ///
        /// Outputs: `n / 32 * 18`.
        fn row_size(&self, n: i64) -> Result<i64, String> {
            Ok(n / 32 * 18)
        }
    }

    /// Purpose: This function builds the fixed scope the expression tests use.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the scope.
    fn vars() -> Vars {
        Vars(HashMap::from([
            ("n_embd", Value::Int(2048)),
            ("n_head", Value::Int(32)),
            ("n_layer", Value::Int(48)),
            ("L", Value::Int(47)),
            ("eps", Value::Float(1e-6)),
            ("q", Value::Bool(true)),
        ]))
    }

    /// Purpose: This function parses and evaluates an expression in the test scope.
    ///
    /// Inputs:
    /// - `src`: the expression's text.
    ///
    /// Outputs: the value. It panics on an error.
    fn eval(src: &str) -> Value {
        Expr::parse(src).unwrap().eval(&vars()).unwrap()
    }

    #[test]
    fn integer_arithmetic_is_c_arithmetic() {
        assert_eq!(eval("n_embd / n_head"), Value::Int(64));
        assert_eq!(eval("7 / 2"), Value::Int(3));
        assert_eq!(eval("2 + 3 * 4"), Value::Int(14));
        assert_eq!(eval("(2 + 3) * 4"), Value::Int(20));
        assert_eq!(eval("(L + 1) % 4"), Value::Int(0));
        assert_eq!(eval("-3 + 5"), Value::Int(2));
    }

    #[test]
    fn floats_and_functions() {
        assert_eq!(eval("1 / sqrt(64)"), Value::Float(0.125));
        assert_eq!(eval("1e-6"), Value::Float(1e-6));
        assert_eq!(eval("eps * 2"), Value::Float(2e-6));
        assert_eq!(eval("row_size(64)"), Value::Int(36));
    }

    #[test]
    fn conditions() {
        assert_eq!(eval("L == n_layer - 1"), Value::Bool(true));
        assert_eq!(eval("q && n_embd % 64 == 0"), Value::Bool(true));
        assert_eq!(eval("!q || L < 3"), Value::Bool(false));
        assert_eq!(eval("has(blk.{L}.attn_q.bias)"), Value::Bool(true));
        assert_eq!(eval("!has(blk.{L}.attn_k.bias)"), Value::Bool(true));
        assert_eq!(eval("n_head != 0 ? n_embd / n_head : 0"), Value::Int(64));
    }

    #[test]
    fn src_names_are_one_token() {
        let e = Expr::parse("src.ne0 * src.nb2").unwrap();
        let mut names = Vec::new();
        e.names(&mut names);
        assert_eq!(names, ["src.ne0", "src.nb2"]);
    }

    #[test]
    fn mistakes_are_errors_not_guesses() {
        assert!(Expr::parse("n_embd +").is_err());
        assert!(Expr::parse("(n_embd").is_err());
        assert!(Expr::parse("n_embd $ 2").is_err());
        assert!(Expr::parse("cbrt(8)").is_err());
        let s = vars();
        assert!(Expr::parse("n_embd / 0").unwrap().eval(&s).is_err());
        assert!(Expr::parse("q + 1").unwrap().eval(&s).is_err());
        assert!(Expr::parse("n_embd && q").unwrap().eval(&s).is_err(), "a number is not a condition");
        assert!(Expr::parse("1.5 * 2").unwrap().eval_int(&s).is_err(), "a float is not a dim");
        assert!(Expr::parse("0 - 1").unwrap().eval_u64(&s).is_err(), "a size is not negative");
    }
}
