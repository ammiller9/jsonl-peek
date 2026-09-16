//! A strict JSON parser for single records.
//!
//! `jsonl-peek` reads one line of a JSONL file at a time, so this parser
//! works on a single `&str` rather than a stream. It is deliberately strict:
//! trailing commas, single-quoted strings, unescaped control characters,
//! `NaN`/`Infinity`, and lone UTF-16 surrogates are all rejected, because a
//! loader downstream would choke on the same things.

use std::fmt;

/// A parsed JSON value. Objects keep insertion order (a plain `Vec` of pairs)
/// rather than sorting or hashing keys, since the whole point is to report
/// the data as it actually appears in the file.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// A short name for the value's type, used when reporting type
    /// distributions (`int:1990 string:10`).
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

/// A parse failure with the 1-based column (in Unicode scalar values) where
/// it occurred. The caller supplies the line number, since this parser only
/// ever sees one line at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub column: usize,
    pub reason: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "col {}: {}", self.column, self.reason)
    }
}

impl std::error::Error for ParseError {}

/// Parse a single JSON value from `input`, requiring that nothing but
/// whitespace surrounds it.
pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut parser = Parser::new(input);
    parser.skip_whitespace();
    if parser.peek().is_none() {
        return Err(parser.error("unexpected end of input"));
    }
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if let Some(c) = parser.peek() {
        return Err(parser.error(format!("unexpected trailing character '{}'", c)));
    }
    Ok(value)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    col: usize,
}

impl Parser {
    fn new(input: &str) -> Self {
        Parser {
            chars: input.chars().collect(),
            pos: 0,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
            self.col += 1;
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n') | Some('\r')) {
            self.advance();
        }
    }

    fn error(&self, reason: impl Into<String>) -> ParseError {
        ParseError {
            column: self.col,
            reason: reason.into(),
        }
    }

    fn error_at(&self, column: usize, reason: impl Into<String>) -> ParseError {
        ParseError {
            column,
            reason: reason.into(),
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_string().map(Value::String),
            Some('t') => self.parse_literal("true", Value::Bool(true)),
            Some('f') => self.parse_literal("false", Value::Bool(false)),
            Some('n') => self.parse_literal("null", Value::Null),
            Some('N') | Some('I') => Err(self.parse_non_finite()),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(self.error(format!("unexpected character '{}'", c))),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_literal(&mut self, text: &str, value: Value) -> Result<Value, ParseError> {
        let start_col = self.col;
        for expected in text.chars() {
            match self.advance() {
                Some(c) if c == expected => {}
                Some(c) => {
                    return Err(self.error_at(
                        start_col,
                        format!("invalid literal, expected '{}', found '{}'", text, c),
                    ));
                }
                None => {
                    return Err(self.error_at(
                        start_col,
                        format!("unexpected end of input while parsing '{}'", text),
                    ));
                }
            }
        }
        Ok(value)
    }

    /// Consumes a run of ASCII letters starting at `NaN`/`Infinity` so the
    /// error names the whole token instead of just its first character.
    fn parse_non_finite(&mut self) -> ParseError {
        let start_col = self.col;
        let mut word = String::new();
        while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
            word.push(self.advance().unwrap());
        }
        self.error_at(start_col, format!("'{}' is not valid JSON", word))
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.advance(); // '{'
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some('"') {
                return Err(self.error("expected string key"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(':') {
                return Err(self.error("expected ':' after object key"));
            }
            self.advance();
            self.skip_whitespace();
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some('}') {
                        return Err(self.error("trailing comma before '}'"));
                    }
                }
                Some('}') => {
                    self.advance();
                    return Ok(Value::Object(entries));
                }
                Some(c) => return Err(self.error(format!("expected ',' or '}}', found '{}'", c))),
                None => return Err(self.error("unexpected end of input in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.advance(); // '['
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            let value = self.parse_value()?;
            items.push(value);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some(']') {
                        return Err(self.error("trailing comma before ']'"));
                    }
                }
                Some(']') => {
                    self.advance();
                    return Ok(Value::Array(items));
                }
                Some(c) => return Err(self.error(format!("expected ',' or ']', found '{}'", c))),
                None => return Err(self.error("unexpected end of input in array")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        let start_col = self.col;
        self.advance(); // opening quote
        let mut out = String::new();
        loop {
            let char_col = self.col;
            match self.advance() {
                Some('"') => return Ok(out),
                Some('\\') => {
                    let esc_col = self.col;
                    match self.advance() {
                        Some('"') => out.push('"'),
                        Some('\\') => out.push('\\'),
                        Some('/') => out.push('/'),
                        Some('b') => out.push('\u{0008}'),
                        Some('f') => out.push('\u{000C}'),
                        Some('n') => out.push('\n'),
                        Some('r') => out.push('\r'),
                        Some('t') => out.push('\t'),
                        Some('u') => {
                            let cp = self.parse_hex4(esc_col)?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if self.peek() != Some('\\') || self.peek_at(1) != Some('u') {
                                    return Err(self.error_at(
                                        esc_col,
                                        "lone UTF-16 surrogate in \\u escape",
                                    ));
                                }
                                self.advance(); // backslash
                                self.advance(); // u
                                let low = self.parse_hex4(esc_col)?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(self.error_at(
                                        esc_col,
                                        "unpaired UTF-16 surrogate in \\u escape",
                                    ));
                                }
                                let combined = 0x10000 + (cp - 0xD800) * 0x400 + (low - 0xDC00);
                                match char::from_u32(combined) {
                                    Some(c) => out.push(c),
                                    None => {
                                        return Err(
                                            self.error_at(esc_col, "invalid surrogate pair")
                                        )
                                    }
                                }
                            } else if (0xDC00..=0xDFFF).contains(&cp) {
                                return Err(
                                    self.error_at(esc_col, "lone UTF-16 surrogate in \\u escape")
                                );
                            } else {
                                match char::from_u32(cp) {
                                    Some(c) => out.push(c),
                                    None => {
                                        return Err(self.error_at(esc_col, "invalid \\u escape"))
                                    }
                                }
                            }
                        }
                        Some(c) => {
                            return Err(
                                self.error_at(esc_col, format!("invalid escape '\\{}'", c))
                            )
                        }
                        None => {
                            return Err(self
                                .error_at(esc_col, "unexpected end of input in string escape"))
                        }
                    }
                }
                Some(c) if (c as u32) < 0x20 => {
                    return Err(self.error_at(
                        char_col,
                        format!("control character U+{:04X} must be escaped", c as u32),
                    ));
                }
                Some(c) => out.push(c),
                None => return Err(self.error_at(start_col, "unterminated string")),
            }
        }
    }

    fn parse_hex4(&mut self, esc_col: usize) -> Result<u32, ParseError> {
        let mut value: u32 = 0;
        for _ in 0..4 {
            match self.advance() {
                Some(c) if c.is_ascii_hexdigit() => {
                    value = value * 16 + c.to_digit(16).unwrap();
                }
                Some(c) => {
                    return Err(self.error_at(
                        esc_col,
                        format!("invalid hex digit '{}' in \\u escape", c),
                    ))
                }
                None => {
                    return Err(self.error_at(esc_col, "unexpected end of input in \\u escape"))
                }
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start_col = self.col;
        let mut text = String::new();
        if self.peek() == Some('-') {
            text.push(self.advance().unwrap());
        }
        match self.peek() {
            Some('0') => {
                text.push(self.advance().unwrap());
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    text.push(self.advance().unwrap());
                }
            }
            _ => return Err(self.error_at(start_col, "invalid number")),
        }
        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            text.push(self.advance().unwrap());
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.error_at(start_col, "expected digit after decimal point"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                text.push(self.advance().unwrap());
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            is_float = true;
            text.push(self.advance().unwrap());
            if matches!(self.peek(), Some('+') | Some('-')) {
                text.push(self.advance().unwrap());
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.error_at(start_col, "expected digit in exponent"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                text.push(self.advance().unwrap());
            }
        }
        if !is_float {
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Int(i));
            }
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| self.error_at(start_col, "invalid number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(input: &str) -> Value {
        parse(input).unwrap_or_else(|e| panic!("expected {:?} to parse, got {}", input, e))
    }

    fn err(input: &str) -> ParseError {
        parse(input).err().unwrap_or_else(|| panic!("expected {:?} to fail to parse", input))
    }

    #[test]
    fn literals() {
        assert_eq!(ok("null"), Value::Null);
        assert_eq!(ok("true"), Value::Bool(true));
        assert_eq!(ok("false"), Value::Bool(false));
    }

    #[test]
    fn integers_and_floats_are_distinct() {
        assert_eq!(ok("0"), Value::Int(0));
        assert_eq!(ok("-0"), Value::Int(0));
        assert_eq!(ok("42"), Value::Int(42));
        assert_eq!(ok("-17"), Value::Int(-17));
        assert_eq!(ok("10.0"), Value::Float(10.0));
        assert_eq!(ok("3.14"), Value::Float(3.14));
        assert_eq!(ok("1e2"), Value::Float(100.0));
        assert_eq!(ok("1E-3"), Value::Float(0.001));
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(ok("\"hello\""), Value::String("hello".to_string()));
        assert_eq!(ok("\"a\\nb\""), Value::String("a\nb".to_string()));
        assert_eq!(ok("\"\\u0041\""), Value::String("A".to_string()));
        // A surrogate pair for U+1F600 (grinning face).
        assert_eq!(
            ok("\"\\ud83d\\ude00\""),
            Value::String("\u{1F600}".to_string())
        );
    }

    #[test]
    fn arrays_and_objects() {
        assert_eq!(ok("[]"), Value::Array(vec![]));
        assert_eq!(
            ok("[1,2,3]"),
            Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );
        assert_eq!(ok("{}"), Value::Object(vec![]));
        assert_eq!(
            ok("{\"b\":1,\"a\":2}"),
            Value::Object(vec![
                ("b".to_string(), Value::Int(1)),
                ("a".to_string(), Value::Int(2)),
            ]),
            "object key order must match the source, not be sorted"
        );
    }

    #[test]
    fn rejects_trailing_commas() {
        assert!(err("[1,]").reason.contains("trailing comma"));
        assert!(err("{\"a\":1,}").reason.contains("trailing comma"));
    }

    #[test]
    fn rejects_single_quotes() {
        assert!(parse("'hi'").is_err());
    }

    #[test]
    fn rejects_non_finite_literals() {
        assert!(err("NaN").reason.contains("not valid JSON"));
        assert!(err("Infinity").reason.contains("not valid JSON"));
        assert!(parse("-Infinity").is_err());
    }

    #[test]
    fn rejects_leading_zeros() {
        assert!(parse("01").is_err());
    }

    #[test]
    fn rejects_malformed_numbers() {
        assert!(parse("1.").is_err());
        assert!(parse(".5").is_err());
        assert!(parse("-").is_err());
    }

    #[test]
    fn rejects_unterminated_and_broken_strings() {
        assert!(err("\"unterminated").reason.contains("unterminated"));
        assert!(err("\"bad\\x\"").reason.contains("invalid escape"));
    }

    #[test]
    fn rejects_unescaped_control_characters() {
        let input = "\"bad\u{0007}char\"";
        assert!(err(input).reason.contains("control character"));
    }

    #[test]
    fn rejects_lone_surrogates() {
        assert!(err("\"\\ud800\"").reason.contains("surrogate"));
        assert!(err("\"\\udc00\"").reason.contains("surrogate"));
    }

    #[test]
    fn rejects_missing_key_quotes_and_commas() {
        assert!(err("{a:1}").reason.contains("expected string key"));
        assert!(err("[1 2]").reason.contains("expected ',' or ']'"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(err("truee").reason.contains("trailing character"));
        assert!(err("null null").reason.contains("trailing character"));
    }

    #[test]
    fn error_column_points_at_the_offending_character() {
        // "[1 2]"
        //  12345
        let e = err("[1 2]");
        assert_eq!(e.column, 4);
    }

    #[test]
    fn type_names() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Int(1).type_name(), "int");
        assert_eq!(Value::Float(1.0).type_name(), "float");
        assert_eq!(Value::String(String::new()).type_name(), "string");
        assert_eq!(Value::Array(vec![]).type_name(), "array");
        assert_eq!(Value::Object(vec![]).type_name(), "object");
    }
}
