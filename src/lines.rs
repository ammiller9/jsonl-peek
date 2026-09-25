//! A line splitter that reuses one buffer instead of allocating a `String`
//! per line, since a multi-gigabyte JSONL file can have tens of millions of
//! them.
//!
//! It also does the small amount of cleanup a hand-rolled `split('\n')`
//! would get wrong: it strips a leading UTF-8 BOM (some export tools still
//! write one), strips the `\r` half of a CRLF ending, and returns the final
//! line of a file that has no trailing newline instead of silently dropping
//! it.

use std::io::{self, BufRead};

const BOM: &str = "\u{FEFF}";

/// Reads successive lines from `R`, handing back a borrowed `&str` into an
/// internal buffer rather than an owned `String`.
pub struct LineReader<R> {
    inner: R,
    buf: String,
    line_no: usize,
    stripped_bom: bool,
}

impl<R: BufRead> LineReader<R> {
    pub fn new(inner: R) -> Self {
        LineReader {
            inner,
            buf: String::new(),
            line_no: 0,
            stripped_bom: false,
        }
    }

    /// Reads the next line, stripping its line ending. Returns `None` at
    /// end of file. The returned `&str` borrows from an internal buffer and
    /// is only valid until the next call to `next_line`.
    pub fn next_line(&mut self) -> io::Result<Option<(usize, &str)>> {
        self.buf.clear();
        let read = self.inner.read_line(&mut self.buf)?;
        if read == 0 {
            return Ok(None);
        }
        self.line_no += 1;

        if !self.stripped_bom {
            self.stripped_bom = true;
            if self.buf.starts_with(BOM) {
                self.buf.drain(..BOM.len());
            }
        }

        if self.buf.ends_with('\n') {
            self.buf.pop();
            if self.buf.ends_with('\r') {
                self.buf.pop();
            }
        }

        Ok(Some((self.line_no, self.buf.as_str())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn collect(input: &str) -> Vec<(usize, String)> {
        let mut reader = LineReader::new(Cursor::new(input.as_bytes()));
        let mut out = Vec::new();
        while let Some((n, line)) = reader.next_line().unwrap() {
            out.push((n, line.to_string()));
        }
        out
    }

    #[test]
    fn splits_on_newline() {
        assert_eq!(
            collect("a\nb\nc\n"),
            vec![(1, "a".to_string()), (2, "b".to_string()), (3, "c".to_string())]
        );
    }

    #[test]
    fn keeps_last_line_without_trailing_newline() {
        assert_eq!(
            collect("a\nb"),
            vec![(1, "a".to_string()), (2, "b".to_string())]
        );
    }

    #[test]
    fn strips_crlf() {
        assert_eq!(
            collect("a\r\nb\r\n"),
            vec![(1, "a".to_string()), (2, "b".to_string())]
        );
    }

    #[test]
    fn strips_leading_bom_once() {
        let input = format!("{}a\nb\n", BOM);
        assert_eq!(
            collect(&input),
            vec![(1, "a".to_string()), (2, "b".to_string())]
        );
    }

    #[test]
    fn empty_input_yields_no_lines() {
        assert_eq!(collect(""), Vec::<(usize, String)>::new());
    }

    #[test]
    fn blank_lines_are_kept_empty() {
        assert_eq!(
            collect("\n\na\n"),
            vec![(1, "".to_string()), (2, "".to_string()), (3, "a".to_string())]
        );
    }

    #[test]
    fn line_without_final_newline_is_not_off_by_one() {
        let lines = collect("only");
        assert_eq!(lines, vec![(1, "only".to_string())]);
    }
}
