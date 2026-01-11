"""Comprehensive tests for pdfminer/psparser.py"""

from io import BytesIO

import pytest

from pdfminer.psexceptions import PSEOF, PSSyntaxError, PSTypeError
from pdfminer.psparser import (
    KWD,
    KEYWORD_ARRAY_BEGIN,
    KEYWORD_ARRAY_END,
    KEYWORD_DICT_BEGIN,
    KEYWORD_DICT_END,
    KEYWORD_PROC_BEGIN,
    KEYWORD_PROC_END,
    LIT,
    PSBaseParser,
    PSKeyword,
    PSKeywordTable,
    PSLiteral,
    PSLiteralTable,
    PSObject,
    PSStackParser,
    PSSymbolTable,
    keyword_name,
    literal_name,
)
from pdfminer import settings


class TestPSObject:
    """Tests for PSObject base class"""

    def test_psobj_instantiation(self):
        obj = PSObject()
        assert isinstance(obj, PSObject)


class TestPSLiteral:
    """Tests for PSLiteral class"""

    def test_literal_with_string_name(self):
        lit = PSLiteral("TestName")
        assert lit.name == "TestName"

    def test_literal_with_bytes_name(self):
        lit = PSLiteral(b"TestName")
        assert lit.name == b"TestName"

    def test_literal_repr(self):
        lit = PSLiteral("Test")
        assert "Test" in repr(lit)
        assert "/" in repr(lit)

    def test_literal_empty_name(self):
        lit = PSLiteral("")
        assert lit.name == ""


class TestPSKeyword:
    """Tests for PSKeyword class"""

    def test_keyword_with_bytes_name(self):
        kwd = PSKeyword(b"stream")
        assert kwd.name == b"stream"

    def test_keyword_repr(self):
        kwd = PSKeyword(b"endobj")
        assert "endobj" in repr(kwd)

    def test_keyword_empty_name(self):
        kwd = PSKeyword(b"")
        assert kwd.name == b""


class TestPSSymbolTable:
    """Tests for PSSymbolTable class"""

    def test_literal_table_intern_string(self):
        lit1 = LIT("TestLiteral")
        lit2 = LIT("TestLiteral")
        assert lit1 is lit2

    def test_literal_table_intern_bytes(self):
        lit1 = LIT(b"BytesLiteral")
        lit2 = LIT(b"BytesLiteral")
        assert lit1 is lit2

    def test_literal_table_different_names(self):
        lit1 = LIT("Name1")
        lit2 = LIT("Name2")
        assert lit1 is not lit2

    def test_keyword_table_intern(self):
        kwd1 = KWD(b"keyword")
        kwd2 = KWD(b"keyword")
        assert kwd1 is kwd2

    def test_keyword_table_different_names(self):
        kwd1 = KWD(b"kwd1")
        kwd2 = KWD(b"kwd2")
        assert kwd1 is not kwd2

    def test_symbol_table_custom(self):
        table = PSSymbolTable(PSLiteral)
        lit1 = table.intern("custom")
        lit2 = table.intern("custom")
        assert lit1 is lit2
        assert isinstance(lit1, PSLiteral)


class TestPredefinedKeywords:
    """Tests for predefined keyword constants"""

    def test_keyword_proc_begin(self):
        assert KEYWORD_PROC_BEGIN.name == b"{"

    def test_keyword_proc_end(self):
        assert KEYWORD_PROC_END.name == b"}"

    def test_keyword_array_begin(self):
        assert KEYWORD_ARRAY_BEGIN.name == b"["

    def test_keyword_array_end(self):
        assert KEYWORD_ARRAY_END.name == b"]"

    def test_keyword_dict_begin(self):
        assert KEYWORD_DICT_BEGIN.name == b"<<"

    def test_keyword_dict_end(self):
        assert KEYWORD_DICT_END.name == b">>"


class TestLiteralName:
    """Tests for literal_name function"""

    def test_literal_name_with_string(self):
        lit = LIT("TestName")
        assert literal_name(lit) == "TestName"

    def test_literal_name_with_bytes_utf8(self):
        lit = LIT(b"TestName")
        assert literal_name(lit) == "TestName"

    def test_literal_name_with_non_utf8_bytes(self):
        lit = LIT(b"\xff\xfe")
        result = literal_name(lit)
        assert isinstance(result, str)

    def test_literal_name_with_non_literal_non_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            result = literal_name("just a string")
            assert result == "just a string"
        finally:
            settings.STRICT = original_strict

    def test_literal_name_with_non_literal_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PSTypeError):
                literal_name("just a string")
        finally:
            settings.STRICT = original_strict


class TestKeywordName:
    """Tests for keyword_name function"""

    def test_keyword_name_with_keyword(self):
        kwd = KWD(b"stream")
        assert keyword_name(kwd) == "stream"

    def test_keyword_name_with_non_keyword_non_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            result = keyword_name("notakeyword")
            assert result == "notakeyword"
        finally:
            settings.STRICT = original_strict

    def test_keyword_name_with_non_keyword_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PSTypeError):
                keyword_name("notakeyword")
        finally:
            settings.STRICT = original_strict


class TestPSBaseParser:
    """Tests for PSBaseParser class"""

    def test_parser_init(self):
        fp = BytesIO(b"test data")
        parser = PSBaseParser(fp)
        assert parser.fp is fp
        assert not parser.eof

    def test_parser_repr(self):
        fp = BytesIO(b"test")
        parser = PSBaseParser(fp)
        repr_str = repr(parser)
        assert "PSBaseParser" in repr_str
        assert "bufpos" in repr_str

    def test_parser_seek(self):
        fp = BytesIO(b"0123456789")
        parser = PSBaseParser(fp)
        parser.seek(5)
        assert parser.bufpos == 5

    def test_parser_flush(self):
        fp = BytesIO(b"test")
        parser = PSBaseParser(fp)
        parser.flush()

    def test_nextline_simple(self):
        fp = BytesIO(b"line1\nline2\n")
        parser = PSBaseParser(fp)
        pos, line = parser.nextline()
        assert line == b"line1\n"

    def test_nextline_crlf(self):
        fp = BytesIO(b"line1\r\nline2\r\n")
        parser = PSBaseParser(fp)
        pos, line = parser.nextline()
        assert line == b"line1\r\n"

    def test_nextline_cr_only(self):
        fp = BytesIO(b"line1\rline2\r")
        parser = PSBaseParser(fp)
        pos, line = parser.nextline()
        assert line == b"line1\r"

    def test_revreadlines(self):
        data = b"line1\nline2\nline3\n"
        fp = BytesIO(data)
        parser = PSBaseParser(fp)
        lines = list(parser.revreadlines())
        assert len(lines) >= 1

    def test_nexttoken_integer(self):
        fp = BytesIO(b"42 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 42

    def test_nexttoken_negative_integer(self):
        fp = BytesIO(b"-123 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == -123

    def test_nexttoken_positive_integer(self):
        fp = BytesIO(b"+456 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 456

    def test_nexttoken_float(self):
        fp = BytesIO(b"3.14 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 3.14

    def test_nexttoken_float_no_integer_part(self):
        fp = BytesIO(b".5 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 0.5

    def test_nexttoken_literal(self):
        fp = BytesIO(b"/Name ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert isinstance(token, PSLiteral)
        assert token == LIT("Name")

    def test_nexttoken_literal_with_hex(self):
        fp = BytesIO(b"/Name#20Space ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert isinstance(token, PSLiteral)
        assert token.name == "Name Space"

    def test_nexttoken_keyword(self):
        fp = BytesIO(b"stream ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert isinstance(token, PSKeyword)
        assert token == KWD(b"stream")

    def test_nexttoken_boolean_true(self):
        fp = BytesIO(b"true ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token is True

    def test_nexttoken_boolean_false(self):
        fp = BytesIO(b"false ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token is False

    def test_nexttoken_string(self):
        fp = BytesIO(b"(hello) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"hello"

    def test_nexttoken_empty_string(self):
        fp = BytesIO(b"() ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b""

    def test_nexttoken_string_with_parens(self):
        fp = BytesIO(b"(hello (world)) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"hello (world)"

    def test_nexttoken_string_with_escapes(self):
        fp = BytesIO(b"(hello\\nworld) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"hello\nworld"

    def test_nexttoken_string_with_octal(self):
        fp = BytesIO(b"(\\101BC) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"ABC"

    def test_nexttoken_hexstring(self):
        fp = BytesIO(b"<48656C6C6F> ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"Hello"

    def test_nexttoken_hexstring_empty(self):
        fp = BytesIO(b"<> ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b""

    def test_nexttoken_hexstring_with_whitespace(self):
        fp = BytesIO(b"< 48 65 6C 6C 6F > ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"Hello"

    def test_nexttoken_dict_begin(self):
        fp = BytesIO(b"<< ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == KEYWORD_DICT_BEGIN

    def test_nexttoken_dict_end(self):
        fp = BytesIO(b">> ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == KEYWORD_DICT_END

    def test_nexttoken_array_begin(self):
        fp = BytesIO(b"[ ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == KEYWORD_ARRAY_BEGIN

    def test_nexttoken_array_end(self):
        fp = BytesIO(b"] ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == KEYWORD_ARRAY_END

    def test_nexttoken_comment_skipped(self):
        fp = BytesIO(b"% this is a comment\n42 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 42

    def test_nexttoken_eof(self):
        fp = BytesIO(b"")
        parser = PSBaseParser(fp)
        with pytest.raises(PSEOF):
            parser.nexttoken()

    def test_nexttoken_null_byte_skipped(self):
        fp = BytesIO(b"\x00 42 ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == 42

    def test_multiple_tokens(self):
        fp = BytesIO(b"1 2 3 ")
        parser = PSBaseParser(fp)
        tokens = []
        for _ in range(3):
            _, token = parser.nexttoken()
            tokens.append(token)
        assert tokens == [1, 2, 3]


class TestPSStackParser:
    """Tests for PSStackParser class"""

    def get_parser(self, data):
        class TestParser(PSStackParser):
            def flush(self):
                self.add_results(*self.popall())

        return TestParser(BytesIO(data))

    def test_parser_init(self):
        parser = self.get_parser(b"test")
        assert parser.curstack == []
        assert parser.results == []
        assert parser.context == []

    def test_parser_reset(self):
        parser = self.get_parser(b"test")
        parser.curstack.append((0, "test"))
        parser.reset()
        assert parser.curstack == []

    def test_parser_push(self):
        parser = self.get_parser(b"test")
        parser.push((0, 42))
        assert len(parser.curstack) == 1
        assert parser.curstack[0] == (0, 42)

    def test_parser_push_multiple(self):
        parser = self.get_parser(b"test")
        parser.push((0, 1), (1, 2), (2, 3))
        assert len(parser.curstack) == 3

    def test_parser_pop(self):
        parser = self.get_parser(b"test")
        parser.push((0, 1), (1, 2), (2, 3))
        popped = parser.pop(2)
        assert len(popped) == 2
        assert len(parser.curstack) == 1

    def test_parser_popall(self):
        parser = self.get_parser(b"test")
        parser.push((0, 1), (1, 2), (2, 3))
        popped = parser.popall()
        assert len(popped) == 3
        assert parser.curstack == []

    def test_parser_add_results(self):
        parser = self.get_parser(b"test")
        parser.add_results((0, 42))
        assert len(parser.results) == 1

    def test_nextobject_integer(self):
        parser = self.get_parser(b"42 ")
        pos, obj = parser.nextobject()
        assert obj == 42

    def test_nextobject_literal(self):
        parser = self.get_parser(b"/Name ")
        pos, obj = parser.nextobject()
        assert isinstance(obj, PSLiteral)

    def test_nextobject_string(self):
        parser = self.get_parser(b"(hello) ")
        pos, obj = parser.nextobject()
        assert obj == b"hello"

    def test_nextobject_array(self):
        parser = self.get_parser(b"[ 1 2 3 ] ")
        pos, obj = parser.nextobject()
        assert obj == [1, 2, 3]

    def test_nextobject_empty_array(self):
        parser = self.get_parser(b"[ ] ")
        pos, obj = parser.nextobject()
        assert obj == []

    def test_nextobject_nested_array(self):
        parser = self.get_parser(b"[ 1 [ 2 3 ] 4 ] ")
        pos, obj = parser.nextobject()
        assert obj == [1, [2, 3], 4]

    def test_nextobject_dict(self):
        parser = self.get_parser(b"<< /Key (value) >> ")
        pos, obj = parser.nextobject()
        assert obj == {"Key": b"value"}

    def test_nextobject_empty_dict(self):
        parser = self.get_parser(b"<< >> ")
        pos, obj = parser.nextobject()
        assert obj == {}

    def test_nextobject_proc(self):
        parser = self.get_parser(b"{ 1 2 add } ")
        pos, obj = parser.nextobject()
        assert isinstance(obj, list)
        assert 1 in obj
        assert 2 in obj

    def test_start_end_type(self):
        parser = self.get_parser(b"test")
        parser.start_type(0, "test_type")
        assert parser.curtype == "test_type"
        pos, objs = parser.end_type("test_type")
        assert objs == []

    def test_end_type_mismatch(self):
        parser = self.get_parser(b"test")
        parser.start_type(0, "type_a")
        with pytest.raises(PSTypeError):
            parser.end_type("type_b")


class TestTokenizationEdgeCases:
    """Tests for edge cases in tokenization"""

    def test_literal_with_special_chars(self):
        fp = BytesIO(b"/Name#2FWith#2FSlashes ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert isinstance(token, PSLiteral)
        assert token.name == "Name/With/Slashes"

    def test_string_multiline(self):
        fp = BytesIO(b"(line1\nline2) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"line1\nline2"

    def test_string_with_backslash_newline(self):
        fp = BytesIO(b"(hello\\\nworld) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"helloworld"

    def test_string_with_backslash_crlf(self):
        fp = BytesIO(b"(hello\\\r\nworld) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"helloworld"

    def test_string_escape_sequences(self):
        fp = BytesIO(b"(\\b\\t\\n\\f\\r\\(\\)\\\\) ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert token == b"\x08\t\n\x0c\r()\\"

    def test_hexstring_odd_length(self):
        fp = BytesIO(b"<ABC> ")
        parser = PSBaseParser(fp)
        pos, token = parser.nexttoken()
        assert len(token) == 2

    def test_adjacent_tokens(self):
        fp = BytesIO(b"/a/b/c ")
        parser = PSBaseParser(fp)
        tokens = []
        for _ in range(3):
            _, token = parser.nexttoken()
            tokens.append(token)
        assert all(isinstance(t, PSLiteral) for t in tokens)

    def test_comment_at_end_of_line(self):
        fp = BytesIO(b"42 % comment\n43 ")
        parser = PSBaseParser(fp)
        _, t1 = parser.nexttoken()
        _, t2 = parser.nexttoken()
        assert t1 == 42
        assert t2 == 43

    def test_single_character_keywords(self):
        fp = BytesIO(b'!"#@')
        parser = PSBaseParser(fp)
        tokens = []
        try:
            while True:
                _, token = parser.nexttoken()
                tokens.append(token)
        except PSEOF:
            pass
        assert all(isinstance(t, PSKeyword) for t in tokens)


class TestDictParsing:
    """Tests for dictionary parsing edge cases"""

    def get_parser(self, data):
        class TestParser(PSStackParser):
            def flush(self):
                self.add_results(*self.popall())

        return TestParser(BytesIO(data))

    def test_dict_with_multiple_entries(self):
        parser = self.get_parser(b"<< /A 1 /B 2 /C 3 >> ")
        pos, obj = parser.nextobject()
        assert obj == {"A": 1, "B": 2, "C": 3}

    def test_dict_with_nested_dict(self):
        parser = self.get_parser(b"<< /Outer << /Inner 42 >> >> ")
        pos, obj = parser.nextobject()
        assert obj == {"Outer": {"Inner": 42}}

    def test_dict_with_array_value(self):
        parser = self.get_parser(b"<< /Array [ 1 2 3 ] >> ")
        pos, obj = parser.nextobject()
        assert obj == {"Array": [1, 2, 3]}

    def test_dict_with_none_values_filtered(self):
        parser = self.get_parser(b"<< /Key /Value >> ")
        pos, obj = parser.nextobject()
        assert "Key" in obj

    def test_dict_odd_number_of_elements_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            parser = self.get_parser(b"<< /Key >> ")
            with pytest.raises(PSSyntaxError):
                parser.nextobject()
        finally:
            settings.STRICT = original_strict


class TestArrayParsing:
    """Tests for array parsing edge cases"""

    def get_parser(self, data):
        class TestParser(PSStackParser):
            def flush(self):
                self.add_results(*self.popall())

        return TestParser(BytesIO(data))

    def test_array_with_mixed_types(self):
        parser = self.get_parser(b"[ 1 (string) /literal true ] ")
        pos, obj = parser.nextobject()
        assert len(obj) == 4
        assert obj[0] == 1
        assert obj[1] == b"string"
        assert isinstance(obj[2], PSLiteral)
        assert obj[3] is True

    def test_deeply_nested_array(self):
        parser = self.get_parser(b"[ [ [ 1 ] ] ] ")
        pos, obj = parser.nextobject()
        assert obj == [[[1]]]

    def test_array_with_dict(self):
        parser = self.get_parser(b"[ << /A 1 >> ] ")
        pos, obj = parser.nextobject()
        assert obj == [{"A": 1}]


class TestProcParsing:
    """Tests for procedure parsing"""

    def get_parser(self, data):
        class TestParser(PSStackParser):
            def flush(self):
                self.add_results(*self.popall())

        return TestParser(BytesIO(data))

    def test_proc_with_keywords(self):
        parser = self.get_parser(b"{ add mul } ")
        pos, obj = parser.nextobject()
        assert isinstance(obj, list)

    def test_nested_proc(self):
        parser = self.get_parser(b"{ { 1 } { 2 } } ")
        pos, obj = parser.nextobject()
        assert isinstance(obj, list)
        assert len(obj) == 2
