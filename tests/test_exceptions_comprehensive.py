"""Comprehensive tests for pdfminer exception classes.

This module tests all exception classes in pdfminer/psexceptions.py and
pdfminer/pdfexceptions.py, including:
- Exception instantiation
- Inheritance hierarchy
- Message formatting and retrieval
- Raising and catching behavior
"""

import pytest

from pdfminer.pdfexceptions import (
    PDFEOFError,
    PDFException,
    PDFIOError,
    PDFKeyError,
    PDFNotImplementedError,
    PDFObjectNotFound,
    PDFTypeError,
    PDFValueError,
)
from pdfminer.psexceptions import (
    PSEOF,
    PSException,
    PSSyntaxError,
    PSTypeError,
    PSValueError,
)


class TestPSExceptionHierarchy:
    """Test the PostScript exception inheritance hierarchy."""

    def test_psexception_inherits_from_exception(self) -> None:
        assert issubclass(PSException, Exception)

    def test_pseof_inherits_from_psexception(self) -> None:
        assert issubclass(PSEOF, PSException)
        assert issubclass(PSEOF, Exception)

    def test_pssyntaxerror_inherits_from_psexception(self) -> None:
        assert issubclass(PSSyntaxError, PSException)
        assert issubclass(PSSyntaxError, Exception)

    def test_pstypeerror_inherits_from_psexception(self) -> None:
        assert issubclass(PSTypeError, PSException)
        assert issubclass(PSTypeError, Exception)

    def test_psvalueerror_inherits_from_psexception(self) -> None:
        assert issubclass(PSValueError, PSException)
        assert issubclass(PSValueError, Exception)


class TestPDFExceptionHierarchy:
    """Test the PDF exception inheritance hierarchy."""

    def test_pdfexception_inherits_from_psexception(self) -> None:
        assert issubclass(PDFException, PSException)
        assert issubclass(PDFException, Exception)

    def test_pdftypeerror_inherits_from_pdfexception_and_typeerror(self) -> None:
        assert issubclass(PDFTypeError, PDFException)
        assert issubclass(PDFTypeError, TypeError)
        assert issubclass(PDFTypeError, PSException)

    def test_pdfvalueerror_inherits_from_pdfexception_and_valueerror(self) -> None:
        assert issubclass(PDFValueError, PDFException)
        assert issubclass(PDFValueError, ValueError)
        assert issubclass(PDFValueError, PSException)

    def test_pdfobjectnotfound_inherits_from_pdfexception(self) -> None:
        assert issubclass(PDFObjectNotFound, PDFException)
        assert issubclass(PDFObjectNotFound, PSException)

    def test_pdfnotimplementederror_inherits_from_pdfexception_and_notimplementederror(
        self,
    ) -> None:
        assert issubclass(PDFNotImplementedError, PDFException)
        assert issubclass(PDFNotImplementedError, NotImplementedError)
        assert issubclass(PDFNotImplementedError, PSException)

    def test_pdfkeyerror_inherits_from_pdfexception_and_keyerror(self) -> None:
        assert issubclass(PDFKeyError, PDFException)
        assert issubclass(PDFKeyError, KeyError)
        assert issubclass(PDFKeyError, PSException)

    def test_pdfeoferror_inherits_from_pdfexception_and_eoferror(self) -> None:
        assert issubclass(PDFEOFError, PDFException)
        assert issubclass(PDFEOFError, EOFError)
        assert issubclass(PDFEOFError, PSException)

    def test_pdfioerror_inherits_from_pdfexception_and_ioerror(self) -> None:
        assert issubclass(PDFIOError, PDFException)
        assert issubclass(PDFIOError, IOError)
        assert issubclass(PDFIOError, PSException)


class TestPSExceptionInstantiation:
    """Test instantiation and message handling for PS exceptions."""

    def test_psexception_no_message(self) -> None:
        exc = PSException()
        assert str(exc) == ""
        assert exc.args == ()

    def test_psexception_with_message(self) -> None:
        exc = PSException("test message")
        assert str(exc) == "test message"
        assert exc.args == ("test message",)

    def test_psexception_with_multiple_args(self) -> None:
        exc = PSException("error", 42, "details")
        assert exc.args == ("error", 42, "details")

    def test_pseof_no_message(self) -> None:
        exc = PSEOF()
        assert str(exc) == ""
        assert exc.args == ()

    def test_pseof_with_message(self) -> None:
        exc = PSEOF("unexpected end of file")
        assert str(exc) == "unexpected end of file"

    def test_pssyntaxerror_no_message(self) -> None:
        exc = PSSyntaxError()
        assert str(exc) == ""

    def test_pssyntaxerror_with_message(self) -> None:
        exc = PSSyntaxError("invalid token at position 42")
        assert str(exc) == "invalid token at position 42"

    def test_pstypeerror_no_message(self) -> None:
        exc = PSTypeError()
        assert str(exc) == ""

    def test_pstypeerror_with_message(self) -> None:
        exc = PSTypeError("expected integer, got string")
        assert str(exc) == "expected integer, got string"

    def test_psvalueerror_no_message(self) -> None:
        exc = PSValueError()
        assert str(exc) == ""

    def test_psvalueerror_with_message(self) -> None:
        exc = PSValueError("value out of range")
        assert str(exc) == "value out of range"


class TestPDFExceptionInstantiation:
    """Test instantiation and message handling for PDF exceptions."""

    def test_pdfexception_no_message(self) -> None:
        exc = PDFException()
        assert str(exc) == ""
        assert exc.args == ()

    def test_pdfexception_with_message(self) -> None:
        exc = PDFException("PDF error occurred")
        assert str(exc) == "PDF error occurred"

    def test_pdfexception_with_multiple_args(self) -> None:
        exc = PDFException("error", "in", "document")
        assert exc.args == ("error", "in", "document")

    def test_pdftypeerror_no_message(self) -> None:
        exc = PDFTypeError()
        assert str(exc) == ""

    def test_pdftypeerror_with_message(self) -> None:
        exc = PDFTypeError("expected dict, got array")
        assert str(exc) == "expected dict, got array"

    def test_pdfvalueerror_no_message(self) -> None:
        exc = PDFValueError()
        assert str(exc) == ""

    def test_pdfvalueerror_with_message(self) -> None:
        exc = PDFValueError("invalid PDF version")
        assert str(exc) == "invalid PDF version"

    def test_pdfobjectnotfound_no_message(self) -> None:
        exc = PDFObjectNotFound()
        assert str(exc) == ""

    def test_pdfobjectnotfound_with_message(self) -> None:
        exc = PDFObjectNotFound("object 5 0 R not found")
        assert str(exc) == "object 5 0 R not found"

    def test_pdfnotimplementederror_no_message(self) -> None:
        exc = PDFNotImplementedError()
        assert str(exc) == ""

    def test_pdfnotimplementederror_with_message(self) -> None:
        exc = PDFNotImplementedError("JBIG2 decoding not supported")
        assert str(exc) == "JBIG2 decoding not supported"

    def test_pdfkeyerror_no_message(self) -> None:
        exc = PDFKeyError()
        assert exc.args == ()

    def test_pdfkeyerror_with_message(self) -> None:
        exc = PDFKeyError("missing_key")
        assert exc.args == ("missing_key",)

    def test_pdfeoferror_no_message(self) -> None:
        exc = PDFEOFError()
        assert str(exc) == ""

    def test_pdfeoferror_with_message(self) -> None:
        exc = PDFEOFError("unexpected end of PDF stream")
        assert str(exc) == "unexpected end of PDF stream"

    def test_pdfioerror_no_message(self) -> None:
        exc = PDFIOError()
        assert str(exc) == ""

    def test_pdfioerror_with_message(self) -> None:
        exc = PDFIOError("failed to read PDF file")
        assert str(exc) == "failed to read PDF file"


class TestExceptionRaising:
    """Test that exceptions can be properly raised and caught."""

    def test_raise_psexception(self) -> None:
        with pytest.raises(PSException, match="PS error"):
            raise PSException("PS error")

    def test_raise_pseof(self) -> None:
        with pytest.raises(PSEOF, match="end of file"):
            raise PSEOF("end of file")

    def test_raise_pssyntaxerror(self) -> None:
        with pytest.raises(PSSyntaxError, match="syntax"):
            raise PSSyntaxError("syntax error")

    def test_raise_pstypeerror(self) -> None:
        with pytest.raises(PSTypeError, match="type mismatch"):
            raise PSTypeError("type mismatch")

    def test_raise_psvalueerror(self) -> None:
        with pytest.raises(PSValueError, match="bad value"):
            raise PSValueError("bad value")

    def test_raise_pdfexception(self) -> None:
        with pytest.raises(PDFException, match="PDF error"):
            raise PDFException("PDF error")

    def test_raise_pdftypeerror(self) -> None:
        with pytest.raises(PDFTypeError, match="wrong type"):
            raise PDFTypeError("wrong type")

    def test_raise_pdfvalueerror(self) -> None:
        with pytest.raises(PDFValueError, match="invalid value"):
            raise PDFValueError("invalid value")

    def test_raise_pdfobjectnotfound(self) -> None:
        with pytest.raises(PDFObjectNotFound, match="object not found"):
            raise PDFObjectNotFound("object not found")

    def test_raise_pdfnotimplementederror(self) -> None:
        with pytest.raises(PDFNotImplementedError, match="not implemented"):
            raise PDFNotImplementedError("not implemented")

    def test_raise_pdfkeyerror(self) -> None:
        with pytest.raises(PDFKeyError):
            raise PDFKeyError("key")

    def test_raise_pdfeoferror(self) -> None:
        with pytest.raises(PDFEOFError, match="eof"):
            raise PDFEOFError("eof")

    def test_raise_pdfioerror(self) -> None:
        with pytest.raises(PDFIOError, match="io error"):
            raise PDFIOError("io error")


class TestCatchingWithBaseClasses:
    """Test that exceptions can be caught using their base classes."""

    def test_catch_pseof_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PSEOF("eof")

    def test_catch_pssyntaxerror_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PSSyntaxError("syntax")

    def test_catch_pstypeerror_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PSTypeError("type")

    def test_catch_psvalueerror_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PSValueError("value")

    def test_catch_pdfexception_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PDFException("pdf error")

    def test_catch_pdftypeerror_as_typeerror(self) -> None:
        with pytest.raises(TypeError):
            raise PDFTypeError("type error")

    def test_catch_pdftypeerror_as_pdfexception(self) -> None:
        with pytest.raises(PDFException):
            raise PDFTypeError("type error")

    def test_catch_pdftypeerror_as_psexception(self) -> None:
        with pytest.raises(PSException):
            raise PDFTypeError("type error")

    def test_catch_pdfvalueerror_as_valueerror(self) -> None:
        with pytest.raises(ValueError):
            raise PDFValueError("value error")

    def test_catch_pdfvalueerror_as_pdfexception(self) -> None:
        with pytest.raises(PDFException):
            raise PDFValueError("value error")

    def test_catch_pdfnotimplementederror_as_notimplementederror(self) -> None:
        with pytest.raises(NotImplementedError):
            raise PDFNotImplementedError("not implemented")

    def test_catch_pdfkeyerror_as_keyerror(self) -> None:
        with pytest.raises(KeyError):
            raise PDFKeyError("key")

    def test_catch_pdfeoferror_as_eoferror(self) -> None:
        with pytest.raises(EOFError):
            raise PDFEOFError("eof")

    def test_catch_pdfioerror_as_ioerror(self) -> None:
        with pytest.raises(IOError):
            raise PDFIOError("io")


class TestExceptionMessageFormatting:
    """Test exception message formatting with various input types."""

    def test_psexception_with_formatted_string(self) -> None:
        obj_id = 42
        exc = PSException(f"error processing object {obj_id}")
        assert str(exc) == "error processing object 42"

    def test_pssyntaxerror_with_position_info(self) -> None:
        pos = 1024
        token = "<<"
        exc = PSSyntaxError(f"unexpected token '{token}' at position {pos}")
        assert str(exc) == "unexpected token '<<' at position 1024"

    def test_pdfobjectnotfound_with_object_reference(self) -> None:
        obj_num = 5
        gen_num = 0
        exc = PDFObjectNotFound(f"object {obj_num} {gen_num} R not found in xref")
        assert str(exc) == "object 5 0 R not found in xref"

    def test_pdfkeyerror_preserves_key(self) -> None:
        key = "/Type"
        exc = PDFKeyError(key)
        assert exc.args[0] == "/Type"

    def test_pdfioerror_with_filename(self) -> None:
        filename = "/path/to/document.pdf"
        exc = PDFIOError(f"failed to open file: {filename}")
        assert str(exc) == "failed to open file: /path/to/document.pdf"

    def test_exception_with_unicode_message(self) -> None:
        exc = PSException("error with unicode: \u00e9\u00e0\u00fc")
        assert str(exc) == "error with unicode: \u00e9\u00e0\u00fc"

    def test_exception_with_empty_string_message(self) -> None:
        exc = PSException("")
        assert str(exc) == ""
        assert exc.args == ("",)


class TestExceptionIsInstance:
    """Test isinstance checks for exception hierarchy."""

    def test_psexception_isinstance(self) -> None:
        exc = PSException()
        assert isinstance(exc, PSException)
        assert isinstance(exc, Exception)
        assert not isinstance(exc, PDFException)

    def test_pseof_isinstance(self) -> None:
        exc = PSEOF()
        assert isinstance(exc, PSEOF)
        assert isinstance(exc, PSException)
        assert isinstance(exc, Exception)

    def test_pdfexception_isinstance(self) -> None:
        exc = PDFException()
        assert isinstance(exc, PDFException)
        assert isinstance(exc, PSException)
        assert isinstance(exc, Exception)

    def test_pdftypeerror_isinstance(self) -> None:
        exc = PDFTypeError()
        assert isinstance(exc, PDFTypeError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, PSException)
        assert isinstance(exc, TypeError)
        assert isinstance(exc, Exception)

    def test_pdfvalueerror_isinstance(self) -> None:
        exc = PDFValueError()
        assert isinstance(exc, PDFValueError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, ValueError)

    def test_pdfkeyerror_isinstance(self) -> None:
        exc = PDFKeyError()
        assert isinstance(exc, PDFKeyError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, KeyError)

    def test_pdfeoferror_isinstance(self) -> None:
        exc = PDFEOFError()
        assert isinstance(exc, PDFEOFError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, EOFError)

    def test_pdfioerror_isinstance(self) -> None:
        exc = PDFIOError()
        assert isinstance(exc, PDFIOError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, IOError)

    def test_pdfnotimplementederror_isinstance(self) -> None:
        exc = PDFNotImplementedError()
        assert isinstance(exc, PDFNotImplementedError)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, NotImplementedError)

    def test_pdfobjectnotfound_isinstance(self) -> None:
        exc = PDFObjectNotFound()
        assert isinstance(exc, PDFObjectNotFound)
        assert isinstance(exc, PDFException)
        assert isinstance(exc, PSException)


class TestExceptionChaining:
    """Test exception chaining with raise ... from ..."""

    def test_chain_pdfexception_from_psexception(self) -> None:
        try:
            try:
                raise PSException("original error")
            except PSException as e:
                raise PDFException("wrapped error") from e
        except PDFException as exc:
            assert str(exc) == "wrapped error"
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, PSException)
            assert str(exc.__cause__) == "original error"

    def test_chain_pdftypeerror_from_typeerror(self) -> None:
        try:
            try:
                raise TypeError("builtin type error")
            except TypeError as e:
                raise PDFTypeError("PDF type error") from e
        except PDFTypeError as exc:
            assert str(exc) == "PDF type error"
            assert isinstance(exc.__cause__, TypeError)

    def test_chain_pdfioerror_from_oserror(self) -> None:
        try:
            try:
                raise OSError("file not found")
            except OSError as e:
                raise PDFIOError("could not read PDF") from e
        except PDFIOError as exc:
            assert str(exc) == "could not read PDF"
            assert isinstance(exc.__cause__, OSError)


class TestExceptionRepr:
    """Test exception repr for debugging purposes."""

    def test_psexception_repr(self) -> None:
        exc = PSException("test")
        repr_str = repr(exc)
        assert "PSException" in repr_str
        assert "test" in repr_str

    def test_pdfexception_repr(self) -> None:
        exc = PDFException("pdf test")
        repr_str = repr(exc)
        assert "PDFException" in repr_str
        assert "pdf test" in repr_str

    def test_pdftypeerror_repr(self) -> None:
        exc = PDFTypeError("type test")
        repr_str = repr(exc)
        assert "PDFTypeError" in repr_str
        assert "type test" in repr_str
