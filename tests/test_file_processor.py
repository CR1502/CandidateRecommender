"""
Unit tests for file processor module.
"""

import pytest
import io
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.file_processor import FileProcessor


class TestFileProcessor:
    """Test suite for FileProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FileProcessor(max_file_size_mb=10)

    def test_init(self):
        """Test FileProcessor initialization."""
        assert self.processor.max_file_size_bytes == 10 * 1024 * 1024
        assert 'pdf' in self.processor.supported_formats
        assert 'docx' in self.processor.supported_formats
        assert 'txt' in self.processor.supported_formats

    def test_validate_file_valid(self):
        """Test file validation with valid file."""
        # Create mock file
        file_content = b"Test content"
        file_obj = io.BytesIO(file_content)

        is_valid, error = self.processor.validate_file(file_obj, "test.txt")
        assert is_valid is True
        assert error is None

    def test_validate_file_invalid_extension(self):
        """Test file validation with invalid extension."""
        file_obj = io.BytesIO(b"content")

        is_valid, error = self.processor.validate_file(file_obj, "test.exe")
        assert is_valid is False
        assert "Unsupported file type" in error

    def test_validate_file_too_large(self):
        """Test file validation with oversized file."""
        # Create large file (11 MB)
        large_content = b"x" * (11 * 1024 * 1024)
        file_obj = io.BytesIO(large_content)

        is_valid, error = self.processor.validate_file(file_obj, "large.txt")
        assert is_valid is False
        assert "File too large" in error

    def test_validate_file_empty(self):
        """Test file validation with empty file."""
        file_obj = io.BytesIO(b"")

        is_valid, error = self.processor.validate_file(file_obj, "empty.txt")
        assert is_valid is False
        assert "File is empty" in error

    def test_extract_from_txt(self):
        """Test text extraction from TXT file."""
        content = "This is a test resume.\nWith multiple lines."
        file_obj = io.BytesIO(content.encode('utf-8'))

        extracted_text = self.processor._extract_from_txt(file_obj)
        assert extracted_text == content

    def test_extract_from_txt_with_encoding(self):
        """Test text extraction with different encoding."""
        content = "Resume with special characters: café, résumé"
        file_obj = io.BytesIO(content.encode('latin-1'))

        extracted_text = self.processor._extract_from_txt(file_obj)
        assert "special characters" in extracted_text

    def test_process_file_txt(self):
        """Test complete file processing for TXT."""
        content = "John Doe\nSoftware Engineer\nPython, Java, SQL"
        file_obj = io.BytesIO(content.encode('utf-8'))

        text, candidate_name = self.processor.process_file(file_obj, "john_doe_resume.txt")

        assert text == content
        assert "John" in candidate_name or "Doe" in candidate_name

    def test_process_multiple_files(self):
        """Test processing multiple files."""
        # Create mock files
        files = []
        for i in range(3):
            content = f"Resume {i}\nCandidate {i}"
            file_obj = io.BytesIO(content.encode('utf-8'))
            file_obj.name = f"resume_{i}.txt"
            files.append(file_obj)

        results = self.processor.process_multiple_files(files)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['filename'] == f"resume_{i}.txt"
            assert result['text'] is not None
            assert result['error'] is None

    def test_process_multiple_files_with_errors(self):
        """Test processing multiple files with some errors."""
        files = []

        # Valid file
        valid_content = "Valid resume"
        valid_file = io.BytesIO(valid_content.encode('utf-8'))
        valid_file.name = "valid.txt"
        files.append(valid_file)

        # Invalid file (empty)
        invalid_file = io.BytesIO(b"")
        invalid_file.name = "invalid.txt"
        files.append(invalid_file)

        results = self.processor.process_multiple_files(files)

        assert len(results) == 2
        assert results[0]['error'] is None
        assert results[1]['error'] is not None


class TestFileProcessorIntegration:
    """Integration tests for FileProcessor with real file formats."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FileProcessor()

    def test_pdf_extraction_mock(self):
        """Test PDF extraction with mock data."""
        # This would require creating actual PDF files or mocking PyPDF2
        # For now, we'll test the error handling
        file_obj = io.BytesIO(b"Not a real PDF")

        with pytest.raises(ValueError):
            self.processor._extract_from_pdf(file_obj)

    def test_docx_extraction_mock(self):
        """Test DOCX extraction with mock data."""
        # This would require creating actual DOCX files or mocking python-docx
        # For now, we'll test the error handling
        file_obj = io.BytesIO(b"Not a real DOCX")

        with pytest.raises(ValueError):
            self.processor._extract_from_docx(file_obj)