"""Tests for images module."""

import base64
import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.images import (
    ImageError,
    get_extension_from_mime,
    parse_data_uri,
    save_base64_image,
    save_images,
)


class TestParseDataURI:
    """Tests for parse_data_uri function."""

    def test_jpeg_data_uri(self):
        """Test parsing a JPEG data URI."""
        data_uri = (
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////"
        )
        mime, b64 = parse_data_uri(data_uri)

        assert mime == "image/jpeg"
        assert b64 == "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////"

    def test_png_data_uri(self):
        """Test parsing a PNG data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        mime, b64 = parse_data_uri(data_uri)

        assert mime == "image/png"
        assert (
            b64
            == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

    def test_webp_data_uri(self):
        """Test parsing a WebP data URI."""
        data_uri = "data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v3AgAA="
        mime, b64 = parse_data_uri(data_uri)

        assert mime == "image/webp"
        assert b64 == "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v3AgAA="

    def test_gif_data_uri(self):
        """Test parsing a GIF data URI."""
        data_uri = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        mime, b64 = parse_data_uri(data_uri)

        assert mime == "image/gif"
        assert b64 == "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

    def test_case_insensitive_mime(self):
        """Test that MIME type is case-sensitive in output."""
        data_uri = "data:image/JPEG;base64,SGVsbG8="
        mime, b64 = parse_data_uri(data_uri)

        # We return the MIME type as-is, but get_extension_from_mime handles case
        assert mime == "image/JPEG"
        assert b64 == "SGVsbG8="

    def test_empty_data_uri(self):
        """Test that empty data URI raises error."""
        with pytest.raises(ImageError) as exc_info:
            parse_data_uri("")

        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_format_no_base64(self):
        """Test that data URI without base64 marker raises error."""
        with pytest.raises(ImageError) as exc_info:
            parse_data_uri("data:image/jpeg,something")

        assert "Invalid data URI format" in str(exc_info.value)

    def test_invalid_format_no_mime(self):
        """Test that data URI without MIME type raises error."""
        with pytest.raises(ImageError) as exc_info:
            parse_data_uri("data:;base64,SGVsbG8=")

        assert "Missing MIME type" in str(exc_info.value)

    def test_invalid_format_no_data(self):
        """Test that data URI without data raises error."""
        with pytest.raises(ImageError) as exc_info:
            parse_data_uri("data:image/jpeg;base64,")

        assert "Missing base64 data" in str(exc_info.value)


class TestGetExtensionFromMime:
    """Tests for get_extension_from_mime function."""

    def test_jpeg_mime(self):
        """Test that image/jpeg maps to .jpeg."""
        assert get_extension_from_mime("image/jpeg") == ".jpeg"

    def test_jpg_mime(self):
        """Test that image/jpg maps to .jpg."""
        assert get_extension_from_mime("image/jpg") == ".jpg"

    def test_png_mime(self):
        """Test that image/png maps to .png."""
        assert get_extension_from_mime("image/png") == ".png"

    def test_webp_mime(self):
        """Test that image/webp maps to .webp."""
        assert get_extension_from_mime("image/webp") == ".webp"

    def test_gif_mime(self):
        """Test that image/gif maps to .gif."""
        assert get_extension_from_mime("image/gif") == ".gif"

    def test_unknown_mime_defaults_to_png(self):
        """Test that unknown MIME type defaults to .png."""
        assert get_extension_from_mime("image/svg+xml") == ".png"
        assert get_extension_from_mime("application/pdf") == ".png"

    def test_case_insensitive_mime(self):
        """Test that MIME type comparison is case-insensitive."""
        assert get_extension_from_mime("IMAGE/JPEG") == ".jpeg"
        assert get_extension_from_mime("Image/PNG") == ".png"


class TestSaveBase64Image:
    """Tests for save_base64_image function."""

    def test_save_jpeg_image(self, tmp_path):
        """Test saving a JPEG image."""
        # Use valid base64 data (simple 1x1 red pixel JPEG would be complex)
        # For testing purposes, we'll use a simple valid base64 string
        jpeg_data = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
        )
        data_uri = f"data:image/jpeg;base64,{jpeg_data}"

        filename = save_base64_image(tmp_path, "test_img", data_uri)

        assert filename == "test_img.jpeg"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()

    def test_save_png_image(self, tmp_path):
        """Test saving a PNG image."""
        # Create a minimal valid PNG (1x1 transparent pixel)
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        data_uri = f"data:image/png;base64,{png_data}"

        filename = save_base64_image(tmp_path, "test_img", data_uri)

        assert filename == "test_img.png"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()

    def test_unknown_mime_defaults_to_png_extension(self, tmp_path):
        """Test that unknown MIME type uses .png extension."""
        # Just use valid PNG data but with unknown MIME type
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        data_uri = f"data:image/svg+xml;base64,{png_data}"

        filename = save_base64_image(tmp_path, "test_img", data_uri)

        assert filename == "test_img.png"

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        nested_dir = tmp_path / "a" / "b" / "c"

        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        data_uri = f"data:image/png;base64,{png_data}"

        filename = save_base64_image(nested_dir, "test_img", data_uri)

        assert nested_dir.exists()
        assert (nested_dir / filename).exists()

    def test_invalid_base64_raises_error(self, tmp_path):
        """Test that invalid base64 data raises ImageError."""
        data_uri = "data:image/png;base64,!!!invalid!!!"

        with pytest.raises(ImageError) as exc_info:
            save_base64_image(tmp_path, "test_img", data_uri)

        assert "Failed to decode base64" in str(exc_info.value)
        assert "test_img" in str(exc_info.value)

    def test_file_write_error_propagates(self, tmp_path):
        """Test that file write errors are caught and wrapped."""
        # This is hard to test without actually causing a write error
        # We'll skip this for now as it requires unusual filesystem conditions
        pass


class TestSaveImages:
    """Tests for save_images function."""

    def test_save_multiple_images(self, tmp_path):
        """Test saving multiple images."""
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        jpeg_data = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
        )

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
            {"id": "img2", "image_base64": f"data:image/jpeg;base64,{jpeg_data}"},
            {"id": "img3", "image_base64": f"data:image/png;base64,{png_data}"},
        ]

        filenames = save_images(tmp_path, images)

        assert len(filenames) == 3
        assert filenames[0] == "img1.png"
        assert filenames[1] == "img2.jpeg"
        assert filenames[2] == "img3.png"

        # Verify all files exist
        for filename in filenames:
            assert (tmp_path / filename).exists()

    def test_image_missing_id(self, tmp_path):
        """Test that missing 'id' field raises ImageError."""
        images = [{"image_base64": "data:image/png;base64,iVBORw0KGgo="}]

        with pytest.raises(ImageError) as exc_info:
            save_images(tmp_path, images)

        assert "missing 'id' field" in str(exc_info.value)

    def test_image_missing_image_base64(self, tmp_path):
        """Test that missing 'image_base64' field raises ImageError."""
        images = [{"id": "test_img"}]

        with pytest.raises(ImageError) as exc_info:
            save_images(tmp_path, images)

        assert "test_img" in str(exc_info.value)
        assert "missing 'image_base64' field" in str(exc_info.value)
