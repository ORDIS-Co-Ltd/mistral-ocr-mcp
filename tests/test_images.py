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


class TestSaveBase64ImagePathTraversal:
    """Tests for path traversal protection in save_base64_image."""

    def test_normal_image_id_preserved(self, tmp_path):
        """Test that normal safe image IDs are preserved."""
        image_id = "img_abc123"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, image_id, data_uri)

        assert result == f"{image_id}.png"
        assert (tmp_path / result).exists()

    def test_path_traversal_attempt_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        malicious_id = "../../../etc/passwd"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, malicious_id, data_uri)

        # The result should not contain path separators
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result
        # File should exist in the intended directory
        assert (tmp_path / result).exists()
        # Should NOT create files outside the intended directory (use safe path check)
        outside_path = tmp_path.parent / "etc" / "passwd.png"
        assert not outside_path.exists()

    def test_null_byte_blocked(self, tmp_path):
        """Test that null bytes in image ID are handled."""
        malicious_id = "image\x00name"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, malicious_id, data_uri)

        # Null bytes should be replaced
        assert "\0" not in result

    def test_special_characters_sanitized(self, tmp_path):
        """Test that special characters are properly sanitized."""
        special_id = "image<script>alert('xss')</script>"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, special_id, data_uri)

        # Dangerous characters should be sanitized
        assert "<" not in result
        assert ">" not in result
        assert "'" not in result
        assert '"' not in result
        assert "(" not in result
        assert ")" not in result

    def test_id_too_long_truncated(self, tmp_path):
        """Test that very long IDs are truncated."""
        long_id = "a" * 300  # Very long ID
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, long_id, data_uri)

        # Should be truncated to 255 chars + extension
        assert len(result) <= 255 + 4  # +4 for ".png"

    def test_hidden_file_attempt_blocked(self, tmp_path):
        """Test that attempts to create hidden files are blocked."""
        hidden_id = ".hidden_file"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, hidden_id, data_uri)

        # Should not start with dot
        assert not result.startswith(".")
        assert result.startswith("_")

    def test_flag_file_attempt_blocked(self, tmp_path):
        """Test that attempts to create flag files are blocked."""
        flag_id = "-flag_file"
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = save_base64_image(tmp_path, flag_id, data_uri)

        # Should not start with dash
        assert not result.startswith("-")


class TestSaveImagesPathTraversal:
    """Tests for path traversal protection in save_images."""

    def test_multiple_images_with_malicious_ids(self, tmp_path):
        """Test that multiple images with malicious IDs are all sanitized."""
        images = [
            {
                "id": "safe_image_1",
                "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            },
            {
                "id": "../../../malicious/path",
                "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            },
            {
                "id": "normal_image_2",
                "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            },
        ]

        result = save_images(tmp_path, images)

        # All filenames should be safe
        for filename in result:
            assert ".." not in filename
            assert "/" not in filename
            assert "\\" not in filename

        # All files should exist in the intended directory
        for filename in result:
            assert (tmp_path / filename).exists()


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

    def test_image_id_with_extension_not_duplicated(self, tmp_path):
        """Test that extension is not appended if image_id already has it."""
        jpeg_data = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
        )
        data_uri = f"data:image/jpeg;base64,{jpeg_data}"

        # image_id already ends with .jpeg
        filename = save_base64_image(tmp_path, "image.jpeg", data_uri)

        assert filename == "image.jpeg"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()

    def test_image_id_with_case_insensitive_extension_not_duplicated(self, tmp_path):
        """Test that extension check is case-insensitive."""
        jpeg_data = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
        )
        data_uri = f"data:image/jpeg;base64,{jpeg_data}"

        # image_id ends with .JPEG (uppercase)
        filename = save_base64_image(tmp_path, "image.JPEG", data_uri)

        assert filename == "image.JPEG"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()

    def test_image_id_without_extension_appends_extension(self, tmp_path):
        """Test that extension is appended when image_id doesn't have it."""
        jpeg_data = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
        )
        data_uri = f"data:image/jpeg;base64,{jpeg_data}"

        # image_id does NOT end with extension
        filename = save_base64_image(tmp_path, "image", data_uri)

        assert filename == "image.jpeg"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()

    def test_image_id_with_different_extension_appends_correct_extension(
        self, tmp_path
    ):
        """Test that extension is appended when image_id has a different extension."""
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        data_uri = f"data:image/png;base64,{png_data}"

        # image_id ends with .jpg but MIME type is .png
        filename = save_base64_image(tmp_path, "image.jpg", data_uri)

        # Should append .png since .jpg != .png
        assert filename == "image.jpg.png"
        saved_path = tmp_path / filename
        assert saved_path.exists()
        assert saved_path.is_file()


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
