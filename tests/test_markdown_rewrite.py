"""Tests for markdown_rewrite module."""

import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.markdown_rewrite import rewrite_markdown


class TestRewriteMarkdown:
    """Tests for rewrite_markdown function."""

    def test_exact_match_replacement(self):
        """Test exact-match replacement of base64 URIs."""
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        jpeg_data = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP/////wAALCAACAgBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AT//Z"

        markdown = f"""
# Document Title

This is a paragraph.

![Image 1](data:image/png;base64,{png_data})

![Image 2](data:image/jpeg;base64,{jpeg_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
            {"id": "img2", "image_base64": f"data:image/jpeg;base64,{jpeg_data}"},
        ]

        filenames = ["img1.png", "img2.jpeg"]

        result = rewrite_markdown(markdown, images, filenames)

        # Check that base64 URIs are replaced
        assert f"data:image/png;base64,{png_data}" not in result
        assert f"data:image/jpeg;base64,{jpeg_data}" not in result

        # Check that relative paths are present
        assert "./img1.png" in result
        assert "./img2.jpeg" in result

        # Check that other markdown is preserved
        assert "# Document Title" in result
        assert "This is a paragraph." in result

    def test_sequential_replacement_no_filenames(self):
        """Test sequential regex replacement when no filenames provided."""
        png_data = "iVBORw0KGgo="
        jpeg_data = "/9j/4AAQSkZJRg=="

        markdown = f"""
# Title

![Image 1](data:image/png;base64,{png_data})

![Image 2](data:image/jpeg;base64,{jpeg_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
            {"id": "img2", "image_base64": f"data:image/jpeg;base64,{jpeg_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        # Check that relative paths are present with correct extensions
        assert "./img1.png" in result
        assert "./img2.jpeg" in result

    def test_sequential_replacement_unknown_mime_defaults_to_png(self):
        """Test that unknown MIME types default to .png extension."""
        svg_data = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4="

        markdown = f"""
![SVG](data:image/svg+xml;base64,{svg_data})
"""

        images = [
            {"id": "svg1", "image_base64": f"data:image/svg+xml;base64,{svg_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        # Should default to .png for unknown MIME type
        assert "./svg1.png" in result

    def test_multiple_images_same_type(self):
        """Test handling multiple images of the same type."""
        png_data1 = "iVBORw0KGgo="
        png_data2 = "iVBORw0KGg="  # Different but similar

        markdown = f"""
![First](data:image/png;base64,{png_data1})
![Second](data:image/png;base64,{png_data2})
"""

        images = [
            {"id": "first", "image_base64": f"data:image/png;base64,{png_data1}"},
            {"id": "second", "image_base64": f"data:image/png;base64,{png_data2}"},
        ]

        filenames = ["first.png", "second.png"]

        result = rewrite_markdown(markdown, images, filenames)

        assert "./first.png" in result
        assert "./second.png" in result

    def test_more_images_than_matches(self):
        """Test handling when there are more images than data URIs in markdown."""
        png_data = "iVBORw0KGgo="

        markdown = f"""
![Only one image](data:image/png;base64,{png_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
            {"id": "img2", "image_base64": "data:image/png;base64,iVBORw0KGg="},
        ]

        filenames = ["img1.png", "img2.png"]

        result = rewrite_markdown(markdown, images, filenames)

        # Only first image should be replaced
        assert "./img1.png" in result
        assert "./img2.png" not in result

    def test_more_matches_than_images(self):
        """Test handling when there are more data URIs than images."""
        png_data = "iVBORw0KGgo="

        markdown = f"""
![First](data:image/png;base64,{png_data})
![Second](data:image/png;base64,{png_data})
![Third](data:image/png;base64,{png_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
        ]

        filenames = ["img1.png"]

        result = rewrite_markdown(markdown, images, filenames)

        # Only first occurrence should be replaced with exact match
        assert "./img1.png" in result
        # The other two should still have the base64 data (they're the same)
        # because exact-match replaces all occurrences of the same data URI

    def test_mismatched_filenames_length_raises_error(self):
        """Test that providing wrong number of filenames raises ValueError."""
        images = [
            {"id": "img1", "image_base64": "data:image/png;base64,iVBORw0KGgo="},
            {"id": "img2", "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
        ]

        filenames = ["img1.png"]  # Only one filename for two images

        with pytest.raises(ValueError) as exc_info:
            rewrite_markdown("some markdown", images, filenames)

        assert "output_filenames length" in str(exc_info.value)
        assert "must match images length" in str(exc_info.value)

    def test_preserves_other_markdown_content(self):
        """Test that other markdown content is preserved."""
        png_data = "iVBORw0KGgo="

        markdown = f"""
# Header 1

## Header 2

This is **bold** and *italic* text.

- List item 1
- List item 2

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |

Code `inline` and block:

```
code block
```

![Image](data:image/png;base64,{png_data})

[Link](https://example.com)
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
        ]

        filenames = ["img1.png"]

        result = rewrite_markdown(markdown, images, filenames)

        # Check all markdown elements are preserved
        assert "# Header 1" in result
        assert "## Header 2" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "- List item 1" in result
        assert "| Column 1 | Column 2 |" in result
        assert "code block" in result
        assert "[Link](https://example.com)" in result

        # Check image is replaced
        assert "./img1.png" in result
        assert f"data:image/png;base64,{png_data}" not in result

    def test_empty_images_list(self):
        """Test handling empty images list."""
        markdown = "# Title\n\nNo images here."

        images = []

        result = rewrite_markdown(markdown, images)

        # Should return unchanged markdown
        assert result == markdown

    def test_empty_markdown(self):
        """Test handling empty markdown."""
        png_data = "iVBORw0KGgo="

        markdown = ""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
        ]

        filenames = ["img1.png"]

        result = rewrite_markdown(markdown, images, filenames)

        # Should return empty string
        assert result == ""

    def test_data_uri_in_various_contexts(self):
        """Test that data URIs are replaced in various markdown contexts."""
        png_data = "iVBORw0KGgo="

        markdown = f"""
# Image as link
[![Alt](data:image/png;base64,{png_data})](https://example.com)

# Image with title
![Alt](data:image/png;base64,{png_data} "Title")

# Image in paragraph
Here is an image data:image/png;base64,{png_data} inline.

# Image in reference
Reference: ![Alt][1]

[1]: data:image/png;base64,{png_data}
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/png;base64,{png_data}"},
        ]

        filenames = ["img1.png"]

        result = rewrite_markdown(markdown, images, filenames)

        # All occurrences should be replaced
        assert result.count(f"data:image/png;base64,{png_data}") == 0
        # Should have at least one replacement
        assert "./img1.png" in result

    def test_webp_extension_from_mime(self):
        """Test that webp images get .webp extension."""
        webp_data = "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v3AgAA="

        markdown = f"""
![WebP](data:image/webp;base64,{webp_data})
"""

        images = [
            {"id": "webp1", "image_base64": f"data:image/webp;base64,{webp_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        assert "./webp1.webp" in result

    def test_gif_extension_from_mime(self):
        """Test that gif images get .gif extension."""
        gif_data = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

        markdown = f"""
![GIF](data:image/gif;base64,{gif_data})
"""

        images = [
            {"id": "gif1", "image_base64": f"data:image/gif;base64,{gif_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        assert "./gif1.gif" in result

    def test_jpg_vs_jpeg_mimes(self):
        """Test that both jpg and jpeg MIME types are handled correctly."""
        jpg_data = "/9j/4AAQSkZJRg=="
        jpeg_data = "/9j/4AAQSkZJRgA=="

        markdown = f"""
![JPG](data:image/jpg;base64,{jpg_data})
![JPEG](data:image/jpeg;base64,{jpeg_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:image/jpg;base64,{jpg_data}"},
            {"id": "img2", "image_base64": f"data:image/jpeg;base64,{jpeg_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        assert "./img1.jpg" in result
        assert "./img2.jpeg" in result

    def test_case_insensitive_mime_in_sequential_mode(self):
        """Test that MIME type case doesn't matter in sequential mode."""
        png_data = "iVBORw0KGgo="

        markdown = f"""
![Mixed case](data:IMAGE/PNG;base64,{png_data})
"""

        images = [
            {"id": "img1", "image_base64": f"data:IMAGE/PNG;base64,{png_data}"},
        ]

        result = rewrite_markdown(markdown, images)

        # Should still get .png extension
        assert "./img1.png" in result
