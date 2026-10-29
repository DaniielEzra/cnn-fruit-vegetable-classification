"""preprocess_pil_image(): the code shared by the notebook and the website."""
import numpy as np
import pytest
from PIL import Image

from app import preprocess_pil_image
from conftest import SAMPLE_DIR


@pytest.mark.parametrize("name", ["apple", "avocado", "banana", "eggplant", "tomato"])
def test_fruits360_images_pass_through_unchanged(name):
    """Training images are already 100x100 RGB: preprocessing must not alter them."""
    with Image.open(SAMPLE_DIR / f"{name}.jpg") as image:
        original = np.asarray(image.convert("RGB"))
        result = preprocess_pil_image(image)
    assert result.shape == (100, 100, 3) and result.dtype == np.uint8
    assert np.array_equal(result, original)


def test_non_square_is_padded_with_white_not_stretched():
    image = Image.new("RGB", (200, 100), (200, 0, 0))
    result = preprocess_pil_image(image)
    assert result.shape == (100, 100, 3)
    assert (result[5, 50] == 255).all()       # top padding is white
    assert (result[50, 50] == [200, 0, 0]).all()  # centre keeps the colour


def test_transparent_png_goes_on_white():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert (preprocess_pil_image(image) == 255).all()


def test_grayscale_becomes_rgb():
    result = preprocess_pil_image(Image.new("L", (50, 50), 100))
    assert result.shape == (100, 100, 3) and (result == 100).all()


def test_exif_orientation_is_applied(tmp_path):
    # Left half red, right half blue; EXIF 6 means "rotate 90 degrees clockwise to display".
    pixels = np.zeros((100, 100, 3), np.uint8)
    pixels[:, :50] = (255, 0, 0)
    pixels[:, 50:] = (0, 0, 255)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[0x0112] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, "JPEG", exif=exif, quality=100)
    with Image.open(path) as loaded:
        result = preprocess_pil_image(loaded)
    top, bottom = result[10, 50].astype(int), result[90, 50].astype(int)
    assert top[0] > 200 and top[2] < 60       # red is now on top
    assert bottom[2] > 200 and bottom[0] < 60  # blue at the bottom
