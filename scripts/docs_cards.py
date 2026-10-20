# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Open Graph cards: one 1200x630 PNG per page, drawn with Pillow from the self-hosted Geist font.

The font ships in the site as WOFF2; Pillow needs a TrueType file, so it is converted once per
build into a temporary directory. No network and no browser are involved.
"""

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
INK, MUTED, TEAL, VIOLET = (10, 13, 20), (154, 163, 181), (92, 225, 230), (167, 139, 250)
_fonts = {}


def _ttf(fonts_dir, name):
    key = (str(fonts_dir), name)
    if key not in _fonts:
        from fontTools.ttLib import TTFont

        font = TTFont(str(Path(fonts_dir) / name))
        font.flavor = None
        tmp = Path(tempfile.mkdtemp(prefix="hms-og-")) / name.replace(".woff2", ".ttf")
        font.save(str(tmp))
        _fonts[key] = str(tmp)
    return _fonts[key]


def _font(fonts_dir, size, weight):
    font = ImageFont.truetype(_ttf(fonts_dir, "geist-latin-wght-normal.woff2"), size)
    try:
        font.set_variation_by_axes([weight])
    except (OSError, AttributeError):
        pass
    return font


def _wrap(draw, text, font, width, max_lines):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "..."
    return lines


def _rings(img):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (cx, cy, color) in ((830, 250, TEAL), (960, 330, VIOLET)):
        for i in range(1, 22):
            r = i * 26
            alpha = max(0, int(120 - i * 5))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (alpha,), width=2)
    img.alpha_composite(layer)


def render(path, fonts_dir, title, section, description):
    img = Image.new("RGBA", (W, H), INK + (255,))
    _rings(img)
    d = ImageDraw.Draw(img)
    # Mark: two interfering ring sets inside a circle.
    cx, cy = 92, 92
    d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=TEAL, width=3)
    for dx, color in ((-8, TEAL), (8, VIOLET)):
        for r in (6, 12, 18):
            d.ellipse([cx + dx - r, cy - r, cx + dx + r, cy + r], outline=color, width=2)
    d.text((140, 74), "Holographic Memory", font=_font(fonts_dir, 30, 600), fill=(232, 236, 244))
    if section:
        d.text((72, 190), section.upper(), font=_font(fonts_dir, 24, 600), fill=TEAL)
    title_font = _font(fonts_dir, 68, 650)
    y = 232
    for line in _wrap(d, title, title_font, 760, 3):
        d.text((72, y), line, font=title_font, fill=(244, 246, 250))
        y += 80
    desc_font = _font(fonts_dir, 28, 400)
    for line in _wrap(d, description, desc_font, 760, 2):
        d.text((72, y + 16), line, font=desc_font, fill=MUTED)
        y += 40
    d.text((72, H - 64), "writerslogic.github.io/holographic-memory", font=_font(fonts_dir, 24, 500), fill=MUTED)
    img.convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
