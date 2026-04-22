#!/usr/bin/env python3
"""Export the final Markdown course report to a readable PDF.

This script intentionally avoids external PDF tools such as pandoc, LaTeX, or
browser engines because they may not be installed in the course environment.
It rasterizes A4 pages with PIL, draws Markdown headings/paragraphs/tables, and
embeds referenced images.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAGE_W = 1654
PAGE_H = 2339
MARGIN_X = 120
MARGIN_Y = 110
LINE_GAP = 10

FONT_CANDIDATES = [
    "/map-vepfs/shuang/EDR-expriment/framwork/DeepResearch/WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/resource/AlibabaPuHuiTi-3-45-Light.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Markdown report to PDF.")
    parser.add_argument("--input", default="docs/final_course_report.md")
    parser.add_argument("--output", default="docs/final_course_report.pdf")
    parser.add_argument("--font", default=None, help="Optional path to a Chinese-capable TTF/TTC font.")
    return parser.parse_args()


def find_font(user_font: str | None) -> str | None:
    candidates = []
    if user_font:
        candidates.append(user_font)
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


class PdfRenderer:
    def __init__(self, font_path: str | None) -> None:
        self.font_path = font_path
        self.fonts = {
            "title": self.load_font(46),
            "h2": self.load_font(34),
            "h3": self.load_font(28),
            "body": self.load_font(24),
            "small": self.load_font(19),
            "code": self.load_font(19),
        }
        self.pages: list[Image.Image] = []
        self.new_page()

    def load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.font_path:
            return ImageFont.truetype(self.font_path, size=size)
        return ImageFont.load_default()

    def new_page(self) -> None:
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y
        self.pages.append(self.page)

    def ensure_space(self, height: int) -> None:
        if self.y + height > PAGE_H - MARGIN_Y:
            self.new_page()

    def text_width(self, text: str, font: ImageFont.ImageFont) -> int:
        if not text:
            return 0
        return int(self.draw.textbbox((0, 0), text, font=font)[2])

    def line_height(self, font: ImageFont.ImageFont) -> int:
        bbox = self.draw.textbbox((0, 0), "测试Ag", font=font)
        return int(bbox[3] - bbox[1] + LINE_GAP)

    def wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        text = text.strip()
        if not text:
            return [""]
        lines: list[str] = []
        current = ""
        for char in text:
            candidate = current + char
            if self.text_width(candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines

    def draw_text_block(
        self,
        text: str,
        font_key: str = "body",
        fill: str = "#1f2933",
        before: int = 8,
        after: int = 10,
        indent: int = 0,
    ) -> None:
        font = self.fonts[font_key]
        max_width = PAGE_W - 2 * MARGIN_X - indent
        lines = self.wrap_text(text, font, max_width)
        h = len(lines) * self.line_height(font) + before + after
        self.ensure_space(h)
        self.y += before
        for line in lines:
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill=fill)
            self.y += self.line_height(font)
        self.y += after

    def draw_horizontal_rule(self) -> None:
        self.ensure_space(30)
        self.draw.line((MARGIN_X, self.y + 12, PAGE_W - MARGIN_X, self.y + 12), fill="#d0d7de", width=2)
        self.y += 34

    def draw_code_block(self, lines: list[str]) -> None:
        font = self.fonts["code"]
        line_h = self.line_height(font)
        wrapped: list[str] = []
        max_width = PAGE_W - 2 * MARGIN_X - 40
        for line in lines:
            wrapped.extend(self.wrap_text(line.rstrip(), font, max_width))
        h = len(wrapped) * line_h + 36
        self.ensure_space(h)
        x0 = MARGIN_X
        y0 = self.y
        x1 = PAGE_W - MARGIN_X
        y1 = self.y + h
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill="#f6f8fa", outline="#d0d7de")
        self.y += 18
        for line in wrapped:
            self.draw.text((MARGIN_X + 20, self.y), line, font=font, fill="#24292f")
            self.y += line_h
        self.y += 18

    def draw_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        font = self.fonts["small"]
        line_h = self.line_height(font)
        col_count = max(len(row) for row in rows)
        available = PAGE_W - 2 * MARGIN_X
        col_w = available // col_count
        row_heights: list[int] = []
        wrapped_rows: list[list[list[str]]] = []
        for row in rows:
            padded = row + [""] * (col_count - len(row))
            wrapped_cells = [self.wrap_text(cell, font, col_w - 18) for cell in padded]
            wrapped_rows.append(wrapped_cells)
            row_heights.append(max(len(cell) for cell in wrapped_cells) * line_h + 18)

        total_h = sum(row_heights)
        self.ensure_space(total_h + 20)
        x = MARGIN_X
        y = self.y + 8
        for r, wrapped_cells in enumerate(wrapped_rows):
            row_h = row_heights[r]
            fill = "#f6f8fa" if r == 0 else "white"
            for c, cell_lines in enumerate(wrapped_cells):
                x0 = x + c * col_w
                x1 = x0 + col_w
                self.draw.rectangle((x0, y, x1, y + row_h), fill=fill, outline="#d0d7de", width=1)
                cy = y + 9
                for line in cell_lines:
                    self.draw.text((x0 + 9, cy), line, font=font, fill="#24292f")
                    cy += line_h
            y += row_h
        self.y = y + 18

    def draw_markdown_image(self, md_path: Path, alt: str, target: str) -> None:
        image_path = (md_path.parent / target).resolve()
        if not image_path.exists():
            self.draw_text_block(f"[missing image: {target}]", "body", fill="#b42318")
            return
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            max_w = PAGE_W - 2 * MARGIN_X
            max_h = 850
            im.thumbnail((max_w, max_h))
            caption_h = self.line_height(self.fonts["small"]) + 8 if alt else 0
            self.ensure_space(im.height + caption_h + 28)
            x = (PAGE_W - im.width) // 2
            self.page.paste(im, (x, self.y))
            self.y += im.height + 8
            if alt:
                caption = f"图：{alt}"
                self.draw.text((MARGIN_X, self.y), caption, font=self.fonts["small"], fill="#57606a")
                self.y += self.line_height(self.fonts["small"])
            self.y += 20

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = self.pages[0], self.pages[1:]
        first.save(output_path, "PDF", resolution=200.0, save_all=True, append_images=rest)


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_table_row(line: str) -> list[str]:
    return [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def normalize_inline(text: str) -> str:
    text = text.strip()
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return text


def render_markdown(input_path: Path, output_path: Path, font_path: str | None) -> None:
    renderer = PdfRenderer(font_path)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            renderer.draw_text_block(normalize_inline(" ".join(paragraph)), "body")
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                renderer.draw_code_block(code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            renderer.draw_markdown_image(input_path, image_match.group(1), image_match.group(2))
            i += 1
            continue

        if is_table_line(stripped):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and is_table_line(lines[i].strip()):
                if not is_table_separator(lines[i]):
                    table_lines.append(split_table_row(lines[i]))
                i += 1
            renderer.draw_table(table_lines)
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            renderer.draw_horizontal_rule()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            renderer.draw_text_block(normalize_inline(stripped[2:]), "title", fill="#0b1f33", before=20, after=22)
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            renderer.draw_text_block(normalize_inline(stripped[3:]), "h2", fill="#0b1f33", before=18, after=14)
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            renderer.draw_text_block(normalize_inline(stripped[4:]), "h3", fill="#0b1f33", before=14, after=10)
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            renderer.draw_text_block("• " + normalize_inline(stripped[2:]), "body", indent=24, before=2, after=2)
            i += 1
            continue

        if re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            renderer.draw_text_block(normalize_inline(stripped), "body", indent=24, before=2, after=2)
            i += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph()
            renderer.draw_text_block(normalize_inline(stripped[2:]), "body", fill="#57606a", indent=28, before=4, after=8)
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    renderer.save(output_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    font_path = find_font(args.font)
    if font_path:
        print(f"Using font: {font_path}")
    else:
        print("Warning: no CJK-capable font found; output may not render Chinese correctly.")
    render_markdown(input_path, output_path, font_path)
    print(f"Wrote PDF: {output_path}")


if __name__ == "__main__":
    main()

