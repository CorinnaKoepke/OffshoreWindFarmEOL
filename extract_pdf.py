"""Extract the paper PDF to a clean, readable .txt file."""
import re
from pathlib import Path
from pypdf import PdfReader

SRC = Path(__file__).with_name("energies-16-04771-v2.pdf")
OUT = Path(__file__).with_name("energies-16-04771-v2.txt")

# Ligatures emitted by the PDF font, mapped to plain ASCII sequences.
LIGATURES = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
}


def clean_line(line: str) -> str:
    # Join typographic ligatures into their plain ASCII letters.
    for lig, rep in LIGATURES.items():
        line = line.replace(lig, rep)
    # Collapse repeated internal whitespace, strip ends.
    return re.sub(r"[ \t\u00a0]+", " ", line).strip()


_URL = re.compile(r"(https?://|\bwww\.)")


def merge_hyphen_broken(lines: list[str]) -> list[str]:
    """Rejoin words the PDF split across a line break (e.g. 'decom-\\nmissioning'),
    but never touch hyphens that belong to a URL."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1]
            is_url = bool(_URL.search(line)) or nxt.startswith("http")
            word_break = bool(nxt) and nxt[0].isalpha() and nxt[0].islower()
            if not is_url and word_break:
                out.append(line[:-1] + nxt)
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def main() -> None:
    reader = PdfReader(str(SRC))
    n = len(reader.pages)
    print(f"pages: {n}")

    parts = []
    for i, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        lines = [clean_line(l) for l in raw.splitlines()]
        lines = merge_hyphen_broken(lines)
        # Drop empty lines but keep a single blank between paragraphs via joining.
        text = "\n".join(lines)
        parts.append(f"\n\n===== PAGE {i} =====\n\n{text}")

    body = "".join(parts)
    # Normalise any remaining double-spaces and tidy excessive blank runs.
    body = re.sub(r"\n{4,}", "\n\n\n", body)

    OUT.write_text(body, encoding="utf-8")
    print(f"wrote: {OUT} ({len(body)} chars)")


if __name__ == "__main__":
    main()
