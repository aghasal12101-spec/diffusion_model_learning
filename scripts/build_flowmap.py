"""Bundle the flow map lecture fragments so the page works when opened from disk.

Writes:
  research_landscape/flowmap/chapters.js            all chapter fragments as one script
  research_landscape/flowmap_lectures_standalone.html  single self-contained file (lab.js + chapters inlined)

Run after editing anything under research_landscape/flowmap/ch/:
  python3 scripts/build_flowmap.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "research_landscape"
CH_DIR = ROOT / "flowmap" / "ch"
IDS = ["intro", "fmm", "shortcut", "meanflow", "selfdistill", "ayf", "tvm", "imf", "pmf",
       "mfvar", "esc", "temf", "soflow", "emf", "cumflow", "otnfm", "summary"]


def script_safe(js: str) -> str:
    # keep "</script>" (and any "</") inside string literals from closing an inline <script>
    return js.replace("</", "<\\/")


def main():
    chapters = {i: (CH_DIR / f"{i}.html").read_text(encoding="utf-8") for i in IDS}
    bundle = ("// 自动生成：python3 scripts/build_flowmap.py。请修改 flowmap/ch/*.html 后重新生成。\n"
              "window.FLOWMAP_CHAPTERS = " + json.dumps(chapters, ensure_ascii=False) + ";\n")
    (ROOT / "flowmap" / "chapters.js").write_text(script_safe(bundle), encoding="utf-8")

    page = (ROOT / "flowmap_lectures.html").read_text(encoding="utf-8")
    lab = (ROOT / "flowmap" / "lab.js").read_text(encoding="utf-8")
    tag_lab = '<script src="flowmap/lab.js" charset="utf-8"></script>'
    tag_ch = '<script src="flowmap/chapters.js" charset="utf-8"></script>'
    assert tag_lab in page and tag_ch in page, "script tags not found in flowmap_lectures.html"
    page = page.replace(tag_lab, "<script>\n" + script_safe(lab) + "\n</script>")
    page = page.replace(tag_ch, "<script>\n" + script_safe(bundle) + "\n</script>")
    head = ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n</head><body>\n')
    (ROOT / "flowmap_lectures_standalone.html").write_text(head + page + "\n</body></html>\n", encoding="utf-8")
    print("wrote chapters.js and flowmap_lectures_standalone.html")


if __name__ == "__main__":
    main()
