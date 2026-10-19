"""Derive submission/final_report.tex from submission/main.tex.

The final report is the paper's body without its appendix: named authors instead of the
double-blind block, a course notice in place of the ICML footer, and every reference into the
appendix rewritten so that the claim it supported is either stated in place or no longer leans
on material the reader cannot see.

It is generated rather than copied by hand so that a later edit to main.tex can be carried over by
rerunning this script. Every replacement must match exactly once; if main.tex has drifted so
that one no longer does, the script stops instead of emitting a report with a dangling \\cref.

    python scripts/make_final_report.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission" / "main.tex"
DST = ROOT / "submission" / "final_report.tex"
REPO = "https://github.com/arrdel/plexus"

APPENDIX_LABELS = ("app:", "fig:graded", "fig:reach", "tab:density", "tab:mcnemar", "tab:noise",
                   "tab:zeroshot")

REPLACEMENTS: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- header and front matter
    ("% PLEXUS -- ICML 2027 submission, main paper.\n%\n% Build:  latexmk -pdf main.tex     "
     "(or: pdflatex; bibtex; pdflatex; pdflatex)",
     "% PLEXUS -- final report. GENERATED from main.tex by scripts/make_final_report.py; edit\n"
     "% main.tex and rerun the script rather than editing this file.\n%\n"
     "% The paper's body without its appendix, with named authors and a course notice.\n%\n"
     "% Build:  pdflatex final_report; bibtex final_report; pdflatex final_report (twice)"),
    ("% ICML 2027, double-blind. The style file anonymises the author block and turns on line numbers\n"
     "% by itself while [accepted] is absent; add [accepted] for the camera-ready. Note that icml2027.sty\n"
     "% loads times and hyperref itself, which is why hyperref is required above it and \\theHalgorithm is\n"
     "% defined before it.\n\\usepackage{icml2027}",
     "% [accepted] prints the author list and drops the review line numbers. Its footer would claim\n"
     "% the paper appears in the ICML proceedings, so the notice is replaced with the course's.\n"
     "\\usepackage[accepted]{icml2027}\n"
     "\\makeatletter\n"
     "\\renewcommand{\\ICML@appearing}{Final report, Graph Machine Learning (Section 002),\n"
     "  Georgia State University, Fall 2026. Code and run artefacts:\n"
     "  \\url{" + REPO + "}.}\n"
     "\\makeatother"),
    ("% ICML's title block is an argument to \\twocolumn, not a \\maketitle. While [accepted] is absent\n"
     "% the style file replaces the author list with ``Anonymous Authors'' and suppresses the\n"
     "% affiliation footnote, so the names below are never shown in review.\n",
     "% ICML's title block is an argument to \\twocolumn, not a \\maketitle.\n"),
    ("\\icmlauthor{Anonymous Author}{anon}\n\\end{icmlauthorlist}\n\n"
     "\\icmlaffiliation{anon}{Anonymous Institution}\n"
     "\\icmlcorrespondingauthor{Anonymous Author}{anon@example.com}",
     "\\icmlauthor{Adele Chinda}{gsu}\n\\icmlauthor{Haoxin Wang}{gsu}\n"
     "\\icmlauthor{Esra Akbas}{gsu}\n\\end{icmlauthorlist}\n\n"
     "\\icmlaffiliation{gsu}{Department of Computer Science, Georgia State University, Atlanta, GA,\n"
     "  USA}\n\\icmlcorrespondingauthor{Adele Chinda}{chindahel1@gmail.com}"),

    # ---------------------------------------------------------------- body
    ("\\item \\textbf{A reproducible harness}, with the provenance chain any reproduction of\n"
     "      \\citet{tagdlm2026} must assume (\\cref{app:provenance}), a measured cost model\n"
     "      (\\cref{app:cost}), and four traps that silently produce a wrong number (\\cref{app:traps}).",
     "\\item \\textbf{A reproducible harness}, released at \\url{" + REPO + "}, with every\n"
     "      preprocessing assumption a reproduction of \\citet{tagdlm2026} must make stated\n"
     "      explicitly, a measured cost model, and tests for the implementation traps that silently\n"
     "      produce a wrong number."),
    ("rather than the floating-point minimum, for a reason given in\n"
     "\\cref{app:traps} that costs a run if ignored.",
     "rather than the floating-point minimum: LLaDA adds its own\n"
     "padding mask to the supplied bias, and the sum of two minima overflows to $-\\infty$, which the\n"
     "attention kernel turns into \\texttt{NaN} rather than into zero weight."),
    (" (\\cref{tab:density}), so the density", ", so the density"),
    ("That is deliberate, since \\cref{app:provenance} shows that matching their absolute\n"
     "numbers requires assumptions their paper does not pin down.",
     "That is deliberate, since matching their absolute numbers requires\n"
     "preprocessing assumptions their paper does not pin down."),
    ("averaged\nover $300$ (\\cref{tab:density}) \\texttt{star}", "averaged\nover $300$ examples \\texttt{star}"),
    ("The proof is immediate ($u \\to 0 \\to w$ is a length-two walk for every pair) and is given\n"
     "with the corresponding statement for \\texttt{ego} in \\cref{app:proofs}.",
     "The proof is immediate: $u \\to 0 \\to w$ is a length-two walk for every pair. Under\n"
     "\\texttt{ego} a walk may only follow observed edges, so the same statement holds with two\n"
     "replaced by the subgraph's diameter."),
    ("\\Cref{fig:reach} computes the reachability curves rather than asserting them:",
     "Computing the reachability curves for a sampled subgraph, rather than asserting\n"
     "them, shows the difference:"),
    (" \\Cref{fig:graded} renders\nthis on the same graph, and \\cref{app:proofs} gives the proof.", ""),
    ("fixed buffer (\\cref{app:cost}).", "fixed buffer."),
    ("\\Cref{app:provenance} explains why this split is an\n"
     "assumption rather than a match, and why we did not inherit an upstream preprocessing bug that\n"
     "would have silently substituted the previous paper's abstract for the $11.7\\%$ of nodes with\n"
     "none.",
     "The split is our assumption rather than a documented\n"
     "match, and we deliberately avoid an upstream preprocessing bug that would silently substitute\n"
     "the previous paper's abstract for the $11.7\\%$ of nodes with none."),
    ("over three seeds or more runs, listed run by run in\n"
     "\\cref{app:results}. No difference smaller than the replicate noise floor of \\cref{app:noise} "
     "is called a\ndifference, whatever its sign, and where per-example predictions exist we test "
     "the difference\nbetween decisions rather than between means (\\cref{app:paired}).",
     "over three seeds or more runs. No difference smaller\n"
     "than the replicate noise floor of \\cref{sec:ncresults} is called a difference, whatever its\n"
     "sign. Where per-example predictions exist we also test the difference between decisions\n"
     "rather than between means, with McNemar's test over the same examples and a Bonferroni\n"
     "correction for the number of tests."),
    ("noise floor of \\cref{tab:noise}, while", "noise floor of \\cref{sec:ncresults}, while"),
    ("before the mask is built, and \\cref{app:provenance} explains why omitting that step would leak\n"
     "the label into \\texttt{ego} and \\texttt{ego\\_star} but not into \\texttt{star}. "
     "\\Cref{app:results}\nlists every run individually and the paired tests between them.",
     "before the mask is built. Otherwise \\texttt{ego} and \\texttt{ego\\_star} would let the two\n"
     "endpoints read each other exactly when the edge exists, writing the label into the mask, while\n"
     "\\texttt{star} connects them either way."),
    ("(\\cref{tab:zeroshot}, three seeds)", "(three seeds)"),
    ("only to\nwithin $\\noisefloor$ points (\\cref{tab:noise}): that is",
     "only to\nwithin $\\noisefloor$ points: that is"),
    ("no comparison in\n\\cref{tab:mcnemar} separates", "no paired comparison\nseparates"),
    ("where the measured densities of\n\\cref{tab:density} predicted",
     "where the measured mask densities\npredicted"),
    ("correction for the $\\nptests$ tests we report (\\cref{app:paired}).",
     "correction for the $\\nptests$ tests we report."),
    ("preprocessing pipeline that \\cref{app:provenance} states as an assumption.",
     "preprocessing pipeline that we state as an assumption."),
    ("\\citet{tagdlm2026} and \\cref{app:provenance} lists five decisions their paper leaves open.",
     "\\citet{tagdlm2026} and their paper leaves five preprocessing decisions open."),
    ("which is the diagnostic \\cref{app:results} sets out and the one thing that\nwould distinguish",
     "which is the one thing that would\ndistinguish"),
    ("and \\cref{app:provenance} sets\nout which assumptions any reproduction of them must supply,",
     "and we state the assumptions any\nreproduction of them must supply,"),

    # ---------------------------------------------------------------- no appendix
    ("\\appendix\n\\input{appendix}\n", ""),
]


def main() -> int:
    text = SRC.read_text()

    # The boxed status note is switched off in main.tex, but its source still points into the
    # appendix and describes a schedule that has since finished, so it is removed outright.
    note = re.compile(r"\\draftnote\{\\textbf\{Status of the empirical program\.\}.*?pin down\.\}\n\n",
                      re.S)
    text, n = note.subn("", text)
    if n != 1:
        print("status draftnote not found exactly once", file=sys.stderr)
        return 1

    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            print(f"expected exactly one match, found {count}:\n{old[:160]}", file=sys.stderr)
            return 1
        text = text.replace(old, new)

    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))
    left = sorted({m for m in re.findall(r"\\[cC]?ref\{([^}]+)\}", body)
                   for lab in APPENDIX_LABELS if lab in m})
    if left or "\\appendix" in body:
        print(f"appendix references remain: {left}", file=sys.stderr)
        return 1

    DST.write_text(text)
    print(f"wrote {DST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
