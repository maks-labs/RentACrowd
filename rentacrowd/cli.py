"""Run a study from the terminal:

    uv run rentacrowd "A $49/mo AI meal planner that auto-orders groceries"
    uv run rentacrowd --file product.txt --notes "focus on rural India, use fresh personas"
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rentacrowd.config import get_settings
from rentacrowd.graph.build import build_graph

console = Console()


def _render(report) -> None:
    console.print(Panel(f"[bold]{report.headline}[/bold]", title="Verdict"))
    console.print(
        f"Overall purchase intent: [bold]{report.overall_mean_purchase_intent}/5[/bold]   "
        f"positive sentiment: [bold]{report.overall_pct_positive}%[/bold]\n"
    )
    t = Table(title="By segment")
    for col in ("Segment", "n", "Intent", "% pos", "Top objections"):
        t.add_column(col)
    for s in report.segment_results:
        t.add_row(
            s.segment, str(s.n), f"{s.mean_purchase_intent}", f"{s.pct_positive}",
            "; ".join(s.top_objections),
        )
    console.print(t)
    console.print("\n[bold]Key objections[/bold]")
    for o in report.key_objections:
        console.print(f"  • {o}")
    console.print("\n[bold]Recommended changes[/bold]")
    for c in report.recommended_changes:
        console.print(f"  • {c}")
    console.print(f"\n[dim]Confidence: {report.confidence_notes}[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(prog="rentacrowd")
    ap.add_argument("product", nargs="?", help="Free-text product description")
    ap.add_argument("--file", help="Read the product description from a file")
    ap.add_argument("--notes", default="", help="Extra instructions for the supervisor")
    args = ap.parse_args()

    raw = open(args.file).read() if args.file else args.product
    if not raw:
        ap.error("provide a product description or --file")

    s = get_settings()
    if not s.nvidia_api_key:
        console.print("[red]NVIDIA_API_KEY is not set (copy .env.example to .env).[/red]")
        sys.exit(1)

    graph = build_graph()
    started = time.time()
    console.print("[dim]Running study…[/dim]")
    final = graph.invoke(
        {"raw_product": raw, "request_notes": args.notes},
        config={"max_concurrency": s.max_concurrency, "recursion_limit": 60},
    )

    _render(final["report"])
    console.print(
        f"\n[green]Done in {time.time() - started:.0f}s[/green]  "
        f"({len(final['responses'])} responses)  ->  {final['study_dir']}"
    )


if __name__ == "__main__":
    main()
