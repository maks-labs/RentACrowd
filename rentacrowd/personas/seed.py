"""Seed the persona library with a broad, deliberately diverse population.

    uv run rentacrowd-seed            # ~100 people, 4 per slice
    uv run rentacrowd-seed --per 8    # ~200 people

Diversity is guaranteed by construction: the slices below span life stage,
income, geography, values and tech comfort, so the model is never asked to "be
diverse" on its own - it fills a scaffold we chose.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from rentacrowd.personas import library
from rentacrowd.personas.generate import make_personas

SLICES: list[tuple[str, str]] = [
    ("University students", "Undergraduates on a tight budget, flat-sharing in a big city, phone-first, time-rich but cash-poor."),
    ("Recent graduates", "First proper job, entry-level salary, house-sharing, still on family phone plans and streaming logins."),
    ("Early-career professionals", "Late 20s, single, urban, meaningful disposable income, eat out often, subscribe to a lot."),
    ("Dual-income couples, no kids", "Two salaries, saving hard for a deposit, optimise spending but will pay for time."),
    ("New parents", "First child under two, chronically sleep-deprived, budgets suddenly rewritten, decisions made at 11pm."),
    ("Parents of primary-school kids", "Two working parents, school runs, packed weekday logistics, fiercely time-poor."),
    ("Parents of teenagers", "Mid-career, feeding hungry teens, watching costs rise, kids influence household purchases."),
    ("Single parents", "One income, no slack in the budget, every subscription is scrutinised monthly."),
    ("Empty nesters", "50s-60s, mortgage nearly done, real disposable income, moderate tech comfort, brand-loyal."),
    ("Retirees on fixed income", "Pension-dependent, cautious with new services, prefer phone or in-person over apps."),
    ("Shift workers", "Nurses, drivers, hospitality and warehouse staff on rotating and night shifts; nothing lines up with normal opening hours."),
    ("Small-business owners", "Independent shop owners, tradespeople and cafe operators; treat every purchase as a business cost."),
    ("Freelancers and gig workers", "Variable monthly income, feast-or-famine, wary of fixed recurring commitments."),
    ("Rural households", "Poor delivery coverage, patchy broadband, long drives to the nearest large shop."),
    ("Small-town households", "Limited local retail choice, strong word-of-mouth culture, price-aware."),
    ("Recent immigrants", "Rebuilding a household from scratch in a new country, sending money home, comparing everything to back home."),
    ("Health and fitness driven", "Train seriously, track macros, dietary rules are non-negotiable, will pay for performance."),
    ("Households with medical dietary needs", "Coeliac, diabetic, allergy or renal diets; errors are dangerous, not just annoying."),
    ("Frugal optimisers", "Deal-hunters with spreadsheets, stack coupons and cashback, enjoy the game of paying less."),
    ("Convenience-first premium buyers", "Outsource everything they can, time is the scarce resource, barely look at price."),
    ("Sustainability-first consumers", "Buy on ethics and footprint, distrust greenwashing, will pay more for provenance."),
    ("Tech early adopters", "Gadget-forward, beta-testers, already pay for a dozen subscriptions and love trying more."),
    ("Privacy-conscious tech sceptics", "Deliberately low-tech, refuse data collection, resent apps replacing simple things."),
    ("Emerging-market urban consumers", "Cities in India, Brazil, Nigeria and Indonesia; mobile-first, price-sensitive, cash and UPI/PIX habits."),
    ("Multi-generational and caregiver households", "Adults caring for an ageing parent alongside their own family; juggling two sets of needs and budgets."),
]


def seed(per_slice: int = 4, workers: int = 8) -> int:
    """Generate every slice concurrently. The shared rate limiter keeps the
    combined request rate under the NIM ceiling, so more workers just means less
    idle time, not more calls."""
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(make_personas, name, brief, per_slice): name
            for name, brief in SLICES
        }
        for fut in as_completed(futures):
            name = futures[fut]
            people = library.add(fut.result())
            total += len(people)
            print(f"  {name:<44} +{len(people):>3}   (library: {library.size()})")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(prog="rentacrowd-seed")
    ap.add_argument("--per", type=int, default=4, help="personas per audience slice")
    args = ap.parse_args()

    print(f"Seeding {len(SLICES)} audience slices x {args.per} = ~{len(SLICES) * args.per} people")
    print(f"into {library.LIBRARY_DIR}\n")
    total = seed(args.per)
    print(f"\nDone. {total} personas written. Library now holds {library.size()}.")


if __name__ == "__main__":
    main()
