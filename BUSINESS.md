# Business Gameplan (working hypothesis)

**Status: not validated.** This is the current best guess for turning Ask Me into a SaaS
product, worked out in one brainstorming session. Nothing here is committed — niche,
pricing, and go-to-market are all things to test and be ready to change. Treat this
document as "what we believe right now and why," not "the plan we're locked into."
See `docs/superpowers/specs/2026-08-10-ask-me-saas-design.md` for how this maps to the
technical build, and `ROADMAP.md` for the phase-by-phase implementation plan.

## Source

Framework extracted from Alex Hormozi's "$0 to $1M" blueprint
(youtube.com/watch?v=AN2KpRBsmRY), via his written companion article (not a raw
transcript — treat exact numbers/rules of thumb as approximate, not gospel).

## The framework (3 levels)

**Level 1 — Fundamentals.** Pick a starting point from Pain (a problem you personally
solved), Passion, or Profession. Enter the smallest, most specific market segment first.
Filter any candidate niche on 5 things: has the problem, has money, feels urgency, has
authority to decide, worth pursuing.

**Level 2 — Revenue generation.** Acquisition via warm outreach, cold outreach, content,
or paid ads (plus "lead getters": customers, employees, affiliates, agencies). Convert
via the CLOSER framework (Clarify → Label → Overview → sell the outcome → address
objections → Reinforce/book next meeting). Increase customer value via 8 levers (price,
delivery cost, frequency, upsell, quantity, tier, downsell, downgrade) — e.g. raise price
~20% every 5 new customers until conversion drops, then hold. Move to Level 3 once the
sales process is predictable, you have ~20+ customers, and unit economics are positive.

**Level 3 — Scaling.** Fix team failures via Document → Demonstrate → Duplicate + explain
the why. Build leverage via brand/people/skills/systems (systems compound hardest).
Expect the 5-stage emotional cycle (uninformed optimism → informed pessimism → valley of
despair → informed optimism → mastery) — most quitting happens in the valley. Don't
chase new ideas ("the woman in the red dress") — most self-made millionaires built one
thing. Improve using your own data: compare your top-10% vs bottom-10% outputs
(calls/content/ads) and iterate on the gap.

## Applied to Ask Me

**Niche (revised — see decision history below):** AI/LLM builders and engineers — people
who'd use this infrastructure to stand up a grounded, branding-focused AI assistant
(for their own portfolio, or for their clients' portfolios/personal brands). This is a
Passion-based pick, not a cold Profession pick — it's a world you're personally
energized by, which matters for sustaining motivation through Hormozi's Level 3 "valley
of despair" stage. Your own instance is intended as tenant #1 / the founding case
study, doubling as the product demo for this exact audience. **Two Owls Tavern** (a
fictional restaurant) currently occupies that slot instead — added as a technical test
tenant to prove the RAG pipeline handles non-personal, business-shaped content (hours,
menu, FAQs) correctly, not because the niche shifted toward restaurants. Niche and offer
below are unchanged; swap in your own content and re-ingest whenever you want it back as
the live demo.

**Offer:** Infrastructure to quickly stand up a grounded (RAG-based), tenant-isolated AI
assistant for personal/professional branding — trained on an entity's own content (bio,
services, past work, FAQs), embeddable on a site, answering visitor questions 24/7. Sold
to AI/LLM builders/engineers who either want one for themselves (proof-of-skill, live
portfolio artifact) or want to build/resell them for their own clients (consultants,
freelancers, personal brands) using this as the underlying platform.

**Positioning within a crowded market:** "Chat with your docs" infra is one of the most
saturated categories in AI tooling right now — most existing players target generic
enterprise knowledge bases or customer support. This is a narrower wedge: infra
purpose-built for personal/professional branding assistants, not generic document Q&A.
Smaller slice of a validated market, less direct competition.

**Pricing (starting anchor, not fixed):** ~$49-99/mo, annual preferred for early cash +
commitment signal. Calibrate up ~20% per new customer while conversion holds, per
Hormozi's Level 2 tactic, rather than guessing the "right" number upfront.

**Go-to-market — concierge first:** Personally find and onboard the first ~10-20
customers via build-in-public content (natural fit — this is your own professional
world) plus warm/cold outreach in AI/LLM builder communities, using the personal
instance as a live demo. Manually help set up their content/bot at first if needed. No
self-serve signup or billing until this motion is proven repeatable. Re-evaluate the
niche itself if these conversations don't confirm real urgency/willingness to pay.

**Definition of "first real validation":** ~20 paying customers via a repeatable,
predictable acquisition path — not a revenue number, not a feature checklist.

## Known tension to solve later, not now

Hormozi's $1M math (25 customers x $10k/year) assumes a high-ticket B2B model. At
$49-99/mo, $1M/year needs roughly 800-1,600 customers — a product-led/content growth
motion, not concierge sales. This is a Level 3 (scaling/systems) problem. The MVP does
not need to solve it, and over-building for it now would be premature.

## Decision history

- **2026-08-10, first pass:** niche = freelancers/consultants/creators (Profession-based
  pick, scored well on the 5 filters). Revised same session after the founder flagged
  "not a space I care about" — motivation/interest wasn't factored into the original
  filter run, and matters for sustaining a solo GTM motion long-term.
- **2026-08-10, revised:** niche = AI/LLM builders and engineers (Passion-based pick),
  offer reframed from "a bot for consultants" to "infra builders use to make branding
  assistants," which also resolves the weak money/urgency signal individual portfolio
  buyers would have had, since builders can serve their own clients with it.

## Open questions / things that could change this whole plan

- Does the "infra for builders" framing actually convert, or do AI/LLM builders see this
  as a fun side tool rather than something worth paying for?
- Is $49-99/mo the right anchor for a builder/platform play, or does infra pricing need
  to look different (usage-based, per-tenant-they-manage, etc.)?
- Does dogfooding your own instance produce a case study that resonates with this
  audience, or does it read as too narrow (just "one engineer's portfolio bot") to sell
  as a platform?
- Have the 5 filters actually been re-run rigorously against this revised niche, or is
  this a case of picking what feels good and backfilling the justification? Worth a
  genuine gut-check before investing real GTM effort.
