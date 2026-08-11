# Restaurant-tenant classify categories and generate persona — design spec

**Date:** 2026-08-11
**Status:** Approved, ready for implementation

## Purpose

`app/workflow.py`'s LangGraph chat pipeline (`classify → retrieve → generate → critique`)
was built and tuned against the app's original single-person-portfolio tenant. The
`_classify_node` category set (`'background' | 'project' | 'general'`) and the
`_generate_node` system prompt ("speaking on behalf of the person described below")
still reflect that tenant, not the restaurant tenant (`two-owls-tavern`) the app now
demos. This is a research/optimization pass on restaurant-tenant response quality
(chunking, data completeness, retrieval, prompting); this spec is the first and
smallest slice of that work — fixing the two places where the pipeline is actively
miscalibrated for a restaurant, before touching chunking/metadata/retrieval strategy.

## Non-goals (captured as backlog, not in this change)

- Structured dietary/allergen metadata on chunks for filtered/hybrid retrieval.
- Expanding tenant content (payment methods, cancellation policy, accessibility,
  delivery platforms, corkage/BYOB, etc.).
- Retrieval strategy changes: category-aware `top_k`, hybrid keyword+vector search.
- Per-tenant persona configuration. There is exactly one tenant today
  (`two-owls-tavern`); hardcoding a restaurant-flavored prompt is correct until a
  second, differently-shaped tenant actually exists. A config field now would be
  speculative.

## Design

### 1. `_classify_node` categories (`backend/app/workflow.py`)

Replace `'background' | 'project' | 'general'` with `'menu' | 'hours_location' |
'policies' | 'general'`, matching the tenant's actual content shape (`menu.md`,
`about.md`'s hours/location section, `about.md`'s house-policies section + `faq.md`).
The classification system prompt gets a one-line description per category so the LLM
has enough signal on ambiguous questions:

- `menu` — dishes, prices, ingredients, allergens, drinks.
- `hours_location` — hours, address, parking, reservations.
- `policies` — dress code, gratuity/split-check, dietary accommodation, pets, private
  events, gift cards, holiday closures.
- `general` — anything else (fallback).

No change to `_retrieve_node`: it already does `f"[{category}] {query}"` before
embedding, which is category-agnostic and works unchanged with the new category
strings.

### 2. `_generate_node` system prompt (`backend/app/workflow.py`)

Reframe from "speaking on behalf of the person described below" to a restaurant-host
voice: the assistant answers as the restaurant itself (first person plural — "we" /
"our"), grounded only in the retrieved context, consistent with how `about.md` and
`faq.md` are already written. Add one explicit guardrail line: never assert a dish is
safe for a given allergy with certainty — defer to asking staff. This isn't new
policy, just making the prompt say outright what the tenant content already implies
(`faq.md`'s "please tell your server directly rather than relying on the menu notes
alone").

### 3. Tests (`backend/tests/test_workflow.py`)

5 of 6 tests script the old category strings (`"general"`, `"background"`) as fake LLM
outputs, including one assertion on the literal search-text prefix
(`"[background] Where did you go to school?"`). These get updated to the new category
names as part of this change — not left referencing categories that no longer exist
in the classifier's output space.

## Testing

Run the backend test suite (`.venv\Scripts\python.exe -m pytest` from `backend/`) after
updating `test_workflow.py`; all tests should pass. No new test cases are required —
this changes prompt content and a fixed string set, not branching logic, so the
existing coverage (classify success/failure, retry behavior, search-text biasing)
still exercises the changed code paths correctly once updated to the new category
names.
