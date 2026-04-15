"""
Isolated LLM integration layer.

All external-LLM calls live under this package so that swapping providers
(Gemini → Haiku → Sonnet → local) is a single-file change in client.py.

Modules:
  client                — thin provider wrapper (Gemini today)
  extract_relationships — Phase 1.5: prose → implicit relationships
  generate_cards        — Phase 1.6: section → metadata card summaries
"""
