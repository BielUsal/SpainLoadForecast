# Data changelog

Every manual fix or exclusion applied to the data, newest first.

| Date | Source | Period affected | Change | Reason |
|---|---|---|---|---|
| 2026-11-12 | ESIOS 1293, 1775 | 2025-12-11 onwards (1775: 2025-12-12 onwards) | Subtracted REE's hourly self-consumption estimate (indicator 1941 minus 545) to obtain `demand_mw` and `ree_forecast_mw`; published values kept in `*_published_mw` | REE started including estimated self-consumption in published demand on 2025-12-11 without revising history |
| 2026-11-12 | ESIOS 1293 | 2025-04-28 to 2025-04-29 | Flagged `is_masked` (values kept) | Iberian blackout: measured demand is not demand |
