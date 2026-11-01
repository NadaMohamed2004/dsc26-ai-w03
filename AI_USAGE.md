# W3 results: Tawseel failed-COD baseline

Data: `make_parcels(4000, seed=0)`, failure rate 0.211 overall. Time split: first 80% of parcels (by `created_at`) train, last 20% test. Test base rate **0.22**, n = **800**.

| Model | ROC AUC | PR-AUC | Brier |
|---|---|---|---|
| Dummy (prior = base rate) | 0.500 | 0.220 | 0.172 |
| Rule: customer_prior_fail_rate (NaN -> train base rate) | 0.612 | 0.292 | n/a (not a probability) |
| Logistic regression + history + address TF-IDF | 0.635 | 0.352 | 0.164 |

PR-AUC lift over the dummy: **+0.132**. On three other seeds (1, 2, 3) the lift was +0.106, +0.141 and +0.185, and the Brier score was better than the dummy's each time.

Sherif, three sentences:
1. On the most recent 20% of parcels, which the model never saw, it finds failures better than guessing: PR-AUC 0.35 against 0.22 for a model that only knows the 22% base rate.
2. The gain is modest on purpose: it uses only what we know when the parcel is created, plus each customer's history strictly before that moment, with no courier notes and no delivery time, and a ROC AUC of 0.64 is what an honest number looks like here.
3. Use it to rank which parcels to confirm by phone first, not to decide automatically, and expect roughly this level in production, not 99%.
