# AI usage

## Tools I used, and for what
I used Claude (Anthropic) to explain the task and the slides, to draft `tawseel/model.py` (history features, time split, pipeline, evaluation), and to help me run everything in Google Colab. I ran the code and the tests myself in Colab (Python 3.13 there, with the pinned numpy, pandas and scikit-learn versions). I rewrote the pipeline helper functions (`extract_time_features`, `convert_to_text`, `convert_to_object`, `convert_to_numeric`) and `build_pipeline` myself so I could read and explain them; the results stayed identical (PR-AUC lift +0.132 on seed 0).

## AI-suggested code I rejected or fixed in add_history_features or time_split
I did not change the AI's `add_history_features` or `time_split`: both passed the 5 public tests and gave the expected results on seeds 0, 1, 2 and 3 (lift +0.132, +0.106, +0.141, +0.185), so no failing test showed a problem. One real issue I hit was not in these functions: after copying the new file into Colab, the notebook raised `NotImplementedError` because the kernel still had the old module loaded; restarting the session fixed it.

## How I checked that my model has no leak
`FEATURE_COLUMNS` excludes `courier_notes`, `delivered_at`, `failed` and `parcel_id`; the split is by time (the last 20% is the test set); history features count only strictly earlier parcels; and the test ROC AUC is 0.635, far below the 0.9 that signals a leak.
