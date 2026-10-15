"""rcassess - assessment engine for corroded RC flexural members (NumPy only).

Shared by the training scripts (IS 456 formulas in analytical.py), the local server (serve.py)
and the Vercel functions in api/. The trained ANN is read from model.json, written by
ann_model/04_export_web_model.py - no scikit-learn or pandas is needed at run time.
"""
