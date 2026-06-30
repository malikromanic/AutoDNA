# -*- coding: utf-8 -*-
"""Manual debug script for inspecting the fuel model's learning curve.

Not imported anywhere else — run directly (``python -m app.learning``)
against whatever drives are currently in the feature store to eyeball how
fuel-model error changes with the number of accumulated drives.

Created on Thu Jun 25 22:07:00 2026

@author: mihal
"""

from AutoDNA.app.fuel_model import compute_and_plot_learning_curve
from AutoDNA.app.feature_loader import load_store

records = load_store()
for r in records:
    print(f"{r['drive_name']:30s}  fuel={r['fuel_l100km']:.2f} L/100km")
compute_and_plot_learning_curve(records)