# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:07:00 2026

@author: mihal
"""

from AutoDNA.app.fuel_model import compute_and_plot_learning_curve
from AutoDNA.app.feature_loader import load_store

records = load_store()
for r in records:
    print(f"{r['drive_name']:30s}  fuel={r['fuel_l100km']:.2f} L/100km")
compute_and_plot_learning_curve(records)