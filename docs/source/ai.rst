AI / Signal Processing (``ai/``)
================================

IMU signal preprocessing, feature extraction, spectrogram conversion, and the
standalone model-training experiments (BiLSTM, GRU, window-based CNN/XGBoost).
These modules are **not** part of the live GPS-only app — they support the
separate IMU/ML work. Heavy dependencies (``torch``, ``xgboost``, ``sklearn``,
``scipy``, ``matplotlib``) are mocked at build time, so the docs render from the
docstrings without those packages installed.

Signal preprocessing & features
-------------------------------

ai.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

ai.features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.features
   :members:
   :undoc-members:
   :show-inheritance:

ai.spectrograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.spectrograms
   :members:
   :undoc-members:
   :show-inheritance:

Dataset & demo pipeline
-----------------------

ai.demo_preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.demo_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

ai.demo_spectrograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.demo_spectrograms
   :members:
   :undoc-members:
   :show-inheritance:

ai.raw_to_inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.raw_to_inputs
   :members:
   :undoc-members:
   :show-inheritance:

BiLSTM experiments
------------------

ai.bilstm.bilstm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.bilstm.bilstm
   :members:
   :undoc-members:
   :show-inheritance:

ai.bilstm.turns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.bilstm.turns
   :members:
   :undoc-members:
   :show-inheritance:

ai.bilstm.hills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.bilstm.hills
   :members:
   :undoc-members:
   :show-inheritance:

ai.bilstm.tmp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.bilstm.tmp
   :members:
   :undoc-members:
   :show-inheritance:

GRU experiment
--------------

ai.gru.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.gru.train
   :members:
   :undoc-members:
   :show-inheritance:

Window-based CNN / XGBoost
--------------------------

ai.window_based.build_dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.window_based.build_dataset
   :members:
   :undoc-members:
   :show-inheritance:

ai.window_based.train_cnn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.window_based.train_cnn
   :members:
   :undoc-members:
   :show-inheritance:

ai.window_based.train_xgboost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.ai.window_based.train_xgboost
   :members:
   :undoc-members:
   :show-inheritance:
