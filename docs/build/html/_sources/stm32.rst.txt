STM32 Data Logger (``stm32/``)
==============================

Decoding the STM32 data-logger ``.BIN`` format — frames delimited by
``0xFF 0xFF``, byte-stuffed payloads, CRC16-protected — into
:class:`~AutoDNA.stm32.bin_parser.packet.Packet` objects and converting to/from
``.npz``, plus the serial service and live stream viewer used to pull recordings
off the device. ``serial`` (pyserial) and ``matplotlib`` are mocked at build
time, so these modules render from their docstrings without those packages
installed.

Binary parser
-------------

stm32.bin_parser.packet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.stm32.bin_parser.packet
   :members:
   :undoc-members:
   :show-inheritance:

stm32.bin_parser.stm_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.stm32.bin_parser.stm_utils
   :members:
   :undoc-members:
   :show-inheritance:

stm32.bin_parser.parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.stm32.bin_parser.parser
   :members:
   :undoc-members:
   :show-inheritance:

Serial service & live stream
----------------------------

stm32.stm_server.stm_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.stm32.stm_server.stm_server
   :members:
   :undoc-members:
   :show-inheritance:

stm32.stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: AutoDNA.stm32.stream
   :members:
   :undoc-members:
   :show-inheritance:
