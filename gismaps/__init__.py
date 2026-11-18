"""gismaps: the code behind the GIS map tutorials (https://github.com/shammun/gis-map-tutorials).

``gismaps.styles``  seven map styles for posters and short animations (glow, epochs, block, relief, spilhaus, cutaway,
                    satellite) on one shared base, ``styles.common`` (one projection + one pixel grid per map)
``gismaps.lens``    lens maps: one base map with round lenses onto other layers of the same place (city posters from
                    OpenStreetMap, a weather poster from ERA5, and their animations)
``gismaps.data``    finds the small input files, downloads them when missing
``gismaps.nb``      notebook helpers (show a picture, STOP checks, PASS/FAIL verification)
``gismaps.prepare_data``  how every reduced input is made from its public source

Maps created by Shammunul Islam. Code: MIT licence. Data: see the README for each source and its licence.
"""
__version__ = "0.1.0"
