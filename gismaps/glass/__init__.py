"""gismaps.glass: the glass-pane globe stack.

A relief globe with a stack of glass panes in front of it. Every pane carries one field of the same weather
forecast (temperature, pressure, wind, precipitable water, cloud, rain) as the round picture of one patch of the
Earth, so the panes line up with the globe behind them.

``gismaps.glass.gfs``      the data: a small shipped subset of one NOAA GFS run, or a live anonymous read from
                           dynamical.org (falls back to the shipped file)
``gismaps.glass.ortho``    the orthographic projection by hand (place <-> pane coordinates) and the grid reader
``gismaps.glass.panes``    one field -> one pane picture (colour, opacity, isobars, coastline); the globe's skin
``gismaps.glass.render``   a small ray caster: the globe and the panes seen through a perspective camera
``gismaps.glass.stack``    ``GlassStack``: the finished poster and the animation (legend, clock, timeline, credits)
``gismaps.glass.remotion`` writes the textures and the manifest that the Remotion version (``remotion/glass_stack``)
                           loads

Map created by Shammunul Islam. Data: dynamical.org / NOAA GFS (CC BY 4.0); relief: ETOPO 2022 (NOAA NCEI);
coastlines: Natural Earth.
"""
