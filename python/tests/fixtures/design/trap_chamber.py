"""A feed channel with one shallow chamber: the geometry of one biopixel trap."""

import microfluidics as mf
from microfluidics.layers import FLOW, TRENCH
from microfluidics.processes import COSINC_SU8_PDMS
from microfluidics.regions import AccessPad, TrapChamber
from microfluidics.units import mbar, mm, mPa_s, pct, ul_min, um


class TrapChamberChip(mf.Design):
    """One 100 by 85 by 1.65 um chamber beside a 100 by 10 um channel 300 um long, fed with
    one species. The chamber is the trapping region Prindle et al. report (Nature 481, 39-44,
    2012); the channel, the slab and how the chip is run are chosen for the tests that read
    this design's scene."""

    processes = (COSINC_SU8_PDMS,)
    slab = mf.Slab(10 * mm, 10 * mm, thickness=4 * mm)
    glass = mf.Glass(24 * mm, 50 * mm, thickness=0.17 * mm)
    layers = mf.Stack(
        {TRENCH: mf.LayerSpec(height=1.65 * um, gds=1), FLOW: mf.LayerSpec(height=10 * um, gds=2)}
    )

    def build(self) -> None:
        width = 100 * um
        self.inlet = AccessPad(channel_width=width).at(5 * mm, 2 * mm)
        self.trap = TrapChamber(
            channel_length=300 * um, channel_width=width, trap_width=100 * um, trap_length=85 * um
        ).attach(lambda trap: trap.inlet, to=self.inlet.opening)
        self.outlet = AccessPad(channel_width=width).attach(
            lambda pad: pad.opening, to=self.trap.outlet
        )

        self.nutrient = mf.Species()
        self.operate(
            viscosity=0.7 * mPa_s,
            drive={self.inlet: 0.01 * ul_min, self.outlet: 0 * mbar},
            feed={self.inlet: {self.nutrient: 100 * pct}},
        )
