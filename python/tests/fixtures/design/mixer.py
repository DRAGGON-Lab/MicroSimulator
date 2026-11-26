"""Three liquids metered into one stream and mixed."""

# This file is the design tool's own example, kept as it is written there.
# ruff: noqa: E501

import microfluidics as mf
from microfluidics.layers import FLOW
from microfluidics.processes import COSINC_SU8_PDMS, FLOWJEM_SU8_PDMS
from microfluidics.regions import AccessPad, Junction, Serpentine
from microfluidics.units import mbar, mm, mPa_s, pct, um


class Mixer(mf.Design):
    """Three liquids meet in shares of a half, three tenths and a fifth, and mix
    along a serpentine.

    Each liquid reaches the junction through a short serpentine, and the length of
    that way sets its share. Every channel is one width, so the shares hold however
    the chip is made.
    """

    processes = (FLOWJEM_SU8_PDMS, COSINC_SU8_PDMS)
    slab = mf.Slab(25 * mm, 20 * mm, thickness=4 * mm)
    glass = mf.Glass(75 * mm, 25 * mm, thickness=1 * mm)
    layers = mf.Stack({FLOW: mf.LayerSpec(height=50 * um, gds=1)})

    def build(self) -> None:
        width, pitch, apart = 200 * um, 600 * um, 4 * mm
        self.meet = Junction(inputs=3, pitch=apart, collector_width=width, input_width=width).at(5 * mm, 12 * mm)

        # Each inlet stands under its input of the junction. A serpentine of three
        # runs leaves two pitches to the left of where it is entered, so each pad
        # stands that far to the right. The junction's last input is nearest its
        # outlet, so the largest share enters there.
        under, clear = 5 * mm + pitch * 2, 600 * um
        self.third_in = AccessPad(channel_width=width).at(under, 2.5 * mm)
        self.second_in = AccessPad(channel_width=width).at(under + apart, 2.5 * mm)
        self.first_in = AccessPad(channel_width=width).at(under + apart * 2, 2.5 * mm)
        self.third_resistor = Serpentine(runs=3, run_length=5.4 * mm, width=width, pitch=pitch).attach(
            lambda resistor: resistor.inlet, to=self.third_in.opening, offset=clear
        )
        self.second_resistor = Serpentine(runs=3, run_length=3.3 * mm, width=width, pitch=pitch).attach(
            lambda resistor: resistor.inlet, to=self.second_in.opening, offset=clear
        )
        self.first_resistor = Serpentine(runs=3, run_length=2 * mm, width=width, pitch=pitch).attach(
            lambda resistor: resistor.inlet, to=self.first_in.opening, offset=clear
        )
        self.third_lead = self.route(FLOW, width, self.third_in.opening, self.third_resistor.inlet)
        self.second_lead = self.route(FLOW, width, self.second_in.opening, self.second_resistor.inlet)
        self.first_lead = self.route(FLOW, width, self.first_in.opening, self.first_resistor.inlet)
        self.third_feed = self.route(FLOW, width, self.third_resistor.outlet, self.meet.input[0])
        self.second_feed = self.route(FLOW, width, self.second_resistor.outlet, self.meet.input[1])
        self.first_feed = self.route(FLOW, width, self.first_resistor.outlet, self.meet.input[2])

        self.mixer = Serpentine(runs=7, run_length=4 * mm, width=width, pitch=pitch).attach(
            lambda mixer: mixer.inlet, to=self.meet.outlet
        )
        self.outlet = AccessPad(channel_width=width).attach(lambda pad: pad.opening, to=self.mixer.outlet, offset=clear)
        self.exit = self.route(FLOW, width, self.mixer.outlet, self.outlet.opening)

        self.first, self.second, self.third = mf.Species(), mf.Species(), mf.Species()
        # Every inlet at one pressure, as from one source.
        self.operate(
            viscosity=1 * mPa_s,
            drive={
                self.first_in: 5 * mbar,
                self.second_in: 5 * mbar,
                self.third_in: 5 * mbar,
                self.outlet: 0 * mbar,
            },
            feed={
                self.first_in: {self.first: 100 * pct},
                self.second_in: {self.second: 100 * pct},
                self.third_in: {self.third: 100 * pct},
            },
        )
        self.first_share = mf.Requirement(
            "The first liquid is 50 % of the mixture",
            mf.fraction(self.first, self.outlet),
            mf.target(50 * pct, within=1 * pct, relative=True),
        )
        self.second_share = mf.Requirement(
            "The second liquid is 30 % of the mixture",
            mf.fraction(self.second, self.outlet),
            mf.target(30 * pct, within=1 * pct, relative=True),
        )
        self.third_share = mf.Requirement(
            "The third liquid is 20 % of the mixture",
            mf.fraction(self.third, self.outlet),
            mf.target(20 * pct, within=1 * pct, relative=True),
        )
        self.tune(self.second_resistor, Serpentine.run_length)
        self.tune(self.third_resistor, Serpentine.run_length)
