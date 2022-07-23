import pint, numpy as np

from .constants import *

ureg = pint.UnitRegistry()
ureg.define('[concentration] = [substance] / [volume]')
ureg.define('molar = mol / L = M')

# TODO(mmd): Maybe overload tuple? or Quantity?
# TODO(mmd): I think I'm doing private members/methods wrong.
class Dose():
    def from_string(s): return Dose.from_quantity(ureg.parse_expression(s))
    def from_quantity(q): return Dose(q.magnitude, unit=q.units)
    def _fix_cmap(unit): return '%sM' % unit[:-1] if len(unit) > 0 and unit[-1] == 'm' else unit
    def _make_quantity(dose): return dose._quantity if type(dose) == Dose else ureg.parse_expression(dose)
    def __init__(self, value, unit_str='', unit=None):
        if np.isnan(value): self._quantity = ureg.Quantity(np.NaN, ureg.dimensionless)
        else: self._quantity = ureg.Quantity(value,
            ureg.Unit(Dose._fix_cmap(unit_str)) if unit is None else unit)

        q_base = self._quantity.to_base_units()
        self._base_args = (q_base.magnitude, q_base.units)

    def __str__(self): return "{:~P}".format(self._quantity)
    def __format__(self, fmt_str): return fmt_str.format(self._quantity)
    def __hash__(self): return self._base_args.__hash__()

    def __add__(self, other): return Dose.from_quantity(self._quantity + other._quantity)
    def __sub__(self, other): return Dose.from_quantity(self._quantity - other._quantity)
    def __mul__(self, other): return Dose.from_quantity(self._quantity * other._quantity)
    def __truediv__(self, other):
        if type(other) == Dose: return Dose.from_quantity(self._quantity / other._quantity)
        else: return Dose.from_quantity(self._quantity / other)
    def __floordiv__(self, other): return Dose.from_quantity(self._quantity // other._quantity)
    def __mod__(self, other): return Dose.from_quantity(self._quantity % other._quantity)

    def __lt__(self, other): return self._quantity <  Dose._make_quantity(other)
    def __le__(self, other): return self._quantity <= Dose._make_quantity(other)
    def __gt__(self, other): return self._quantity >  Dose._make_quantity(other)
    def __ge__(self, other): return self._quantity >= Dose._make_quantity(other)
    def __eq__(self, other): return self._quantity == Dose._make_quantity(other)
    def __ne__(self, other): return self._quantity != Dose._make_quantity(other)

    def magnitude(self): return self._quantity.magnitude
    def units(self): return self._quantity.units
