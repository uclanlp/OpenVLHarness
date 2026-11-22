import math
import re
from datetime import datetime, timedelta

from openvlharness.toolsmith import GeneratedTool

UNITS = {
    "length": {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "km": 1000.0,
               "kilometer": 1000.0, "kilometers": 1000.0, "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
               "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "mi": 1609.344, "mile": 1609.344,
               "miles": 1609.344, "yd": 0.9144, "yard": 0.9144, "yards": 0.9144, "ft": 0.3048, "foot": 0.3048,
               "feet": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "nmi": 1852.0, "nauticalmile": 1852.0},
    "mass": {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001, "grams": 0.001,
             "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6, "t": 1000.0, "tonne": 1000.0, "tonnes": 1000.0,
             "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
             "oz": 0.028349523125, "ounce": 0.028349523125, "ounces": 0.028349523125, "st": 6.35029318,
             "stone": 6.35029318},
    "area": {"m2": 1.0, "sqm": 1.0, "squaremeter": 1.0, "squaremeters": 1.0, "km2": 1e6, "sqkm": 1e6,
             "squarekilometer": 1e6, "ha": 1e4, "hectare": 1e4, "hectares": 1e4, "acre": 4046.8564224,
             "acres": 4046.8564224, "ft2": 0.09290304, "sqft": 0.09290304, "squarefoot": 0.09290304,
             "squarefeet": 0.09290304, "mi2": 2589988.110336, "sqmi": 2589988.110336, "squaremile": 2589988.110336},
}
KELVIN = {"c": (1.0, 273.15), "celsius": (1.0, 273.15), "k": (1.0, 0.0), "kelvin": (1.0, 0.0),
          "f": (5 / 9, 255.3722222222222), "fahrenheit": (5 / 9, 255.3722222222222)}
OPERATORS = {"add": sum, "multiply": math.prod, "mean": lambda values: sum(values) / len(values),
             "subtract": lambda values: values[0] - sum(values[1:]),
             "divide": lambda values: values[0] / math.prod(values[1:]),
             "power": lambda values: values[0] ** values[1]}
DATES = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y")


def unit(name):
    return re.sub(r"[\s.]+", "", str(name).lower())


def date(text):
    for pattern in DATES:
        try:
            return datetime.strptime(str(text).strip(), pattern).date()
        except ValueError:
            continue
    return None


class UnitCompute(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="unit_compute",
            tool_description=(
                "Offline arithmetic. op='convert' converts value between from_unit and to_unit (length, mass, area, "
                "temperature); op='arith' applies operator (add, subtract, multiply, divide, power, mean) to operands; "
                "op='date_diff' returns the days from date1 to date2, or the date add_days after date1. "
                "DO NOT USE FOR: currency conversion or looking up quantities."),
            input_schema={"type": "object", "properties": {
                "op": {"type": "string", "enum": ["convert", "arith", "date_diff"]},
                "value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"},
                "operands": {"type": "array", "items": {"type": "number"}}, "operator": {"type": "string"},
                "date1": {"type": "string"}, "date2": {"type": "string"}, "add_days": {"type": "integer"}},
                "required": ["op"]})

    def execute(self, op="", value=None, from_unit="", to_unit="", operands=None, operator="", date1="", date2="",
                add_days=None):
        try:
            if op == "convert":
                return self.convert(float(value), unit(from_unit), unit(to_unit))
            if op == "arith":
                values = [float(operand) for operand in operands or []]
                if operator not in OPERATORS or not values:
                    return {"error": f"operator must be one of {sorted(OPERATORS)} with at least one operand"}
                return {"result": OPERATORS[operator](values), "operator": operator, "operands": values}
            if op == "date_diff":
                return self.dates(date1, date2, add_days)
        except (TypeError, ValueError, ZeroDivisionError, IndexError, OverflowError) as exc:
            return {"error": f"{op} failed: {exc}"}
        return {"error": f"unknown op {op!r} (expected convert, arith or date_diff)"}

    @staticmethod
    def convert(value, source, target):
        if source in KELVIN and target in KELVIN:
            kelvin = value * KELVIN[source][0] + KELVIN[source][1]
            result = (kelvin - KELVIN[target][1]) / KELVIN[target][0]
            return {"result": round(result, 6), "family": "temperature"}
        for family, factors in UNITS.items():
            if source in factors and target in factors:
                return {"result": round(value * factors[source] / factors[target], 6), "family": family}
        return {"error": f"cannot convert {source} to {target}"}

    @staticmethod
    def dates(first, second, add_days):
        start = date(first)
        if start is None:
            return {"error": f"unreadable date1 {first!r}"}
        if add_days is not None:
            return {"result_date": (start + timedelta(days=int(add_days))).isoformat()}
        end = date(second)
        if end is None:
            return {"error": f"unreadable date2 {second!r}"}
        return {"result_days": (end - start).days, "abs_days": abs((end - start).days)}
