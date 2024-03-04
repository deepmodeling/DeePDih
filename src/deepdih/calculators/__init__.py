from ase.calculators.mixing import SumCalculator
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from .bias import OpenMMBiasCalculator
from .gromacs import GromacsTopCalculator

def merge_calculators(calc1: Calculator, calc2: Calculator) -> Calculator:
    return SumCalculator([calc1, calc2])