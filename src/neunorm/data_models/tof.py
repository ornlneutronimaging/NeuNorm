"""
TOF-specific data models for NeuNorm 2.0.

Includes binning configurations for TOF/Energy/Wavelength spaces.
"""

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class BinningConfig(BaseModel):
    """
    Configuration for TOF/Energy/Wavelength binning in neutron imaging.

    Supports three binning domains:
    1. Energy (eV) - for resonance imaging (nuclear cross-sections)
    2. Wavelength (Å) - for Bragg edge imaging (crystallography)
    3. TOF (ns) - for raw time-of-flight binning

    Each domain can use linear or logarithmic spacing.

    Parameters
    ----------
    bins : int
        Number of bins (default: 5000)
    bin_space : Literal['tof', 'energy', 'wavelength']
        Binning domain (default: 'energy')
    use_log_bin : bool
        Use logarithmic spacing (default: True)
    energy_range : tuple[float, float] or None
        Energy range (E_min, E_max) in eV. Required for bin_space='energy'
    wavelength_range : tuple[float, float] or None
        Wavelength range (λ_min, λ_max) in Angstrom. Required for bin_space='wavelength'
    tof_range : tuple[float, float] or None
        TOF range (t_min, t_max) in nanoseconds. Optional for bin_space='tof'

    Examples
    --------
    >>> # Resonance imaging (energy space, logarithmic)
    >>> config = BinningConfig(bins=5000, bin_space='energy', energy_range=(1.0, 100.0))

    >>> # Bragg edge imaging (wavelength space, linear)
    >>> config = BinningConfig(
    ...     bins=3000,
    ...     bin_space='wavelength',
    ...     wavelength_range=(0.5, 3.0),
    ...     use_log_bin=False
    ... )

    >>> # Raw TOF binning (full range)
    >>> config = BinningConfig(bins=10000, bin_space='tof')
    """

    bins: int = Field(default=5000, gt=0, description="Number of bins")
    bin_space: Literal["tof", "energy", "wavelength"] = Field(
        default="energy", description="Binning domain: 'tof' (ns), 'energy' (eV), or 'wavelength' (Å)"
    )
    use_log_bin: bool = Field(default=True, description="Use logarithmic spacing (True) or linear spacing (False)")
    energy_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Energy range (E_min, E_max) in eV. Required for bin_space='energy'"
    )
    wavelength_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Wavelength range (λ_min, λ_max) in Angstrom. Required for bin_space='wavelength'"
    )
    tof_range: Optional[Tuple[float, float]] = Field(
        default=None, description="TOF range (t_min, t_max) in nanoseconds. Optional for bin_space='tof'"
    )

    @field_validator("energy_range")
    @classmethod
    def validate_energy_range_values(cls, v):
        """Validate energy range has correct format and values"""
        if v is not None:
            if len(v) != 2:
                raise ValueError("energy_range must be a tuple of (E_min, E_max)")
            emin, emax = v
            if emin <= 0:
                raise ValueError(f"E_min must be positive, got {emin}")
            if emin >= emax:
                raise ValueError(f"E_min ({emin}) must be less than E_max ({emax})")
        return v

    @field_validator("wavelength_range")
    @classmethod
    def validate_wavelength_range_values(cls, v):
        """Validate wavelength range has correct format and values"""
        if v is not None:
            if len(v) != 2:
                raise ValueError("wavelength_range must be a tuple of (wl_min, wl_max)")
            wl_min, wl_max = v
            if wl_min <= 0:
                raise ValueError(f"wl_min must be positive, got {wl_min}")
            if wl_min >= wl_max:
                raise ValueError(f"wl_min ({wl_min}) must be less than wl_max ({wl_max})")
        return v

    @field_validator("tof_range")
    @classmethod
    def validate_tof_range_values(cls, v):
        """Validate TOF range has correct format and values"""
        if v is not None:
            if len(v) != 2:
                raise ValueError("tof_range must be a tuple of (t_min, t_max)")
            t_min, t_max = v
            if t_min < 0:
                raise ValueError(f"t_min must be non-negative, got {t_min}")
            if t_min >= t_max:
                raise ValueError(f"t_min ({t_min}) must be less than t_max ({t_max})")
        return v

    @model_validator(mode="after")
    def validate_range_required(self):
        """Ensure required range is provided based on bin_space"""
        if self.bin_space == "energy" and self.energy_range is None:
            raise ValueError(
                "energy_range required when bin_space='energy'. Provide energy_range=(E_min, E_max) in eV."
            )
        if self.bin_space == "wavelength" and self.wavelength_range is None:
            raise ValueError(
                "wavelength_range required when bin_space='wavelength'. "
                "Provide wavelength_range=(wl_min, wl_max) in Angstrom."
            )
        # tof_range is optional (can use full detector range)
        return self
