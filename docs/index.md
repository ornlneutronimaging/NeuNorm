# NeuNorm 2.0

Neutron imaging normalization and time-of-flight (TOF) data processing for ORNL
imaging facilities — MARS at HFIR and VENUS at SNS.

NeuNorm 2.0 is a complete, scipp-based rewrite of the library. It provides
end-to-end pipelines (TIFF/FITS/event loading, dark/gamma correction, flat-field
normalization, TOF binning, resonance and Bragg-edge analysis) with HDF5 as the
primary output format.

```{note}
NeuNorm 2.0 is a breaking change from the 1.x series. Code written against the
1.x `NeuNorm.normalization.Normalization` API will not run unchanged on 2.0.
Pin `NeuNorm<2` if you depend on the legacy API.
```

```{toctree}
:maxdepth: 2
:caption: Workflows

workflows/README
workflows/venus_ccd_cmos
workflows/venus_tpx1
workflows/venus_tpx3
workflows/mars_ccd_cmos
workflows/mars_tpx3
```

```{toctree}
:maxdepth: 1
:caption: Development

development/DEVELOPMENT_PLAN
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
