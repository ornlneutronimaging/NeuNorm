# Resonance mode: a spectrum instead of images

The normal reduction divides sample by open beam pixel by pixel and gives you a stack of normalized
images. Resonance mode gives you a **1-D transmission spectrum**: you name a region of interest, and
each spectral bin becomes one data point — the sample's mean counts in that region divided by the open
beam's mean counts in the same region. The result is written as a three-column ASCII file, which is
what the resonance and Bragg-edge fitting tools read.

Pass `spectrum_roi` to any of the three VENUS TOF pipelines:

```python
from pathlib import Path

from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline

spectrum = run_venus_tpx1_pipeline(
    sample_hdf5_paths=sample_hdf5_paths,
    ob_hdf5_paths=ob_hdf5_paths,
    sample_tiff_paths=sample_tiff_paths,
    ob_tiff_paths=ob_tiff_paths,
    output_path=Path("spectrum.txt"),
    spectrum_roi=(10, 10, 26, 26),
)
print(spectrum.dims, spectrum.sizes)
```

`spectrum_roi` is a fourth ROI role, distinct from the three that already exist:

| argument | what it does to the region |
|---|---|
| `roi` | **crops** to it; everything downstream sees only that region |
| `background_roi` | takes its mean as a **flux proxy**, standing in for proton charge |
| `air_roi` | takes its mean as a **scale correction**, so open beam reads 1.0 there |
| `spectrum_roi` | takes its mean as **the measurement itself** |

## The file it writes

Comma-delimited, exactly one plain header line, one row per bin:

```text
bin_index,transmission,uncertainty
0,0.448760,0.010663
1,0.451203,0.010701
2,0.449881,0.010684
```

This is the `*_Spectra.txt` shape the ORNL imaging tools already exchange, so NeuNorm's own reader
parses it with no special casing:

```python
import numpy as np

data = np.loadtxt("spectrum.txt", skiprows=1, delimiter=",")
bin_index, transmission, uncertainty = data.T
```

- **`bin_index`** is each point's **first input frame index**. With no rebinning that is simply the
  file index, which is what the column is usually read as. Under a rebinning it stays traceable — see
  "Binning by frame index" below.
- **`transmission`** is dimensionless, at six decimal places.
- **`uncertainty`** is **1 sigma**, `sqrt(variance)` — the same quantity the HDF5 writer stores as
  `/uncertainty`.

An HDF5 file is written **alongside** it, at the same path with a `.hdf5` suffix. Three columns cannot
hold a time axis, the masks, or the provenance, and those are worth keeping: the HDF5 carries
`/transmission`, `/uncertainty`, the `N+1` `/tof` bin edges, the per-bin `/spectra_tof` mean time,
`/wavelength`, `/energy`, and `/metadata/spectrum_roi`. Ask for `output_path="spectrum.hdf5"` instead
and you get only the HDF5.

TIFF output is refused. A 1-D spectrum is not an image stack, and the TIFF writer would not object —
it would quietly produce a multi-page file of 1×1-pixel images that even reads back cleanly.

## Why the region is collapsed before the division

This is the whole point of the mode, and it is an order of operations rather than a new algorithm. The
region is reduced to one number per bin **first**, and the division happens **once**:

$$T_i = \frac{\langle S_i \rangle_{\text{ROI}}}{\langle O_i \rangle_{\text{ROI}}}
\qquad \text{not} \qquad
T_i = \Big\langle \frac{S_i}{O_i} \Big\rangle_{\text{ROI}}$$

The two are different quantities — the ratio of means is not the mean of ratios. On synthetic Poisson
counts over a 64-pixel region they sit about 1.2% apart. The per-pixel form has a second problem at
fine TOF binning: wherever an open-beam pixel recorded zero counts in a bin its ratio is undefined, so
averaging ratios spreads NaN through the spectrum, while summing counts first does not notice.

The mean is **mask-aware and pooled** — `sum(counts over the region) / count(unmasked pixels)` — so a
dead or hot pixel inside the region is excluded from the numerator and the denominator alike, and the
remaining pixels still give the region's true mean. Several regions can be pooled by passing a list;
where they overlap, each pixel is counted once.

Both sides are given the **union of both sides' masks** before the collapse. This matters more than it
looks: a region mean divides by its own count of unmasked pixels, so a pixel masked on the sample alone
leaves the numerator and the denominator averaging over different pixels — and the dead pixel goes on
inflating the open beam's mean. Measured on four pixels with one dead, under non-uniform flux: 0.400
from a ratio of sums, 0.533 from a ratio of means, 0.800 once the masks match, against a true 0.800.
Switching from sums to means is not by itself the fix.

## Binning by frame index

Frame-index binning works exactly as it does in image mode, and for the same reason — the stacks are
binned **before** the region is collapsed, so a spectrum run and an image run of the same data see the
same counts:

```python
from pathlib import Path

from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline

# every 2 frames into one point
spectrum = run_venus_tpx1_pipeline(
    sample_hdf5_paths=sample_hdf5_paths,
    ob_hdf5_paths=ob_hdf5_paths,
    sample_tiff_paths=sample_tiff_paths,
    ob_tiff_paths=ob_tiff_paths,
    output_path=Path("binned.txt"),
    spectrum_roi=(10, 10, 26, 26),
    rebin_by_tof=2,
)
print(spectrum.sizes)
```

`rebin_by_tof` takes the same arguments as in image mode: an integer factor, `True` for the
statistics-based recommendation, or an explicit `[[start, stop], ...]` list of half-open frame ranges.
`rebin_reduction` chooses how frames combine and keeps its usual defaults — a factor **sums**, a bin
list takes the **mean**.

The `bin_index` column follows the input, not the row number. With `rebin_by_tof=2` over six frames you
get rows indexed `0, 2, 4`. With a gapped list the gap shows:

```text
rebin_by_tof=[[0, 2], [4, 6]]   ->   bin_index 0 and 4; frames 2-3 have no row at all
```

Frames no range covers are dropped silently, as in image mode. A gapped axis is no longer a continuous
spectrum, and resonance fitting over it is not valid — prefer contiguous ranges when the output will be
fitted. The full bin-to-frame mapping is recorded in the HDF5 under
`/metadata/spectrum_bin_first_frame`.

Sum and mean binning **commute** with the region collapse, so binning first costs nothing. A median
reduction does not commute, and neither does any reduction when a per-frame `(tof, y, x)` mask is
present — binning consumes that mask and changes each bin's unmasked pixel count. Where the two orders
differ, the documented order above is what runs.

### Binning a stack that has no timing coordinate

The three TOF pipelines always attach a `tof` axis, so this does not arise in a pipeline run. It does
arise if you load a stack yourself and bin it directly: the rebinner rebuilds its output axis from the
input's bin edges, so a stack with no coordinate on the binned dimension is refused with
`ValueError: data must carry a 'N_image' coordinate to rebuild the rebinned axis`. It does not invent an
index axis for you.

Attach one and binning by pure file index works, with the frame index becoming the axis:

```python
import numpy as np
import scipp as sc

from neunorm.tof.histogram_rebinner import reduce_tof_bins

n_frames = 6
values = np.arange(1, n_frames * 4 + 1, dtype=float).reshape(n_frames, 2, 2)
stack = sc.DataArray(
    sc.array(dims=["N_image", "y", "x"], values=values, variances=values.copy(), unit="counts"),
    coords={"N_image": sc.arange("N_image", 0, n_frames + 1, unit=None)},
)

binned = reduce_tof_bins(stack, [(0, 2), (2, 4), (4, 6)], reduction="sum", tof_dim="N_image")
print(binned.coords["N_image"].values)  # [0 2 4 6] -- bin boundaries as file indices
print(binned.coords["spectra_tof"].values)  # [0.5 2.5 4.5] -- each bin's mean member index
```

## Which pixel frame `spectrum_roi` is resolved in

The crop and the spatial rebin both run **before** the region is collapsed, so `spectrum_roi` indices
are resolved against the array as it is at that point — not against detector pixels. With
`roi=(4, 4, 60, 60)` the region's origin is that crop's corner; with `rebin_by_spatial=2` one index
step is two detector pixels. `background_roi` carries the same caveat.

A run that combines them says so in the log, because the numbers look entirely plausible either way:

```text
WARNING  spectrum_roi indices are resolved AFTER the roi=(4, 4, 60, 60) crop, so they are offsets
         into the cropped image, not detector pixels.
```

## Zero open beam

An open-beam region mean of zero would make the transmission infinite. By default that raises. Pass
`spectrum_roi_strict=False` to let `inf`/`nan` through instead, which is the 1.x behaviour and is there
for reproducing legacy output:

```python
from pathlib import Path

from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline

spectrum = run_venus_tpx1_pipeline(
    sample_hdf5_paths=sample_hdf5_paths,
    ob_hdf5_paths=ob_hdf5_paths,
    sample_tiff_paths=sample_tiff_paths,
    ob_tiff_paths=ob_tiff_paths,
    output_path=Path("legacy.txt"),
    spectrum_roi=(10, 10, 26, 26),
    spectrum_roi_strict=False,
)
```

The guard applies to the **open beam only**. A zero *sample* mean is a real measurement — a fully
absorbing bin, a black resonance — and gives transmission 0.0, never an error.

## Using it without a pipeline

The reduction is a public function, if you already have two stacks in memory:

```python
import numpy as np
import scipp as sc

from neunorm.processing.spectrum_reducer import normalize_roi_spectrum, roi_mean_spectrum

edges = sc.array(dims=["tof"], values=np.arange(4, dtype=float) * 10.0, unit="us")


def stack(scale):
    values = np.full((3, 8, 8), scale)
    return sc.DataArray(
        sc.array(dims=["tof", "y", "x"], values=values, variances=values.copy(), unit="counts"),
        coords={"tof": edges},
    )


spectrum = normalize_roi_spectrum(stack(50.0), stack(100.0), (2, 2, 6, 6))
print(spectrum.values)  # one point per TOF bin

# or just the region mean of one stack
mean_counts = roi_mean_spectrum(stack(50.0), (2, 2, 6, 6))
print(mean_counts.values)
```

`normalize_roi_spectrum` reuses the same normalization the image mode uses, so the proton-charge
correction and its 0.5% systematic come with it.

## Feeding the resonance detector

`detect_resonances` takes the same `spectrum_roi`, so peaks are looked for where the sample actually is
rather than across the whole detector:

```python
from neunorm.tof.resonance import detect_resonances

result = detect_resonances(hist_sample, hist_ob, spectrum_roi=(10, 10, 26, 26))
print(result["resonance_energies"])
```

Its integrated spectrum is now the same mask-aware pooled region mean, with the masks symmetrized, and
its SNR is computed from the propagated transmission variance rather than from a Poisson formula over
raw counts. For counting data the two agree to floating-point round-off, so detection on existing data
is unchanged; what changes is that a region mean and an asymmetric mask no longer bias it.

## What resonance mode does not do

- It does not fit anything. It produces the spectrum; `detect_resonances` finds peaks in one, and
  fitting lives downstream.
- It does not give you images as well. One run produces one output — ask for image mode if you want
  the stack.
- It is not available on the CCD pipelines or MARS TPX3. Those collapse or lack a spectral axis at
  normalization time, so there is no per-bin open beam to divide by; a "spectrum" over their image
  index would not be a TOF spectrum.
- It does not represent masks in the ASCII file. Three columns cannot, and inventing a fourth would
  break the format every downstream reader expects. The HDF5 written alongside carries them.
- It does not correct the shared-dark correlation. When a dark frame is subtracted from both sides
  before the collapse, the two region means are correlated through it, and the uncertainty reported
  here treats them as independent — so it is slightly **conservative**. `normalize_with_dark` has the
  analytic correction for the image mode.
