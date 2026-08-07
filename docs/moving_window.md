# Moving window

A moving window replaces each pixel by the average — or, with `kind="sum"`, the total — of a box of
pixels around it. It is applied to the sample and the open beam once they have been combined and
immediately before they are divided, on the three VENUS TOF pipelines.

```python
from pathlib import Path

from neunorm.data_models.moving_window import MovingWindow
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline

transmission = run_venus_tpx1_pipeline(
    sample_hdf5_paths=sample_hdf5_paths,
    ob_hdf5_paths=ob_hdf5_paths,
    sample_tiff_paths=sample_tiff_paths,
    ob_tiff_paths=ob_tiff_paths,
    output_path=Path("transmission.hdf5"),
    moving_window=MovingWindow(x=3, y=3),
)
```

`MovingWindow` takes the window length per axis **by dimension name**, the kind, and the edge policy:

```python
from neunorm.data_models.moving_window import MovingWindow

MovingWindow(x=3, y=3)                          # 3x3 average
MovingWindow(x=3, y=3, kind="sum")              # 3x3 sum
MovingWindow(x=5, y=3)                          # 5 wide, 3 tall
MovingWindow(x=3, y=3, tof=3, dimension="3D")   # also averages 3 TOF bins
MovingWindow(x=3, y=3, mode="nearest")          # a different edge policy
```

Sizes are named rather than positional because the detectors do not agree on axis order: the event
path produces `(tof, x, y)` and the histogram path `(tof, y, x)`. A positional tuple copied between
them would transpose the two spatial axes, which for a non-square window is a wrong answer that still
looks plausible.

## What it costs

A window improves per-pixel precision and coarsens spatial resolution, at the same rate. Measured on
a 128x128 pixel-wise transmission map with a step edge at 60% transmission over 200 open-beam counts
per pixel:

| kernel | noise sigma | precision gain | 90-10 edge width | resolution cost |
|---|---|---|---|---|
| 1x1 | 0.1010 | 1.00x | 0.8 px | 1.00x |
| 3x3 | 0.0338 | 2.99x | 2.4 px | 3.00x |
| 5x5 | 0.0204 | 4.94x | 4.0 px | 5.01x |
| 7x7 | 0.0146 | 6.90x | 5.6 px | 7.01x |

For a `k x k` window the per-pixel noise falls by a factor `k` and the edge widens by the same factor
`k`. It is an even exchange, and it is the same exchange `rebin_by_spatial` makes.

The difference is what the result looks like afterwards. A `k x k` rebin returns an array `k` times
smaller on each axis, so the resolution it gave up is visible in the shape. A moving window returns an
array of the original shape: a 512x512 map still has 512x512 pixels, but carries roughly one
independent value per `k**2` of them. Nothing in the array says so, which is why the window is
recorded in the output file's metadata under `moving_window`.

### A feature smaller than the window

Below the window size, a feature does not merely blur — it loses depth. The fraction of the true
transmission depth recovered by a fit over the feature's own extent:

| kernel | 3-pixel feature | 9-pixel feature |
|---|---|---|
| 1x1 | 0.99 | 1.00 |
| 3x3 | 0.61 | 0.86 |
| 5x5 | 0.36 | 0.75 |
| 7x7 | 0.19 | 0.65 |

A 3-pixel feature under a 5x5 window keeps about a third of its depth, and under 7x7 about a fifth.
The feature is still clearly visible and will still be fitted; the depth that comes back is the
window's, not the sample's.

### Neighbouring pixels are no longer independent

The per-pixel variance NeuNorm reports after a window is correct for that pixel. What it cannot
express is that the pixel now shares most of its window with its neighbour:

| kernel | reported per-pixel sigma | lag-1 neighbour correlation |
|---|---|---|
| 1x1 | 14.131 | +0.000 |
| 3x3 | 4.713 | +0.671 |
| 5x5 | 2.828 | +0.805 |

A `k x k` window shares `k - 1` of its `k` columns with the window one pixel over, so the correlation
is `(k - 1) / k`. scipp carries no covariance, so this cannot be propagated.

That is why `moving_window` and `spectrum_roi` are refused together. A region reduction divides by an
`n` the pixels no longer have, and under-reports its uncertainty by roughly `sqrt(kernel pixels)` —
measured at x2.9 for a 3x3 window and x4.7 for 5x5, over a 32x32 region. The reported error bar
shrinks while the true spread of the estimator barely moves, so the two are alternative answers to
low counting statistics rather than stages of one workflow.

## Where the window sits, and what its sizes mean

The window runs after the `roi` crop and after `rebin_by_spatial`, so its sizes are in **post-crop,
post-rebin** pixels — the same frame `spectrum_roi` uses. A window of 3 on a stack rebinned by 2
therefore spans 6 detector pixels, and the two coarsenings compound. NeuNorm says so at run time when
both are requested.

## Dead and hot pixels

Masked pixels are excluded from the window, from both the value and the count of what was averaged.
This matters more than it sounds: a filter that simply averaged its neighbourhood would let one dead
pixel drag down every pixel within reach of it — `k**2` of them, reading 88.89 rather than 100.00 in a
uniform 100-count field. Here a masked pixel contributes nothing and the surrounding pixels keep their
true level. The masks themselves pass through unchanged: a dead pixel is still flagged afterwards,
whatever value the window computed for it.

## moving_sum and moving_average

Applied before normalization, the two are indistinguishable in the result. The window is applied to
both stacks, so a sum is `k` times the average on each side and the `k` cancels in the ratio; the
transmission and the relative uncertainty agree to float round-off, measured at 1.5e-8 relative on
float32 pipeline data. `kind="sum"` differs only in the intermediate counts, and is recorded in the
output metadata so a run is identifiable either way.

This holds only at this point in the pipeline. A sum applied *after* normalization would scale
transmission by `k`, which is not a transmission, so NeuNorm offers no after-normalization variant.

## Edge policy and even sizes

The frame edge is mirrored, as iBeatles does it (`mode="reflect"`). Mirroring and a
shrink-to-real-pixels edge differ only within `k // 2` of the boundary, so the choice is a convention
confined to the border — unless the window is large relative to the axis, which NeuNorm warns about,
having measured the axis as it is at the filter rather than as the detector shipped it. Every edge
mode `scipy.ndimage.uniform_filter` accepts is available.

Even window lengths are accepted, again as iBeatles accepts them. A window with no centre pixel leans
one way: the response shifts by exactly -0.50 px along that axis. That is inherent to an even window
rather than an error, so it is documented rather than refused.

## Scope

`moving_window` is available on `run_venus_tpx1_pipeline`, `run_venus_tpx3_histogram_pipeline` and
`run_venus_tpx3_event_pipeline`. The MARS pipelines do not split along the energy axis and so do not
have the per-bin counting-statistics problem this addresses, and `venus_ccd` has no TOF axis.

## Relation to `rebin_by_spatial`

| | `moving_window` | `rebin_by_spatial` |
|---|---|---|
| window | overlapping, one output per input pixel | non-overlapping blocks |
| output shape | unchanged | smaller by the factor on each axis |
| independent values | about one per `k**2` pixels | one per output pixel |
| resolution/precision | exchanged at rate `k` | exchanged at rate `k` |
| visible in the result | no | yes, in the shape |

They make the same trade. `rebin_by_spatial` leaves the array honest about how many independent
measurements it holds; `moving_window` keeps the original sampling and records the window in the
file's metadata instead. They compound if both are used.
