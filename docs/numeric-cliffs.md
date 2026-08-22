# Numeric correctness cliffs

ANE programs use fp16, but the largest finite fp16 value is not a guarantee that
every operator can produce or carry every value up to that limit. Some hardware
paths have lower correctness cliffs: a mathematically finite result becomes
infinite, or an integer result is rounded. These are correctness limits rather
than performance limits.

The values below describe the untouched lowering (`opt=0`). They have been
measured on M2 Pro (A14 family) and M5 (A16 family); run the sweep on other chips
instead of assuming an unmeasured family behaves identically.

| Operation | Cliff | Observed silicon | Failure |
| --- | --- | --- | --- |
| `matmul` | output magnitude around 32752 | M2 Pro and M5 | finite output becomes `+/-inf` |
| width-offset slice | input magnitude above 4094 | pre-A16; absent on A16/M5 | sliced value becomes `+/-inf` |
| `reshape` -> `reduce_sum` | integer totals above 2048 | M2 Pro and M5 | integer result may round to the fp16 grid |

## Matmul output saturation

A matmul output becomes infinite near `fp16_max / 2`, approximately 32752,
even though fp16 can represent finite values up to 65504. The boundary depends
on the true output magnitude, not the contraction size: sweeps at several values
of K find the same limit.

This is an accumulator/output-path limit. It has been measured at the same point
on M2 Pro and M5, but other families should still be measured before treating it
as universal.

To avoid it, keep a conservative bound on each dot product below 32752. Scale
activations or weights before the matmul and undo the scale afterwards when the
following computation has enough headroom. Checking only that the inputs are
finite is not sufficient; the relevant quantity is the accumulated output.

See [issue #115](https://github.com/sbryngelson/ANEForge/issues/115) for the
controlled sweeps and cross-generation measurements.

## Width-offset slice saturation

On pre-A16 hardware, a slice with a nonzero last-axis begin offset can use a Q.4
fixed-point crop-DMA path with an implied x16 scale. Values with
`|value| > 4094` (`65504 / 16`) then clamp to `+/-inf`. The A16 path measured on
M5 is exact and does not have this cliff.

Avoid this path by keeping sliced values within `[-4094, 4094]`, using a zero
last-axis begin offset when possible, or moving the offset to another axis. A
graph intended for both older and newer chips should follow the pre-A16 limit.
The [cross-chip deployment guide](cross-chip.md#the-h13-slice-saturation-warning)
describes the corresponding compile-time warning.

## Integer reduction exactness

Fp16 represents every integer only through 2048. Above that point its spacing
grows to two, so odd integer totals are not representable. A
`reshape` -> `reduce_sum` sweep is exact through 2048, then a target sum of 2049
can read back as 2050.

This is a precision cliff, not overflow: the result remains finite but no longer
necessarily equals the integer sum bit for bit. It has been reproduced on M2 Pro
and M5.

Keep integer totals at or below 2048 when bit-exact equality matters. Otherwise,
compare with an fp16-appropriate tolerance or restructure the computation so
that exact integer decisions are made before a large reduction.

## Measure your chip

The committed sweep in
[`bench/numeric_cliffs.py`](https://github.com/sbryngelson/ANEForge/blob/main/bench/numeric_cliffs.py)
measures all three boundaries:

```bash
PYTHONPATH=. python3 bench/numeric_cliffs.py
```

It prints JSON and writes `bench/results/numeric_cliffs_results.json`. A missing
slice cliff is a meaningful result on modern silicon. Add your chip and output
to the [numeric-cliffs datapoint drive](https://github.com/sbryngelson/ANEForge/issues/164)
so the per-family map can be tightened as more generations are measured.
