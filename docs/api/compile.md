# Compile, optimize, estimate

Lowering a graph to one ANE program, the accuracy-preserving autotuner, and the
measurement-free cost model. All are reached from the top level (`af.compile`, `af.tune`,
`af.estimate`, ...).

::: aneforge
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - compile
        - Model
        - SegmentedModel
        - PrecisionWarning
        - CrossChipFP16Warning
        - DispatchFloorWarning
        - tune
        - tune_precision
        - estimate
        - estimate_provenance
        - project_peak
        - precision_risk
        - CompileBackoffError
        - reset_compile_breaker
