# Native ELM provenance

The numerical code in src/elm/ was authored for Gail from the equations and
references below. It was not translated from another implementation. Earlier
repository research included third-party source inspection, so this work is not
described as a formal clean-room implementation and makes no patent-clearance
claim.

## Derivation

For training rows X in R^(N×d) and targets Y in R^(N×k), Gail fits the
normaliser on the training partition only, then creates a seeded fixed
projection H = phi(XW + b). The output weights solve:

    (Hᵀ S H + λI) B = Hᵀ S Y

where S is the diagonal matrix of non-negative sample weights and λ > 0. Gail
uses f64 accumulation, regularised Cholesky solves, a diagonal condition
estimate and a relative residual check. It forms no inverse. Because normal
equations square the design matrix condition number, Gail retries with larger
regularisation and rejects a fit when its finite-value, residual or condition
checks fail. The ridge comparator uses the same solver; the classification
comparator is a regularised multinomial logistic fit.

The projection generator is versioned as splitmix64-v1. The serialised
artefact contains the actual input weights, biases, readout weights, schema,
normaliser, calibration and solver diagnostics. Replaying a seed alone is not
treated as sufficient provenance.

Regression uses a split-conformal absolute-residual radius fitted on the
calibration partition. Classification uses a temperature fitted on that
partition and reports Brier score, log loss, accuracy and a calibration gap.
The final partition is not used to choose an estimator or fit calibration.
Paired improvements are bootstrapped by independent group, not by row.

## References

- G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, “Extreme learning machine: Theory
  and applications”, Neurocomputing 70 (2006), 489–501,
  doi:10.1016/j.neucom.2005.12.126.
- A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for
  nonorthogonal problems”, Technometrics 12 (1970), 55–67,
  doi:10.1080/00401706.1970.10488634.
- V. Vovk, A. Gammerman and G. Shafer, Algorithmic Learning in a Random
  World, Springer (2005), split-conformal prediction foundations.
- B. Efron, “Bootstrap methods: Another look at the jackknife”, The Annals
  of Statistics 7 (1979), 1–26, doi:10.1214/aos/1176344552.
- G. L. Steele, D. Lea and C. H. Flood, “Fast splittable pseudorandom number
  generators”, OOPSLA (2014), 453–472, doi:10.1145/2660193.2660195.

## Dependencies and authorship

The ELM implementation adds no numerical runtime dependency. It uses Gail's
existing rayon, serde, serde_json, sha2 and uuid dependencies. Their licences
remain governed by the repository lockfile and existing dependency review.
Gail-specific solver, fitting, calibration, manifest, registry, gate and
integration code is native Rust.

Online recursive updates, kernel models and deep ELM variants are not part of
this release.
