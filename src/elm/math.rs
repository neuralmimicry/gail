//! Small, bounded dense numerical routines used by ELM and baseline fits.
//!
//! Ridge systems use regularised Cholesky solves, with diagonal regularisation
//! and an explicit residual check. No matrix inverse is formed.

use super::{ElmError, Result};

pub const MAX_MATRIX_CELLS: usize = 16_000_000;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SolverDiagnostics {
    pub solver: String,
    pub regularisation: f64,
    pub diagonal_condition_estimate: f64,
    pub relative_residual: f64,
    pub retries: usize,
}

pub fn validate_matrix(matrix: &[Vec<f64>], name: &str) -> Result<(usize, usize)> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(ElmError::InvalidInput(format!("{name} must not be empty")));
    }
    let cols = matrix[0].len();
    let cells = matrix
        .len()
        .checked_mul(cols)
        .ok_or_else(|| ElmError::InvalidInput(format!("{name} dimensions overflow")))?;
    if cells > MAX_MATRIX_CELLS || matrix.iter().any(|row| row.len() != cols) {
        return Err(ElmError::InvalidInput(format!(
            "{name} is ragged or exceeds the matrix cell limit"
        )));
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ElmError::InvalidInput(format!(
            "{name} contains non-finite values"
        )));
    }
    Ok((matrix.len(), cols))
}

/// Fit a weighted ridge readout against an already constructed design matrix.
#[allow(clippy::needless_range_loop)] // Matrix indices follow the documented row/column equations.
pub fn weighted_ridge(
    design: &[Vec<f64>],
    targets: &[Vec<f64>],
    sample_weights: Option<&[f64]>,
    lambda: f64,
) -> Result<(Vec<Vec<f64>>, SolverDiagnostics)> {
    let (rows, columns) = validate_matrix(design, "design matrix")?;
    let (target_rows, outputs) = validate_matrix(targets, "target matrix")?;
    if rows != target_rows || lambda <= 0.0 || !lambda.is_finite() {
        return Err(ElmError::InvalidInput(
            "ridge row counts must match and lambda must be finite and positive".into(),
        ));
    }
    if let Some(weights) = sample_weights
        && (weights.len() != rows
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || weights.iter().all(|weight| *weight == 0.0))
    {
        return Err(ElmError::InvalidInput(
            "sample weights must be finite, non-negative, correctly sized and not all zero".into(),
        ));
    }
    let cells = columns
        .checked_mul(columns)
        .and_then(|value| value.checked_add(columns.saturating_mul(outputs)))
        .ok_or_else(|| ElmError::InvalidInput("ridge workspace dimensions overflow".into()))?;
    if cells > MAX_MATRIX_CELLS {
        return Err(ElmError::InvalidInput(
            "ridge workspace exceeds limit".into(),
        ));
    }

    let mut gram = vec![vec![0.0; columns]; columns];
    let mut rhs = vec![vec![0.0; outputs]; columns];
    for row_index in 0..rows {
        let weight = sample_weights.map_or(1.0, |values| values[row_index]);
        if weight == 0.0 {
            continue;
        }
        for left in 0..columns {
            for right in 0..=left {
                gram[left][right] += weight * design[row_index][left] * design[row_index][right];
            }
            for output in 0..outputs {
                rhs[left][output] += weight * design[row_index][left] * targets[row_index][output];
            }
        }
    }
    for left in 0..columns {
        for right in 0..left {
            gram[right][left] = gram[left][right];
        }
    }

    let mut effective_lambda = lambda;
    let mut last_error = None;
    for retry in 0..4 {
        let mut regularised = gram.clone();
        for (index, row) in regularised.iter_mut().enumerate() {
            row[index] += effective_lambda;
        }
        match cholesky_solve(&regularised, &rhs) {
            Ok((solution, condition)) => {
                let residual = relative_residual(&regularised, &solution, &rhs);
                if residual.is_finite()
                    && residual <= 1e-7
                    && condition.is_finite()
                    && condition <= 1e12
                {
                    return Ok((
                        solution,
                        SolverDiagnostics {
                            solver: "regularised_cholesky_normal_equations".into(),
                            regularisation: effective_lambda,
                            diagonal_condition_estimate: condition,
                            relative_residual: residual,
                            retries: retry,
                        },
                    ));
                }
                last_error = Some(format!(
                    "unstable ridge solve: residual={residual:.3e}, condition_estimate={condition:.3e}"
                ));
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        effective_lambda *= 10.0;
        if !effective_lambda.is_finite() {
            break;
        }
    }
    Err(ElmError::Numerical(last_error.unwrap_or_else(|| {
        "ridge solve failed without a numerical result".into()
    })))
}

#[allow(clippy::needless_range_loop)] // Triangular substitution deliberately indexes matrix coordinates.
fn cholesky_solve(matrix: &[Vec<f64>], rhs: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64)> {
    let n = matrix.len();
    let outputs = rhs[0].len();
    let mut lower = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for index in 0..column {
                value -= lower[row][index] * lower[column][index];
            }
            if row == column {
                if !value.is_finite() || value <= f64::EPSILON {
                    return Err(ElmError::Numerical(
                        "regularised Gram matrix is not positive definite".into(),
                    ));
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }
    let minimum_diagonal = (0..n)
        .map(|index| lower[index][index])
        .fold(f64::INFINITY, f64::min);
    let maximum_diagonal = (0..n).map(|index| lower[index][index]).fold(0.0, f64::max);
    let condition_estimate = (maximum_diagonal / minimum_diagonal).powi(2);
    let mut forward = vec![vec![0.0; outputs]; n];
    for row in 0..n {
        for output in 0..outputs {
            let mut value = rhs[row][output];
            for column in 0..row {
                value -= lower[row][column] * forward[column][output];
            }
            forward[row][output] = value / lower[row][row];
        }
    }
    let mut solution = vec![vec![0.0; outputs]; n];
    for row in (0..n).rev() {
        for output in 0..outputs {
            let mut value = forward[row][output];
            for column in row + 1..n {
                value -= lower[column][row] * solution[column][output];
            }
            solution[row][output] = value / lower[row][row];
            if !solution[row][output].is_finite() {
                return Err(ElmError::Numerical("non-finite ridge coefficient".into()));
            }
        }
    }
    Ok((solution, condition_estimate))
}

fn relative_residual(matrix: &[Vec<f64>], solution: &[Vec<f64>], rhs: &[Vec<f64>]) -> f64 {
    let mut residual_sq = 0.0;
    let mut rhs_sq = 0.0;
    for row in 0..matrix.len() {
        for output in 0..rhs[0].len() {
            let predicted: f64 = matrix[row]
                .iter()
                .zip(solution)
                .map(|(coefficient, weights)| coefficient * weights[output])
                .sum();
            residual_sq += (predicted - rhs[row][output]).powi(2);
            rhs_sq += rhs[row][output].powi(2);
        }
    }
    residual_sq.sqrt() / rhs_sq.sqrt().max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_matches_known_one_feature_reference() {
        let x = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0]];
        let y = vec![vec![1.0], vec![3.0], vec![5.0]];
        let (weights, diagnostics) = weighted_ridge(&x, &y, None, 1e-10).expect("ridge fit");
        assert!((weights[0][0] - 1.0).abs() < 1e-8);
        assert!((weights[1][0] - 2.0).abs() < 1e-8);
        assert!(diagnostics.relative_residual < 1e-8);
    }

    #[test]
    fn malformed_and_negative_weight_inputs_are_rejected() {
        assert!(validate_matrix(&[vec![1.0], vec![]], "x").is_err());
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![vec![1.0], vec![2.0]];
        assert!(weighted_ridge(&x, &y, Some(&[1.0, -1.0]), 0.1).is_err());
        assert!(weighted_ridge(&x, &y, None, 0.0).is_err());
    }

    #[test]
    fn ill_conditioned_repeated_column_is_regularised_and_reported() {
        let x = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let y = vec![vec![1.0], vec![2.0], vec![3.0]];
        let (_, diagnostics) = weighted_ridge(&x, &y, None, 1e-8).expect("regularised fit");
        assert!(diagnostics.regularisation >= 1e-8);
        assert!(diagnostics.relative_residual < 1e-7);
    }
}
