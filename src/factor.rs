//! Serial Gauss-Seidel factor fitting over a sparse observed-edge graph.
//!
//! Each row solves its quadratic loss on a Euclidean ball. A numerical row
//! update is accepted only when its measured row loss does not increase; a
//! complete sweep is likewise checked before it enters the objective history.

pub struct FitResult {
    pub loadings: Vec<f64>,
    pub history: Vec<f64>,
    pub converged: bool,
    pub reason: &'static str,
    pub rejected: usize,
    pub max_update: f64,
}

fn norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |total, value| total.hypot(*value))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn kkt_valid(gram: &[f64], linear: &[f64], result: &[f64], multiplier: f64) -> bool {
    let rank = linear.len();
    let mut residual = 0.0_f64;
    for j in 0..rank {
        residual = residual.hypot(
            dot(&gram[j * rank..(j + 1) * rank], result) + multiplier * result[j] - linear[j],
        );
    }
    let result_norm = norm(result);
    let scale =
        (norm(gram) * result_norm + norm(linear) + multiplier * result_norm).max(f64::MIN_POSITIVE);
    residual.is_finite() && residual <= 128.0 * f64::EPSILON * rank.max(1) as f64 * scale
}

/// Cheap unconstrained solution for a numerically positive-definite Gram.
/// Singular/poor pivots, boundary solutions and failed residual checks all
/// fall through to the constrained eigensolve; no ridge changes the objective.
fn cholesky_interior(gram: &[f64], linear: &[f64], radius: f64) -> Option<Vec<f64>> {
    let rank = linear.len();
    let scale = gram.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return None;
    }
    let mut lower = vec![0.0; rank * rank];
    let threshold = 64.0 * f64::EPSILON * rank.max(1) as f64;
    for i in 0..rank {
        for j in 0..=i {
            let mut entry = gram[i * rank + j] / scale;
            for k in 0..j {
                entry -= lower[i * rank + k] * lower[j * rank + k];
            }
            if i == j {
                if !entry.is_finite() || entry <= threshold {
                    return None;
                }
                lower[i * rank + j] = entry.sqrt();
            } else {
                lower[i * rank + j] = entry / lower[j * rank + j];
            }
        }
    }
    let mut result = vec![0.0; rank];
    for i in 0..rank {
        let mut entry = linear[i] / scale;
        for j in 0..i {
            entry -= lower[i * rank + j] * result[j];
        }
        result[i] = entry / lower[i * rank + i];
    }
    for i in (0..rank).rev() {
        let mut entry = result[i];
        for j in i + 1..rank {
            entry -= lower[j * rank + i] * result[j];
        }
        result[i] = entry / lower[i * rank + i];
    }
    if result.iter().any(|value| !value.is_finite())
        || norm(&result) > radius
        || !kkt_valid(gram, linear, &result, 0.0)
    {
        return None;
    }
    Some(result)
}

struct EigenResult {
    values: Vec<f64>,
    // Row-major orthogonal matrix; its columns are eigenvectors.
    vectors: Vec<f64>,
}

/// A scaled cyclic Jacobi solve for the small symmetric row Gram matrix.
///
/// Residual and orthogonality checks establish the numerical decomposition
/// contract independently of the iteration's stopping rule.
fn symmetric_eigen(gram: &[f64], rank: usize) -> Result<EigenResult, String> {
    let scale = gram.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let mut vectors = vec![0.0; rank * rank];
    for i in 0..rank {
        vectors[i * rank + i] = 1.0;
    }
    if scale == 0.0 {
        return Ok(EigenResult {
            values: vec![0.0; rank],
            vectors,
        });
    }
    let original: Vec<f64> = gram.iter().map(|value| value / scale).collect();
    let mut matrix = original.clone();
    let original_norm = norm(&original);
    let tolerance = f64::EPSILON * rank.max(1) as f64 * original_norm;
    let mut converged = false;
    for _ in 0..64 {
        let mut off_diagonal = 0.0_f64;
        for p in 0..rank {
            for q in p + 1..rank {
                off_diagonal = off_diagonal.hypot(matrix[p * rank + q]);
            }
        }
        if off_diagonal <= tolerance {
            converged = true;
            break;
        }
        for p in 0..rank {
            for q in p + 1..rank {
                let pp = matrix[p * rank + p];
                let qq = matrix[q * rank + q];
                let pq = matrix[p * rank + q];
                if pq.abs() <= f64::EPSILON * (pp.abs() + qq.abs()) {
                    continue;
                }
                let tau = (qq - pp) / (2.0 * pq);
                let tangent = (1.0 / (tau.abs() + tau.hypot(1.0))).copysign(tau);
                let cosine = 1.0 / tangent.hypot(1.0);
                let sine = tangent * cosine;
                matrix[p * rank + p] = pp - tangent * pq;
                matrix[q * rank + q] = qq + tangent * pq;
                matrix[p * rank + q] = 0.0;
                matrix[q * rank + p] = 0.0;
                for i in 0..rank {
                    if i != p && i != q {
                        let ip = matrix[i * rank + p];
                        let iq = matrix[i * rank + q];
                        let new_ip = cosine * ip - sine * iq;
                        let new_iq = sine * ip + cosine * iq;
                        matrix[i * rank + p] = new_ip;
                        matrix[p * rank + i] = new_ip;
                        matrix[i * rank + q] = new_iq;
                        matrix[q * rank + i] = new_iq;
                    }
                    let ip = vectors[i * rank + p];
                    let iq = vectors[i * rank + q];
                    vectors[i * rank + p] = cosine * ip - sine * iq;
                    vectors[i * rank + q] = sine * ip + cosine * iq;
                }
            }
        }
    }
    if !converged {
        return Err("row symmetric eigensolver did not converge in 64 sweeps".to_string());
    }
    let mut residual = 0.0_f64;
    let mut orthogonality = 0.0_f64;
    for i in 0..rank {
        for j in 0..rank {
            let product: f64 = (0..rank)
                .map(|k| original[i * rank + k] * vectors[k * rank + j])
                .sum();
            residual = residual.hypot(product - vectors[i * rank + j] * matrix[j * rank + j]);
            let product: f64 = (0..rank)
                .map(|k| vectors[k * rank + i] * vectors[k * rank + j])
                .sum();
            orthogonality = orthogonality.hypot(product - if i == j { 1.0 } else { 0.0 });
        }
    }
    let check_tolerance = 64.0 * f64::EPSILON * rank.max(1) as f64;
    if !residual.is_finite()
        || residual > check_tolerance * original_norm
        || !orthogonality.is_finite()
        || orthogonality > check_tolerance
    {
        return Err("row eigensolver residual or orthogonality check failed".to_string());
    }
    Ok(EigenResult {
        values: (0..rank).map(|i| matrix[i * rank + i] * scale).collect(),
        vectors,
    })
}

/// Solve min z'Gz - 2b'z subject to ||z|| <= radius, without a ridge.
fn quadratic_ball(gram: &[f64], linear: &[f64], radius: f64) -> Result<Vec<f64>, String> {
    let rank = linear.len();
    if radius == 0.0 || linear.iter().all(|value| *value == 0.0) {
        return Ok(vec![0.0; rank]);
    }
    if let Some(result) = cholesky_interior(gram, linear, radius) {
        return Ok(result);
    }
    let eigen = symmetric_eigen(gram, rank)?;
    let values: Vec<f64> = eigen.values.iter().map(|value| value.max(0.0)).collect();
    let mut coordinates = vec![0.0; rank];
    for (k, coordinate) in coordinates.iter_mut().enumerate() {
        *coordinate = (0..rank)
            .map(|j| eigen.vectors[j * rank + k] * linear[j])
            .sum();
    }
    let threshold =
        f64::EPSILON * rank.max(1) as f64 * values.iter().copied().fold(0.0_f64, f64::max);
    let mut free = vec![0.0; rank];
    let mut null_norm = 0.0_f64;
    for k in 0..rank {
        if values[k] > threshold {
            free[k] = coordinates[k] / values[k];
        } else {
            null_norm = null_norm.hypot(coordinates[k]);
        }
    }
    let linear_norm = norm(linear);
    let roundoff = 64.0 * f64::EPSILON * linear_norm.max(f64::MIN_POSITIVE);
    let mut multiplier = 0.0;
    if null_norm > roundoff || norm(&free) > radius {
        let mut lower = 0.0;
        let mut upper = (linear_norm / radius).max(f64::MIN_POSITIVE);
        for _ in 0..80 {
            let candidate_multiplier = 0.5 * (lower + upper);
            let candidate_norm = (0..rank).fold(0.0_f64, |total, k| {
                total.hypot(coordinates[k] / (values[k] + candidate_multiplier))
            });
            if candidate_norm > radius {
                lower = candidate_multiplier;
            } else {
                upper = candidate_multiplier;
            }
            if upper - lower <= 4.0 * f64::EPSILON * upper {
                break;
            }
        }
        for k in 0..rank {
            free[k] = coordinates[k] / (values[k] + upper);
        }
        multiplier = upper;
    }
    let mut result = vec![0.0; rank];
    for j in 0..rank {
        result[j] = (0..rank)
            .map(|k| eigen.vectors[j * rank + k] * free[k])
            .sum();
    }
    // Orthogonal transforms can add an outward last-bit norm error.
    let result_norm = norm(&result);
    if result_norm > radius {
        for value in &mut result {
            *value *= radius / result_norm;
        }
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Err("row trust-region solution is not finite".to_string());
    }
    if !kkt_valid(gram, linear, &result, multiplier) {
        return Err("row trust-region KKT residual check failed".to_string());
    }
    Ok(result)
}

fn objective(
    rows: &[usize],
    cols: &[usize],
    targets: &[f64],
    weights: &[f64],
    loadings: &[f64],
    rank: usize,
    total_weight: f64,
) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }
    (0..rows.len())
        .map(|edge| {
            let residual = dot(
                &loadings[rows[edge] * rank..(rows[edge] + 1) * rank],
                &loadings[cols[edge] * rank..(cols[edge] + 1) * rank],
            ) - targets[edge];
            (weights[edge] / total_weight) * residual * residual
        })
        .sum()
}

#[allow(clippy::too_many_arguments)]
pub fn optimize(
    rows: &[i64],
    cols: &[i64],
    targets: &[f64],
    weights: &[f64],
    initial: &[f64],
    n: usize,
    rank: usize,
    radius: f64,
    max_iter: usize,
    tol: f64,
) -> Result<FitResult, String> {
    let edge_count = rows.len();
    if n == 0 || rank >= n {
        return Err("n must be positive and rank must be less than n".to_string());
    }
    if cols.len() != edge_count || targets.len() != edge_count || weights.len() != edge_count {
        return Err("edge arrays must have equal lengths".to_string());
    }
    let parameter_count = n
        .checked_mul(rank)
        .ok_or_else(|| "factor dimensions overflow".to_string())?;
    if initial.len() != parameter_count || initial.iter().any(|value| !value.is_finite()) {
        return Err("initial must contain finite row-major n-by-rank loadings".to_string());
    }
    if !radius.is_finite() || !(0.0..=1.0).contains(&radius) || !tol.is_finite() || tol < 0.0 {
        return Err("radius must lie in [0,1] and tol must be finite and nonnegative".to_string());
    }
    let target_limit = f64::MAX.sqrt() / 4.0;
    if targets
        .iter()
        .any(|value| !value.is_finite() || value.abs() > target_limit)
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(
            "targets must be finite within squared-loss range and weights positive finite"
                .to_string(),
        );
    }
    let mut row_ids = Vec::with_capacity(edge_count);
    let mut col_ids = Vec::with_capacity(edge_count);
    let mut previous_pair = None;
    let mut offsets = vec![0_usize; n + 1];
    for (&row, &col) in rows.iter().zip(cols) {
        if row < 0 || col <= row || col as usize >= n {
            return Err("edges must have canonical indices 0 <= row < col < n".to_string());
        }
        if previous_pair.is_some_and(|pair| pair >= (row, col)) {
            return Err("edges must be sorted with unique unordered pairs".to_string());
        }
        previous_pair = Some((row, col));
        row_ids.push(row as usize);
        col_ids.push(col as usize);
        offsets[row as usize + 1] += 1;
        offsets[col as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let adjacency_len = edge_count
        .checked_mul(2)
        .ok_or_else(|| "edge count overflow".to_string())?;
    let mut neighbors = vec![0_usize; adjacency_len];
    let mut edge_ids = vec![0_usize; adjacency_len];
    let mut cursor = offsets[..n].to_vec();
    for edge in 0..edge_count {
        for (vertex, other) in [
            (row_ids[edge], col_ids[edge]),
            (col_ids[edge], row_ids[edge]),
        ] {
            neighbors[cursor[vertex]] = other;
            edge_ids[cursor[vertex]] = edge;
            cursor[vertex] += 1;
        }
    }
    let maximum_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    let weights: Vec<f64> = weights.iter().map(|value| value / maximum_weight).collect();
    if weights.iter().any(|value| *value == 0.0) {
        return Err("positive weights underflow on normalization".to_string());
    }
    let total_weight = weights.iter().sum();
    let mut loadings = initial.to_vec();
    for i in 0..n {
        let row = &mut loadings[i * rank..(i + 1) * rank];
        let row_norm = norm(row);
        if offsets[i] == offsets[i + 1] {
            row.fill(0.0);
        } else if row_norm > radius {
            for value in row {
                *value *= radius / row_norm;
            }
        }
    }
    let mut history = vec![objective(
        &row_ids,
        &col_ids,
        targets,
        &weights,
        &loadings,
        rank,
        total_weight,
    )];
    let mut converged = rank == 0 || edge_count == 0 || radius == 0.0;
    let mut reason = if converged {
        "no_factor_work"
    } else {
        "max_iter"
    };
    let mut rejected = 0;
    let mut max_update = 0.0_f64;
    let mut gram = vec![0.0; rank * rank];
    let mut linear = vec![0.0; rank];
    for _ in 0..if converged { 0 } else { max_iter } {
        let previous = loadings.clone();
        max_update = 0.0;
        for i in 0..n {
            let begin = offsets[i];
            let end = offsets[i + 1];
            if begin == end {
                continue;
            }
            gram.fill(0.0);
            linear.fill(0.0);
            let row_scale: f64 = edge_ids[begin..end].iter().map(|edge| weights[*edge]).sum();
            for adjacency in begin..end {
                let edge = edge_ids[adjacency];
                let other =
                    &loadings[neighbors[adjacency] * rank..(neighbors[adjacency] + 1) * rank];
                let weight = weights[edge] / row_scale;
                for j in 0..rank {
                    linear[j] += (weight * targets[edge]) * other[j];
                    for k in 0..=j {
                        gram[j * rank + k] += (weight * other[j]) * other[k];
                    }
                }
            }
            for j in 0..rank {
                for k in 0..j {
                    gram[k * rank + j] = gram[j * rank + k];
                }
            }
            let trial = quadratic_ball(&gram, &linear, radius)?;
            let current = &loadings[i * rank..(i + 1) * rank];
            let mut before = 0.0;
            let mut after = 0.0;
            for adjacency in begin..end {
                let edge = edge_ids[adjacency];
                let other =
                    &loadings[neighbors[adjacency] * rank..(neighbors[adjacency] + 1) * rank];
                let weight = weights[edge] / row_scale;
                let old_residual = dot(other, current) - targets[edge];
                let new_residual = dot(other, &trial) - targets[edge];
                before += weight * old_residual * old_residual;
                after += weight * new_residual * new_residual;
            }
            if after <= before {
                let update = (0..rank).fold(0.0_f64, |total, k| total.hypot(trial[k] - current[k]));
                max_update = max_update.max(update);
                loadings[i * rank..(i + 1) * rank].copy_from_slice(&trial);
            } else {
                rejected += 1;
            }
        }
        let loss = objective(
            &row_ids,
            &col_ids,
            targets,
            &weights,
            &loadings,
            rank,
            total_weight,
        );
        if !loss.is_finite() {
            return Err("factor objective is not finite".to_string());
        }
        if loss > *history.last().unwrap() {
            loadings = previous;
            reason = "floating_point_stall";
            converged = false;
            break;
        }
        let improvement = history.last().unwrap() - loss;
        history.push(loss);
        if improvement <= tol * history[0].max(1.0) {
            converged = true;
            reason = "objective_tolerance";
            break;
        }
    }
    Ok(FitResult {
        loadings,
        history,
        converged,
        reason,
        rejected,
        max_update,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_fast_path_accepts_only_certified_interior_solutions() {
        let gram = [2.0, 0.5, 0.5, 1.0];
        let linear = [0.2, 0.1];
        let result = cholesky_interior(&gram, &linear, 1.0).unwrap();
        assert!((result[0] - 0.08571428571428572).abs() < 1e-14);
        assert!((result[1] - 0.05714285714285714).abs() < 1e-14);
        assert!(cholesky_interior(&gram, &linear, 0.01).is_none());
        assert!(cholesky_interior(&[1.0, 0.0, 0.0, 0.0], &linear, 1.0).is_none());
        assert!(cholesky_interior(&[1.0, 0.0, 0.0, 1e-20], &linear, 1.0).is_none());
        let tiny_gram: Vec<f64> = gram.iter().map(|value| value * 1e-280).collect();
        let tiny_linear: Vec<f64> = linear.iter().map(|value| value * 1e-280).collect();
        let tiny_result = cholesky_interior(&tiny_gram, &tiny_linear, 1.0).unwrap();
        assert!(norm(&[tiny_result[0] - result[0], tiny_result[1] - result[1]]) < 1e-14);
    }

    #[test]
    fn trust_region_boundary_and_singular() {
        let result = quadratic_ball(&[1.0, 0.0, 0.0, 0.0], &[0.2, 0.0], 1.0).unwrap();
        assert!((result[0] - 0.2).abs() < 1e-14);
        assert!(result[1].abs() < 1e-14);
        let result = quadratic_ball(&[0.0; 4], &[1.0, 2.0], 0.7).unwrap();
        assert!((result[0] - 0.7 / 5.0_f64.sqrt()).abs() < 1e-14);
        assert!(norm(&result) <= 0.7 + 1e-15);
    }

    #[test]
    fn missing_pair_fit_is_monotone() {
        let result = optimize(
            &[0, 1],
            &[1, 2],
            &[0.25, 0.25],
            &[1.0, 1.0],
            &[0.5; 3],
            3,
            1,
            0.95_f64.sqrt(),
            20,
            1e-13,
        )
        .unwrap();
        assert!(result.history.windows(2).all(|pair| pair[1] <= pair[0]));
        assert!(*result.history.last().unwrap() < 1e-25);
        assert!(result.converged);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        assert!(optimize(&[0], &[1], &[0.2], &[1.0], &[], 2, 1, 0.9, 1, 1e-8).is_err());
        assert!(optimize(&[1], &[0], &[0.2], &[1.0], &[0.1; 2], 2, 1, 0.9, 1, 1e-8).is_err());
        assert!(optimize(&[0], &[1], &[0.2], &[0.0], &[0.1; 2], 2, 1, 0.9, 1, 1e-8).is_err());
    }
    #[test]
    fn conditioned_boundary_kkt_regression() {
        let gram = [
            0.009878003172403641,
            0.006889124626360955,
            0.00022978049474765039,
            -0.005041809951372721,
            -0.009041679420643477,
            -0.0005281165588462089,
            -0.006029498414952759,
            0.020212849602229865,
            0.006889124626360956,
            0.014985465107915904,
            -0.0030658272803145916,
            -0.025420883764505613,
            -0.00809564852969316,
            0.0017210794935629643,
            0.002881277896852003,
            0.007359989965781675,
            0.00022978049474765039,
            -0.0030658272803145916,
            0.0012266592979838355,
            0.006856909957447962,
            0.000653611328551985,
            -0.0005538334473238558,
            -0.0023840877449639477,
            0.003078407061983633,
            -0.0050418099513727214,
            -0.02542088376450561,
            0.006856909957447962,
            0.049890807507794274,
            0.008433708871212024,
            -0.004223852665319008,
            -0.012261996706837919,
            0.004224264511110683,
            -0.009041679420643479,
            -0.00809564852969316,
            0.0006536113285519851,
            0.008433708871212024,
            0.00917351263322326,
            0.00033212261489432455,
            0.004305637150191714,
            -0.016583481082333144,
            -0.0005281165588462089,
            0.0017210794935629643,
            -0.0005538334473238557,
            -0.004223852665319008,
            0.0003321226148943246,
            0.0005626369751678215,
            0.0017683538636218202,
            -0.0021273956446267004,
            -0.006029498414952759,
            0.0028812778968520035,
            -0.002384087744963948,
            -0.01226199670683792,
            0.004305637150191715,
            0.0017683538636218202,
            0.008686872526961462,
            -0.01705301812023111,
            0.020212849602229865,
            0.007359989965781673,
            0.003078407061983633,
            0.0042242645111106834,
            -0.01658348108233314,
            -0.0021273956446267004,
            -0.017053018120231106,
            0.047071330468388066,
        ];
        let linear = [
            -0.0025946056353538297,
            -0.0028765886684153047,
            -0.0001663681848256409,
            0.003473859462048171,
            0.0017821315656026213,
            -0.0004364698892382044,
            0.0009167485041861745,
            -0.0057425768568314375,
        ];
        let result = quadratic_ball(&gram, &linear, 0.95_f64.sqrt()).unwrap();
        let product: Vec<f64> = (0..8)
            .map(|i| dot(&gram[i * 8..(i + 1) * 8], &result))
            .collect();
        let lambda = (0..8)
            .map(|i| result[i] * (linear[i] - product[i]))
            .sum::<f64>()
            / dot(&result, &result);
        let residual: Vec<f64> = (0..8)
            .map(|i| product[i] + lambda * result[i] - linear[i])
            .collect();
        assert!(
            norm(&residual) / norm(&linear) < 1e-10,
            "KKT residual {}",
            norm(&residual) / norm(&linear)
        );
        for scale in [1e-280, 1e-100, 1e100] {
            let scaled_gram: Vec<f64> = gram.iter().map(|value| value * scale).collect();
            let scaled_linear: Vec<f64> = linear.iter().map(|value| value * scale).collect();
            let scaled_result =
                quadratic_ball(&scaled_gram, &scaled_linear, 0.95_f64.sqrt()).unwrap();
            let difference: Vec<f64> = scaled_result
                .iter()
                .zip(&result)
                .map(|(a, b)| a - b)
                .collect();
            assert!(
                norm(&difference) < 1e-10,
                "scale-dependent solution at {scale}"
            );
        }
    }
}
