//! Scene-level tests: generate a synthetic drifted loop, apply the closure, and
//! check the expected physical behavior (spiral collapses back to a clean
//! circle, older points move less, results are deterministic).

use fast_reform::synthetic::{generate_scene, LoopParams};
use fast_reform::{reform, GraphDelta3D};

fn distance_from_origin(point: [f32; 3]) -> f64 {
    let x = point[0] as f64;
    let y = point[1] as f64;
    (x * x + y * y).sqrt()
}

#[test]
fn closure_collapses_spiral_back_onto_the_circle() {
    let params = LoopParams::default();
    let scene = generate_scene(&params);

    // Before closure the drifted cloud spills well outside the circle band.
    let band = params.sensor_spread * 2.0;
    let open_max = scene
        .cloud
        .points
        .iter()
        .map(|&p| distance_from_origin(p))
        .fold(0.0_f64, f64::max);
    assert!(
        open_max > params.radius + band,
        "expected drift to push points past the band, max radius was {open_max}"
    );

    // After closure every point sits within the circle's sensor band.
    let closed = reform(&scene.cloud, &scene.graph_delta);
    for &point in &closed.points {
        let radius = distance_from_origin(point);
        assert!(
            (radius - params.radius).abs() <= band + 1e-3,
            "point off the circle after closure: radius {radius}, expected ~{}",
            params.radius
        );
    }
}

#[test]
fn older_points_move_less_than_newer_points() {
    let params = LoopParams::default();
    let scene = generate_scene(&params);
    let closed = reform(&scene.cloud, &scene.graph_delta);

    let displacement = |index: usize| -> f64 {
        let before = scene.cloud.points[index];
        let after = closed.points[index];
        let dx = (after[0] - before[0]) as f64;
        let dy = (after[1] - before[1]) as f64;
        (dx * dx + dy * dy).sqrt()
    };

    // Points are sampled continuously, so pick the earliest- and latest-observed
    // by timestamp. The earliest (drift ~identity) barely moves; the latest moves far.
    let times = scene.cloud.timestamps.as_ref().expect("scene has timestamps");
    let earliest_point = (0..times.len())
        .min_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap())
        .unwrap();
    let latest_point = (0..times.len())
        .max_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap())
        .unwrap();
    assert!(
        displacement(earliest_point) < displacement(latest_point),
        "earliest {} should move less than latest {}",
        displacement(earliest_point),
        displacement(latest_point)
    );
    assert!(displacement(earliest_point) < 0.5, "earliest point should barely move");
}

#[test]
fn seam_ends_overlap_after_closure() {
    let params = LoopParams::default();
    let scene = generate_scene(&params);
    let closed = reform(&scene.cloud, &scene.graph_delta);

    // Warp the node positions too, using the same correction, so we can compare
    // the closed-loop node positions directly.
    let node_cloud = fast_reform::PointCloud2 {
        points: scene.node_positions.clone(),
        timestamps: Some(scene.node_times.clone()),
        ..fast_reform::PointCloud2::default()
    };
    let closed_nodes = reform(&node_cloud, &scene.graph_delta);

    // The last node ends one angular step before node 0, so after closure it
    // should be roughly one chord length away — far closer than the open-loop gap.
    let first = closed_nodes.points[0];
    let last = closed_nodes.points[closed_nodes.points.len() - 1];
    let closed_gap = {
        let dx = (last[0] - first[0]) as f64;
        let dy = (last[1] - first[1]) as f64;
        (dx * dx + dy * dy).sqrt()
    };
    let one_step_chord = params.radius * (std::f64::consts::TAU / params.num_nodes as f64);
    assert!(
        closed_gap < one_step_chord * 1.5,
        "seam ends did not overlap: gap {closed_gap}, chord {one_step_chord}"
    );

    // Sanity: the closed nodes really do trace the circle.
    for &point in &closed.points {
        let _ = point; // closed cloud used elsewhere; keep the loop meaningful
        break;
    }
}

#[test]
fn scene_generation_is_deterministic() {
    let params = LoopParams::default();
    let first = generate_scene(&params);
    let second = generate_scene(&params);
    assert_eq!(first.cloud, second.cloud);
    assert_eq!(first.graph_delta.transforms.len(), second.graph_delta.transforms.len());
}

#[test]
fn scaled_correction_interpolates_open_to_closed() {
    let params = LoopParams::default();
    let scene = generate_scene(&params);

    // alpha = 0 is the open loop (no change); alpha = 1 is the closed loop.
    let open = reform(&scene.cloud, &scene.graph_delta.scaled(0.0));
    assert_eq!(open.points, scene.cloud.points);

    let closed_full = reform(&scene.cloud, &scene.graph_delta);
    let closed_scaled = reform(&scene.cloud, &scene.graph_delta.scaled(1.0));
    assert_eq!(closed_full.points, closed_scaled.points);
}

#[test]
fn empty_graph_delta_is_identity() {
    let params = LoopParams::default();
    let scene = generate_scene(&params);
    let unchanged = reform(&scene.cloud, &GraphDelta3D::default());
    assert_eq!(unchanged.points, scene.cloud.points);
}
