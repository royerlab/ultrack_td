use petgraph::algo::min_spanning_tree;
use petgraph::data::FromElements;
use petgraph::graph::Graph;
use petgraph::prelude::*;
use petgraph::unionfind::UnionFind;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashSet;

#[pyfunction]
pub fn hierarchical_segmentation(
    edges: Vec<(usize, usize, f32)>,
    num_nodes: usize,
    min_frontier: f32,
    min_size: usize,
    max_size: usize,
) -> PyResult<Vec<Vec<usize>>> {
    // Build the graph from edge data
    let mut graph = Graph::new_undirected();

    // Add nodes
    let node_indices: Vec<_> = (0..num_nodes).map(|_| graph.add_node(())).collect();

    // Add edges
    for (u, v, weight) in edges {
        if u < num_nodes && v < num_nodes {
            graph.add_edge(node_indices[u], node_indices[v], weight);
        }
    }

    let mst = Graph::from_elements(min_spanning_tree(&graph));
    let components = segment_mst(&mst, min_frontier, min_size, max_size);

    Ok(components)
}

fn segment_mst(
    mst: &Graph<(), f32, Undirected>,
    min_frontier: f32,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let n = mst.node_count();
    let mut uf = UnionFind::new(n);
    let mut comp_size: Vec<usize> = vec![1; n];

    let mut rng: StdRng = StdRng::seed_from_u64(42);

    let mut edges: Vec<(usize, usize, f32, f32)> = mst
        .edge_references()
        .map(|e| {
            (
                e.source().index(),
                e.target().index(),
                *e.weight(),
                rng.random_range(0.0..1.0),
            )
        })
        .collect();

    edges.sort_by(|a, b| {
        let res = a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal);
        if res == Ordering::Equal {
            // random tie breaker
            a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal)
        } else {
            res
        }
    });

    let mut leaves_in_comp: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let mut components = vec![];
    let mut seen = vec![false; n];

    // Pass 1: merge by min_frontier & max_size.
    for (u, v, w, _) in &edges {
        // if the weight is less than min_frontier, skip
        let ru = uf.find(*u);
        let rv = uf.find(*v);
        if ru == rv {
            continue;
        }
        if *w >= min_frontier {
            continue;
        }

        let new_size = comp_size[ru] + comp_size[rv];
        if new_size <= max_size {
            uf.union(ru, rv);
            let rnew = uf.find(ru);
            comp_size[rnew] = new_size;

            // move leaves from the non-root component into the root component without
            // cloning and without violating the borrow checker
            let (from, to) = if rnew == ru { (rv, ru) } else { (ru, rv) };
            let moved = std::mem::take(&mut leaves_in_comp[from]);
            leaves_in_comp[to].extend(moved);
        }
    }

    // Pass 2: ensure all segments >= min_size by merging small ones with nearest neighbours (by weight order).
    for (u, v, _, _) in &edges {
        let ru = uf.find(*u);
        let rv = uf.find(*v);
        if ru == rv {
            continue;
        }

        // If either component is smaller than min_size, merge them (even if weight > threshold).
        if comp_size[ru] < min_size || comp_size[rv] < min_size {
            let new_size = comp_size[ru] + comp_size[rv];
            uf.union(ru, rv);
            let rnew = uf.find(ru);
            comp_size[rnew] = new_size;

            let (from, to) = if rnew == ru { (rv, ru) } else { (ru, rv) };
            let moved = std::mem::take(&mut leaves_in_comp[from]);
            leaves_in_comp[to].extend(moved);

            if new_size >= min_size {
                components.push(leaves_in_comp[rnew].clone());
                seen[rnew] = true;
            }
        }
    }

    for (u, v, w, _) in &edges {
        let ru = uf.find(*u);
        let rv = uf.find(*v);
        if ru == rv {
            continue;
        }

        let new_size = comp_size[ru] + comp_size[rv];
        if new_size <= max_size {
            if !seen[ru] {
                components.push(leaves_in_comp[ru].clone());
                seen[ru] = true;
            }
            if !seen[rv] {
                components.push(leaves_in_comp[rv].clone());
                seen[rv] = true;
            }

            uf.union(ru, rv);
            let rnew = uf.find(ru);
            let (from, to) = if rnew == ru { (rv, ru) } else { (ru, rv) };

            comp_size[rnew] = new_size;

            // move, don’t copy
            let moved = std::mem::take(&mut leaves_in_comp[from]);
            leaves_in_comp[to].extend(moved);

            components.push(leaves_in_comp[to].clone());
        }
    }
    components
}
