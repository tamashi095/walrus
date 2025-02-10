// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// #![cfg(feature = "quilt")]
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0


//! Module for quilt encoding and decoding.
//!
//! This module provides functions to encode a set of blobs into a quilt by padding them to a
//! systematic size and concatenating them into a single blob.

/// Returns the next systematic size that can accommodate the given length.
///
/// The size is calculated based on the number of shards (`n_shards`). It uses an initial base
/// value computed from `n_shards` and then finds the smallest power of two multiplier such that
/// the product is at least `len`. If `len` is 0, returns 0.
fn next_systematic_size(n_shards: usize, len: usize) -> usize {
    // Base value is n_shards - (n_shards - 1) / 3.
    debug_assert!((n_shards - 1) % 3 == 0);
    let base = n_shards - (n_shards - 1) / 3;

    // If len is 0, return 0.
    if len == 0 {
        return 0;
    }

    // Find the smallest multiplier (power of 2) that gives at least len.
    let mut result = base;
    while result < len {
        result *= 2;
    }

    result
}

/// Constructs a quilt from a list of blob slices.
///
/// Each blob is padded to the next systematic size determined by the maximum blob length and
/// the number of shards (`n_shards`). The padded blobs are then concatenated into a single blob.
///
/// # Arguments
///
/// * `n_shards` - Number of shards used to calculate the systematic size.
/// * `blobs` - A slice of blob slices to be concatenated.
///
/// # Returns
///
/// A vector containing the concatenated quilt blob.
pub fn construct_quilt(n_shards: usize, blobs: &[&[u8]]) -> Vec<u8> {
    // Find the maximum length among all blobs.
    let max_len = blobs.iter().map(|blob| blob.len()).max().unwrap_or(0);

    // Get the next systematic size that can accommodate the max length.
    let target_len = next_systematic_size(n_shards, max_len);

    // Create a vector to store the concatenated quilt blob.
    let mut concatenated_blob = Vec::new();

    // Process each blob: pad it to the target length and append it.
    for blob in blobs {
        let mut padded = Vec::from(*blob);
        padded.resize(target_len, 0);  // Pad with zeros to target length.
        concatenated_blob.extend_from_slice(&padded);
    }

    concatenated_blob
}
