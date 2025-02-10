// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::client::Client;
use walrus_sui::client::SuiContractClient;

use super::error::QuiltError;

struct QuiltEncoder {
    n_shards: usize,
    client: Arc<Client<SuiContractClient>>,
}

impl QuiltEncoder {
    pub fn new(n_shards: usize, client: Arc<Client<SuiContractClient>>) -> Self {
        Self { n_shards, client }
    }

    pub fn encode(&self, blob: &[u8]) -> Result<Vec<u8>, QuiltError> {
        let mut quilt = vec![0; self.n_shards];
        for (i, shard) in quilt.iter_mut().enumerate() {
            shard.copy_from_slice(&blob[i * shard_size..(i + 1) * shard_size]);
        }
        Ok(quilt)
    }
}

/// Returns the next systematic size that can accommodate the given length.
///
/// The size is calculated based on the number of shards (`n_shards`). It uses an initial base
/// value computed from `n_shards` and then finds the smallest power of two multiplier such that
/// the product is at least `len`. If `len` is 0, returns 0.
pub(crate) fn next_systematic_size(n_shards: usize, len: usize) -> usize {
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
pub fn construct_quilt(n_shards: usize, blobs: &[&[u8]]) -> Result<Vec<u8>, QuiltError> {
    // Validate input parameters
    if n_shards == 0 {
        return Err(QuiltError::InvalidShardCount);
    }

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

    Ok(concatenated_blob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_systematic_size() {
        assert_eq!(next_systematic_size(13, 0), 0);
        assert_eq!(next_systematic_size(13, 100), 128);
        assert_eq!(next_systematic_size(13, 1000), 1024);
    }

    #[test]
    fn test_construct_quilt() {
        let blobs = vec![&[1, 2, 3][..], &[4, 5][..], &[6][..]];
        let result = construct_quilt(13, &blobs).unwrap();
        assert!(!result.is_empty());
    }
} 