// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Errors that may be encountered while interacting with the core library.

use std::{
    fmt::Display,
    string::{String, ToString},
};

use thiserror::Error;

use super::{BlobId, SliverIndex};

/// Errors that may be encountered while interacting with quilt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuiltError {
    /// The blob is not found in the quilt.
    #[error("the blob is not found in the quilt: {0}")]
    BlobNotFoundInQuilt(String),
    /// The blob is not aligned with the quilt.
    #[error("the blob is not aligned with the quilt")]
    InvalidFormatNotAligned,
    /// Failed to extract the quilt index size.
    #[error("failed to extract the quilt index size")]
    FailedToExtractQuiltIndexSize,
    /// Failed to decode the quilt index.
    #[error("failed to decode the quilt index: {0}")]
    QuiltIndexDerSerError(String),
    /// Missing sliver.
    #[error("missing sliver: {0}")]
    MissingSliver(SliverIndex),
    /// Quilt index is missing.
    #[error("quilt index is missing")]
    MissingQuiltIndex,
    /// Too many blobs to fit in the quilt.
    #[error("too many blobs to fit in the quilt: {0} > max number of blobs: {1}")]
    TooManyBlobs(usize, usize),
    /// The quilt is too large.
    #[error("the quilt is too large: {0}")]
    TooLargeQuilt(usize),
    /// Index is out of bounds.
    #[error("index is out of bounds: {0} > max index: {1}")]
    IndexOutOfBounds(usize, usize),
}

impl QuiltError {
    /// The blob is not found in the quilt.
    pub fn blob_not_found_in_quilt<T: Display>(blob_identifier: &T) -> Self {
        Self::BlobNotFoundInQuilt(blob_identifier.to_string())
    }
    /// The blob is not aligned with the quilt.
    pub fn invalid_format_not_aligned() -> Self {
        Self::InvalidFormatNotAligned
    }
    /// Failed to extract the quilt index size.
    pub fn failed_to_extract_quilt_index_size() -> Self {
        Self::FailedToExtractQuiltIndexSize
    }
    /// Failed to extract the quilt index.
    pub fn quilt_index_der_ser_error(reason: String) -> Self {
        Self::QuiltIndexDerSerError(reason)
    }
    /// Quilt index is missing.
    pub fn missing_quilt_index() -> Self {
        Self::MissingQuiltIndex
    }
    /// Missing sliver.
    pub fn missing_sliver(sliver_index: SliverIndex) -> Self {
        Self::MissingSliver(sliver_index)
    }
    /// Too many blobs to fit in the quilt.
    pub fn too_many_blobs(num_blobs: usize, max_blobs: usize) -> Self {
        Self::TooManyBlobs(num_blobs, max_blobs)
    }
    /// The quilt is too large.
    pub fn too_large_quilt(quilt_size: usize) -> Self {
        Self::TooLargeQuilt(quilt_size)
    }
    /// Index is out of bounds.
    pub fn index_out_of_bounds(index: usize, max_index: usize) -> Self {
        Self::IndexOutOfBounds(index, max_index)
    }
}
