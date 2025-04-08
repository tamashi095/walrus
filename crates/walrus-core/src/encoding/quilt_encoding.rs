// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{cmp, fmt, num::NonZeroU16};

use crc::{Crc, CRC_8_CDMA2000};
use enum_dispatch::enum_dispatch;
use hex;
use serde::{Deserialize, Serialize};
use tracing::{Level, Span};

use super::{EncodingConfig, EncodingConfigEnum, Primary, Secondary, SliverData, SliverPair};
use crate::{
    encoding::{blob_encoding::BlobEncoder, config::EncodingConfigTrait as _, QuiltError},
    metadata::{QuiltBlock, QuiltIndex, QuiltIndexV1, QuiltMetadata, QuiltVersion},
    SliverIndex,
};

/// The number of bytes to store the size of the quilt index.
const QUILT_INDEX_SIZE_PREFIX_SIZE: usize = 8;

/// The maximum number of columns a quilt index can have.
const MAX_NUM_COLUMNS_FOR_QUILT_INDEX: usize = 1024;

/// The configuration of a quilt.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum QuiltConfig {
    /// Version 1 of the quilt configuration.
    V1,
}

impl QuiltConfig {
    /// Returns a new [`QuiltConfig`] for the given version.
    pub fn new(version: QuiltVersion) -> Self {
        match version {
            QuiltVersion::V1 => QuiltConfig::V1,
        }
    }

    /// Returns a new [`QuiltEncoder`] for the given configuration.
    pub fn get_encoder<'a>(
        &self,
        encoding_config: EncodingConfigEnum<'a>,
        blobs: &'a [BlobWithIdentifier<'a>],
    ) -> QuiltEncoder<'a> {
        match self {
            QuiltConfig::V1 => QuiltEncoder::V1(QuiltEncoderV1::new(encoding_config, blobs)),
        }
    }

    /// Returns a new [`QuiltDecoderV1`] for the given configuration.
    pub fn get_decoder<'a>(&self, slivers: &'a [&'a SliverData<Secondary>]) -> QuiltDecoder<'a> {
        match self {
            QuiltConfig::V1 => QuiltDecoder::V1(QuiltDecoderV1::new(slivers)),
        }
    }

    /// Constructs a [`Quilt`] from a quilt blob.
    ///
    /// `quilt_blob` is a [`Quilt`] constructed from a set of blobs.
    /// This function loads the quilt blob to access its internal structures without having
    /// to re-encode the blobs.
    pub fn new_from_quilt_blob(
        &self,
        quilt_blob: Vec<u8>,
        metadata: &QuiltMetadata,
        n_shards: NonZeroU16,
    ) -> Result<Quilt, QuiltError> {
        assert_eq!(self, &QuiltConfig::V1);
        let encoding_config = EncodingConfig::new(n_shards);
        let config = encoding_config.get_for_type(metadata.metadata().encoding_type());

        let n_primary_source_symbols = config.n_primary_source_symbols().get();
        let n_secondary_source_symbols = config.n_secondary_source_symbols().get();
        let n_source_symbols = n_primary_source_symbols * n_secondary_source_symbols;

        // Verify data alignment.
        if quilt_blob.len() % n_source_symbols as usize != 0 {
            return Err(QuiltError::invalid_format_not_aligned());
        }

        // Calculate matrix dimensions.
        let row_size = quilt_blob.len() / n_primary_source_symbols as usize;
        let symbol_size = row_size / n_secondary_source_symbols as usize;

        // Extract quilt index size and calculate required columns.
        let quilt_index = utils::get_quilt_index_from_data(&quilt_blob, row_size, symbol_size)?;
        assert_eq!(quilt_index, metadata.index);

        Ok(Quilt::V1(QuiltV1 {
            data: quilt_blob,
            row_size,
            quilt_index,
            symbol_size,
        }))
    }
}

/// Apis for the quilt.
#[enum_dispatch]
pub trait QuiltApi {
    /// Get the blob by identifier.
    fn get_blob_by_identifier(&self, identifier: &str) -> Result<Vec<u8>, QuiltError>;
    /// Get the quilt index.
    fn quilt_index(&self) -> QuiltIndex;
    /// Get the data of the quilt.
    fn data(&self) -> &[u8];
    /// Get the symbol size of the quilt.
    fn symbol_size(&self) -> usize;
}

/// A quilt is a collection of encoded blobs stored together in a unified structure.
#[enum_dispatch(QuiltApi)]
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum Quilt {
    V1(QuiltV1),
}

impl Quilt {
    /// Returns the version of the quilt.
    pub fn version(&self) -> QuiltVersion {
        match self {
            Quilt::V1(_) => QuiltVersion::V1,
        }
    }
}

impl Default for Quilt {
    fn default() -> Self {
        Quilt::V1(QuiltV1::default())
    }
}

#[enum_dispatch]
pub trait QuiltEncoderApi {
    fn construct_quilt(&self) -> Result<Quilt, QuiltError>;
    fn encode(&self) -> Result<Vec<SliverPair>, QuiltError>;
    fn encode_with_metadata(&self) -> Result<(Vec<SliverPair>, QuiltMetadata), QuiltError>;
}

#[enum_dispatch(QuiltEncoderApi)]
pub enum QuiltEncoder<'a> {
    V1(QuiltEncoderV1<'a>),
}

/// A quilt is a collection of encoded blobs stored together in a unified structure.
///
/// The data is organized as a 2D matrix where:
/// - Each blob occupies a continuous range of columns (secondary slivers).
/// - The first column's initial `QUILT_INDEX_SIZE_PREFIX_SIZE` bytes contain the unencoded
///   length of the [`QuiltIndexV1`]. It is guaranteed the column size is more than
///   `QUILT_INDEX_SIZE_PREFIX_SIZE`.
/// - The [`QuiltIndexV1`] is stored in the first one or multiple columns.
/// - The blob layout is defined by the [`QuiltIndexV1`].
#[derive(Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuiltV1 {
    /// The data of the quilt.
    data: Vec<u8>,
    /// The size of each row in bytes.
    row_size: usize,
    /// The size of each symbol in bytes.
    symbol_size: usize,
    /// The internal structure of the quilt.
    quilt_index: QuiltIndexV1,
}

impl QuiltApi for QuiltV1 {
    /// Gets the blob by description.
    fn get_blob_by_identifier(&self, identifier: &str) -> Result<Vec<u8>, QuiltError> {
        self.quilt_index
            .get_quilt_block_by_identifier(identifier)
            .and_then(|quilt_block| self.get_blob(quilt_block))
    }

    /// Returns the quilt index.
    fn quilt_index(&self) -> QuiltIndex {
        QuiltIndex::V1(self.quilt_index.clone())
    }

    /// Returns the data of the quilt.
    fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the symbol size of the quilt.
    fn symbol_size(&self) -> usize {
        self.symbol_size
    }
}

impl QuiltV1 {
    /// Gets the blob represented by the given quilt block.
    fn get_blob(&self, quilt_block: &QuiltBlock) -> Result<Vec<u8>, QuiltError> {
        let start_col = quilt_block.start_index() as usize;
        let end_col = quilt_block.end_index() as usize;
        let mut blob = vec![0u8; quilt_block.unencoded_length() as usize];

        let mut written = 0;
        for col in start_col..end_col {
            for row in 0..(self.data.len() / self.row_size) {
                let remaining = blob.len() - written;
                if remaining == 0 {
                    break;
                }
                let chunk_size = cmp::min(self.symbol_size, remaining);
                let start_idx = row * self.row_size + col * self.symbol_size;
                let end_idx = start_idx + chunk_size;

                blob[written..written + chunk_size].copy_from_slice(&self.data[start_idx..end_idx]);
                written += chunk_size;
            }
        }
        Ok(blob)
    }
}

impl fmt::Debug for QuiltV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("QuiltV1");

        ds.field(
            "\ndata",
            &format_args!(
                "\n{:#?}",
                DebugMatrix {
                    data: &self.data,
                    row_size: self.row_size,
                    symbol_size: self.symbol_size
                }
            ),
        );

        ds.field(
            "quilt_index",
            &format_args!("\n{:#?}", DebugQuiltIndex(&self.quilt_index)),
        );

        ds.field("symbol_size", &self.symbol_size).finish()?;

        writeln!(f)
    }
}

struct DebugMatrix<'a> {
    data: &'a [u8],
    row_size: usize,
    symbol_size: usize,
}

impl fmt::Debug for DebugMatrix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (i, row) in self.data.chunks(self.row_size).enumerate() {
            let entries = row
                .chunks(self.symbol_size)
                .map(|chunk| format!("0x{}", hex::encode(chunk)))
                .collect::<Vec<_>>();
            list.entry(&DebugRow {
                index: i,
                entries: &entries,
            });
        }
        list.finish()?;
        writeln!(f)
    }
}

struct DebugRow<'a> {
    index: usize,
    entries: &'a [String],
}

impl fmt::Debug for DebugRow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex_width = self.entries.first().map_or(4, |e| e.len());
        let entries_per_line = 200 / (hex_width + 2); // +2 for ", " separator.

        write!(f, "\nRow {:0>2}:\n", self.index)?;
        for (i, entry) in self.entries.iter().enumerate() {
            if i % entries_per_line == 0 {
                if i > 0 {
                    writeln!(f)?;
                }
                write!(f, "    ")?;
            }

            write!(f, "{:width$}", entry, width = hex_width)?;

            if i < self.entries.len() - 1 {
                write!(f, ", ")?;
            }

            if i == 5 && self.entries.len() > QUILT_INDEX_SIZE_PREFIX_SIZE {
                write!(f, "... (+{} more)", self.entries.len() - i - 1)?;
                break;
            }
        }
        Ok(())
    }
}

struct DebugQuiltIndex<'a>(&'a QuiltIndexV1);

impl fmt::Debug for DebugQuiltIndex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for block in self.0.quilt_blocks.iter() {
            list.entry(&format_args!(
                "\nQuiltBlock {{\n    crc8: {:x?},\n    unencoded_length: {},\
                \n    end_index: {}\n    identifier: {:?}\n}}",
                block.crc8(),
                block.unencoded_length(),
                block.end_index(),
                block.identifier()
            ));
        }
        list.finish()?;
        writeln!(f)
    }
}

/// A wrapper around a blob and its identifier.
#[derive(Debug)]
pub struct BlobWithIdentifier<'a> {
    blob: &'a [u8],
    identifier: String,
}

impl<'a> BlobWithIdentifier<'a> {
    /// Creates a new `BlobWithIdentifier` from a blob and an identifier.
    pub fn new(blob: &'a [u8], identifier: impl Into<String>) -> Self {
        Self {
            blob,
            identifier: identifier.into(),
        }
    }

    /// Returns the length of the blob.
    pub fn len(&self) -> usize {
        self.blob.len()
    }
}

/// Encodes a set of blobs into a single quilt blob.
///
/// The blobs are first quilted into a 2D matrix, then the matrix is encoded into a single
/// quilt blob.
/// The quilt blob can be decoded into the original blobs by using the [`QuiltDecoderV1`].
#[derive(Debug)]
pub struct QuiltEncoderV1<'a> {
    /// The blobs to encode.
    blobs: &'a [BlobWithIdentifier<'a>],
    /// The encoding configuration.
    config: EncodingConfigEnum<'a>,
    /// A tracing span associated with this quilt encoder.
    span: Span,
}

impl QuiltEncoderApi for QuiltEncoderV1<'_> {
    /// Constructs a [`QuiltV1`].
    fn construct_quilt(&self) -> Result<Quilt, QuiltError> {
        let _guard = self.span.enter();

        let n_rows = self.config.n_source_symbols::<Primary>().get().into();
        let n_columns = self.config.n_source_symbols::<Secondary>().get().into();
        tracing::trace!(
            "Constructing quilt with n_columns: {}, n_rows: {}",
            n_columns,
            n_rows
        );

        // Compute crc8.
        let mut blobs_with_crc8 = Vec::new();
        for blob_with_identifier in self.blobs.iter() {
            blobs_with_crc8.push((blob_with_identifier, get_crc8(blob_with_identifier.blob)));
        }

        // Sort blobs based on their crc8.
        blobs_with_crc8
            .sort_by_key(|(blob_with_identifier, _)| blob_with_identifier.identifier.as_str());

        // Create initial QuiltBlocks.
        let quilt_blocks: Vec<QuiltBlock> = blobs_with_crc8
            .iter()
            .map(|(blob_with_identifier, crc8)| {
                QuiltBlock::new(
                    blob_with_identifier.blob.len() as u64,
                    blob_with_identifier.identifier.clone(),
                    *crc8,
                )
            })
            .collect();

        let mut quilt_index = QuiltIndexV1 {
            version: QuiltVersion::V1,
            quilt_blocks,
        };

        // Get the serialized quilt index size.
        let serialized_index_size = bcs::serialized_size(&quilt_index).map_err(|e| {
            QuiltError::quilt_index_der_ser_error(format!("failed to serialize quilt index: {}", e))
        })? as u64;

        // Calculate total size including the 8-byte size prefix.
        let index_total_size = QUILT_INDEX_SIZE_PREFIX_SIZE + serialized_index_size as usize;

        // Collect blob sizes for symbol size computation.
        let all_sizes: Vec<usize> = core::iter::once(index_total_size)
            .chain(blobs_with_crc8.iter().map(|(bwd, _)| bwd.blob.len()))
            .collect();

        let required_alignment = self.config.encoding_type().required_alignment() as usize;
        let symbol_size = compute_symbol_size(
            &all_sizes,
            n_columns,
            n_rows,
            MAX_NUM_COLUMNS_FOR_QUILT_INDEX,
            required_alignment,
        )?;

        let row_size = symbol_size * n_columns;
        let column_size = symbol_size * n_rows;
        let mut data = vec![0u8; row_size * n_rows];

        // Calculate columns needed for the index.
        let index_cols_needed = index_total_size.div_ceil(column_size);
        assert!(index_cols_needed <= MAX_NUM_COLUMNS_FOR_QUILT_INDEX);
        let mut current_col = index_cols_needed;

        // Adds a blob to the data as consecutive columns, starting at the given column.
        let mut add_blob_to_data = |blob: &[u8], current_col: usize| {
            let mut offset = 0;
            let mut row = 0;
            let mut col = current_col;
            while offset < blob.len() {
                let end = cmp::min(offset + symbol_size, blob.len());
                let chunk = &blob[offset..end];
                let dest_idx = row * row_size + col * symbol_size;
                data[dest_idx..dest_idx + chunk.len()].copy_from_slice(chunk);
                row = (row + 1) % n_rows;
                if row == 0 {
                    col += 1;
                }
                offset += chunk.len();
            }
        };

        // First pass: Fill data with actual blobs and populate quilt blocks.
        for (i, (blob_with_identifier, crc8)) in blobs_with_crc8.iter().enumerate() {
            let cols_needed = blob_with_identifier.blob.len().div_ceil(column_size);
            tracing::debug!(
                "Blob: {:?} needs {} columns, current_col: {}",
                blob_with_identifier.identifier,
                cols_needed,
                current_col
            );
            assert!(current_col + cols_needed <= n_columns);

            add_blob_to_data(blob_with_identifier.blob, current_col);

            assert_eq!(*crc8, quilt_index.quilt_blocks[i].crc8());
            quilt_index.quilt_blocks[i].set_start_index(current_col as u16);
            current_col += cols_needed;
            quilt_index.quilt_blocks[i].set_end_index(current_col as u16);
        }

        let mut final_index_data = Vec::with_capacity(index_total_size);
        let index_size_u64 = index_total_size as u64;
        final_index_data.extend_from_slice(&index_size_u64.to_le_bytes());
        final_index_data
            .extend_from_slice(&bcs::to_bytes(&quilt_index).expect("Serialization should succeed"));

        // Add the index data to the data.
        add_blob_to_data(&final_index_data, 0);

        Ok(Quilt::V1(QuiltV1 {
            data,
            row_size,
            quilt_index,
            symbol_size,
        }))
    }

    /// Encodes the blobs into a quilt and returns the slivers.
    fn encode(&self) -> Result<Vec<SliverPair>, QuiltError> {
        let _guard = self.span.enter();
        tracing::debug!("starting to encode quilt");

        let quilt = self.construct_quilt()?;
        if quilt.version() != QuiltVersion::V1 {
            return Err(QuiltError::InvalidQuiltVersion(
                QuiltVersion::V1,
                quilt.version(),
            ));
        }
        let encoder = BlobEncoder::new(self.config.clone(), quilt.data()).map_err(|_| {
            QuiltError::quilt_oversize(format!("quilt is too large: {}", quilt.data().len()))
        })?;
        assert_eq!(encoder.symbol_usize(), quilt.symbol_size());
        Ok(encoder.encode())
    }

    /// Encodes the blobs into a quilt and returns the slivers and metadata.
    fn encode_with_metadata(&self) -> Result<(Vec<SliverPair>, QuiltMetadata), QuiltError> {
        let _guard = self.span.enter();
        tracing::debug!("starting to encode quilt with metadata");

        let quilt = self.construct_quilt()?;
        assert_eq!(quilt.version(), QuiltVersion::V1);
        let Quilt::V1(quilt_v1) = quilt;
        let encoder =
            BlobEncoder::new(self.config.clone(), quilt_v1.data.as_slice()).map_err(|_| {
                QuiltError::quilt_oversize(format!("quilt is too large: {}", quilt_v1.data.len()))
            })?;

        assert_eq!(encoder.symbol_usize(), quilt_v1.symbol_size);

        let (sliver_pairs, metadata) = encoder.encode_with_metadata();
        let quilt_metadata = QuiltMetadata {
            quilt_id: *metadata.blob_id(),
            metadata: metadata.metadata().clone(),
            index: quilt_v1.quilt_index.clone(),
        };

        Ok((sliver_pairs, quilt_metadata))
    }
}

impl<'a> QuiltEncoderV1<'a> {
    /// Creates a new [`QuiltEncoderV1`] from a encoding config and a set of blobs.
    pub fn new(config: EncodingConfigEnum<'a>, blobs: &'a [BlobWithIdentifier<'a>]) -> Self {
        Self {
            blobs,
            config,
            span: tracing::span!(Level::ERROR, "QuiltEncoderV1"),
        }
    }
}

/// Finds the minimum symbol size needed to store blobs in a fixed number of columns.
/// Each blob must be stored in consecutive columns exclusively.
///
/// The first column is used to store the quilt index, so the symbol size has to be
/// large enough to store the quilt index in a single column.
///
/// # Arguments
/// * `blobs_sizes` - Slice of blob lengths.
/// * `nc` - Number of columns available.
/// * `nr` - Number of rows available.
///
/// # Returns
/// * `Result<usize, QuiltError>` - The minimum symbol size needed, or an error if impossible.
fn compute_symbol_size(
    blobs_sizes: &[usize],
    nc: usize,
    nr: usize,
    max_num_columns_for_quilt_index: usize,
    required_alignment: usize,
) -> Result<usize, QuiltError> {
    if blobs_sizes.len() > nc {
        // The first column is not user data.
        return Err(QuiltError::too_many_blobs(blobs_sizes.len(), nc - 1));
    }

    if blobs_sizes.is_empty() {
        return Err(QuiltError::other(
            "failed to compute symbol size: blobs are empty".to_string(),
        ));
    }

    let mut min_val = cmp::max(
        blobs_sizes
            .iter()
            .sum::<usize>()
            .div_ceil(nc)
            .div_ceil(nr * max_num_columns_for_quilt_index),
        blobs_sizes
            .first()
            .expect("blobs_sizes is not empty")
            .div_ceil(nr),
    );
    let mut max_val = blobs_sizes.iter().max().copied().unwrap_or(0).div_ceil(nr);
    min_val = cmp::max(min_val, QUILT_INDEX_SIZE_PREFIX_SIZE.div_ceil(nr));

    while min_val < max_val {
        let mid = (min_val + max_val) / 2;
        if can_blobs_fit_into_matrix(blobs_sizes, nc, mid * nr) {
            max_val = mid;
        } else {
            min_val = mid + 1;
        }
    }

    // Ensure the quilt index prefix can fit in the first column.
    min_val = cmp::max(min_val, QUILT_INDEX_SIZE_PREFIX_SIZE.div_ceil(nr));

    let symbol_size = min_val.div_ceil(required_alignment) * required_alignment;
    if symbol_size > u16::MAX as usize {
        return Err(QuiltError::quilt_oversize(format!(
            "the resulting symbol size {} is too large, remove some blobs",
            symbol_size
        )));
    }

    Ok(symbol_size)
}

/// Checks if the blobs can fit in the given number of columns.
///
/// # Arguments
/// * `blobs_sizes` - The sizes of the blobs.
/// * `nc` - The number of columns available.
/// * `length` - The size of the column.
///
/// # Returns
fn can_blobs_fit_into_matrix(blobs_sizes: &[usize], nc: usize, column_size: usize) -> bool {
    let mut used_cols = 0;
    for &blob in blobs_sizes {
        let cur = blob.div_ceil(column_size);
        if used_cols + cur > nc {
            return false;
        }
        used_cols += cur;
    }

    true
}

/// A trait for quilt decoders.
#[enum_dispatch]
pub trait QuiltDecoderApi<'a> {
    /// Decodes the quilt index from received slivers.
    ///
    /// The decoded [`QuiltIndexV1`] is stored in the decoder and can be retrieved
    /// using the [`QuiltDecoderV1::get_quilt_index`] method after the method returns.
    /// Returns
    fn decode_quilt_index(&mut self) -> Result<&QuiltIndexV1, QuiltError>;
    /// Get the quilt index.
    fn get_quilt_index(&self) -> Option<&QuiltIndexV1>;
    /// Get the blob by identifier.
    fn get_blob_by_identifier(&self, identifier: &str) -> Result<Vec<u8>, QuiltError>;
    /// Adds slivers to the decoder.
    fn add_slivers(&mut self, slivers: &'a [&'a SliverData<Secondary>]);
}

/// A quilt encoder.
#[enum_dispatch(QuiltDecoderApi)]
pub enum QuiltDecoder<'a> {
    V1(QuiltDecoderV1<'a>),
}

/// A quilt decoder of version V1.
#[derive(Debug)]
pub struct QuiltDecoderV1<'a> {
    slivers: Vec<&'a SliverData<Secondary>>,
    quilt_index: Option<QuiltIndexV1>,
}

impl<'a> QuiltDecoderApi<'a> for QuiltDecoderV1<'a> {
    fn decode_quilt_index(&mut self) -> Result<&QuiltIndexV1, QuiltError> {
        let index = SliverIndex(0);

        let first_sliver = self
            .slivers
            .iter()
            .find(|s| s.index == index)
            .ok_or_else(|| QuiltError::missing_sliver(index))?;

        let data_size = utils::get_quilt_index_data_size(first_sliver.symbols.data())?;

        // Calculate how many slivers we need based on the data size.
        let num_slivers_needed = (data_size as usize).div_ceil(first_sliver.symbols.data().len());
        let index_size = data_size as usize - QUILT_INDEX_SIZE_PREFIX_SIZE;
        let mut combined_data = Vec::with_capacity(index_size);

        // Add data from the first sliver (skipping the 8-byte size prefix).
        let bytes_to_take =
            index_size.min(first_sliver.symbols.data().len() - QUILT_INDEX_SIZE_PREFIX_SIZE);
        combined_data.extend_from_slice(
            &first_sliver.symbols.data()
                [QUILT_INDEX_SIZE_PREFIX_SIZE..QUILT_INDEX_SIZE_PREFIX_SIZE + bytes_to_take],
        );

        // Collect data from subsequent slivers if needed.
        for i in 1..num_slivers_needed {
            let next_index = SliverIndex(i as u16);
            let next_sliver = self
                .slivers
                .iter()
                .find(|s| s.index == next_index)
                .ok_or_else(|| {
                    utils::missing_sliver_error(
                        SliverIndex(i as u16),
                        SliverIndex(num_slivers_needed as u16),
                    )
                })?;

            let remaining_needed = index_size - combined_data.len();
            let sliver_data = next_sliver.symbols.data();
            let to_take = remaining_needed.min(sliver_data.len());
            combined_data.extend_from_slice(&sliver_data[..to_take]);
        }

        debug_assert_eq!(combined_data.len(), index_size);

        // Decode the QuiltIndexV1 from the collected data.
        self.quilt_index = Some(
            bcs::from_bytes(&combined_data)
                .map_err(|e| QuiltError::quilt_index_der_ser_error(e.to_string()))?,
        );

        // After successful deserialization, sort the blocks by end_index.
        if let Some(quilt_index) = &mut self.quilt_index {
            #[cfg(debug_assertions)]
            for i in 1..quilt_index.quilt_blocks.len() {
                assert!(
                    quilt_index.quilt_blocks[i].end_index()
                        >= quilt_index.quilt_blocks[i - 1].end_index()
                );
            }
            quilt_index.populate_start_indices(num_slivers_needed as u16);
        }

        Ok(self
            .quilt_index
            .as_ref()
            .expect("quilt index should be decoded"))
    }

    fn get_quilt_index(&self) -> Option<&QuiltIndexV1> {
        self.quilt_index.as_ref()
    }

    /// Get the blob by description.
    fn get_blob_by_identifier(&self, identifier: &str) -> Result<Vec<u8>, QuiltError> {
        self.quilt_index
            .as_ref()
            .ok_or(QuiltError::missing_quilt_index())
            .and_then(|quilt_index| quilt_index.get_quilt_block_by_identifier(identifier))
            .and_then(|quilt_block| self.get_blob_by_quilt_block(quilt_block))
    }

    /// Adds slivers to the decoder.
    fn add_slivers(&mut self, slivers: &'a [&'a SliverData<Secondary>]) {
        self.slivers.extend(slivers);
    }
}

impl<'a> QuiltDecoderV1<'a> {
    /// Get the start index of the first blob inside the quilt.
    pub fn get_start_index(first_sliver: &SliverData<Secondary>) -> Result<u16, QuiltError> {
        let data_size = utils::get_quilt_index_data_size(first_sliver.symbols.data())?;

        let sliver_size = first_sliver.symbols.data().len();
        let total_size_needed = data_size as usize;
        let num_slivers_needed = total_size_needed.div_ceil(sliver_size);
        Ok(num_slivers_needed as u16)
    }

    /// Creates a new QuiltDecoderV1 with the given slivers.
    pub fn new(slivers: &'a [&'a SliverData<Secondary>]) -> Self {
        Self {
            slivers: slivers.to_vec(),
            quilt_index: None,
        }
    }

    /// Creates a new QuiltDecoderV1 with the given slivers, and a quilt index.
    pub fn new_with_quilt_index(
        slivers: &'a [&'a SliverData<Secondary>],
        quilt_index: QuiltIndexV1,
    ) -> Self {
        Self {
            slivers: slivers.to_vec(),
            quilt_index: Some(quilt_index),
        }
    }

    /// Get the blob represented by the quilt block.
    fn get_blob_by_quilt_block(&self, quilt_block: &QuiltBlock) -> Result<Vec<u8>, QuiltError> {
        let start_idx = quilt_block.start_index() as usize;
        let end_idx = quilt_block.end_index() as usize;

        // Extract and reconstruct the blob.
        let mut blob = Vec::with_capacity(quilt_block.unencoded_length() as usize);

        // Collect data from the appropriate slivers.
        for i in start_idx..end_idx {
            let sliver_idx = SliverIndex(i as u16);
            if let Some(sliver) = self.slivers.iter().find(|s| s.index == sliver_idx) {
                let remaining_needed = quilt_block.unencoded_length() as usize - blob.len();
                blob.extend_from_slice(
                    &sliver.symbols.data()[..remaining_needed.min(sliver.symbols.data().len())],
                );
            } else {
                return Err(utils::missing_sliver_error(
                    sliver_idx,
                    SliverIndex(end_idx as u16),
                ));
            }
        }

        Ok(blob)
    }
}

mod utils {
    use super::*;

    /// Returns the missing sliver error.
    pub fn missing_sliver_error(begin: SliverIndex, end: SliverIndex) -> QuiltError {
        if begin == end {
            QuiltError::missing_sliver(begin)
        } else {
            QuiltError::missing_sliver_range(begin, end)
        }
    }

    /// Get the data size of the quilt index.
    pub fn get_quilt_index_data_size(combined_data: &[u8]) -> Result<usize, QuiltError> {
        if combined_data.len() < QUILT_INDEX_SIZE_PREFIX_SIZE {
            return Err(QuiltError::failed_to_extract_quilt_index_size());
        }

        let data_size = u64::from_le_bytes(
            combined_data[0..QUILT_INDEX_SIZE_PREFIX_SIZE]
                .try_into()
                .map_err(|_| QuiltError::failed_to_extract_quilt_index_size())?,
        ) as usize;

        Ok(data_size)
    }

    /// Get the quilt index from the data.
    pub fn get_quilt_index_from_data(
        data: &[u8],
        row_size: usize,
        symbol_size: usize,
    ) -> Result<QuiltIndexV1, QuiltError> {
        // Get the first column and extract the size prefix.
        let first_column = get_column(0, data, row_size, symbol_size)
            .map_err(|_| QuiltError::failed_to_extract_quilt_index_size())?;

        let data_size = utils::get_quilt_index_data_size(&first_column)?;
        let quilt_index_size = data_size - QUILT_INDEX_SIZE_PREFIX_SIZE;

        // Initialize our collection buffer with the first column's data (skipping size prefix).
        let mut collected_data = Vec::with_capacity(quilt_index_size);
        collected_data.extend_from_slice(&first_column[QUILT_INDEX_SIZE_PREFIX_SIZE..]);

        // Keep collecting data from subsequent columns until we have enough bytes.
        let mut current_column = 1;
        while collected_data.len() < quilt_index_size {
            let column_data = get_column(current_column, data, row_size, symbol_size)?;
            collected_data.extend_from_slice(&column_data);
            current_column += 1;
        }

        // Truncate to exact size needed
        collected_data.truncate(quilt_index_size);

        // Decode the QuiltIndexV1
        let mut quilt_index: QuiltIndexV1 = bcs::from_bytes(&collected_data)
            .map_err(|e| QuiltError::quilt_index_der_ser_error(e.to_string()))?;
        quilt_index.populate_start_indices(current_column as u16);
        Ok(quilt_index)
    }

    /// Gets the ith column of data, as if `data` is a 2D matrix.
    ///
    /// # Arguments
    /// * `i` - The column index.
    /// * `data` - The data to extract the column from.
    /// * `row_size` - The size of each row in bytes.
    /// * `symbol_size` - The size of each symbol in bytes.
    ///
    /// # Returns
    /// A vector containing the bytes from the ith column.
    fn get_column(
        i: usize,
        data: &[u8],
        row_size: usize,
        symbol_size: usize,
    ) -> Result<Vec<u8>, QuiltError> {
        // Verify inputs make sense.
        if row_size == 0
            || data.is_empty()
            || symbol_size == 0
            || row_size % symbol_size != 0
            || data.len() % row_size != 0
        {
            return Err(QuiltError::invalid_format_not_aligned());
        }

        let n_rows = data.len() / row_size;
        if i >= (row_size / symbol_size) {
            return Err(QuiltError::index_out_of_bounds(i, row_size / symbol_size));
        }

        let mut column = Vec::with_capacity(n_rows * symbol_size);

        for row in 0..n_rows {
            let start_idx = row * row_size + i * symbol_size;
            let end_idx = start_idx + symbol_size;

            // Check if we have enough data for this chunk.
            if end_idx > data.len() {
                break;
            }

            column.extend_from_slice(&data[start_idx..end_idx]);
        }

        Ok(column)
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber;
    use walrus_test_utils::param_test;

    use super::*;
    use crate::{
        encoding::{RaptorQEncodingConfig, ReedSolomonEncodingConfig},
        metadata::BlobMetadataApi as _,
    };

    /// Get the minimum required columns.
    fn min_required_columns(blobs: &[usize], length: usize) -> usize {
        if length == 0 {
            return usize::MAX;
        }
        let mut used_cols = 0;
        for &blob in blobs {
            used_cols += blob.div_ceil(length);
        }
        used_cols
    }

    param_test! {
        test_quilt_find_min_length: [
            case_1: (&[2, 1, 2, 1], 3, 3, 1, Err(QuiltError::too_many_blobs(4, 2))),
            case_2: (&[1000, 1, 1], 4, 7, 2, Ok(144)),
            case_3: (
                &[],
                3,
                1,
                1,
                Err(QuiltError::Other("failed to compute symbol size: blobs are empty".to_string())),
            ),
            case_4: (&[1], 3, 2, 1, Ok(4)),
            case_5: (&[115, 80, 4], 17, 9, 1, Ok(13)),
            case_6: (&[20, 20, 20], 3, 5, 1, Ok(4)),
            case_7: (&[5, 5, 5], 5, 1, 2, Ok(8)),
            case_8: (&[25, 35, 45], 200, 1, 2, Ok(26)),
            case_9: (&[10, 0, 0, 0], 17, 9, 1, Ok(2)),
            case_10: (&[10, 0, 0, 0], 17, 9, 2, Ok(2)),
        ]
    }
    fn test_quilt_find_min_length(
        blobs: &[usize],
        nc: usize,
        nr: usize,
        required_alignment: usize,
        expected: Result<usize, QuiltError>,
    ) {
        // Initialize tracing subscriber for this test
        let _guard = tracing_subscriber::fmt().try_init();
        let res = compute_symbol_size(
            blobs,
            nc,
            nr,
            MAX_NUM_COLUMNS_FOR_QUILT_INDEX,
            required_alignment,
        );
        assert_eq!(res, expected);
        if let Ok(min_size) = res {
            assert!(min_required_columns(blobs, min_size * nr) <= nc);
        }
    }

    param_test! {
        test_quilt_construct_quilt: [
            case_0: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_0_random_order: (
                &[
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_1: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_1_random_order: (
                &[
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_2: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 3][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[255u8; 1024][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                4, 8, 12
            ),
            case_3: (
                &[
                    BlobWithIdentifier {
                        blob: &[9, 8, 7, 6, 5, 4, 3, 2, 1][..],
                        identifier: "test blob 0".to_string(),
                    },
                ],
                3, 5, 7
            ),
        ]
    }
    fn test_quilt_construct_quilt(
        blobs_with_identifiers: &[BlobWithIdentifier<'_>],
        source_symbols_primary: u16,
        source_symbols_secondary: u16,
        n_shards: u16,
    ) {
        let raptorq_config = RaptorQEncodingConfig::new_for_test(
            source_symbols_primary,
            source_symbols_secondary,
            n_shards,
        );
        let reed_solomon_config = ReedSolomonEncodingConfig::new_for_test(
            source_symbols_primary,
            source_symbols_secondary,
            n_shards,
        );

        construct_quilt(
            blobs_with_identifiers,
            EncodingConfigEnum::RaptorQ(&raptorq_config),
        );
        construct_quilt(
            blobs_with_identifiers,
            EncodingConfigEnum::ReedSolomon(&reed_solomon_config),
        );
    }

    fn construct_quilt(
        blobs_with_identifiers: &[BlobWithIdentifier<'_>],
        config: EncodingConfigEnum,
    ) {
        let _guard = tracing_subscriber::fmt().try_init();

        let quilt_config = QuiltConfig::new(QuiltVersion::V1);
        let encoder = quilt_config.get_encoder(config, blobs_with_identifiers);

        let quilt = encoder.construct_quilt().expect("Should construct quilt");
        tracing::debug!("QuiltV1: {:?}", quilt);

        // Verify each blob and its description.
        for blob_with_identifier in blobs_with_identifiers {
            // Verify blob data matches.
            let extracted_blob = quilt
                .get_blob_by_identifier(blob_with_identifier.identifier.as_str())
                .expect("Block should exist for this blob identifier");
            let stored_crc8 = quilt
                .quilt_index()
                .get_quilt_block_by_identifier(blob_with_identifier.identifier.as_str())
                .expect("Block should exist for this blob identifier")
                .crc8();
            assert_eq!(
                extracted_blob, blob_with_identifier.blob,
                "Mismatch in encoded blob"
            );
            assert_eq!(stored_crc8, get_crc8(blob_with_identifier.blob));

            assert_eq!(
                quilt
                    .quilt_index()
                    .get_quilt_block_by_identifier(blob_with_identifier.identifier.as_str())
                    .expect("Block should exist for this blob ID")
                    .identifier(),
                &blob_with_identifier.identifier,
                "Mismatch in blob description"
            );
        }

        assert_eq!(quilt.quilt_index().len(), blobs_with_identifiers.len());
    }

    param_test! {
        test_quilt_encoder_and_decoder: [
            case_0: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_0_random_order: (
                &[
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_1: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_1_random_order: (
                &[
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_2: (
                &[
                    BlobWithIdentifier {
                        blob: &[1, 3][..],
                        identifier: "test blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[255u8; 1024][..],
                        identifier: "test blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[1, 2, 3][..],
                        identifier: "test blob 2".to_string(),
                    },
                ],
                5, 9, 13
            ),
            case_3: (
                &[
                    BlobWithIdentifier {
                        blob: &[9, 8, 7, 6, 5, 4, 3, 2, 1][..],
                        identifier: "test blob 0".to_string(),
                    },
                ],
                3, 5, 7
            ),
            case_4: (
                &[
                    BlobWithIdentifier {
                        blob: &[][..],
                        identifier: "empty blob 0".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[][..],
                        identifier: "empty blob 1".to_string(),
                    },
                    BlobWithIdentifier {
                        blob: &[][..],
                        identifier: "empty blob 2".to_string(),
                    }
                ],
                5, 9, 13
            ),
        ]
    }
    fn test_quilt_encoder_and_decoder(
        blobs_with_identifiers: &[BlobWithIdentifier<'_>],
        source_symbols_primary: u16,
        source_symbols_secondary: u16,
        n_shards: u16,
    ) {
        let raptorq_config = RaptorQEncodingConfig::new_for_test(
            source_symbols_primary,
            source_symbols_secondary,
            n_shards,
        );
        let reed_solomon_config = ReedSolomonEncodingConfig::new_for_test(
            source_symbols_primary,
            source_symbols_secondary,
            n_shards,
        );

        encode_decode_quilt(
            blobs_with_identifiers,
            EncodingConfigEnum::RaptorQ(&raptorq_config),
        );
        encode_decode_quilt(
            blobs_with_identifiers,
            EncodingConfigEnum::ReedSolomon(&reed_solomon_config),
        );
    }

    fn encode_decode_quilt(
        blobs_with_identifiers: &[BlobWithIdentifier<'_>],
        config: EncodingConfigEnum,
    ) {
        let _guard = tracing_subscriber::fmt().try_init();

        let quilt_config = QuiltConfig::new(QuiltVersion::V1);
        let encoder = quilt_config.get_encoder(config.clone(), blobs_with_identifiers);

        let (sliver_pairs, quilt_metadata) = encoder
            .encode_with_metadata()
            .expect("Should encode with quilt index and metadata");
        tracing::debug!(
            "Sliver pairs: {:?}\nQuilt metadata: {:?}",
            sliver_pairs,
            quilt_metadata
        );

        let slivers: Vec<&SliverData<Secondary>> = sliver_pairs
            .iter()
            .map(|sliver_pair| &sliver_pair.secondary)
            .collect();

        let mut quilt_decoder = quilt_config.get_decoder(&[]);
        assert!(matches!(
            quilt_decoder.decode_quilt_index(),
            Err(QuiltError::MissingSliver(_))
        ));

        quilt_decoder.add_slivers(&slivers);
        assert_eq!(
            quilt_decoder.decode_quilt_index(),
            Ok(&quilt_metadata.index)
        );

        for blob_with_identifier in blobs_with_identifiers {
            let blob =
                quilt_decoder.get_blob_by_identifier(blob_with_identifier.identifier.as_str());
            assert_eq!(blob, Ok(blob_with_identifier.blob.to_vec()));
        }

        let mut decoder = config
            .get_blob_decoder::<Secondary>(quilt_metadata.metadata().unencoded_length())
            .expect("Should create decoder");

        let (quilt_blob, metadata_with_id) = decoder
            .decode_and_verify(
                quilt_metadata.blob_id(),
                sliver_pairs
                    .iter()
                    .map(|s| s.secondary.clone())
                    .collect::<Vec<_>>(),
            )
            .expect("Should decode and verify quilt")
            .expect("Should decode quilt");

        assert_eq!(metadata_with_id.metadata(), quilt_metadata.metadata());

        let quilt = quilt_config
            .new_from_quilt_blob(quilt_blob, &quilt_metadata, config.n_shards())
            .expect("Should create quilt");
        assert_eq!(
            quilt.data(),
            encoder
                .construct_quilt()
                .expect("Should construct quilt")
                .data()
        );
    }
}

fn get_crc8(data: &[u8]) -> u8 {
    let crc = Crc::<u8>::new(&CRC_8_CDMA2000);
    let mut digest = crc.digest();
    digest.update(data);
    digest.finalize()
}
