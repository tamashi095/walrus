// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{cmp, fmt, marker::PhantomData, num::NonZeroU16, slice::Chunks};

use fastcrypto::hash::Blake2b256;
use hex;
use serde::{Deserialize, Serialize};
use tracing::{Level, Span};

use super::{
    basic_encoding::{raptorq::RaptorQDecoder, reed_solomon::ReedSolomonDecoder, Decoder},
    utils,
    DataTooLargeError,
    DecodingSymbol,
    DecodingVerificationError,
    EncodingAxis,
    EncodingConfig,
    EncodingConfigEnum,
    Primary,
    Secondary,
    SliverData,
    SliverPair,
    Symbols,
};
use crate::{
    encoding::{
        blob_encoding::{BlobDecoderEnum, BlobEncoder},
        config::EncodingConfigTrait as _,
    },
    error::QuiltError,
    merkle::{leaf_hash, MerkleTree},
    metadata::{
        QuiltBlock,
        QuiltIndex,
        QuiltMetadata,
        QuiltMetadataWithIndex,
        SliverPairMetadata,
        VerifiedBlobMetadataWithId,
    },
    BlobId,
    SliverIndex,
    SliverPairIndex,
};

/// A quilt is a collection of encoded blobs stored together in a unified structure.
///
/// The data is organized as a 2D matrix where:
/// - Each blob occupies a continuous range of columns (secondary slivers).
/// - The first column's initial 8 bytes contain the unencoded length of the [`QuiltIndex`].
/// - The [`QuiltIndex`] is stored in the first columns(s).
/// - The blob layout is defined by the [`QuiltIndex`].
#[derive(Default, Serialize, Deserialize)]
pub struct Quilt {
    /// The data of the quilt.
    pub data: Vec<u8>,
    /// The size of each row in bytes.
    pub row_size: usize,
    /// The internal structure of the quilt.
    pub blocks: Vec<QuiltBlock>,
    /// The size of each symbol in bytes.
    pub symbol_size: usize,
}

impl Quilt {
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
    fn get_column(i: usize, data: &[u8], row_size: usize, symbol_size: usize) -> Vec<u8> {
        // Verify inputs make sense.
        if row_size == 0
            || data.len() == 0
            || symbol_size == 0
            || row_size % symbol_size != 0
            || data.len() % row_size != 0
        {
            return Vec::new();
        }

        let n_rows = data.len() / row_size;
        if i >= (row_size / symbol_size) {
            return Vec::new();
        }

        let mut column = Vec::with_capacity(n_rows * symbol_size);

        for row in 0..n_rows {
            let start_idx = row * row_size + i * symbol_size;
            let end_idx = start_idx + symbol_size;

            // Check if we have enough data for this chunk
            if end_idx > data.len() {
                break;
            }

            column.extend_from_slice(&data[start_idx..end_idx]);
        }

        column
    }

    /// Constructs a [`Quilt`] from a quilt blob.
    ///
    /// `quilt_blob` is a [`Quilt`] constructed from a set of blobs.
    /// This function loads the quilt blob to access its internal structures without having
    /// to re-encode the blobs.
    pub fn new_from_quilt_blob(
        quilt_blob: Vec<u8>,
        metadata: &QuiltMetadataWithIndex,
        n_shards: NonZeroU16,
    ) -> Result<Self, QuiltError> {
        let encoding_config = EncodingConfig::new(n_shards);
        let config = encoding_config.get_for_type(metadata.metadata().metadata().encoding_type());

        let n_primary_source_symbols = config.n_primary_source_symbols().get();
        let n_secondary_source_symbols = config.n_secondary_source_symbols().get();
        let n_source_symbols = n_primary_source_symbols * n_secondary_source_symbols;

        // Verify data alignment.
        if quilt_blob.len() % n_source_symbols as usize != 0 {
            return Err(QuiltError::invalid_format_not_aligned());
        }

        // Calculate matrix dimensions
        let row_size = quilt_blob.len() / n_primary_source_symbols as usize;
        let symbol_size = row_size / n_secondary_source_symbols as usize;

        // Extract quilt index size and calculate required columns.
        let data_size = u64::from_le_bytes(
            Self::get_column(0, &quilt_blob, row_size, symbol_size)[0..8]
                .try_into()
                .map_err(|_| QuiltError::failed_to_extract_quilt_index_size())?,
        ) as u16;

        // Construct quilt
        let mut blocks = vec![QuiltBlock {
            blob_id: BlobId::ZERO,
            unencoded_length: data_size as u64,
            end_index: (data_size as usize)
                .div_ceil(symbol_size * config.n_primary_source_symbols().get() as usize)
                as u16,
            desc: "".to_string(),
        }];
        blocks.extend(metadata.index.quilt_blocks.iter().cloned());

        Ok(Self {
            data: quilt_blob,
            row_size,
            blocks,
            symbol_size,
        })
    }

    /// Gets the ith blob from the quilt.
    pub fn get_blob(&self, index: usize) -> Option<Vec<u8>> {
        if index >= self.blocks.len() {
            return None;
        }
        let block = &self.blocks[index];
        let start_col = if index == 0 {
            0
        } else {
            self.blocks[index - 1].end_index as usize
        };
        let end_col = block.end_index as usize;
        let mut blob = vec![0u8; block.unencoded_length as usize];

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
        Some(blob)
    }

    /// Gets the blob with the given blob id.
    pub fn get_blob_by_id(&self, id: &BlobId) -> Option<Vec<u8>> {
        let index = self
            .blocks
            .iter()
            .enumerate()
            .find(|(_, block)| block.blob_id == *id)
            .map(|(i, _)| i);
        index.and_then(|i| self.get_blob(i))
    }
}

impl fmt::Debug for Quilt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("Quilt");

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
            "blocks",
            &format_args!("\n{:#?}", DebugBlocks(&self.blocks)),
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

            // Pad entries to fixed width for alignment
            write!(f, "{:width$}", entry, width = hex_width)?;

            if i < self.entries.len() - 1 {
                write!(f, ", ")?;
            }

            // Add truncation indicator.
            if i == 5 && self.entries.len() > 8 {
                write!(f, "... (+{} more)", self.entries.len() - i - 1)?;
                break;
            }
        }
        Ok(())
    }
}

struct DebugBlocks<'a>(&'a [QuiltBlock]);

impl fmt::Debug for DebugBlocks<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for block in self.0 {
            list.entry(&format_args!(
                "\nQuiltBlock {{\n    blob_id: {:x?},\n    unencoded_length: {},\
                \n    end_index: {}\n    desc: {:?}\n}}",
                block.blob_id, block.unencoded_length, block.end_index, block.desc
            ));
        }
        list.finish()?;
        writeln!(f)
    }
}

/// A wrapper around a blob and its description.
#[derive(Debug)]
pub struct BlobWithDesc<'a> {
    blob: &'a [u8],
    desc: &'a str,
}

impl<'a> BlobWithDesc<'a> {
    /// Creates a new `BlobWithDesc` from a blob and a description.
    pub fn new(blob: &'a [u8], desc: &'a str) -> Self {
        Self { blob, desc }
    }

    /// Creates a new `BlobWithDesc` from a blob.
    pub fn new_from_blob(blob: &'a [u8]) -> Self {
        Self { blob, desc: "" }
    }

    /// Returns the length of the blob.
    pub fn len(&self) -> usize {
        self.blob.len()
    }
}

/// Encodes a set of blobs into a single quilt blob.
#[derive(Debug)]
pub struct QuiltEncoder<'a> {
    /// The blobs to encode.
    blobs: &'a [BlobWithDesc<'a>],
    /// The encoding configuration.
    config: EncodingConfigEnum<'a>,
    /// A tracing span associated with this quilt encoder.
    span: Span,
}

impl<'a> QuiltEncoder<'a> {
    /// Creates a new `QuiltEncoder` from a encoding confi and a set of blobs.
    pub fn new(config: EncodingConfigEnum<'a>, blobs: &'a [BlobWithDesc<'a>]) -> Self {
        Self {
            blobs,
            config,
            span: tracing::span!(Level::ERROR, "QuiltEncoder"),
        }
    }

    /// Constructs a quilt from a set of blobs.
    pub fn construct_quilt(&self) -> Result<Quilt, QuiltError> {
        let _guard = self.span.enter();
        let n_rows = self.config.n_source_symbols::<Primary>().get().into();
        let n_columns = self.config.n_source_symbols::<Secondary>().get().into();
        tracing::debug!(
            "Constructing quilt blob with n_columns: {}, n_rows: {}",
            n_columns,
            n_rows
        );

        let mut blob_with_ids: Vec<_> = self
            .blobs
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(self.config.clone(), blob_with_desc.blob).unwrap();
                let metadata = encoder.compute_metadata();
                (blob_with_desc, *metadata.blob_id())
            })
            .collect();

        // Sort blobs based on their blob_ids.
        blob_with_ids.sort_by_key(|(_, id)| *id);

        let blob_sizes = blob_with_ids
            .iter()
            .map(|(b, _)| b.blob.len())
            .collect::<Vec<_>>();
        let symbol_size = compute_symbol_size(&blob_sizes, n_columns, n_rows)?;
        let required_alignment = self.config.encoding_type().required_alignment() as usize;
        let symbol_size = symbol_size.div_ceil(required_alignment) * required_alignment;

        let row_size = symbol_size * n_columns;
        let mut data = vec![0u8; row_size * n_rows];
        let mut quilt_blocks = Vec::new();
        let mut current_col = 0;

        // 4. Data filling section (preserved with blob access updates)
        for (blob_with_desc, blob_id) in &blob_with_ids {
            let mut cur = current_col;
            let cols_needed = blob_with_desc.blob.len().div_ceil(symbol_size * n_rows);
            tracing::debug!(
                "Blob: {:?} needs {} columns, current_col: {}",
                blob_id,
                cols_needed,
                cur
            );
            if cur + cols_needed > n_columns {
                return Err(QuiltError::too_many_blobs(
                    blob_with_desc.blob.len(),
                    n_columns,
                ));
            }

            // Copy blob data into columns (updated blob access)
            let mut row = 0;
            let mut offset = 0;
            while offset < blob_with_desc.blob.len() {
                let end = cmp::min(offset + symbol_size, blob_with_desc.blob.len());
                let chunk = &blob_with_desc.blob[offset..end];
                let dest_idx = row * row_size + cur * symbol_size;

                data[dest_idx..dest_idx + chunk.len()].copy_from_slice(chunk);

                row = (row + 1) % n_rows;
                if row == 0 {
                    cur += 1;
                }
                offset += symbol_size;
            }

            current_col += cols_needed;

            // Create QuiltBlock with description (new field)
            quilt_blocks.push(QuiltBlock {
                blob_id: *blob_id,
                unencoded_length: blob_with_desc.blob.len() as u64,
                end_index: current_col as u16,
                desc: blob_with_desc.desc.to_string(),
            });
        }

        Ok(Quilt {
            data,
            row_size,
            blocks: quilt_blocks,
            symbol_size,
        })
    }

    /// Constructs a quilt blob with a container quilt index.
    pub fn construct_container_quilt(&self) -> Result<Quilt, QuiltError> {
        let _guard = self.span.enter();
        let n_rows = self.config.n_source_symbols::<Primary>().get().into();
        let n_columns = self.config.n_source_symbols::<Secondary>().get().into();
        tracing::info!(
            "Constructing quilt blob with n_columns: {}, n_rows: {}",
            n_columns,
            n_rows
        );

        // 1. Compute blob_ids and create mapping
        let mut blob_with_ids: Vec<_> = self
            .blobs
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(self.config.clone(), blob_with_desc.blob).unwrap();
                let metadata = encoder.compute_metadata();
                (blob_with_desc, *metadata.blob_id())
            })
            .collect();

        // 2. Sort blobs based on their blob_ids
        blob_with_ids.sort_by_key(|(_, id)| *id);

        // Create initial QuiltBlocks with default end_index
        let quilt_blocks: Vec<QuiltBlock> = blob_with_ids
            .iter()
            .map(|(blob_with_desc, blob_id)| QuiltBlock {
                blob_id: *blob_id,
                unencoded_length: blob_with_desc.blob.len() as u64,
                end_index: Default::default(),
                desc: blob_with_desc.desc.to_string(),
            })
            .collect();

        // Create the container quilt index
        let mut container_quilt_index = QuiltIndex {
            quilt_blocks,
            start_index: 0,
        };

        // Get just the serialized size without actually serializing
        let serialized_index_size = bcs::serialized_size(&container_quilt_index)
            .expect("Size calculation should succeed") as u64;

        // Calculate total size including the 8-byte size prefix
        let final_index_data_size = 8 + serialized_index_size as usize;

        // Calculate blob sizes for symbol size computation
        let blob_sizes: Vec<usize> = blob_with_ids
            .iter()
            .map(|(bwd, _)| bwd.blob.len())
            .collect();
        let mut all_sizes = vec![final_index_data_size];
        all_sizes.extend(blob_sizes.clone());

        let symbol_size = compute_symbol_size(&all_sizes, n_columns, n_rows)
            .map_err(|_| QuiltError::too_large_quilt(all_sizes.len()))?;

        let required_alignment = self.config.encoding_type().required_alignment() as usize;
        let symbol_size = symbol_size.div_ceil(required_alignment) * required_alignment;

        let row_size = symbol_size * n_columns;
        let mut data = vec![0u8; row_size * n_rows];

        // Calculate columns needed for the index
        let index_cols_needed = final_index_data_size.div_ceil(symbol_size * n_rows);
        let mut current_col = index_cols_needed;

        // First pass: Fill data with actual blobs and collect quilt blocks
        for (i, (blob_with_desc, blob_id)) in blob_with_ids.iter().enumerate() {
            let mut cur = current_col;
            let cols_needed = blob_with_desc.blob.len().div_ceil(symbol_size * n_rows);
            tracing::info!(
                "Blob: {:?} needs {} columns, current_col: {}",
                blob_id,
                cols_needed,
                cur
            );
            if cur + cols_needed > n_columns {
                return Err(QuiltError::too_large_quilt(
                    self.blobs.iter().map(|b| b.blob.len()).sum(),
                ));
            }

            // Copy blob data into columns
            let mut row = 0;
            let mut offset = 0;
            while offset < blob_with_desc.blob.len() {
                let end = cmp::min(offset + symbol_size, blob_with_desc.blob.len());
                let chunk = &blob_with_desc.blob[offset..end];
                let dest_idx = row * row_size + cur * symbol_size;

                data[dest_idx..dest_idx + chunk.len()].copy_from_slice(chunk);

                row = (row + 1) % n_rows;
                if row == 0 {
                    cur += 1;
                }
                offset += symbol_size;
            }

            current_col += cols_needed;

            assert_eq!(blob_id, &container_quilt_index.quilt_blocks[i].blob_id);
            assert_eq!(blob_id, &container_quilt_index.quilt_blocks[i].blob_id);
            container_quilt_index.quilt_blocks[i].end_index = current_col as u16;
        }

        // Create the final index data with size prefix
        let mut final_index_data = Vec::new();
        final_index_data.extend_from_slice(&final_index_data_size.to_le_bytes());
        tracing::info!("final_index_data_size: {}", final_index_data_size);
        final_index_data.extend_from_slice(
            &bcs::to_bytes(&container_quilt_index).expect("Serialization should succeed"),
        );

        // Second pass: Fill the beginning of data with the index blob
        let mut row = 0;
        let mut offset = 0;
        let mut cur = 0;

        while offset < final_index_data_size {
            let end = cmp::min(offset + symbol_size, final_index_data_size);
            let chunk = &final_index_data[offset..end];
            let dest_idx = row * row_size + cur * symbol_size;

            data[dest_idx..dest_idx + chunk.len()].copy_from_slice(chunk);

            row = (row + 1) % n_rows;
            if row == 0 {
                cur += 1;
            }
            offset += symbol_size;
        }

        let mut quilt_blocks = Vec::new();
        quilt_blocks.push(QuiltBlock {
            blob_id: BlobId::ZERO,
            unencoded_length: final_index_data.len() as u64,
            end_index: index_cols_needed as u16,
            desc: "QuiltBlobIndex".to_string(),
        });
        quilt_blocks.extend(container_quilt_index.quilt_blocks);

        Ok(Quilt {
            data,
            row_size,
            blocks: quilt_blocks,
            symbol_size,
        })
    }

    pub fn encode(&self) -> Result<Vec<SliverPair>, DataTooLargeError> {
        let _guard = self.span.enter();
        let quilt = self
            .construct_quilt()
            .expect("should be able to construct quilt");

        let encoder = BlobEncoder::new(self.config.clone(), quilt.data.as_slice())?;
        assert_eq!(encoder.symbol_usize(), quilt.symbol_size);
        Ok(encoder.encode())
    }

    pub fn encode_with_quilt_index(&self) -> Result<Vec<SliverPair>, DataTooLargeError> {
        let _guard = self.span.enter();
        let quilt = self
            .construct_container_quilt()
            .expect("should be able to construct quilt");
        let encoder = BlobEncoder::new(self.config.clone(), quilt.data.as_slice())?;
        assert_eq!(encoder.symbol_usize(), quilt.symbol_size);
        Ok(encoder.encode())
    }

    pub fn encode_with_metadata(&self) -> (Vec<SliverPair>, QuiltMetadata) {
        let _guard = self.span.enter();
        tracing::debug!("starting to encode blob with metadata");
        let quilt = self
            .construct_quilt()
            .expect("should be able to construct quilt");
        let encoder = BlobEncoder::new(self.config.clone(), quilt.data.as_slice())
            .expect("should be able to create encoder");
        assert_eq!(encoder.symbol_usize(), quilt.symbol_size);
        let (sliver_pairs, metadata) = encoder.encode_with_metadata();
        let quilt_metadata = QuiltMetadata::new(
            metadata.blob_id().clone(),
            metadata.metadata().clone(),
            quilt.blocks,
        );
        (sliver_pairs, quilt_metadata)
    }

    /// Encodes the blobs into a quilt and returns the slivers and metadata.
    pub fn encode_with_quilt_index_and_metadata(
        &self,
    ) -> Result<(Vec<SliverPair>, QuiltMetadata), QuiltError> {
        let _guard = self.span.enter();
        tracing::debug!("starting to encode blob with metadata");
        let quilt = self.construct_container_quilt()?;
        let encoder = BlobEncoder::new(self.config.clone(), quilt.data.as_slice())
            .map_err(|_| QuiltError::too_large_quilt(quilt.data.len()))?;
        assert_eq!(encoder.symbol_usize(), quilt.symbol_size);
        let (sliver_pairs, metadata) = encoder.encode_with_metadata();
        let quilt_metadata = QuiltMetadata::new(
            metadata.blob_id().clone(),
            metadata.metadata().clone(),
            quilt.blocks,
        );
        Ok((sliver_pairs, quilt_metadata))
    }
}

/// Finds the minimum symbol size needed to store blobs in a fixed number of columns.
/// Each blob must be stored in consecutive columns exclusively.
///
/// # Arguments
/// * `blobs` - Slice of blob lengths.
/// * `nc` - Number of columns available.
/// * `base` - Base of the encoding, the column size must be a multiple of this.
///
/// # Returns
/// * `Result<usize, QuiltError>` - The minimum length needed, or an error if impossible.
fn compute_symbol_size(blobs_sizes: &[usize], nc: usize, nr: usize) -> Result<usize, QuiltError> {
    if blobs_sizes.len() > nc {
        return Err(QuiltError::too_many_blobs(blobs_sizes.len(), nc));
    }

    let mut min_val = blobs_sizes.iter().sum::<usize>().div_ceil(nc).div_ceil(nr);
    let mut max_val = blobs_sizes.iter().max().copied().unwrap_or(0).div_ceil(nr);

    while min_val < max_val {
        let mid = (min_val + max_val) / 2;
        if can_fit(blobs_sizes, nc, mid * nr) {
            max_val = mid;
        } else {
            min_val = mid + 1;
        }
    }
    min_val = cmp::max(min_val, (8 as usize).div_ceil(nr));
    Ok(min_val)
}

fn can_fit(blobs_sizes: &[usize], nc: usize, length: usize) -> bool {
    tracing::info!("Blobs: {:?}, nc: {}, length: {}", blobs_sizes, nc, length);
    let mut used_cols = 0;
    for &blob in blobs_sizes {
        let cur = (blob - 1) / length + 1;
        if used_cols + cur > nc {
            return false;
        }
        used_cols += cur;
    }
    tracing::info!("Can fit: {}, length: {}", used_cols, length);
    true
}

/// A decoder for a quilt.
#[derive(Debug)]
pub struct QuiltDecoder<'a> {
    n_shards: NonZeroU16,
    slivers: Vec<&'a SliverData<Secondary>>,
    quilt_index: Option<QuiltIndex>,
}

impl<'a> QuiltDecoder<'a> {
    /// Get the start index of the quilt index from the first sliver.
    pub fn get_start_index(first_sliver: &SliverData<Secondary>) -> Option<u16> {
        if first_sliver.symbols.data().len() < 8 {
            return None;
        }

        let data_size = u64::from_le_bytes(
            first_sliver.symbols.data()[0..8]
                .try_into()
                .expect("slice with incorrect length"),
        ) as u16;

        let sliver_size = first_sliver.symbols.data().len();
        let total_size_needed = data_size as usize; // 8 bytes for prefix + data
        let num_slivers_needed = total_size_needed.div_ceil(sliver_size); // Ceiling division
        Some(num_slivers_needed as u16)
    }

    /// Creates a new QuiltDecoder with the given number of shards and slivers.
    pub fn new(n_shards: NonZeroU16, slivers: &'a [&'a SliverData<Secondary>]) -> Self {
        Self {
            n_shards,
            slivers: slivers.to_vec(),
            quilt_index: None,
        }
    }

    /// Creates a new QuiltDecoder with the given number of shards and slivers, and a quilt index.
    pub fn new_with_quilt_index(
        n_shards: NonZeroU16,
        slivers: &'a [&'a SliverData<Secondary>],
        quilt_index: QuiltIndex,
    ) -> Self {
        Self {
            n_shards,
            slivers: slivers.to_vec(),
            quilt_index: Some(quilt_index),
        }
    }

    /// Decodes the quilt index.
    pub fn decode_quilt_index(&mut self) -> Result<&QuiltIndex, QuiltError> {
        let index = SliverIndex(0);

        let first_sliver = self
            .slivers
            .iter()
            .find(|s| s.index == index)
            .ok_or_else(|| QuiltError::missing_sliver(index))?;

        if first_sliver.symbols.data().len() < 8 {
            return Err(QuiltError::failed_to_extract_quilt_index_size());
        }

        // Read the first 8 bytes to get the data size.
        let data_size = u64::from_le_bytes(
            first_sliver.symbols.data()[0..8]
                .try_into()
                .map_err(|_| QuiltError::failed_to_extract_quilt_index_size())?,
        );

        tracing::info!("quilt index data_size: {}", data_size);

        // Calculate how many slivers we need based on the data size.
        let num_slivers_needed = (data_size as usize).div_ceil(first_sliver.symbols.data().len());
        let index_size = (data_size - 8) as usize;

        // Otherwise, we need to collect data from multiple slivers.
        let mut combined_data = Vec::with_capacity(index_size);

        // Add data from the first sliver (skipping the 8-byte size prefix).
        let bytes_to_take = index_size.min(first_sliver.symbols.data().len() - 8);
        combined_data.extend_from_slice(&first_sliver.symbols.data()[8..8 + bytes_to_take]);

        // Find and add data from subsequent slivers.
        for i in 1..num_slivers_needed {
            let next_index = SliverIndex(i as u16);
            let next_sliver = self
                .slivers
                .iter()
                .find(|s| s.index == next_index)
                .ok_or_else(|| QuiltError::missing_sliver(next_index))?;

            // Add data from this sliver.
            let remaining_needed = index_size - combined_data.len();
            let sliver_data = next_sliver.symbols.data();
            let to_take = remaining_needed.min(sliver_data.len());
            combined_data.extend_from_slice(&sliver_data[..to_take]);
        }

        debug_assert_eq!(combined_data.len(), index_size);

        // Decode the QuiltIndex from the collected data.
        self.quilt_index = bcs::from_bytes(&combined_data).ok();

        // After successful deserialization, sort the blocks by end_index.
        if let Some(quilt_index) = &mut self.quilt_index {
            quilt_index
                .quilt_blocks
                .sort_by_key(|block| block.end_index);
            quilt_index.start_index = num_slivers_needed as u16;
        }

        Ok(self
            .quilt_index
            .as_ref()
            .expect("quilt index should be decoded"))
    }

    /// Get the quilt index.
    pub fn get_quilt_index(&self) -> Option<&QuiltIndex> {
        self.quilt_index.as_ref()
    }

    /// Get the blob by id.
    pub fn get_blob_by_id(&self, id: &BlobId) -> Option<Vec<u8>> {
        self.get_blob_by_predicate(|block| &block.blob_id == id)
    }

    /// Get the blob by description.
    pub fn get_blob_by_desc(&self, desc: &str) -> Option<Vec<u8>> {
        self.get_blob_by_predicate(|block| &block.desc == desc)
    }

    /// Get the block by predicate.
    fn get_blob_by_predicate<F>(&self, predicate: F) -> Option<Vec<u8>>
    where
        F: Fn(&QuiltBlock) -> bool,
    {
        let quilt_index = self.get_quilt_index()?;

        // Find the block matching the predicate.
        let (block_idx, block) = quilt_index
            .quilt_blocks
            .iter()
            .enumerate()
            .find(|(_, block)| predicate(block))?;

        // Determine start index (0 for first block, previous block's end_index otherwise).
        let start_idx = if block_idx == 0 {
            quilt_index.start_index
        } else {
            quilt_index.quilt_blocks[block_idx - 1].end_index
        };

        let end_idx = block.end_index;

        // Extract and reconstruct the blob.
        let mut blob = Vec::with_capacity(block.unencoded_length as usize);

        // Collect data from the appropriate slivers.
        for i in start_idx..end_idx {
            let sliver_idx = SliverIndex(i as u16);
            if let Some(sliver) = self.slivers.iter().find(|s| s.index == sliver_idx) {
                let remaining_needed = block.unencoded_length as usize - blob.len();
                blob.extend_from_slice(
                    &sliver.symbols.data()[..remaining_needed.min(sliver.symbols.data().len())],
                );
            }
        }

        Some(blob)
    }

    /// Adds slivers to the decoder.
    pub fn add_slivers(mut self, slivers: &'a [&'a SliverData<Secondary>]) -> Self {
        self.slivers.extend(slivers);
        self
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber;
    use walrus_test_utils::{param_test, random_data, random_subset};

    use super::*;
    use crate::{
        encoding::{EncodingConfig, RaptorQEncodingConfig, ReedSolomonEncodingConfig},
        metadata::{BlobMetadataApi as _, UnverifiedBlobMetadataWithId},
        EncodingType,
    };

    /// Get the minimum required columns.
    fn min_required_columns(blobs: &[usize], length: usize) -> usize {
        if length == 0 {
            return usize::MAX;
        }
        let mut used_cols = 0;
        for &blob in blobs {
            used_cols += (blob - 1) / length + 1;
        }
        used_cols
    }

    /// Test the find minimum length.
    param_test! {
        test_find_min_length: [
            case_1: (&[2, 1, 2, 1], 3, 3, Err(QuiltError::too_many_blobs(4, 3))),
            case_2: (&[1000, 1, 1], 4, 7, Ok(72)),
            case_3: (&[], 3, 1, Ok(8)),
            case_4: (&[1], 3, 2, Ok(4)),
            case_5: (&[115, 80, 4], 17, 9, Ok(2)),
            case_6: (&[20, 20, 20], 3, 5, Ok(4)),
            case_7: (&[5, 5, 5], 5, 1, Ok(8)),
            case_8: (&[25, 35, 45], 200, 1, Ok(8))
        ]
    }
    fn test_find_min_length(
        blobs: &[usize],
        nc: usize,
        nr: usize,
        expected: Result<usize, QuiltError>,
    ) {
        // Initialize tracing subscriber for this test
        let _guard = tracing_subscriber::fmt().try_init();
        let res = compute_symbol_size(blobs, nc, nr);
        assert_eq!(res, expected);
        if let Ok(min_size) = res {
            assert!(min_required_columns(blobs, min_size * nr) <= nc);
        }
    }

    #[test]
    fn test_quilt_encoder_rows() {
        let _guard = tracing_subscriber::fmt().try_init();
        tracing::info!("Starting test_quilt_encoder_rows");

        // Create BlobWithDesc directly with static strings.
        let blobs_with_desc = &[
            BlobWithDesc {
                blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                desc: "test blob 0",
            },
            BlobWithDesc {
                blob: &[5, 68, 3, 2, 5][..],
                desc: "test blob 1",
            },
            BlobWithDesc {
                blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                desc: "test blob 2",
            },
        ];

        let config = RaptorQEncodingConfig::new_for_test(3, 5, 7);
        let config_enum = EncodingConfigEnum::RaptorQ(&config);
        let nr = config.source_symbols_primary.get() as usize;

        // Calculate blob IDs directly from blobs_with_desc.
        let blob_ids: Vec<BlobId> = blobs_with_desc
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(config_enum.clone(), blob_with_desc.blob)
                    .expect("Should create encoder");
                *encoder.compute_metadata().blob_id()
            })
            .collect();

        let encoder = QuiltEncoder::new(config_enum, blobs_with_desc);
        let quilt = encoder
            .construct_quilt()
            .expect("should be able to construct quilt");
        tracing::info!("Quilt: {:?}", quilt);

        // Verify each blob and its description.
        for (blob_with_desc, blob_id) in blobs_with_desc.iter().zip(blob_ids.iter()) {
            let retrieved_blob = quilt
                .get_blob_by_id(blob_id)
                .expect("Should find blob by ID");

            // Verify blob data matches
            assert_eq!(
                &retrieved_blob, blob_with_desc.blob,
                "Mismatch in encoded blob"
            );

            // Verify description matches
            let block = quilt
                .blocks
                .iter()
                .find(|block| block.blob_id == *blob_id)
                .expect("Block should exist for this blob ID");

            assert_eq!(
                block.desc, blob_with_desc.desc,
                "Mismatch in blob description"
            );
        }

        let (sliver_pairs, metadata) = encoder.encode_with_metadata();

        // Format primary slivers as a 2D matrix (rows)
        let mut primary_matrix = String::from("Primary Slivers Matrix (rows):\n");

        // Add column headers
        primary_matrix.push_str("     ");
        for col_idx in 0..sliver_pairs[0].primary.symbols.data().len() {
            primary_matrix.push_str(&format!("{:02} ", col_idx));
        }
        primary_matrix.push('\n');

        // Add separator line
        primary_matrix.push_str("    ");
        for _ in 0..sliver_pairs[0].primary.symbols.data().len() {
            primary_matrix.push_str("---");
        }
        primary_matrix.push('\n');

        // Add each row with row number
        for (row_idx, sliver_pair) in sliver_pairs.iter().enumerate() {
            let primary_data = sliver_pair.primary.symbols.data();

            // Add row number
            primary_matrix.push_str(&format!("{:02}: ", row_idx));

            // Format each byte in the row
            for byte in primary_data {
                primary_matrix.push_str(&format!("{:02x} ", byte));
            }
            primary_matrix.push('\n');
        }

        // Log the primary matrix
        tracing::info!("\n{}", primary_matrix);

        // Format secondary slivers as a 2D matrix (columns)
        let mut secondary_matrix = String::from("Secondary Slivers Matrix (columns):\n");

        // Add column headers
        secondary_matrix.push_str("     ");
        for col_idx in 0..sliver_pairs.len() {
            secondary_matrix.push_str(&format!("{:02} ", col_idx));
        }
        secondary_matrix.push('\n');

        // Add separator line
        secondary_matrix.push_str("    ");
        for _ in 0..sliver_pairs.len() {
            secondary_matrix.push_str("---");
        }
        secondary_matrix.push('\n');

        // Find the maximum length of any secondary sliver
        let max_secondary_len = sliver_pairs
            .iter()
            .map(|pair| pair.secondary.symbols.data().len())
            .max()
            .unwrap_or(0);

        // Add each row of the secondary slivers matrix with row number
        for row_idx in 0..max_secondary_len {
            // Add row number
            secondary_matrix.push_str(&format!("{:02}: ", row_idx));

            for sliver_pair in &sliver_pairs {
                let secondary_data = sliver_pair.secondary.symbols.data();

                // Format the byte at this position, or a placeholder if out of bounds
                if row_idx < secondary_data.len() {
                    secondary_matrix.push_str(&format!("{:02x} ", secondary_data[row_idx]));
                } else {
                    secondary_matrix.push_str("-- ");
                }
            }
            secondary_matrix.push('\n');
        }

        // Log the secondary matrix

        let row_size = quilt.row_size;

        // Verify primary source slivers match quilt data.
        for (i, sliver) in sliver_pairs.iter().take(nr).enumerate() {
            let expected_row = &quilt.data[i * row_size..(i + 1) * row_size];
            assert_eq!(
                sliver.primary.symbols.data(),
                expected_row,
                "Row {} mismatch in primary sliver",
                i
            );
        }

        // Sort QuiltBlocks by end_index to ensure we process them in order
        let quilt_blocks = metadata.blocks.clone();
        // quilt_blocks.sort_by_key(|block| block.end_index);

        // Verify that each blob can be reconstructed from the secondary slivers
        let mut prev_end_index = 0;

        for quilt_block in quilt_blocks {
            // Find the original blob with this blob_id for comparison
            let original_blob = blobs_with_desc
                .iter()
                .zip(blob_ids.iter())
                .find(|(_blob_with_desc, id)| *id == &quilt_block.blob_id)
                .expect("Should find original blob")
                .0
                .blob;

            // Extract the blob data from the secondary slivers
            let mut reconstructed_data = Vec::new();

            // Simply concatenate all secondary slivers from prev_end_index to end_index
            let n_slivers = sliver_pairs.len();
            for i in prev_end_index..quilt_block.end_index {
                let idx = i as usize;
                if idx < n_slivers {
                    let secondary_data = sliver_pairs[n_slivers - 1 - idx].secondary.symbols.data();
                    reconstructed_data.extend_from_slice(secondary_data);
                }
            }

            // Truncate to unencoded_length if needed
            if reconstructed_data.len() > quilt_block.unencoded_length as usize {
                reconstructed_data.truncate(quilt_block.unencoded_length as usize);
            }

            // Verify the reconstructed blob matches the original
            assert_eq!(
                reconstructed_data, original_blob,
                "Blob with ID {:?} could not be correctly reconstructed from secondary slivers",
                quilt_block.blob_id
            );

            // Update prev_end_index for the next blob
            prev_end_index = quilt_block.end_index;
        }
    }

    param_test! {
        test_quilt_encoder: [
            case_0: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_0_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_2: (
                &[
                    BlobWithDesc { blob: &[1, 3][..], desc: "test blob 0" },
                    BlobWithDesc { blob: &[255u8; 1024][..], desc: "test blob 1" },
                    BlobWithDesc { blob: &[1, 2, 3][..], desc: "test blob 2" },
                ],
                4, 8, 12
            ),
            case_3: (
                &[
                    BlobWithDesc { blob: &[9, 8, 7, 6, 5, 4, 3, 2, 1][..], desc: "test blob 0" },
                ],
                3, 5, 7
            ),
        ]
    }
    fn test_quilt_encoder(
        blobs_with_desc: &[BlobWithDesc<'_>],
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

        test_construct_quilt(
            blobs_with_desc,
            EncodingConfigEnum::RaptorQ(&raptorq_config),
        );
        test_construct_quilt(
            blobs_with_desc,
            EncodingConfigEnum::ReedSolomon(&reed_solomon_config),
        );
    }

    fn test_construct_quilt(blobs_with_desc: &[BlobWithDesc<'_>], config: EncodingConfigEnum) {
        let _guard = tracing_subscriber::fmt().try_init();
        let nr = config.n_source_symbols::<Primary>().get() as usize;

        // Calculate blob IDs directly from blobs_with_desc.
        let blob_ids: Vec<BlobId> = blobs_with_desc
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(config.clone(), blob_with_desc.blob)
                    .expect("Should create encoder");
                *encoder.compute_metadata().blob_id()
            })
            .collect();

        let encoder = QuiltEncoder::new(config, blobs_with_desc);
        let quilt = encoder.construct_quilt().expect("Should construct quilt");
        tracing::info!("Quilt: {:?}", quilt);

        // Verify each blob and its description.
        for (blob_with_desc, blob_id) in blobs_with_desc.iter().zip(blob_ids.iter()) {
            let retrieved_blob = quilt
                .get_blob_by_id(blob_id)
                .expect("Should find blob by ID");

            // Verify blob data matches
            assert_eq!(
                &retrieved_blob, blob_with_desc.blob,
                "Mismatch in encoded blob"
            );

            // Verify description matches
            let block = quilt
                .blocks
                .iter()
                .find(|block| block.blob_id == *blob_id)
                .expect("Block should exist for this blob ID");

            assert_eq!(
                block.desc, blob_with_desc.desc,
                "Mismatch in blob description"
            );
        }

        let (sliver_pairs, metadata) = encoder.encode_with_metadata();

        // Format primary slivers as a 2D matrix (rows)
        let mut primary_matrix = String::from("Primary Slivers Matrix (rows):\n");

        // Add column headers
        primary_matrix.push_str("     ");
        for col_idx in 0..sliver_pairs[0].primary.symbols.data().len() {
            primary_matrix.push_str(&format!("{:02} ", col_idx));
        }
        primary_matrix.push('\n');

        // Add separator line
        primary_matrix.push_str("    ");
        for _ in 0..sliver_pairs[0].primary.symbols.data().len() {
            primary_matrix.push_str("---");
        }
        primary_matrix.push('\n');

        // Add each row with row number
        for (row_idx, sliver_pair) in sliver_pairs.iter().enumerate() {
            let primary_data = sliver_pair.primary.symbols.data();

            // Add row number
            primary_matrix.push_str(&format!("{:02}: ", row_idx));

            // Format each byte in the row
            for byte in primary_data {
                primary_matrix.push_str(&format!("{:02x} ", byte));
            }
            primary_matrix.push('\n');
        }

        // Log the primary matrix
        tracing::info!("\n{}", primary_matrix);

        // Format secondary slivers as a 2D matrix (columns)
        let mut secondary_matrix = String::from("Secondary Slivers Matrix (columns):\n");

        // Add column headers
        secondary_matrix.push_str("     ");
        for col_idx in 0..sliver_pairs.len() {
            secondary_matrix.push_str(&format!("{:02} ", col_idx));
        }
        secondary_matrix.push('\n');

        // Add separator line
        secondary_matrix.push_str("    ");
        for _ in 0..sliver_pairs.len() {
            secondary_matrix.push_str("---");
        }
        secondary_matrix.push('\n');

        // Find the maximum length of any secondary sliver
        let max_secondary_len = sliver_pairs
            .iter()
            .map(|pair| pair.secondary.symbols.data().len())
            .max()
            .unwrap_or(0);

        // Add each row of the secondary slivers matrix with row number
        for row_idx in 0..max_secondary_len {
            // Add row number
            secondary_matrix.push_str(&format!("{:02}: ", row_idx));

            for sliver_pair in &sliver_pairs {
                let secondary_data = sliver_pair.secondary.symbols.data();

                // Format the byte at this position, or a placeholder if out of bounds
                if row_idx < secondary_data.len() {
                    secondary_matrix.push_str(&format!("{:02x} ", secondary_data[row_idx]));
                } else {
                    secondary_matrix.push_str("-- ");
                }
            }
            secondary_matrix.push('\n');
        }

        // Log the secondary matrix
        tracing::info!("\n{}", secondary_matrix);

        let row_size = quilt.row_size;

        // Verify primary source slivers match quilt data.
        for (i, sliver) in sliver_pairs.iter().take(nr).enumerate() {
            let expected_row = &quilt.data[i * row_size..(i + 1) * row_size];
            assert_eq!(
                sliver.primary.symbols.data(),
                expected_row,
                "Row {} mismatch in primary sliver",
                i
            );
        }

        // Sort QuiltBlocks by end_index to ensure we process them in order
        let mut quilt_blocks = metadata.blocks.clone();
        quilt_blocks.sort_by_key(|block| block.end_index);

        // Verify that each blob can be reconstructed from the secondary slivers
        let mut prev_end_index = 0;

        for quilt_block in quilt_blocks {
            // Find the original blob with this blob_id for comparison
            let original_blob = blobs_with_desc
                .iter()
                .zip(blob_ids.iter())
                .find(|(_blob_with_desc, id)| *id == &quilt_block.blob_id)
                .expect("Should find original blob")
                .0
                .blob;

            // Extract the blob data from the secondary slivers
            let mut reconstructed_data = Vec::new();

            // Simply concatenate all secondary slivers from prev_end_index to end_index
            for i in prev_end_index..quilt_block.end_index {
                let idx = i as usize;
                if idx < sliver_pairs.len() {
                    let secondary_data = sliver_pairs[sliver_pairs.len() - 1 - idx]
                        .secondary
                        .symbols
                        .data();
                    reconstructed_data.extend_from_slice(secondary_data);
                }
            }

            // Truncate to unencoded_length if needed
            if reconstructed_data.len() > quilt_block.unencoded_length as usize {
                reconstructed_data.truncate(quilt_block.unencoded_length as usize);
            }

            // Verify the reconstructed blob matches the original
            assert_eq!(
                reconstructed_data, original_blob,
                "Blob with ID {:?} could not be correctly reconstructed from secondary slivers",
                quilt_block.blob_id
            );

            // Update prev_end_index for the next blob
            prev_end_index = quilt_block.end_index;
        }
    }
    param_test! {
        test_construct_container_quilt: [
            case_0: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_0_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_2: (
                &[
                    BlobWithDesc { blob: &[1, 3][..], desc: "test blob 0" },
                    BlobWithDesc { blob: &[255u8; 1024][..], desc: "test blob 1" },
                    BlobWithDesc { blob: &[1, 2, 3][..], desc: "test blob 2" },
                ],
                4, 8, 12
            ),
            case_3: (
                &[
                    BlobWithDesc { blob: &[9, 8, 7, 6, 5, 4, 3, 2, 1][..], desc: "test blob 0" },
                ],
                3, 5, 7
            ),
        ]
    }
    fn test_construct_container_quilt(
        blobs_with_desc: &[BlobWithDesc<'_>],
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

        construct_container_quilt(
            blobs_with_desc,
            EncodingConfigEnum::RaptorQ(&raptorq_config),
        );
        construct_container_quilt(
            blobs_with_desc,
            EncodingConfigEnum::ReedSolomon(&reed_solomon_config),
        );
    }

    fn construct_container_quilt(blobs_with_desc: &[BlobWithDesc<'_>], config: EncodingConfigEnum) {
        let _guard = tracing_subscriber::fmt().try_init();

        // Calculate blob IDs directly from blobs_with_desc.
        let blob_ids: Vec<BlobId> = blobs_with_desc
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(config.clone(), blob_with_desc.blob)
                    .expect("Should create encoder");
                *encoder.compute_metadata().blob_id()
            })
            .collect();

        let encoder = QuiltEncoder::new(config, blobs_with_desc);
        let quilt = encoder
            .construct_container_quilt()
            .expect("Should construct quilt");
        tracing::info!("Quilt: {:?}", quilt);

        // Verify each blob and its description.
        for (blob_with_desc, blob_id) in blobs_with_desc.iter().zip(blob_ids.iter()) {
            let retrieved_blob = quilt
                .get_blob_by_id(blob_id)
                .expect("Should find blob by ID");

            // Verify blob data matches
            assert_eq!(
                &retrieved_blob, blob_with_desc.blob,
                "Mismatch in encoded blob"
            );

            // Verify description matches
            let block = quilt
                .blocks
                .iter()
                .find(|block| block.blob_id == *blob_id)
                .expect("Block should exist for this blob ID");

            assert_eq!(
                block.desc, blob_with_desc.desc,
                "Mismatch in blob description"
            );
        }

        let quilt_index_blob = quilt
            .get_blob_by_id(&BlobId::ZERO)
            .expect("Should find quilt index blob");

        // When decoding the quilt index:

        // First, read the first 8 bytes to get the data size
        let data_size = u64::from_le_bytes(quilt_index_blob[0..8].try_into().unwrap());

        // Then, decode the QuiltIndex from the next data_size bytes
        let container_quilt_index: QuiltIndex =
            bcs::from_bytes(&quilt_index_blob[8..(data_size as usize)])
                .expect("Failed to decode QuiltIndex");
        tracing::info!("QuiltIndex: {:?}", container_quilt_index);
        assert_eq!(
            container_quilt_index.quilt_blocks.len(),
            blobs_with_desc.len()
        );
    }

    param_test! {
        test_quilt_decoder_with_quilt_index: [
            case_0: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_0_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1: (
                &[
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_1_random_order: (
                &[
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5][..],
                        desc: "test blob 0",
                    },
                    BlobWithDesc {
                        blob: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
                        desc: "test blob 1",
                    },
                    BlobWithDesc {
                        blob: &[5, 68, 3, 2, 5, 6, 78, 8][..],
                        desc: "test blob 2",
                    },
                ],
                3, 5, 7
            ),
            case_2: (
                &[
                    BlobWithDesc { blob: &[1, 3][..], desc: "test blob 0" },
                    BlobWithDesc { blob: &[255u8; 1024][..], desc: "test blob 1" },
                    BlobWithDesc { blob: &[1, 2, 3][..], desc: "test blob 2" },
                ],
                4, 8, 12
            ),
            case_3: (
                &[
                    BlobWithDesc { blob: &[9, 8, 7, 6, 5, 4, 3, 2, 1][..], desc: "test blob 0" },
                ],
                3, 5, 7
            ),
        ]
    }
    fn test_quilt_decoder_with_quilt_index(
        blobs_with_desc: &[BlobWithDesc<'_>],
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

        quilt_decoder_with_quilt_index(
            blobs_with_desc,
            EncodingConfigEnum::RaptorQ(&raptorq_config),
        );
        quilt_decoder_with_quilt_index(
            blobs_with_desc,
            EncodingConfigEnum::ReedSolomon(&reed_solomon_config),
        );
    }

    fn quilt_decoder_with_quilt_index(
        blobs_with_desc: &[BlobWithDesc<'_>],
        config: EncodingConfigEnum,
    ) {
        let _guard = tracing_subscriber::fmt().try_init();
        let n_shards = config.n_shards();

        // Calculate blob IDs directly from blobs_with_desc.
        let blob_ids: Vec<BlobId> = blobs_with_desc
            .iter()
            .map(|blob_with_desc| {
                let encoder = BlobEncoder::new(config.clone(), blob_with_desc.blob)
                    .expect("Should create encoder");
                *encoder.compute_metadata().blob_id()
            })
            .collect();

        let encoder = QuiltEncoder::new(config, blobs_with_desc);
        let quilt = encoder
            .construct_container_quilt()
            .expect("Should construct quilt");
        tracing::info!("Quilt: {:?}", quilt);

        let (sliver_pairs, quilt_metadata) = encoder
            .encode_with_quilt_index_and_metadata()
            .expect("Should encode with quilt index and metadata");
        tracing::info!("Sliver pairs: {:?}", sliver_pairs);
        tracing::info!("Quilt metadata: {:?}", quilt_metadata);
        let slivers: Vec<&SliverData<Secondary>> = sliver_pairs
            .iter()
            .map(|sliver_pair| &sliver_pair.secondary)
            .collect();

        let mut decoder = QuiltDecoder::new(n_shards, &[]);
        let result = decoder.decode_quilt_index();
        tracing::info!("Result: {:?}", result);
        assert!(matches!(
            decoder.decode_quilt_index(),
            Err(QuiltError::MissingSliver(_))
        ));
        decoder = decoder.add_slivers(&slivers);
        let quilt_index = decoder
            .decode_quilt_index()
            .expect("Should decode quilt index");
        tracing::info!("Quilt index: {:?}", quilt_index);

        for (blob_with_desc, blob_id) in blobs_with_desc.iter().zip(blob_ids.iter()) {
            let blob = decoder.get_blob_by_id(blob_id);
            assert_eq!(blob, Some(blob_with_desc.blob.to_vec()));
            let blob = decoder.get_blob_by_desc(blob_with_desc.desc);
            assert_eq!(blob, Some(blob_with_desc.blob.to_vec()));
        }
    }
}
