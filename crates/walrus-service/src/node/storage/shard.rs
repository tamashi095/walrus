// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Walrus shard storage.

use core::fmt::{self, Display};
use std::{
    collections::HashSet,
    ops::Bound::{Excluded, Unbounded},
    path::Path,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use fastcrypto::traits::KeyPair;
use futures::{stream::FuturesUnordered, StreamExt};
use regex::Regex;
use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};
use typed_store::{
    rocks::{
        be_fix_int_ser as to_rocks_db_key,
        errors::typed_store_err_from_rocks_err,
        DBBatch,
        DBMap,
        ReadWriteOptions,
        RocksDB,
    },
    Map,
    TypedStoreError,
};
use walrus_core::{
    by_axis::ByAxis,
    encoding::{EncodingAxis, Primary, PrimarySliver, Secondary, SecondarySliver},
    BlobId,
    Epoch,
    InconsistencyProof,
    ShardIndex,
    Sliver,
    SliverType,
};
use walrus_utils::metrics::Registry;

use super::{
    blob_info::{BlobInfo, BlobInfoIterator},
    constants,
    metrics::{CommonDatabaseMetrics, Labels, OperationType},
    DatabaseConfig,
};
use crate::node::{
    blob_retirement_notifier::ExecutionResultWithRetirementCheck,
    config::ShardSyncConfig,
    errors::SyncShardClientError,
    StorageNodeInner,
};

type ShardMetrics = CommonDatabaseMetrics;

/// A cache of the family names created for an instance of a shard, to avoid recomputing them for
/// metrics.
#[derive(Debug)]
struct ShardColumnFamilyNames {
    pending_recover_slivers: String,
    primary_slivers: String,
    secondary_slivers: String,
    shard_status: String,
    shard_sync_progress: String,
}

impl ShardColumnFamilyNames {
    fn slivers(&self, axis: SliverType) -> &str {
        if axis.is_primary() {
            &self.primary_slivers
        } else {
            &self.secondary_slivers
        }
    }
}

impl ShardColumnFamilyNames {
    fn new(id: ShardIndex) -> Self {
        Self {
            pending_recover_slivers: constants::pending_recover_slivers_column_family_name(id),
            primary_slivers: constants::primary_slivers_column_family_name(id),
            secondary_slivers: constants::secondary_slivers_column_family_name(id),
            shard_status: constants::shard_status_column_family_name(id),
            shard_sync_progress: constants::shard_sync_progress_column_family_name(id),
        }
    }
}

// Important: this enum is committed to database. Do not modify the existing fields. Only add new
// fields at the end.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShardStatus {
    /// Initial status of the shard when just created.
    None,

    /// The shard is active in this node serving reads and writes.
    Active,

    /// The shard is being synced to the last epoch.
    ActiveSync,

    /// The shard is being synced to the last epoch and recovering missing blobs using sliver
    /// recovery mechanism.
    ActiveRecover,

    /// The shard is locked for moving to another node. Shard does not accept any more writes in
    /// this status.
    LockedToMove,
}

impl Display for ShardStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl ShardStatus {
    pub fn is_owned_by_node(&self) -> bool {
        self != &ShardStatus::LockedToMove
    }

    /// Provides a string representation of the enum variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            ShardStatus::None => "None",
            ShardStatus::Active => "Active",
            ShardStatus::ActiveSync => "ActiveSync",
            ShardStatus::ActiveRecover => "ActiveRecover",
            ShardStatus::LockedToMove => "LockedToMove",
        }
    }
}

// When syncing a shard, the task first requests the primary slivers following
// the order of the blob IDs, and then all the secondary slivers.
// This struct represents the current progress of syncing a shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ShardSyncProgressV1 {
    last_synced_blob_id: BlobId,
    sliver_type: SliverType,
}

// Represents the progress of syncing a shard. It is used to resume syncing a shard
// if it was interrupted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ShardSyncProgress {
    V1(ShardSyncProgressV1),
}

impl ShardSyncProgress {
    fn new(last_synced_blob_id: BlobId, sliver_type: SliverType) -> Self {
        Self::V1(ShardSyncProgressV1 {
            last_synced_blob_id,
            sliver_type,
        })
    }
}

// Represents the last synced status of the shard after restart.
#[derive(Debug)]
pub(crate) enum ShardLastSyncStatus {
    // The last synced blob ID for the primary slivers.
    // If the last_synced_blob_id is None, it means the shard has not synced any primary slivers.
    Primary { last_synced_blob_id: Option<BlobId> },
    // The last synced blob ID for the secondary slivers.
    Secondary { last_synced_blob_id: Option<BlobId> },
    // The shard is in recovery mode.
    Recovery,
}

/// Primary sliver data stored in the database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimarySliverData {
    V1(PrimarySliver),
}

impl From<PrimarySliver> for PrimarySliverData {
    fn from(sliver: PrimarySliver) -> Self {
        Self::V1(sliver)
    }
}

impl From<PrimarySliverData> for PrimarySliver {
    fn from(data: PrimarySliverData) -> Self {
        match data {
            PrimarySliverData::V1(sliver) => sliver,
        }
    }
}

/// Secondary sliver data stored in the database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondarySliverData {
    V1(SecondarySliver),
}

impl From<SecondarySliver> for SecondarySliverData {
    fn from(sliver: SecondarySliver) -> Self {
        Self::V1(sliver)
    }
}

impl From<SecondarySliverData> for SecondarySliver {
    fn from(data: SecondarySliverData) -> Self {
        match data {
            SecondarySliverData::V1(sliver) => sliver,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardStorage {
    id: ShardIndex,
    shard_status: DBMap<(), ShardStatus>,
    primary_slivers: DBMap<BlobId, PrimarySliverData>,
    secondary_slivers: DBMap<BlobId, SecondarySliverData>,
    shard_sync_progress: DBMap<(), ShardSyncProgress>,
    pending_recover_slivers: DBMap<(SliverType, BlobId), ()>,
    metrics: ShardMetrics,
    cf_names: Arc<ShardColumnFamilyNames>,
}

macro_rules! reopen_cf {
    ($cf_options:expr, $db:expr, $rw_options:expr) => {{
        let (cf_name, cf_db_option) = $cf_options;
        if $db.cf_handle(&cf_name).is_none() {
            #[cfg(msim)]
            sui_macros::fail_point!("create-cf-before");
            $db.create_cf(&cf_name, &cf_db_option)
                .map_err(typed_store_err_from_rocks_err)?;
        }
        DBMap::reopen($db, Some(&cf_name), &$rw_options, false)?
    }};
}

/// Storage corresponding to a single shard.
impl ShardStorage {
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %id))]
    pub(crate) fn create_or_reopen(
        id: ShardIndex,
        database: &Arc<RocksDB>,
        db_config: &DatabaseConfig,
        initial_shard_status: Option<ShardStatus>,
        registry: &Registry,
    ) -> Result<Self, TypedStoreError> {
        let start = Instant::now();

        let collection_name = constants::base_column_family_name(id);
        let labels = Labels {
            collection_name: &collection_name,
            operation_name: OperationType::Create,
            query_summary: "CREATE shard tables",
            ..Labels::default()
        };

        let metrics = ShardMetrics::new_with_id(registry, collection_name.clone());
        let response = Self::create_or_reopen_inner(
            id,
            database,
            db_config,
            initial_shard_status,
            metrics.clone(),
        );

        metrics.observe_operation_duration(
            labels.with_response_as_unit(response.as_ref()),
            start.elapsed(),
        );

        response
    }

    fn create_or_reopen_inner(
        id: ShardIndex,
        database: &Arc<RocksDB>,
        db_config: &DatabaseConfig,
        initial_shard_status: Option<ShardStatus>,
        metrics: ShardMetrics,
    ) -> Result<Self, TypedStoreError> {
        let cf_names = ShardColumnFamilyNames::new(id);
        let rw_options = ReadWriteOptions::default();

        let shard_status = reopen_cf!(
            (
                &cf_names.shard_status,
                shard_status_column_family_options(db_config),
            ),
            database,
            rw_options
        );
        let shard_sync_progress = reopen_cf!(
            (
                &cf_names.shard_sync_progress,
                shard_sync_progress_column_family_options(db_config),
            ),
            database,
            rw_options
        );
        let pending_recover_slivers = reopen_cf!(
            (
                &cf_names.pending_recover_slivers,
                pending_recover_slivers_column_family_options(db_config),
            ),
            database,
            rw_options
        );

        // Make sure that sliver column families are created last. They are used to identify
        // whether the shard storage is initialized in `existing_cf_shards_ids`.
        let primary_slivers = reopen_cf!(
            (
                &cf_names.primary_slivers,
                primary_slivers_column_family_options(db_config)
            ),
            database,
            rw_options
        );
        let secondary_slivers = reopen_cf!(
            (
                &cf_names.secondary_slivers,
                secondary_slivers_column_family_options(db_config)
            ),
            database,
            rw_options
        );

        if let Some(status) = initial_shard_status {
            shard_status.insert(&(), &status)?;
        }

        Ok(Self {
            id,
            shard_status,
            primary_slivers,
            secondary_slivers,
            shard_sync_progress,
            pending_recover_slivers,
            metrics,
            cf_names: Arc::new(cf_names),
        })
    }

    /// Stores the provided primary or secondary sliver for the given blob ID.
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn put_sliver(
        &self,
        blob_id: &BlobId,
        sliver: &Sliver,
    ) -> Result<(), TypedStoreError> {
        let start = Instant::now();
        let labels = Labels {
            collection_name: self.cf_names.slivers(sliver.r#type()),
            operation_name: OperationType::Insert,
            query_summary: "INSERT (blob_id, sliver)",
            ..Default::default()
        };

        let response = match sliver {
            Sliver::Primary(primary) => self
                .primary_slivers
                .insert(blob_id, &PrimarySliverData::from(primary.clone())),
            Sliver::Secondary(secondary) => self
                .secondary_slivers
                .insert(blob_id, &SecondarySliverData::from(secondary.clone())),
        };

        self.metrics
            .observe_operation_duration(labels.with_response(response.as_ref()), start.elapsed());

        response
    }

    pub(crate) fn id(&self) -> ShardIndex {
        self.id
    }

    /// Returns the sliver of the specified type that is stored for that Blob ID, if any.
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn get_sliver(
        &self,
        blob_id: &BlobId,
        sliver_type: SliverType,
    ) -> Result<Option<Sliver>, TypedStoreError> {
        match sliver_type {
            SliverType::Primary => self
                .get_primary_sliver(blob_id)
                .map(|s| s.map(Sliver::Primary)),
            SliverType::Secondary => self
                .get_secondary_sliver(blob_id)
                .map(|s| s.map(Sliver::Secondary)),
        }
    }

    /// Retrieves the stored primary sliver for the given blob ID.
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn get_primary_sliver(
        &self,
        blob_id: &BlobId,
    ) -> Result<Option<PrimarySliver>, TypedStoreError> {
        let start = Instant::now();
        let labels = Labels {
            collection_name: &self.cf_names.primary_slivers,
            operation_name: OperationType::Get,
            query_summary: "GET primary_sliver BY blob_id",
            ..Labels::default()
        };

        let response = self
            .primary_slivers
            .get(blob_id)
            .map(|s| s.map(|s| s.into()));

        self.metrics
            .observe_operation_duration(labels.with_response(response.as_ref()), start.elapsed());

        response
    }

    /// Retrieves the stored secondary sliver for the given blob ID.
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn get_secondary_sliver(
        &self,
        blob_id: &BlobId,
    ) -> Result<Option<SecondarySliver>, TypedStoreError> {
        let start = Instant::now();
        let labels = Labels {
            collection_name: &self.cf_names.secondary_slivers,
            operation_name: OperationType::Get,
            query_summary: "GET secondary_sliver BY blob_id",
            ..Labels::default()
        };

        let response = self
            .secondary_slivers
            .get(blob_id)
            .map(|s| s.map(|s| s.into()));

        self.metrics
            .observe_operation_duration(labels.with_response(response.as_ref()), start.elapsed());

        response
    }

    /// Returns true iff the sliver-pair for the given blob ID is stored by the shard.
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn is_sliver_pair_stored(&self, blob_id: &BlobId) -> Result<bool, TypedStoreError> {
        Ok(self.is_sliver_stored::<Primary>(blob_id)?
            && self.is_sliver_stored::<Secondary>(blob_id)?)
    }

    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn is_sliver_stored<A: EncodingAxis>(
        &self,
        blob_id: &BlobId,
    ) -> Result<bool, TypedStoreError> {
        self.is_sliver_type_stored(blob_id, A::sliver_type())
    }

    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn is_sliver_type_stored(
        &self,
        blob_id: &BlobId,
        type_: SliverType,
    ) -> Result<bool, TypedStoreError> {
        let start = Instant::now();
        let labels = Labels {
            collection_name: self.cf_names.slivers(type_),
            operation_name: OperationType::ContainsKey,
            query_summary: "CONTAINS_KEY blob_id",
            ..Labels::default()
        };

        let response = match type_ {
            SliverType::Primary => self.primary_slivers.contains_key(blob_id),
            SliverType::Secondary => self.secondary_slivers.contains_key(blob_id),
        };

        self.metrics
            .observe_operation_duration(labels.with_response(response.as_ref()), start.elapsed());

        response
    }

    /// Deletes the sliver pair for the given [`BlobId`].
    #[tracing::instrument(skip_all, fields(walrus.shard_index = %self.id), err)]
    pub(crate) fn delete_sliver_pair(
        &self,
        batch: &mut DBBatch,
        blob_id: &BlobId,
    ) -> Result<(), TypedStoreError> {
        batch.delete_batch(&self.primary_slivers, std::iter::once(blob_id))?;
        batch.delete_batch(&self.secondary_slivers, std::iter::once(blob_id))?;
        Ok(())
    }

    /// Returns the ids of existing shards that are fully initialized in the database at the
    /// provided path.
    pub(crate) fn existing_cf_shards_ids(path: &Path, options: &Options) -> HashSet<ShardIndex> {
        // RocksDb internal uses real clock to start and initialize the database. Wrap this call
        // in a nondeterministic block to make the test deterministic.
        sui_macros::nondeterministic!(DB::list_cf(options, path)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|cf_name| match id_from_column_family_name(&cf_name) {
                // To check existing shards, we only need to look at whether the secondary sliver
                // column was created or not, as it was created after the primary sliver column.
                Some((shard_index, SliverType::Secondary)) => Some(shard_index),
                Some((_, SliverType::Primary)) | None => None,
            })
            .collect())
    }

    pub(crate) fn status(&self) -> Result<ShardStatus, TypedStoreError> {
        self.shard_status
            .get(&())
            .map(|s| s.unwrap_or(ShardStatus::None))
    }

    /// Sets the shard db to prepare for a new shard sync.
    ///
    /// This function will delete the existing sync progress for the shard and reset the shard
    /// sync from scratch.
    pub(crate) fn record_start_shard_sync(&self) -> Result<(), TypedStoreError> {
        let mut batch = self.shard_status.batch();
        batch.delete_batch(&self.shard_sync_progress, [()])?;
        batch.insert_batch(&self.shard_status, [((), ShardStatus::ActiveSync)])?;
        batch.write()
    }

    pub(crate) fn set_active_status(&self) -> Result<(), TypedStoreError> {
        self.shard_status.insert(&(), &ShardStatus::Active)
    }

    pub(crate) fn resume_active_shard_sync(&self) -> Result<ShardLastSyncStatus, TypedStoreError> {
        // Note that when active shard sync is stopped in the middle and shard recover starts,
        // the shard sync progress recorded in the db is not deleted, until the shard is completely
        // synced. By setting the shard status to active sync, the shard will resume the sync from
        // the last synced blob id.
        self.shard_status.insert(&(), &ShardStatus::ActiveSync)?;
        self.get_last_sync_status(&ShardStatus::ActiveSync)
    }

    /// Fetches the slivers with `sliver_type` for the provided blob IDs.
    pub(crate) fn fetch_slivers(
        &self,
        sliver_type: SliverType,
        slivers_to_fetch: &[BlobId],
    ) -> Result<Vec<(BlobId, Sliver)>, TypedStoreError> {
        let start = Instant::now();
        let labels = Labels {
            collection_name: self.cf_names.slivers(sliver_type),
            operation_name: OperationType::MultiGet,
            query_summary: "MULTI_GET sliver BY blob_id_list",
            ..Labels::default()
        };

        #[cfg(msim)]
        {
            let mut return_empty = false;
            sui_macros::fail_point_if!("fail_point_sync_shard_return_empty", || return_empty =
                true);
            if return_empty {
                return Ok(Vec::new());
            }
        }

        let response = ByAxis::from(sliver_type)
            .map(
                // TODO(#648): compare multi_get with scan for large value size.
                |_| self.primary_slivers.multi_get(slivers_to_fetch),
                |_| self.secondary_slivers.multi_get(slivers_to_fetch),
            )
            .transpose();

        self.metrics.observe_operation_duration(
            labels.with_response(response.as_ref().map(|_| &())),
            start.elapsed(),
        );

        let output = match response? {
            ByAxis::Primary(slivers) => slivers_to_fetch
                .iter()
                .zip(slivers)
                .filter_map(|(&blob_id, sliver)| {
                    let PrimarySliverData::V1(sliver) = sliver?;
                    Some((blob_id, Sliver::Primary(sliver)))
                })
                .collect(),
            ByAxis::Secondary(slivers) => slivers_to_fetch
                .iter()
                .zip(slivers)
                .filter_map(|(&blob_id, sliver)| {
                    let SecondarySliverData::V1(sliver) = sliver?;
                    Some((blob_id, Sliver::Secondary(sliver)))
                })
                .collect(),
        };

        Ok(output)
    }

    /// Syncs the shard to the current epoch from the previous shard owner.
    #[tracing::instrument(
        skip_all,
        fields(
            walrus.node = %node.protocol_key_pair.as_ref().public(),
            walrus.shard_index = %self.id
        ),
        err
    )]
    pub async fn start_sync_shard_before_epoch(
        &self,
        epoch: Epoch,
        node: Arc<StorageNodeInner>,
        config: &ShardSyncConfig,
        directly_recover_shard: bool,
    ) -> Result<(), SyncShardClientError> {
        tracing::info!(walrus.epoch = epoch, %directly_recover_shard, "syncing shard");

        #[cfg(msim)]
        {
            // This fail point is used to signal that direct shard sync recovery is triggered.
            if directly_recover_shard {
                sui_macros::fail_point!("fail_point_direct_shard_sync_recovery");
            }
        }

        if self.status()? == ShardStatus::None {
            self.shard_status.insert(&(), &ShardStatus::ActiveSync)?
        }

        let shard_status = self.status()?;
        assert!(
            shard_status == ShardStatus::ActiveSync || shard_status == ShardStatus::ActiveRecover
        );

        match self.get_last_sync_status(&shard_status)? {
            ShardLastSyncStatus::Primary {
                last_synced_blob_id,
            } => {
                self.sync_shard_before_epoch_internal(
                    epoch,
                    node.clone(),
                    SliverType::Primary,
                    last_synced_blob_id,
                    config,
                    directly_recover_shard,
                )
                .await?;
                self.sync_shard_before_epoch_internal(
                    epoch,
                    node.clone(),
                    SliverType::Secondary,
                    None,
                    config,
                    directly_recover_shard,
                )
                .await?;
            }
            ShardLastSyncStatus::Secondary {
                last_synced_blob_id,
            } => {
                self.sync_shard_before_epoch_internal(
                    epoch,
                    node.clone(),
                    SliverType::Secondary,
                    last_synced_blob_id,
                    config,
                    directly_recover_shard,
                )
                .await?;
            }
            ShardLastSyncStatus::Recovery => {}
        }

        self.recovery_any_missing_slivers(node, config, epoch)
            .await?;

        let mut batch = self.shard_status.batch();
        batch.insert_batch(&self.shard_status, [((), ShardStatus::Active)])?;
        batch.delete_batch(&self.shard_sync_progress, [()])?;
        batch.write()?;

        Ok(())
    }

    /// Returns the last sync status for the shard.
    fn get_last_sync_status(
        &self,
        shard_status: &ShardStatus,
    ) -> Result<ShardLastSyncStatus, TypedStoreError> {
        let last_sync_status = if shard_status == &ShardStatus::ActiveRecover {
            // We are in recovery mode. Skip happy path syncing and directly jump to recover
            // missing blobs.
            ShardLastSyncStatus::Recovery
        } else {
            match self.shard_sync_progress.get(&())? {
                Some(ShardSyncProgress::V1(ShardSyncProgressV1 {
                    last_synced_blob_id,
                    sliver_type,
                })) => {
                    tracing::info!(%last_synced_blob_id, %sliver_type, "resuming shard sync");
                    match sliver_type {
                        SliverType::Primary => ShardLastSyncStatus::Primary {
                            last_synced_blob_id: Some(last_synced_blob_id),
                        },
                        SliverType::Secondary => ShardLastSyncStatus::Secondary {
                            last_synced_blob_id: Some(last_synced_blob_id),
                        },
                    }
                }
                None => ShardLastSyncStatus::Primary {
                    last_synced_blob_id: None,
                },
            }
        };
        Ok(last_sync_status)
    }

    #[tracing::instrument(
        skip_all,
        fields(
            walrus.sliver_type = %sliver_type
        ),
        err
    )]
    async fn sync_shard_before_epoch_internal(
        &self,
        epoch: Epoch,
        node: Arc<StorageNodeInner>,
        sliver_type: SliverType,
        mut last_synced_blob_id: Option<BlobId>,
        config: &ShardSyncConfig,
        directly_recover_shard: bool,
    ) -> Result<(), SyncShardClientError> {
        // Helper to track the number of scanned blobs to test recovery. Not used in production.
        #[cfg(msim)]
        let mut scan_count: u64 = 0;

        let mut blob_info_iter = node
            .storage
            .blob_info
            .certified_blob_info_iter_before_epoch(
                epoch,
                last_synced_blob_id.map_or(Unbounded, Excluded),
            );

        let mut next_blob_info = blob_info_iter.next().transpose()?;
        // For transitioning from GENESIS epoch to epoch 1, since GENESIS epoch does not have
        // any committees and should not receive any user blobs, there shouldn't be any certified
        // blobs. In case, the shard sync should finish immediately and transition to Active state.
        assert!(epoch != 0 && (epoch > 1 || next_blob_info.is_none()));
        if !directly_recover_shard {
            while let Some((next_starting_blob_id, _)) = next_blob_info {
                tracing::debug!(
                    "syncing shard to before epoch: {}. Starting blob id: {}",
                    epoch,
                    next_starting_blob_id,
                );
                let mut batch = match sliver_type {
                    SliverType::Primary => self.primary_slivers.batch(),
                    SliverType::Secondary => self.secondary_slivers.batch(),
                };

                walrus_utils::with_label!(
                    node.metrics.sync_shard_sync_sliver_progress,
                    &self.id.to_string(),
                    &sliver_type.to_string()
                )
                .set(next_starting_blob_id.first_two_bytes() as i64);

                let fetched_slivers = node
                    .committee_service
                    .sync_shard_before_epoch(
                        self.id(),
                        next_starting_blob_id,
                        sliver_type,
                        config.sliver_count_per_sync_request,
                        epoch,
                        &node.protocol_key_pair,
                    )
                    .await?;

                next_blob_info = self.batch_fetched_slivers_and_check_missing_blobs(
                    epoch,
                    &node,
                    &fetched_slivers,
                    sliver_type,
                    next_blob_info,
                    &mut blob_info_iter,
                    &mut batch,
                )?;

                #[cfg(msim)]
                {
                    scan_count += fetched_slivers.len() as u64;
                    inject_failure(scan_count, sliver_type)?;
                }

                // Record sync progress.
                last_synced_blob_id = fetched_slivers.last().map(|(id, _)| *id);
                if let Some(last_synced_blob_id) = last_synced_blob_id {
                    batch.insert_batch(
                        &self.shard_sync_progress,
                        [((), ShardSyncProgress::new(last_synced_blob_id, sliver_type))],
                    )?;
                }
                batch.write()?;

                walrus_utils::with_label!(
                    node.metrics.sync_shard_sync_sliver_total,
                    &self.id.to_string(),
                    &sliver_type.to_string()
                )
                .inc_by(fetched_slivers.len() as u64);

                if last_synced_blob_id.is_none() {
                    break;
                }
            }
        }

        if next_blob_info.is_none() {
            return Ok(());
        }

        let mut batch = self.pending_recover_slivers.batch();
        while let Some((blob_id, _)) = next_blob_info {
            batch.insert_batch(
                &self.pending_recover_slivers,
                [((sliver_type, blob_id), ())],
            )?;
            next_blob_info = blob_info_iter.next().transpose()?;
        }
        batch.write()?;

        Ok(())
    }

    /// Helper function to add fetched slivers to the db batch and check for missing blobs.
    /// Advance `blob_info_iter`` to the next blob that is greater than the last fetched blob id,
    /// which is the next expected blob to fetch, and return the next expected blob.
    #[allow(clippy::too_many_arguments)]
    fn batch_fetched_slivers_and_check_missing_blobs(
        &self,
        epoch: Epoch,
        node: &Arc<StorageNodeInner>,
        fetched_slivers: &[(BlobId, Sliver)],
        sliver_type: SliverType,
        mut next_blob_info: Option<(BlobId, BlobInfo)>,
        blob_info_iter: &mut BlobInfoIterator,
        batch: &mut DBBatch,
    ) -> Result<Option<(BlobId, BlobInfo)>, SyncShardClientError> {
        for (blob_id, sliver) in fetched_slivers.iter() {
            tracing::debug!(
                walrus.blob_id = %blob_id,
                epoch,
                %sliver_type,
                "synced blob",
            );
            //TODO(#705): verify sliver validity.
            //  - blob is certified
            //  - metadata is correct

            #[cfg(any(test, feature = "test-utils"))]
            {
                debug_assert!(node.storage.has_metadata(blob_id)?);
            }

            match sliver {
                Sliver::Primary(primary) => {
                    assert_eq!(sliver_type, SliverType::Primary);
                    batch.insert_batch(
                        &self.primary_slivers,
                        [(blob_id, &PrimarySliverData::from(primary.clone()))],
                    )?;
                }
                Sliver::Secondary(secondary) => {
                    assert_eq!(sliver_type, SliverType::Secondary);
                    batch.insert_batch(
                        &self.secondary_slivers,
                        [(blob_id, &SecondarySliverData::from(secondary.clone()))],
                    )?;
                }
            }

            next_blob_info = self.check_and_record_missing_blobs(
                blob_info_iter,
                next_blob_info,
                *blob_id,
                sliver_type,
                batch,
            )?;
        }
        Ok(next_blob_info)
    }

    /// Given a iterator over blob info table that is currently pointing to `next_blob_info`,
    /// and `fetched_blob_id` returned from the remote storage node, checks if there are any
    /// missing blobs between `next_blob_info` and `fetched_blob_id`.
    ///
    /// If there are missing blobs, add them in `db_batch` to be stored in
    /// `pending_recover_slivers` table.
    ///
    /// Returns the next blob info that is greater than `fetched_blob_id` or None if there are no.
    fn check_and_record_missing_blobs(
        &self,
        blob_info_iter: &mut BlobInfoIterator,
        mut next_blob_info: Option<(BlobId, BlobInfo)>,
        fetched_blob_id: BlobId,
        sliver_type: SliverType,
        db_batch: &mut DBBatch,
    ) -> Result<Option<(BlobId, BlobInfo)>, TypedStoreError> {
        let Some((mut next_blob_id, _)) = next_blob_info else {
            // `next_blob_info is the last item read from `blob_info_iter`.
            // If it is None, there are no more blobs to check.
            debug_assert!(blob_info_iter.next().transpose()?.is_none());
            return Ok(None);
        };

        if next_blob_id == fetched_blob_id {
            return blob_info_iter.next().transpose();
        }

        while to_rocks_db_key(&next_blob_id) < to_rocks_db_key(&fetched_blob_id) {
            db_batch.insert_batch(
                &self.pending_recover_slivers,
                [((sliver_type, next_blob_id), ())],
            )?;

            next_blob_info = blob_info_iter.next().transpose()?;
            if let Some((blob_id, _)) = next_blob_info {
                next_blob_id = blob_id;
            } else {
                return Ok(None);
            }
        }

        if next_blob_id == fetched_blob_id {
            return blob_info_iter.next().transpose();
        }

        Ok(next_blob_info)
    }

    /// Recovers any missing blobs stored in `pending_recover_slivers` table.
    async fn recovery_any_missing_slivers(
        &self,
        node: Arc<StorageNodeInner>,
        config: &ShardSyncConfig,
        epoch: Epoch,
    ) -> Result<(), SyncShardClientError> {
        if self.pending_recover_slivers.is_empty() {
            return Ok(());
        }

        #[cfg(msim)]
        sui_macros::fail_point!("fail_point_shard_sync_recovery");

        tracing::info!("shard sync is done; still has missing blobs; shard enters recovery mode");
        self.shard_status.insert(&(), &ShardStatus::ActiveRecover)?;

        #[cfg(msim)]
        {
            let mut inject_error = false;
            sui_macros::fail_point_if!("fail_point_after_start_recovery", || inject_error = true);
            if inject_error {
                return Err(SyncShardClientError::Internal(anyhow::anyhow!(
                    "sync shard simulated sync failure after start recovery"
                )));
            }
        }

        loop {
            self.recover_missing_blobs(node.clone(), config, epoch)
                .await?;

            if self.pending_recover_slivers.is_empty() {
                // TODO: in test, check that we have recovered all the certified blobs.
                break;
            }
            tracing::warn!("recovering missing blobs still misses blobs. Retrying in 60 seconds.",);
            tokio::time::sleep(Duration::from_secs(60)).await;
        }

        Ok(())
    }

    /// Recovers missing blob slivers stored in the `pending_recover_slivers` table.
    async fn recover_missing_blobs(
        &self,
        node: Arc<StorageNodeInner>,
        config: &ShardSyncConfig,
        epoch: Epoch,
    ) -> Result<(), SyncShardClientError> {
        let mut futures = FuturesUnordered::new();

        // Update the metric for the total number of blobs pending recovery, so that we know how
        // many blobs are pending recovery.
        let mut total_blobs_pending_recovery = self.pending_recover_slivers.safe_iter().count();
        self.record_pending_recovery_metrics(&node, total_blobs_pending_recovery);

        for recover_blob in self.pending_recover_slivers.safe_iter() {
            let ((sliver_type, blob_id), _) = recover_blob?;

            #[allow(unused_mut)]
            let mut skip_certified_check_in_test = false;
            sui_macros::fail_point_if!(
                "shard_recovery_skip_initial_blob_certification_check",
                || { skip_certified_check_in_test = true }
            );

            // Given that node may redo shard transfer, there may be pending recover slivers that
            // are already stored in the shard. Check their existence before starting recovery.
            let sliver_is_stored = match sliver_type {
                SliverType::Primary => self.is_sliver_stored::<Primary>(&blob_id)?,
                SliverType::Secondary => self.is_sliver_stored::<Secondary>(&blob_id)?,
            };

            if sliver_is_stored {
                self.skip_recover_blob(blob_id, sliver_type, &node, "already_stored")?;
            } else if !skip_certified_check_in_test && !node.is_blob_certified(&blob_id)? {
                self.skip_recover_blob(blob_id, sliver_type, &node, "not_certified")?;
            } else {
                futures.push(self.recover_blob(blob_id, sliver_type, node.clone(), epoch));
            }

            total_blobs_pending_recovery -= 1;
            self.record_pending_recovery_metrics(&node, total_blobs_pending_recovery);

            // Wait for some futures to complete if we reach the concurrent blob recovery limit
            if futures.len() >= config.max_concurrent_blob_recovery_during_shard_recovery {
                if let Some(Err(error)) = futures.next().await {
                    tracing::error!(
                        ?error,
                        "error recovering missing blob sliver. \
                        blob is not removed from pending_recover_slivers"
                    );
                }
            }
        }

        while let Some(result) = futures.next().await {
            if let Err(error) = result {
                tracing::error!(
                    ?error,
                    "error recovering missing blob sliver. \
                    blob is not removed from pending_recover_slivers"
                );
            }
        }

        Ok(())
    }

    fn record_pending_recovery_metrics(
        &self,
        node: &Arc<StorageNodeInner>,
        total_blobs_pending_recovery: usize,
    ) {
        walrus_utils::with_label!(
            node.metrics.sync_shard_recover_sliver_pending_total,
            &self.id.to_string()
        )
        .set(total_blobs_pending_recovery as i64);
    }

    /// Skips recovering a blob that is no longer certified.
    fn skip_recover_blob(
        &self,
        blob_id: BlobId,
        sliver_type: SliverType,
        node: &Arc<StorageNodeInner>,
        reason: &str,
    ) -> Result<(), TypedStoreError> {
        tracing::debug!(
            %blob_id,
            shard_index = %self.id,
            skip_reason = %reason,
            "skip blob recovery"
        );
        walrus_utils::with_label!(
            node.metrics.sync_shard_recover_sliver_skip_total,
            &self.id.to_string(),
            &sliver_type.to_string(),
            reason
        )
        .inc();
        self.pending_recover_slivers
            .remove(&(sliver_type, blob_id))?;
        Ok(())
    }

    /// Recovers the missing blob sliver for the given blob ID.
    async fn recover_blob(
        &self,
        blob_id: BlobId,
        sliver_type: SliverType,
        node: Arc<StorageNodeInner>,
        epoch: Epoch,
    ) -> Result<(), SyncShardClientError> {
        tracing::info!(
            walrus.blob_id = %blob_id,
            walrus.shard_index = %self.id,
            "start recovering missing blob"
        );

        #[cfg(msim)]
        sui_macros::fail_point!("fail_point_shard_sync_recover_blob");

        // Get the metadata for the blob. If metadata is not found, it will be recovered.
        let metadata = {
            let result = node
                .blob_retirement_notifier
                .execute_with_retirement_check(&node, blob_id, || {
                    node.get_or_recover_blob_metadata(&blob_id, epoch)
                })
                .await?;

            match result {
                ExecutionResultWithRetirementCheck::Executed(metadata) => metadata?,
                ExecutionResultWithRetirementCheck::BlobRetired => {
                    self.skip_recover_blob(blob_id, sliver_type, &node, "blob_retired")?;
                    return Ok(());
                }
            }
        };

        let sliver_id = self
            .id
            .to_pair_index(node.encoding_config.n_shards(), &blob_id);

        walrus_utils::with_label!(
            node.metrics.sync_shard_recover_sliver_total,
            &self.id.to_string(),
            &sliver_type.to_string()
        )
        .inc();

        // Recover the blob sliver.
        let execution_result = node
            .blob_retirement_notifier
            .execute_with_retirement_check(&node, blob_id, || {
                node.committee_service.recover_sliver(
                    metadata.into(),
                    sliver_id,
                    sliver_type,
                    epoch,
                )
            })
            .await?;

        match execution_result {
            ExecutionResultWithRetirementCheck::Executed(result) => {
                match result {
                    Ok(sliver) => {
                        self.handle_successful_recovery(&node, sliver_type, blob_id, sliver)?;
                    }
                    Err(inconsistency_proof) => {
                        self.handle_inconsistency(&node, sliver_type, blob_id, inconsistency_proof)
                            .await;
                    }
                }
                self.pending_recover_slivers
                    .remove(&(sliver_type, blob_id))?;
            }
            ExecutionResultWithRetirementCheck::BlobRetired => {
                self.skip_recover_blob(blob_id, sliver_type, &node, "blob_retired")?;
            }
        };

        Ok(())
    }

    /// Handles the successful recovery of a blob.
    fn handle_successful_recovery(
        &self,
        node: &StorageNodeInner,
        sliver_type: SliverType,
        blob_id: BlobId,
        sliver: Sliver,
    ) -> Result<(), TypedStoreError> {
        walrus_utils::with_label!(
            node.metrics.sync_shard_recover_sliver_success_total,
            &self.id.to_string(),
            &sliver_type.to_string()
        )
        .inc();
        self.put_sliver(&blob_id, &sliver)
    }

    /// Handles the inconsistency of a blob.
    async fn handle_inconsistency(
        &self,
        node: &Arc<StorageNodeInner>,
        sliver_type: SliverType,
        blob_id: BlobId,
        inconsistency_proof: InconsistencyProof,
    ) {
        tracing::debug!("received an inconsistency proof when recovering sliver");
        walrus_utils::with_label!(
            node.metrics.sync_shard_recover_sliver_error_total,
            &self.id.to_string(),
            &sliver_type.to_string()
        )
        .inc();

        let invalid_blob_certificate = node
            .committee_service
            .get_invalid_blob_certificate(blob_id, &inconsistency_proof)
            .await;
        node.contract_service
            .invalidate_blob_id(&invalid_blob_certificate)
            .await
    }

    /// Deletes the storage for the shard.
    pub fn delete_shard_storage(&self) -> Result<(), TypedStoreError> {
        let rocksdb = self.primary_slivers.rocksdb.clone();

        // Drop column families in reverse order of creation in ShardStorage::create_or_reopen.
        rocksdb
            .drop_cf(&self.cf_names.secondary_slivers)
            .map_err(typed_store_err_from_rocks_err)?;
        rocksdb
            .drop_cf(&self.cf_names.primary_slivers)
            .map_err(typed_store_err_from_rocks_err)?;
        rocksdb
            .drop_cf(&self.cf_names.pending_recover_slivers)
            .map_err(typed_store_err_from_rocks_err)?;
        rocksdb
            .drop_cf(&self.cf_names.shard_sync_progress)
            .map_err(typed_store_err_from_rocks_err)?;
        rocksdb
            .drop_cf(&self.cf_names.shard_status)
            .map_err(typed_store_err_from_rocks_err)?;
        Ok(())
    }

    /// Locks the shard for moving to another node.
    pub(crate) fn lock_shard_for_epoch_change(&self) -> Result<(), TypedStoreError> {
        self.shard_status.insert(&(), &ShardStatus::LockedToMove)
    }

    #[cfg(test)]
    pub(crate) fn sliver_count(&self, sliver_type: SliverType) -> Result<usize, TypedStoreError> {
        match sliver_type {
            SliverType::Primary => self
                .primary_slivers
                .safe_iter()
                .try_fold(0, |count, e| e.map(|_| count + 1)),
            SliverType::Secondary => self
                .secondary_slivers
                .safe_iter()
                .try_fold(0, |count, e| e.map(|_| count + 1)),
        }
    }

    #[cfg(test)]
    pub(crate) fn update_status_in_test(&self, status: ShardStatus) -> Result<(), TypedStoreError> {
        self.shard_status.insert(&(), &status)
    }

    #[cfg(test)]
    pub(crate) fn check_and_record_missing_blobs_in_test(
        &self,
        blob_info_iter: &mut BlobInfoIterator,
        next_blob_info: Option<(BlobId, BlobInfo)>,
        fetched_blob_id: BlobId,
        sliver_type: SliverType,
    ) -> Result<Option<(BlobId, BlobInfo)>, TypedStoreError> {
        let mut db_batch = self.pending_recover_slivers.batch();
        let next_blob_info = self.check_and_record_missing_blobs(
            blob_info_iter,
            next_blob_info,
            fetched_blob_id,
            sliver_type,
            &mut db_batch,
        )?;
        db_batch.write()?;
        Ok(next_blob_info)
    }

    #[cfg(test)]
    pub(crate) fn all_pending_recover_slivers(
        &self,
    ) -> Result<Vec<(SliverType, BlobId)>, TypedStoreError> {
        self.pending_recover_slivers
            .safe_iter()
            .map(|r| r.map(|(k, _)| k))
            .collect()
    }
}

fn id_from_column_family_name(name: &str) -> Option<(ShardIndex, SliverType)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^shard-(\d+)/(primary|secondary)-slivers$").expect("valid static regex")
    })
    .captures(name)
    .and_then(|captures| {
        let Ok(id) = captures.get(1)?.as_str().parse() else {
            tracing::warn!(%name, "ignoring shard-like column family with an ID out of range");
            return None;
        };
        let sliver_type = match captures.get(2)?.as_str() {
            "primary" => SliverType::Primary,
            "secondary" => SliverType::Secondary,
            _ => panic!("Invalid sliver type in regex capture"),
        };
        Some((ShardIndex(id), sliver_type))
    })
}

/// Returns the name and options for the column families for a shard's primary
/// sliver with the specified index.
pub fn primary_slivers_column_family_options(db_config: &DatabaseConfig) -> Options {
    db_config.shard().to_options()
}

/// Returns the name and options for the column families for a shard's secondary
/// sliver with the specified index.
pub fn secondary_slivers_column_family_options(db_config: &DatabaseConfig) -> Options {
    db_config.shard().to_options()
}

/// Returns the name and options for the column families for a shard's operating status
/// with the specified index.
pub fn shard_status_column_family_options(db_config: &DatabaseConfig) -> Options {
    db_config.shard_status().to_options()
}

/// Returns the name and options for the column families for a shard's sync progress
/// with the specified index.
pub fn shard_sync_progress_column_family_options(db_config: &DatabaseConfig) -> Options {
    db_config.shard_sync_progress().to_options()
}

/// Returns the name and options for the column families for a shard's pending recover slivers
/// with the specified index.
pub fn pending_recover_slivers_column_family_options(db_config: &DatabaseConfig) -> Options {
    db_config.pending_recover_slivers().to_options()
}

#[cfg(msim)]
fn inject_failure(scan_count: u64, sliver_type: SliverType) -> Result<(), anyhow::Error> {
    // Inject a failure point to simulate a sync failure.

    let mut injected_status = Ok(());
    sui_macros::fail_point_arg!("fail_point_fetch_sliver", |(
        trigger_sliver_type,
        trigger_at,
    ): (SliverType, u64)| {
        tracing::info!(
            ?trigger_sliver_type,
            trigger_index = ?trigger_at,
            blob_count = ?scan_count,
            fail_point = "fail_point_fetch_sliver",
        );
        if trigger_sliver_type == sliver_type && trigger_at <= scan_count {
            injected_status = Err(anyhow::anyhow!("fetch_sliver simulated sync failure"));
        }
    });
    injected_status
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use walrus_core::test_utils::random_blob_id;
    use walrus_sui::test_utils::event_id_for_testing;
    use walrus_test_utils::{async_param_test, param_test, Result as TestResult, WithTempDir};

    use super::*;
    use crate::{
        node::{
            storage::{
                blob_info::BlobCertificationStatus,
                tests::{empty_storage, get_sliver, BLOB_ID, OTHER_SHARD_INDEX, SHARD_INDEX},
            },
            Storage,
        },
        test_utils::empty_storage_with_shards,
    };

    async_param_test! {
        can_store_and_retrieve_sliver -> TestResult: [
            primary: (SliverType::Primary),
            secondary: (SliverType::Secondary),
        ]
    }
    async fn can_store_and_retrieve_sliver(sliver_type: SliverType) -> TestResult {
        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");
        let sliver = get_sliver(sliver_type, 1);

        shard.put_sliver(&BLOB_ID, &sliver)?;
        let retrieved = shard.get_sliver(&BLOB_ID, sliver_type)?;

        assert_eq!(retrieved, Some(sliver));

        Ok(())
    }

    #[tokio::test]
    async fn stores_separate_primary_and_secondary_sliver() -> TestResult {
        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        let primary = get_sliver(SliverType::Primary, 1);
        let secondary = get_sliver(SliverType::Secondary, 2);

        shard.put_sliver(&BLOB_ID, &primary)?;
        shard.put_sliver(&BLOB_ID, &secondary)?;

        let retrieved_primary = shard.get_sliver(&BLOB_ID, SliverType::Primary)?;
        let retrieved_secondary = shard.get_sliver(&BLOB_ID, SliverType::Secondary)?;

        assert_eq!(retrieved_primary, Some(primary), "invalid primary sliver");
        assert_eq!(
            retrieved_secondary,
            Some(secondary),
            "invalid secondary sliver"
        );

        Ok(())
    }

    #[tokio::test]
    async fn stores_and_deletes_slivers() -> TestResult {
        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        let primary = get_sliver(SliverType::Primary, 1);
        let secondary = get_sliver(SliverType::Secondary, 2);

        shard.put_sliver(&BLOB_ID, &primary)?;
        shard.put_sliver(&BLOB_ID, &secondary)?;

        assert!(shard.is_sliver_pair_stored(&BLOB_ID)?);

        let mut batch = storage.inner.metadata.batch();
        shard.delete_sliver_pair(&mut batch, &BLOB_ID)?;
        batch.write()?;

        assert!(!shard.is_sliver_stored::<Primary>(&BLOB_ID)?);
        assert!(!shard.is_sliver_stored::<Secondary>(&BLOB_ID)?);

        Ok(())
    }

    #[tokio::test]
    async fn delete_on_empty_slivers_does_not_error() -> TestResult {
        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        assert!(!shard.is_sliver_stored::<Primary>(&BLOB_ID)?);
        assert!(!shard.is_sliver_stored::<Secondary>(&BLOB_ID)?);

        let mut batch = storage.inner.metadata.batch();
        shard
            .delete_sliver_pair(&mut batch, &BLOB_ID)
            .expect("delete should not error");
        batch.write()?;
        Ok(())
    }

    async_param_test! {
        stores_and_retrieves_for_multiple_shards -> TestResult: [
            primary_primary: (SliverType::Primary, SliverType::Primary),
            secondary_secondary: (SliverType::Secondary, SliverType::Secondary),
            mixed: (SliverType::Primary, SliverType::Secondary),
        ]
    }
    async fn stores_and_retrieves_for_multiple_shards(
        type_first: SliverType,
        type_second: SliverType,
    ) -> TestResult {
        let storage = empty_storage_with_shards(&[SHARD_INDEX, OTHER_SHARD_INDEX]).await;

        let first_shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");
        let first_sliver = get_sliver(type_first, 1);

        let second_shard = storage
            .as_ref()
            .shard_storage(OTHER_SHARD_INDEX)
            .await
            .expect("shard should exist");
        let second_sliver = get_sliver(type_second, 2);

        first_shard.put_sliver(&BLOB_ID, &first_sliver)?;
        second_shard.put_sliver(&BLOB_ID, &second_sliver)?;

        let first_retrieved = first_shard.get_sliver(&BLOB_ID, type_first)?;
        let second_retrieved = second_shard.get_sliver(&BLOB_ID, type_second)?;

        assert_eq!(
            first_retrieved,
            Some(first_sliver),
            "invalid sliver from first shard"
        );
        assert_eq!(
            second_retrieved,
            Some(second_sliver),
            "invalid sliver from second shard"
        );

        Ok(())
    }

    async_param_test! {
        indicates_when_sliver_pair_is_stored -> TestResult: [
            neither: (false, false),
            only_primary: (true, false),
            only_secondary: (false, true),
            both: (true, true),
        ]
    }
    async fn indicates_when_sliver_pair_is_stored(
        store_primary: bool,
        store_secondary: bool,
    ) -> TestResult {
        let is_pair_stored: bool = store_primary & store_secondary;

        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        if store_primary {
            shard.put_sliver(&BLOB_ID, &get_sliver(SliverType::Primary, 3))?;
        }
        if store_secondary {
            shard.put_sliver(&BLOB_ID, &get_sliver(SliverType::Secondary, 4))?;
        }

        assert_eq!(shard.is_sliver_pair_stored(&BLOB_ID)?, is_pair_stored);

        Ok(())
    }

    param_test! {
        test_parse_column_family_name: [
            primary: ("shard-10/primary-slivers", Some((ShardIndex(10), SliverType::Primary))),
            secondary: (
                "shard-20/secondary-slivers",
                Some((ShardIndex(20), SliverType::Secondary))
            ),
            invalid_id: ("shard-a/primary-slivers", None),
            invalid_sliver_type: ("shard-20/random-slivers", None),
            invalid_sliver_name: ("shard-20/slivers", None),
        ]
    }
    fn test_parse_column_family_name(
        cf_name: &str,
        expected_output: Option<(ShardIndex, SliverType)>,
    ) {
        assert_eq!(id_from_column_family_name(cf_name), expected_output);
    }

    struct ShardStorageFetchSliversSetup {
        storage: WithTempDir<Storage>,
        blob_ids: [BlobId; 4],
        data: HashMap<BlobId, HashMap<SliverType, Sliver>>,
    }

    async fn setup_storage() -> Result<ShardStorageFetchSliversSetup, TypedStoreError> {
        let storage = empty_storage().await;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        let blob_ids = [
            BlobId([0; 32]),
            BlobId([1; 32]),
            BlobId([2; 32]),
            BlobId([3; 32]),
        ];

        // Only generates data for first and third blob IDs.
        let data = [
            (
                blob_ids[0],
                [
                    (SliverType::Primary, get_sliver(SliverType::Primary, 0)),
                    (SliverType::Secondary, get_sliver(SliverType::Secondary, 1)),
                ]
                .iter()
                .cloned()
                .collect::<HashMap<SliverType, Sliver>>(),
            ),
            (
                blob_ids[2],
                [
                    (SliverType::Primary, get_sliver(SliverType::Primary, 2)),
                    (SliverType::Secondary, get_sliver(SliverType::Secondary, 3)),
                ]
                .iter()
                .cloned()
                .collect::<HashMap<SliverType, Sliver>>(),
            ),
        ]
        .iter()
        .cloned()
        .collect::<HashMap<BlobId, HashMap<SliverType, Sliver>>>();

        // Populates the shard with the generated data.
        for blob_data in data.iter() {
            for (_sliver_type, sliver) in blob_data.1.iter() {
                shard.put_sliver(blob_data.0, sliver)?;
            }
        }

        Ok(ShardStorageFetchSliversSetup {
            storage,
            blob_ids,
            data,
        })
    }

    async_param_test! {
        test_shard_storage_fetch_single_sliver -> TestResult: [
            primary: (SliverType::Primary),
            secondary: (SliverType::Secondary),
        ]
    }
    async fn test_shard_storage_fetch_single_sliver(sliver_type: SliverType) -> TestResult {
        let ShardStorageFetchSliversSetup {
            storage,
            blob_ids,
            data,
        } = setup_storage().await?;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");
        assert_eq!(
            shard.fetch_slivers(sliver_type, &[blob_ids[0]])?,
            vec![(blob_ids[0], data[&blob_ids[0]][&sliver_type].clone())]
        );

        assert_eq!(
            shard.fetch_slivers(sliver_type, &[blob_ids[2]])?,
            vec![(blob_ids[2], data[&blob_ids[2]][&sliver_type].clone())]
        );

        Ok(())
    }

    async_param_test! {
        test_shard_storage_fetch_multiple_slivers -> TestResult: [
            primary: (SliverType::Primary),
            secondary: (SliverType::Secondary),
        ]
    }
    async fn test_shard_storage_fetch_multiple_slivers(sliver_type: SliverType) -> TestResult {
        let ShardStorageFetchSliversSetup {
            storage,
            blob_ids,
            data,
        } = setup_storage().await?;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        assert_eq!(
            shard.fetch_slivers(sliver_type, &[blob_ids[0], blob_ids[2]])?,
            vec![
                (blob_ids[0], data[&blob_ids[0]][&sliver_type].clone()),
                (blob_ids[2], data[&blob_ids[2]][&sliver_type].clone())
            ]
        );

        Ok(())
    }

    async_param_test! {
        test_shard_storage_fetch_non_existing_slivers -> TestResult: [
            primary: (SliverType::Primary),
            secondary: (SliverType::Secondary),
        ]
    }
    async fn test_shard_storage_fetch_non_existing_slivers(sliver_type: SliverType) -> TestResult {
        let ShardStorageFetchSliversSetup {
            storage, blob_ids, ..
        } = setup_storage().await?;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        assert!(shard.fetch_slivers(sliver_type, &[blob_ids[1]])?.is_empty());

        Ok(())
    }

    async_param_test! {
        test_shard_storage_fetch_mix_existing_non_existing_slivers -> TestResult: [
            primary: (SliverType::Primary),
            secondary: (SliverType::Secondary),
        ]
    }
    async fn test_shard_storage_fetch_mix_existing_non_existing_slivers(
        sliver_type: SliverType,
    ) -> TestResult {
        let ShardStorageFetchSliversSetup {
            storage,
            blob_ids,
            data,
        } = setup_storage().await?;
        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");

        assert_eq!(
            shard.fetch_slivers(sliver_type, &blob_ids)?,
            vec![
                (blob_ids[0], data[&blob_ids[0]][&sliver_type].clone()),
                (blob_ids[2], data[&blob_ids[2]][&sliver_type].clone())
            ]
        );

        Ok(())
    }

    // Tests for `check_and_record_missing_blobs` method.
    // We use 8 randomly generated blob IDs, and for the index = 4 and 7,
    // we remove the corresponding blob info from the client side.
    async_param_test! {
        test_check_and_record_missing_blobs -> TestResult: [
            missing_in_between: (0, 2, Some(3)),
            no_missing: (2, 2, Some(3)),
            next_certified_1: (2, 3, Some(5)),
            next_certified_2: (2, 4, Some(5)),
            no_next_1: (0, 6, None),
            no_next_2: (0, 7, None),
        ]
    }
    async fn test_check_and_record_missing_blobs(
        next_blob_info_index: usize,
        fetched_blob_id_index: usize,
        expected_next_blob_info_index: Option<usize>,
    ) -> TestResult {
        let storage = empty_storage().await;
        let blob_info = storage.inner.blob_info.clone();
        let new_epoch = 3;

        for _ in 0..8 {
            let blob_id = random_blob_id();
            blob_info.insert(
                &blob_id,
                &BlobInfo::new_for_testing(
                    10,
                    BlobCertificationStatus::Certified,
                    event_id_for_testing(),
                    Some(0),
                    Some(1),
                    None,
                ),
            )?;
        }

        let mut sorted_blob_ids = blob_info.keys()?;
        sorted_blob_ids.sort();
        blob_info.remove(&sorted_blob_ids[4])?;
        blob_info.remove(&sorted_blob_ids[7])?;

        let shard = storage
            .as_ref()
            .shard_storage(SHARD_INDEX)
            .await
            .expect("shard should exist");
        let mut blob_info_iter = storage
            .inner
            .blob_info
            .certified_blob_info_iter_before_epoch(
                new_epoch,
                Excluded(sorted_blob_ids[next_blob_info_index]),
            );
        let next_certified_blob_to_check = shard.check_and_record_missing_blobs_in_test(
            &mut blob_info_iter,
            Some((
                sorted_blob_ids[next_blob_info_index],
                blob_info
                    .get(&sorted_blob_ids[next_blob_info_index])
                    .unwrap()
                    .unwrap(),
            )),
            sorted_blob_ids[fetched_blob_id_index],
            SliverType::Primary,
        )?;

        assert_eq!(
            next_certified_blob_to_check.map(|(blob_id, _)| blob_id),
            expected_next_blob_info_index.map(|i| sorted_blob_ids[i])
        );

        let missing_blob_ids = shard
            .all_pending_recover_slivers()?
            .iter()
            .map(|(sliver_type, id)| {
                assert_eq!(sliver_type, &SliverType::Primary);
                *id
            })
            .collect::<Vec<_>>();
        let mut expected_missing_blobs = Vec::new();
        for (i, blob_id) in sorted_blob_ids
            .iter()
            .enumerate()
            .take(fetched_blob_id_index)
            .skip(next_blob_info_index)
        {
            if i != 4 && i != 7 {
                expected_missing_blobs.push(*blob_id);
            }
        }
        assert_eq!(missing_blob_ids, expected_missing_blobs);

        Ok(())
    }
}
