// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

use prometheus::{
    core::{AtomicU64, GenericGauge, GenericGaugeVec},
    Histogram,
    HistogramVec,
    IntCounter,
    IntCounterVec,
    IntGauge,
    IntGaugeVec,
};
use walrus_sui::types::{
    BlobCertified,
    BlobEvent,
    ContractEvent,
    DenyListEvent,
    EpochChangeEvent,
    PackageEvent,
};

use crate::{
    client::ClientErrorKind,
    common::telemetry::{CurrentEpochMetric, CurrentEpochStateMetric},
    node::events::EventStreamElement,
};

pub(crate) const STATUS_FAILURE: &str = "failure";
pub(crate) const STATUS_SUCCESS: &str = "success";
pub(crate) const STATUS_ABORTED: &str = "aborted";
pub(crate) const STATUS_CANCELLED: &str = "cancelled";
pub(crate) const STATUS_SKIPPED: &str = "skip";
pub(crate) const STATUS_INCONSISTENT: &str = "inconsistent";
pub(crate) const STATUS_QUEUED: &str = "queued";
pub(crate) const STATUS_PENDING: &str = "pending";
pub(crate) const STATUS_PERSISTED: &str = "persisted";
pub(crate) const STATUS_IN_PROGRESS: &str = "in-progress";
pub(crate) const STATUS_HIGHEST_FINISHED: &str = "highest_finished";

type U64GaugeVec = GenericGaugeVec<AtomicU64>;
type U64Gauge = GenericGauge<AtomicU64>;

walrus_utils::metrics::define_metric_set! {
    #[namespace = "walrus"]
    /// Metrics exported by the storage node.
    pub(crate) struct NodeMetricSet {
        #[help = "The total number of metadata stored"]
        metadata_stored_total: IntCounter[],

        #[help = "The total number of metadata instances returned"]
        metadata_retrieved_total: IntCounter[],

        #[help = "The total number of storage confirmations issued"]
        storage_confirmations_issued_total: IntCounter[],

        #[help = "The number of shard sync per status"]
        shard_sync_total: IntCounterVec["status"],

        #[help = "Total number of slivers synced during shard sync"]
        sync_shard_sync_sliver_total: IntCounterVec["shard", "sliver_type"],

        #[help = "The progress of the shard sync."]
        sync_shard_sync_sliver_progress: IntGaugeVec["shard", "sliver_type"],

        #[help = "Total number of slivers pending recovery during shard sync"]
        sync_shard_recover_sliver_pending_total: IntGaugeVec["shard"],

        #[help = "Total number of slivers started recovery during shard sync"]
        sync_shard_recover_sliver_total: IntCounterVec["shard", "sliver_type"],

        #[help = "Total number of slivers successfully recovered during shard sync"]
        sync_shard_recover_sliver_success_total: IntCounterVec["shard", "sliver_type"],

        #[help = "Total number of slivers failed to recover during shard sync"]
        sync_shard_recover_sliver_error_total: IntCounterVec["shard", "sliver_type"],

        #[help = "Total number of slivers skipped during shard sync"]
        sync_shard_recover_sliver_skip_total: IntCounterVec["shard", "sliver_type"],

        #[help = "Total number of cancelled sliver recoveries during shard sync"]
        sync_shard_recover_sliver_cancellation_total: IntCounterVec["shard", "sliver_type"],

        #[help = "The total number of slivers stored"]
        slivers_stored_total: IntCounterVec["sliver_type"],

        #[help = "Total number of sliver instances returned"]
        slivers_retrieved_total: IntCounterVec["sliver_type"],

        #[help = "The number of Walrus events processed"]
        event_cursor_progress: U64GaugeVec["state"],

        #[help = "The number of blob recoveries currently pending"]
        recover_blob_backlog: IntGaugeVec["state"],

        #[help = "Time (in seconds) spent processing events"]
        event_process_duration_seconds: HistogramVec["event_type"],

        #[help = "Time (in seconds) spent recovering blobs"]
        recover_blob_duration_seconds: HistogramVec {
            labels: ["status"],
            buckets: default_buckets_for_slow_operations(),
        },

        #[help = "Time (in seconds) spent recovering metadata or slivers of blobs"]
        recover_blob_part_duration_seconds: HistogramVec {
            labels: ["part", "status"],
            buckets: default_buckets_for_slow_operations(),
        },

        #[help = "Unencoded size (in bytes) of the blob associated with uploaded metadata"]
        uploaded_metadata_unencoded_blob_bytes: Histogram {
            // Buckets from 2^18 (256 KiB) to 2^34 (16 GiB, inclusive), which covers the ~212 KiB to
            // 14 GiB unencoded blobs which are possible in a system with 1000 shards.
            buckets: (18..=34).map(|power| (1u64 << power) as f64).collect::<Vec<_>>()
        },

        #[help = "Indicates the current node status"]
        current_node_status: IntGauge[],

        #[help = "The number of blob metadata synced"]
        sync_blob_metadata_count: IntCounter[],

        #[help = "The number of blob metadata skipped"]
        sync_blob_metadata_skipped: IntCounter[],

        #[help = "The progress of the blob metadata sync. It is represented by the first two bytes \
        of the blob ID since the sync job is sequential over blob IDs."]
        sync_blob_metadata_progress: IntGauge[],

        #[help = "For checking consistency of processed events. Each bucket maps to a recent \
        recording of event source. The event source is the combination of checkpoint sequence \
        number and counter."]
        periodic_event_source_for_deterministic_events: IntGaugeVec["bucket"],

        #[help = "The hash of the list of certified blobs at the beginning of the epoch. Note that \
        the label is the last two digits of the epoch number."]
        blob_info_consistency_check: IntGaugeVec["epoch"],

        #[help = "The number of errors occurred when checking the consistency of the blob info \
        table."]
        blob_info_consistency_check_error: IntCounter[],

        #[help = "The number of certified blobs scanned during the blob info consistency check."]
        blob_info_consistency_check_certified_scanned: IntCounterVec["epoch"],

        #[help = "Status metric indicating the node's ID"]
        node_id: IntGaugeVec["walrus_node_id"],
    }
}

/// Returns 14 buckets from ~31 ms to 256 seconds (~4m 15s).
///
/// As prometheus includes a bucket to +Inf, values over 256 seconds are still counted.
/// The number of buckets was chosen to be consistent with the default number of buckets in
/// histograms created by Prometheus.
fn default_buckets_for_slow_operations() -> Vec<f64> {
    prometheus::exponential_buckets(0.03125, 2.0, 14).expect("count, start, and factor are valid")
}

walrus_utils::metrics::define_metric_set! {
    #[namespace = "walrus"]
    /// Metrics exported by the default committee service.
    pub(crate) struct CommitteeServiceMetricSet {
        current_epoch: CurrentEpochMetric,
        current_epoch_state: CurrentEpochStateMetric,

        #[help = "The number shards currently owned by this node"]
        shards_owned: U64Gauge[],
    }
}

pub(crate) trait TelemetryLabel {
    fn label(&self) -> &'static str;
}

impl TelemetryLabel for BlobEvent {
    fn label(&self) -> &'static str {
        match self {
            BlobEvent::Registered(_) => "registered",
            BlobEvent::Certified(event) => event.label(),
            BlobEvent::Deleted(_) => "deleted",
            BlobEvent::InvalidBlobID(_) => "invalid-blob",
            BlobEvent::DenyListBlobDeleted(_) => "deny-list-deleted",
        }
    }
}

impl TelemetryLabel for EpochChangeEvent {
    fn label(&self) -> &'static str {
        match self {
            EpochChangeEvent::EpochParametersSelected(_) => "epoch-parameters-selected",
            EpochChangeEvent::EpochChangeStart(_) => "epoch-change-start",
            EpochChangeEvent::EpochChangeDone(_) => "epoch-change-done",
            EpochChangeEvent::ShardsReceived(_) => "shards-received",
            EpochChangeEvent::ShardRecoveryStart(_) => "shard-recovery-start",
        }
    }
}

impl TelemetryLabel for PackageEvent {
    fn label(&self) -> &'static str {
        match self {
            PackageEvent::ContractUpgraded(_) => "contract-upgraded",
            PackageEvent::ContractUpgradeProposed(_) => "contract-upgrade-proposed",
            PackageEvent::ContractUpgradeQuorumReached(_) => "contract-upgrade-quorum-reached",
            _ => "unknown-package-event",
        }
    }
}

impl TelemetryLabel for DenyListEvent {
    fn label(&self) -> &'static str {
        match self {
            DenyListEvent::DenyListUpdate(_) => "deny-list-updated",
            DenyListEvent::RegisterDenyListUpdate(_) => "register-deny-list-update",
        }
    }
}

impl TelemetryLabel for ContractEvent {
    fn label(&self) -> &'static str {
        match self {
            ContractEvent::BlobEvent(event) => event.label(),
            ContractEvent::EpochChangeEvent(event) => event.label(),
            ContractEvent::PackageEvent(event) => event.label(),
            ContractEvent::DenyListEvent(event) => event.label(),
        }
    }
}

impl TelemetryLabel for EventStreamElement {
    fn label(&self) -> &'static str {
        match self {
            EventStreamElement::ContractEvent(event) => event.label(),
            EventStreamElement::CheckpointBoundary => "end-of-checkpoint",
        }
    }
}

impl TelemetryLabel for BlobCertified {
    fn label(&self) -> &'static str {
        "certified"
    }
}

impl TelemetryLabel for ClientErrorKind {
    fn label(&self) -> &'static str {
        match self {
            ClientErrorKind::CertificationFailed(_) => "certification-failed",
            ClientErrorKind::NotEnoughConfirmations(_, _) => "not-enough-confirmations",
            ClientErrorKind::NotEnoughSlivers => "not-enough-slivers",
            ClientErrorKind::BlobIdDoesNotExist => "blob-id-does-not-exist",
            ClientErrorKind::NoMetadataReceived => "no-metadata-received",
            ClientErrorKind::NoValidStatusReceived => "no-valid-status-received",
            ClientErrorKind::InvalidConfig => "invalid-config",
            ClientErrorKind::BlobIdBlocked(_) => "blob-id-blocked",
            ClientErrorKind::NoCompatiblePaymentCoin => "no-compatible-payment-coin",
            ClientErrorKind::NoCompatibleGasCoins(_, _, _) => "no-compatible-gas-coins",
            ClientErrorKind::AllConnectionsFailed(_) => "all-connections-failed",
            ClientErrorKind::BehindCurrentEpoch { .. } => "behind-current-epoch",
            ClientErrorKind::UnsupportedEncodingType(_) => "unsupported-encoding-type",
            ClientErrorKind::CommitteeChangeNotified => "committee-change-notified",
            ClientErrorKind::StakeBelowThreshold(_) => "stake-below-threshold",
            ClientErrorKind::FailedToLoadCerts(_) => "failed-to-load-certs",
            ClientErrorKind::Other(_) => "unknown",
        }
    }
}
