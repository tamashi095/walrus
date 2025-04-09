// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Walrus committee service and associated types.

use std::{num::NonZeroU16, sync::Arc};

use async_trait::async_trait;
use walrus_core::{
    encoding::EncodingConfig,
    keys::ProtocolKeyPair,
    merkle::MerkleProof,
    messages::InvalidBlobCertificate,
    metadata::VerifiedBlobMetadataWithId,
    BlobId,
    Epoch,
    InconsistencyProof as InconsistencyProofEnum,
    PublicKey,
    ShardIndex,
    Sliver,
    SliverPairIndex,
    SliverType,
};
use walrus_rest_client::error::ClientBuildError;
use walrus_sui::{
    client::ReadClient,
    types::{Committee, StorageNode},
};

use self::node_service::NodeService;
use crate::common::active_committees::ActiveCommittees;

mod committee_service;
mod node_service;
mod request_futures;

pub(crate) use self::{
    committee_service::NodeCommitteeService,
    node_service::DefaultNodeServiceFactory,
};
use super::errors::SyncShardClientError;

/// Alias to the default type used for recovery symbols.
pub(crate) type DefaultRecoverySymbol = walrus_core::RecoverySymbol<MerkleProof>;

/// Service used to query the current, previous, and next committees.
#[async_trait]
pub(crate) trait CommitteeLookupService: Send + Sync + std::fmt::Debug {
    /// Returns the active committees, which are possibly already transitioning.
    async fn get_active_committees(&self) -> Result<ActiveCommittees, anyhow::Error>;
}

#[async_trait]
impl<T: ReadClient + std::fmt::Debug> CommitteeLookupService for T {
    async fn get_active_committees(&self) -> Result<ActiveCommittees, anyhow::Error> {
        let committees_and_state = self.get_committees_and_state().await?;
        ActiveCommittees::try_from(committees_and_state)
    }
}

/// Errors returned by [`CommitteeService::begin_committee_change`].
#[derive(Debug, thiserror::Error)]
pub enum BeginCommitteeChangeError {
    /// Error returned when the caller requests to begin a committee change to an epoch that is in
    /// the past.
    ///
    /// The caller is lagging in its expectation as to what is the current epoch.
    #[error("the provided epoch is less than the expected epoch: {actual} ({expected})")]
    EpochIsLess {
        /// The expected next epoch based on the state of the committee service.
        expected: Epoch,
        /// The epoch provided by the caller.
        actual: Epoch,
    },
    /// Error returned when the caller requests to begin a committee change to the current epoch.
    #[error("the provided epoch is the same as the current epoch")]
    EpochIsTheSameAsCurrent,
    /// The provided epoch is not 1 greater than the current epoch.
    ///
    /// This indicates that the caller has failed to update the epoch when it should have.
    #[error("the provided epoch skips the next epoch: {actual} ({expected})")]
    EpochIsNotSequential {
        /// The expected epoch based on the state of the committee service.
        expected: Epoch,
        /// The epoch provided by the caller.
        actual: Epoch,
    },
    /// The requested epoch matches the expected epoch, however, we are already transitioning to
    /// that epoch so no change is necessary.
    #[error("the committees are already changing to the specified epoch")]
    ChangeAlreadyInProgress,
    /// The caller's expected epoch matches that stored in the committee service, however, the
    /// latest fetched committee has a different epoch.
    #[error("the epoch of the latest committee differs from the next expected epoch")]
    LatestCommitteeEpochDiffers {
        /// The latest committee retrieved.
        latest_committee: Committee,
        /// The expected epoch based on the state of the committee service.
        expected_epoch: Epoch,
    },
    /// Failed to lookup the committees.
    #[error(transparent)]
    LookupError(anyhow::Error),
    /// Failed to create any service
    #[error(transparent)]
    AllServicesFailed(anyhow::Error),
}

/// Errors returned when completing a committee change.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EndCommitteeChangeError {
    /// The epoch provided by the caller is in the past relative to the committee service.
    #[error(
        "the provided epoch is in the past relative to the current epoch: {provided} < {expected}"
    )]
    ProvidedEpochIsInThePast {
        /// The epoch provided by the caller.
        provided: Epoch,
        /// The epoch expected based on the internal state of the committee service.
        expected: Epoch,
    },
    /// The epoch provided by the caller is in the future relative to the committee service.
    #[error(
        "the provided epoch is in the future relative to the current epoch: {provided} < {expected}"
    )]
    ProvidedEpochIsInTheFuture {
        /// The epoch provided by the caller.
        provided: Epoch,
        /// The epoch expected based on the internal state of the committee service.
        expected: Epoch,
    },
    /// The new epoch matches that expected by the caller, but the change has already completed and
    /// the committee is no longer transitioning.
    #[error("the committee is not currently transitioning")]
    EpochChangeAlreadyDone,
}

/// A `CommitteeService` provides information on the current committee, as well as interactions
/// with committee members.
///
/// It is associated with a single storage epoch.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait CommitteeService: std::fmt::Debug + Send + Sync {
    /// Returns the epoch associated with the committee.
    fn get_epoch(&self) -> Epoch;

    /// Returns the number of shards in the committee.
    fn get_shard_count(&self) -> NonZeroU16;

    /// Returns the active committees.
    fn active_committees(&self) -> ActiveCommittees;

    /// Returns the encoding config associated with the shards in the committee size.
    fn encoding_config(&self) -> &Arc<EncodingConfig>;

    /// Begin the committee transition to the specified epoch.
    ///
    /// This fetches the committees and and ensures that the change is exactly one epoch forward,
    /// and that it matches the new epoch.
    async fn begin_committee_change(
        &self,
        new_epoch: Epoch,
    ) -> Result<(), BeginCommitteeChangeError>;

    /// Ends the current transition of the committee.
    ///
    /// If a transition was in progress with the specified epoch, then it is ended.
    /// Otherwise, an error is returned.
    fn end_committee_change(&self, epoch: Epoch) -> Result<(), EndCommitteeChangeError>;

    /// Update the committee in the node to the latest committee on chain.
    async fn begin_committee_change_to_latest_committee(
        &self,
    ) -> Result<(), BeginCommitteeChangeError>;

    /// Update information about committee members from the latest committee on chain.
    ///
    /// This does not change the committee, but updates the configurations of the nodes.
    /// This is used to update the committee members address, public key, etc, when the
    /// node changes its config.
    async fn sync_committee_members(&self) -> Result<(), anyhow::Error>;

    /// Get and verify metadata.
    async fn get_and_verify_metadata(
        &self,
        blob_id: BlobId,
        certified_epoch: Epoch,
    ) -> VerifiedBlobMetadataWithId;

    /// Recovers a sliver from symbols stored by the committee.
    async fn recover_sliver(
        &self,
        metadata: Arc<VerifiedBlobMetadataWithId>,
        sliver_id: SliverPairIndex,
        sliver_type: SliverType,
        certified_epoch: Epoch,
    ) -> Result<Sliver, InconsistencyProofEnum<MerkleProof>>;

    /// Sends the inconsistency proofs to other nodes and gets a certificate of
    /// the blob's invalidity.
    async fn get_invalid_blob_certificate(
        &self,
        blob_id: BlobId,
        inconsistency_proof: &InconsistencyProofEnum,
    ) -> InvalidBlobCertificate;

    /// Syncs a shard to the given epoch.
    async fn sync_shard_before_epoch(
        &self,
        shard: ShardIndex,
        starting_blob_id: BlobId,
        sliver_type: SliverType,
        sliver_count: u64,
        epoch: Epoch,
        key_pair: &ProtocolKeyPair,
    ) -> Result<Vec<(BlobId, Sliver)>, SyncShardClientError>;

    /// Checks if the given public key belongs to a Walrus storage node.
    fn is_walrus_storage_node(&self, public_key: &PublicKey) -> bool;
}

/// Interface for creating new [`NodeService`]s, such as during epoch change.
#[async_trait]
pub(crate) trait NodeServiceFactory: Send {
    type Service: NodeService;

    async fn make_service(
        &mut self,
        info: &StorageNode,
        encoding_config: &Arc<EncodingConfig>,
    ) -> Result<Self::Service, ClientBuildError>;
}
