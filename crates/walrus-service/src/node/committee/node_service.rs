// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0
//! Services for communicating with Storage Nodes.
//!
//! This module defines the [`NodeService`] trait. It is a marker trait for a [`Clone`]-able
//! [`tower::Service`] trait implementation on the defined [`Request`] and [`Response`] types.
//!
//! It also defines [`RemoteStorageNode`] which implements the trait for
//! [`walrus_rest_client::client::Client`], and implements the trait for [`LocalStorageNode`], an
//! alias to [`Arc<StorageNodeInner>`][StorageNodeInner].
//!
//! The use of [`tower::Service`] will allow us to add layers to monitor a given node's
//! communication with all others, to monitor and disable nodes which fail frequently, and to later
//! apply back-pressure.
//
// NB: Ideally we would *additionally* have a single service trait which would represent the
// storage node. Both clients and servers would implement this trait. This would allow us to treat
// the remote and storage local nodes the same.
use std::{
    fmt::Debug,
    num::NonZero,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use futures::{future::BoxFuture, FutureExt};
use rustls::pki_types::CertificateDer;
use rustls_native_certs::CertificateResult;
use tokio::time::Instant;
use tower::{
    balance::p2c::Balance,
    discover::ServiceList,
    load::{CompleteOnResponse, PendingRequestsDiscover},
    util::BoxCloneSyncService,
    Service,
    ServiceBuilder,
};
use walrus_core::{
    encoding::{EncodingConfig, GeneralRecoverySymbol, Primary, Secondary},
    keys::ProtocolKeyPair,
    messages::InvalidBlobIdAttestation,
    metadata::VerifiedBlobMetadataWithId,
    BlobId,
    Epoch,
    InconsistencyProof as InconsistencyProofEnum,
    PublicKey,
    ShardIndex,
    Sliver,
    SliverIndex,
    SliverPairIndex,
    SliverType,
};
use walrus_rest_client::{
    client::{Client, RecoverySymbolsFilter},
    error::{ClientBuildError, NodeError},
};
use walrus_sui::types::StorageNode as SuiStorageNode;
use walrus_utils::metrics::Registry;

use super::{DefaultRecoverySymbol, NodeServiceFactory};
use crate::node::config::defaults;

/// Requests used with a [`NodeService`].
#[derive(Debug, Clone)]
pub(crate) enum Request {
    GetVerifiedMetadata(BlobId),
    GetVerifiedRecoverySymbol {
        sliver_type: SliverType,
        metadata: Arc<VerifiedBlobMetadataWithId>,
        sliver_pair_at_remote: SliverPairIndex,
        intersecting_pair_index: SliverPairIndex,
    },
    SubmitProofForInvalidBlobAttestation {
        blob_id: BlobId,
        proof: InconsistencyProofEnum,
        epoch: Epoch,
        public_key: PublicKey,
    },
    SyncShardAsOfEpoch {
        shard: ShardIndex,
        starting_blob_id: BlobId,
        sliver_count: u64,
        sliver_type: SliverType,
        current_epoch: Epoch,
        key_pair: ProtocolKeyPair,
    },
    ListVerifiedRecoverySymbols {
        filter: RecoverySymbolsFilter,
        metadata: Arc<VerifiedBlobMetadataWithId>,
        target_index: SliverIndex,
        target_type: SliverType,
    },
}

/// Duration for which to cache native certificates when building multiple clients.
const NATIVE_CERTS_TTL: Duration = Duration::from_secs(60);

/// Responses to [`Request`]s sent to a node service.
///
/// The convenience method [`into_value::<T>()`][Self::into_value] can be used to convert the
/// response into the expected response type.
#[derive(Debug)]
pub(crate) enum Response {
    VerifiedMetadata(VerifiedBlobMetadataWithId),
    VerifiedRecoverySymbol(DefaultRecoverySymbol),
    InvalidBlobAttestation(InvalidBlobIdAttestation),
    ShardSlivers(Vec<(BlobId, Sliver)>),
    VerifiedRecoverySymbols(Vec<GeneralRecoverySymbol>),
}

impl Response {
    /// Convert a response to its inner type and panic if there is a mismatch.
    ///
    /// As responses correspond to the request, this should never panic except in the case of
    /// programmer error.
    pub fn into_value<T>(self) -> T
    where
        T: TryFrom<Self>,
        <T as TryFrom<Self>>::Error: std::fmt::Debug,
    {
        self.try_into()
            .expect("response must be of the correct type")
    }
}

#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("the response variant does not match the expected type")]
pub(crate) struct InvalidResponseVariant;

macro_rules! impl_response_conversion {
    ($type:ty, $($variant:tt)+) => {
        impl TryFrom<Response> for $type {
            type Error = InvalidResponseVariant;

            fn try_from(value: Response) -> Result<Self, Self::Error> {
                if let $($variant)+(inner) = value {
                    Ok(inner)
                } else {
                    Err(InvalidResponseVariant)
                }
            }
        }

        impl From<$type> for Response {
            fn from(value: $type) -> Self {
                $($variant)+(value)
            }
        }
    };
}

impl_response_conversion!(VerifiedBlobMetadataWithId, Response::VerifiedMetadata);
impl_response_conversion!(DefaultRecoverySymbol, Response::VerifiedRecoverySymbol);
impl_response_conversion!(
    Vec<GeneralRecoverySymbol>,
    Response::VerifiedRecoverySymbols
);
impl_response_conversion!(InvalidBlobIdAttestation, Response::InvalidBlobAttestation);
impl_response_conversion!(Vec<(BlobId, Sliver)>, Response::ShardSlivers);

#[derive(Debug, thiserror::Error)]
pub(crate) enum NodeServiceError {
    #[error(transparent)]
    Node(#[from] NodeError),
    #[allow(unused)]
    #[error(transparent)]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl From<Box<dyn std::error::Error + Sync + Send>> for NodeServiceError {
    fn from(value: Box<dyn std::error::Error + Sync + Send>) -> Self {
        match value.downcast::<NodeError>() {
            Ok(node_error) => Self::Node(*node_error),
            Err(other) => Self::Other(other),
        }
    }
}

/// Marker trait for types implementing the [`tower::Service`] signature expected of services
/// used for communication with the committee.
pub(crate) trait NodeService
where
    Self: Send + Clone,
    Self: Service<Request, Response = Response, Error = NodeServiceError, Future: Send>,
{
}

impl<T> NodeService for T
where
    T: Send + Clone,
    T: Service<Request, Response = Response, Error = NodeServiceError, Future: Send>,
{
}

/// A [`NodeService`] that is reachable via a [`walrus_rest_client::client::Client`].
#[derive(Clone, Debug)]
pub(crate) struct RemoteStorageNode {
    client: Client,
    encoding_config: Arc<EncodingConfig>,
}

impl Service<Request> for RemoteStorageNode {
    type Error = NodeServiceError;
    type Response = Response;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let client = self.client.clone();
        let encoding_config = self.encoding_config.clone();
        async move {
            let response = match req {
                Request::GetVerifiedMetadata(blob_id) => client
                    .get_and_verify_metadata(&blob_id, &encoding_config)
                    .await
                    .map(Response::VerifiedMetadata)?,

                Request::GetVerifiedRecoverySymbol {
                    sliver_type,
                    metadata,
                    sliver_pair_at_remote,
                    intersecting_pair_index,
                } => {
                    let symbol = if sliver_type == SliverType::Primary {
                        client
                            .get_and_verify_recovery_symbol::<Primary>(
                                &metadata,
                                &encoding_config,
                                sliver_pair_at_remote,
                                intersecting_pair_index,
                            )
                            .await
                            .map(DefaultRecoverySymbol::Primary)
                    } else {
                        client
                            .get_and_verify_recovery_symbol::<Secondary>(
                                &metadata,
                                &encoding_config,
                                sliver_pair_at_remote,
                                intersecting_pair_index,
                            )
                            .await
                            .map(DefaultRecoverySymbol::Secondary)
                    };
                    symbol.map(Response::VerifiedRecoverySymbol)?
                }

                Request::SubmitProofForInvalidBlobAttestation {
                    blob_id,
                    proof,
                    epoch,
                    public_key,
                } => client
                    .submit_inconsistency_proof_and_verify_attestation(
                        &blob_id,
                        &proof,
                        epoch,
                        &public_key,
                    )
                    .await
                    .map(Response::from)?,

                Request::SyncShardAsOfEpoch {
                    shard,
                    starting_blob_id,
                    sliver_count,
                    sliver_type,
                    current_epoch,
                    key_pair,
                } => {
                    let result = if sliver_type == SliverType::Primary {
                        client
                            .sync_shard::<Primary>(
                                shard,
                                starting_blob_id,
                                sliver_count,
                                current_epoch,
                                &key_pair,
                            )
                            .await
                    } else {
                        client
                            .sync_shard::<Secondary>(
                                shard,
                                starting_blob_id,
                                sliver_count,
                                current_epoch,
                                &key_pair,
                            )
                            .await
                    };
                    result.map(|value| Response::ShardSlivers(value.into()))?
                }

                Request::ListVerifiedRecoverySymbols {
                    filter,
                    metadata,
                    target_index,
                    target_type,
                } => client
                    .list_and_verify_recovery_symbols(
                        filter,
                        metadata.clone(),
                        encoding_config.clone(),
                        target_index,
                        target_type,
                    )
                    .await
                    .map(Response::VerifiedRecoverySymbols)?,
            };
            Ok(response)
        }
        .boxed()
    }
}

// TODO(jsmith): Define a LocalStorageNode that can be used within process.
// Such a service would need to hold only a `Weak` to `StorageNodeInner`, so as to avoid a memory
// leak due to cyclic Arcs.
//
// /// A *trusted* [`NodeService`] that can be communicated with within the process.
// pub(crate) type LocalStorageNode = Weak<StorageNodeInner>;

/// A [`NodeServiceFactory`] creating [`RemoteStorageNode`] services.
#[derive(Debug, Clone)]
pub(crate) struct DefaultNodeServiceFactory {
    disable_use_proxy: bool,
    disable_loading_native_certs: bool,
    connect_timeout: Option<Duration>,
    connections_per_node: NonZero<usize>,
    buffer_length: NonZero<usize>,
    registry: Option<Registry>,
    /// Cached certificates along with the time at which they were loaded.
    native_certs: Option<(Instant, Vec<CertificateDer<'static>>)>,
}

impl Default for DefaultNodeServiceFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultNodeServiceFactory {
    /// Returns a new default instance of the factory.
    pub const fn new() -> Self {
        Self {
            disable_use_proxy: false,
            disable_loading_native_certs: false,
            connect_timeout: None,
            registry: None,
            connections_per_node: defaults::CONNECTIONS_PER_NODE,
            buffer_length: defaults::STORAGE_NODE_REQUEST_QUEUE_LENGTH,
            native_certs: None,
        }
    }

    /// Creates services with metrics written to the provided registry.
    pub fn metrics_registry(&mut self, registry: Registry) -> &mut Self {
        self.registry = Some(registry);
        self
    }

    /// Sets the timeout for connecting to nodes, for all subsequently created nodes.
    pub fn connect_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Skips the use of proxies and the loading of native certificates, as these require
    /// interacting with the operating system and can significantly slow down the construction of
    /// new instances.
    pub fn avoid_system_services(&mut self) -> &mut Self {
        self.disable_use_proxy = true;
        self.disable_loading_native_certs = true;
        self
    }

    /// The number of connections to open to each storage node.
    pub fn connections_per_node(&mut self, count: NonZero<usize>) -> &mut Self {
        self.connections_per_node = count;
        self
    }

    /// The length of the queue for node requests served by the connection pool.
    pub fn storage_node_request_queue_length(&mut self, count: NonZero<usize>) -> &mut Self {
        self.buffer_length = count;
        self
    }

    /// Creates a new client for the specified storage node.
    pub fn build(
        &mut self,
        member: &SuiStorageNode,
        encoding_config: &Arc<EncodingConfig>,
    ) -> Result<BoxCloneSyncService<Request, Response, NodeServiceError>, ClientBuildError> {
        if self.connections_per_node.get() == 1 {
            return Ok(BoxCloneSyncService::new(
                self.build_single_service(member, encoding_config)?,
            ));
        }

        let services: Vec<_> = (0..self.connections_per_node.get())
            .map(|_| self.build_single_service(member, encoding_config))
            .collect::<Result<_, _>>()?;

        let balance = Balance::new(PendingRequestsDiscover::new(
            ServiceList::new(services),
            CompleteOnResponse::default(),
        ));

        let service = ServiceBuilder::default()
            .layer_fn(BoxCloneSyncService::new)
            .map_err(NodeServiceError::from)
            .buffer(self.buffer_length.get())
            .service(balance);

        Ok(service)
    }

    fn build_single_service(
        &mut self,
        member: &SuiStorageNode,
        encoding_config: &Arc<EncodingConfig>,
    ) -> Result<RemoteStorageNode, ClientBuildError> {
        let mut builder = walrus_rest_client::client::Client::builder()
            .authenticate_with_public_key(member.network_public_key.clone());

        if self.disable_loading_native_certs {
            builder = builder.tls_built_in_root_certs(false);
        } else if let Some(certificates) = self.load_native_certs() {
            builder = builder.add_root_certificates(certificates);
            // If the above is None, the client will load the certificates themselves and err if
            // they also fail to load any certificates.
        }

        if self.disable_use_proxy {
            builder = builder.no_proxy();
        }
        if let Some(timeout) = self.connect_timeout.as_ref() {
            builder = builder.connect_timeout(*timeout);
        }
        if let Some(registry) = self.registry.as_ref() {
            builder = builder.metric_registry(registry.clone());
        }

        builder
            .build(&member.network_address.0)
            .map(|client| RemoteStorageNode {
                client,
                encoding_config: encoding_config.clone(),
            })
    }

    /// Loads and returns the native certificates, potentially caching them. Returns `None` if
    /// loading fails or certificates are disabled.
    fn load_native_certs(&mut self) -> Option<&Vec<CertificateDer<'static>>> {
        // Do not provide any certs if loading is disabled.
        if self.disable_loading_native_certs {
            return None;
        }

        // Clear the certificates if they are too old.
        if let Some((load_time, _)) = self.native_certs.as_ref() {
            if load_time.elapsed() > NATIVE_CERTS_TTL {
                self.native_certs = None;
            }
        }

        if self.native_certs.is_none() {
            let CertificateResult { certs, errors, .. } = rustls_native_certs::load_native_certs();
            if !errors.is_empty() {
                tracing::warn!(
                    "encountered {} errors when trying to load native certs",
                    errors.len(),
                );
                tracing::debug!(?errors, "errors encountered when loading native certs");
            }
            if certs.is_empty() {
                tracing::warn!("failed to load any native certificates");
            };
            self.native_certs = Some((Instant::now(), certs));
        }

        self.native_certs.as_ref().map(|(_, certs)| certs)
    }
}

#[async_trait::async_trait]
impl NodeServiceFactory for DefaultNodeServiceFactory {
    type Service = BoxCloneSyncService<Request, Response, NodeServiceError>;

    async fn make_service(
        &mut self,
        member: &SuiStorageNode,
        encoding_config: &Arc<EncodingConfig>,
    ) -> Result<Self::Service, ClientBuildError> {
        self.build(member, encoding_config)
    }
}
