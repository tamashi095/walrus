// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Infrastructure for retrying RPC calls with backoff, in case there are network errors.
//!
//! Wraps the [`SuiClient`] to introduce retries.
use std::{collections::BTreeMap, fmt, str::FromStr, sync::Arc, time::Duration};

use anyhow::Context;
use futures::{future, stream, Stream, StreamExt};
use rand::{
    rngs::{StdRng, ThreadRng},
    Rng as _,
};
use serde::{de::DeserializeOwned, Serialize};
use sui_sdk::{
    rpc_types::{
        Balance,
        Coin,
        DryRunTransactionBlockResponse,
        ObjectsPage,
        SuiCommittee,
        SuiEvent,
        SuiMoveNormalizedModule,
        SuiMoveNormalizedType,
        SuiObjectDataOptions,
        SuiObjectResponse,
        SuiObjectResponseQuery,
        SuiRawData,
        SuiTransactionBlockEffectsAPI,
        SuiTransactionBlockResponse,
        SuiTransactionBlockResponseOptions,
    },
    wallet_context::WalletContext,
    SuiClient,
    SuiClientBuilder,
};
#[cfg(msim)]
use sui_types::transaction::TransactionDataAPI;
use sui_types::{
    base_types::{ObjectID, SuiAddress, TransactionDigest},
    dynamic_field::derive_dynamic_field_id,
    quorum_driver_types::ExecuteTransactionRequestType::WaitForLocalExecution,
    sui_serde::BigInt,
    transaction::{Transaction, TransactionData, TransactionKind},
    TypeTag,
};
use tracing::Level;
use walrus_core::ensure;
use walrus_utils::backoff::{ExponentialBackoff, ExponentialBackoffConfig};

use super::{
    failover::{FailoverError, LazyClientBuilder},
    retry_rpc_errors,
    FailoverWrapper,
    SuiClientError,
    SuiClientResult,
    GAS_SAFE_OVERHEAD,
    MULTI_GET_OBJ_LIMIT,
};
use crate::{
    client::SuiClientMetricSet,
    contracts::{self, AssociatedContractStruct, TypeOriginMap},
    types::move_structs::{Key, Subsidies, SuiDynamicField, SystemObjectForDeserialization},
    utils::get_sui_object_from_object_response,
};

/// The maximum gas allowed in a transaction, in MIST (50 SUI). Used for gas budget estimation.
const MAX_GAS_BUDGET: u64 = 50_000_000_000;

/// [`LazySuiClientBuilder`] has enough information to create a [`SuiClient`], when its
/// [`LazyClientBuilder`] trait implementation is used.
#[derive(Clone)]
pub enum LazySuiClientBuilder {
    /// A client that is built dynamically from a URL.
    Url {
        /// The URL of the RPC server.
        rpc_url: String,
        /// Override the default timeout for any requests.
        request_timeout: Option<Duration>,
    },
    /// Use a pre-existing client.
    Client(Arc<SuiClient>),
}

impl fmt::Debug for LazySuiClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url {
                rpc_url,
                request_timeout,
            } => f
                .debug_tuple("Url")
                .field(rpc_url)
                .field(request_timeout)
                .finish(),
            Self::Client(_) => f.debug_tuple("Client").finish(),
        }
    }
}

impl From<SuiClient> for LazySuiClientBuilder {
    fn from(client: SuiClient) -> Self {
        Self::Client(Arc::new(client))
    }
}

impl LazySuiClientBuilder {
    /// Creates a new [`LazySuiClientBuilder`] from a URL and an optional `request_timeout`.
    pub fn new(rpc_url: impl AsRef<str>, request_timeout: Option<Duration>) -> Self {
        Self::Url {
            rpc_url: rpc_url.as_ref().to_string(),
            request_timeout,
        }
    }
}

impl LazyClientBuilder<SuiClient> for LazySuiClientBuilder {
    async fn lazy_build_client(&self) -> Result<Arc<SuiClient>, FailoverError> {
        match self {
            Self::Client(client) => Ok(client.clone()),
            Self::Url {
                rpc_url,
                request_timeout: Some(request_timeout),
            } => Ok(Arc::new(
                SuiClientBuilder::default()
                    .request_timeout(*request_timeout)
                    .build(rpc_url)
                    .await
                    .map_err(|e| FailoverError::FailedToGetClient(e.to_string()))?,
            )),
            Self::Url {
                rpc_url,
                request_timeout: None,
            } => Ok(Arc::new(
                SuiClientBuilder::default()
                    .build(rpc_url)
                    .await
                    .map_err(|e| FailoverError::FailedToGetClient(e.to_string()))?,
            )),
        }
    }
    fn get_rpc_url(&self) -> Option<&str> {
        match self {
            Self::Url { rpc_url, .. } => Some(rpc_url.as_str()),
            Self::Client(_) => None,
        }
    }
}

/// A [`SuiClient`] that retries RPC calls with backoff in case of network errors.
///
/// This retriable client wraps functions from the [`CoinReadApi`][sui_sdk::apis::CoinReadApi] and
/// the [`ReadApi`][sui_sdk::apis::ReadApi] of the [`SuiClient`], and
/// additionally provides some convenience methods.
#[allow(missing_debug_implementations)]
#[derive(Clone)]
pub struct RetriableSuiClient {
    failover_sui_client: FailoverWrapper<SuiClient, LazySuiClientBuilder>,
    backoff_config: ExponentialBackoffConfig,
    metrics: Option<Arc<SuiClientMetricSet>>,
}

impl RetriableSuiClient {
    /// Creates a new retriable client.
    ///
    /// NB: If you are creating the sui client from a wallet context, you should use
    /// [`RetriableSuiClient::new_from_wallet`] instead. This is because the wallet context will
    /// make a call to the RPC server in [`WalletContext::get_client`], which may fail without any
    /// retries. `new_from_wallet` will handle this case correctly.
    pub async fn new(
        lazy_client_builders: Vec<LazySuiClientBuilder>,
        backoff_config: ExponentialBackoffConfig,
    ) -> anyhow::Result<Self> {
        Ok(RetriableSuiClient {
            failover_sui_client: FailoverWrapper::new(lazy_client_builders)
                .await
                .context("creating failover wrapper")?,
            backoff_config,
            metrics: None,
        })
    }

    /// Sets the metrics for the client.
    pub fn with_metrics(mut self, metrics: Option<Arc<SuiClientMetricSet>>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Returns a reference to the inner backoff configuration.
    pub fn backoff_config(&self) -> &ExponentialBackoffConfig {
        &self.backoff_config
    }

    /// Creates a new retriable client from an RCP address.
    pub async fn new_for_rpc_urls<S: AsRef<str>>(
        rpc_addresses: &[S],
        backoff_config: ExponentialBackoffConfig,
        request_timeout: Option<Duration>,
    ) -> SuiClientResult<Self> {
        let failover_clients = rpc_addresses
            .iter()
            .map(|rpc_url| LazySuiClientBuilder::new(rpc_url, request_timeout))
            .collect::<Vec<_>>();
        Ok(Self::new(failover_clients, backoff_config).await?)
    }

    /// Creates a new retriable client from a wallet context.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn new_from_wallet(
        wallet: &WalletContext,
        backoff_config: ExponentialBackoffConfig,
    ) -> SuiClientResult<Self> {
        let strategy = backoff_config.get_strategy(ThreadRng::default().gen());
        let client = retry_rpc_errors(
            strategy,
            || async { wallet.get_client().await },
            None,
            "get_client",
        )
        .await?;
        Ok(Self::new(vec![client.into()], backoff_config).await?)
    }

    /// Creates a new retriable client from a wallet context with metrics.
    pub async fn new_from_wallet_with_metrics(
        wallet: &WalletContext,
        backoff_config: ExponentialBackoffConfig,
        metrics: Arc<SuiClientMetricSet>,
    ) -> SuiClientResult<Self> {
        let client = Self::new_from_wallet(wallet, backoff_config).await?;
        Ok(client.with_metrics(Some(metrics)))
    }
    // Reimplementation of the `SuiClient` methods.

    /// Return a list of coins for the given address, or an error upon failure.
    ///
    /// Reimplements the functionality of [`sui_sdk::apis::CoinReadApi::select_coins`] with the
    /// addition of retries on network errors.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn select_coins(
        &self,
        address: SuiAddress,
        coin_type: Option<String>,
        amount: u128,
        exclude: Vec<ObjectID>,
    ) -> SuiClientResult<Vec<Coin>> {
        retry_rpc_errors(
            self.get_strategy(),
            || async {
                self.select_coins_inner(address, coin_type.clone(), amount, exclude.clone())
                    .await
            },
            self.metrics.clone(),
            "select_coins",
        )
        .await
    }

    /// Returns a list of coins for the given address, or an error upon failure.
    ///
    /// This is a reimplementation of the [`sui_sdk::apis::CoinReadApi::select_coins`] method, but
    /// using [`get_coins_stream_retry`] to handle retriable failures.
    async fn select_coins_inner(
        &self,
        address: SuiAddress,
        coin_type: Option<String>,
        amount: u128,
        exclude: Vec<ObjectID>,
    ) -> SuiClientResult<Vec<Coin>> {
        let mut total = 0u128;
        let coins = self
            .get_coins_stream_retry(address, coin_type)
            .filter(|coin: &Coin| future::ready(!exclude.contains(&coin.coin_object_id)))
            .take_while(|coin: &Coin| {
                let ready = future::ready(total < amount);
                total += coin.balance as u128;
                ready
            })
            .collect::<Vec<_>>()
            .await;

        if total < amount {
            return Err(sui_sdk::error::Error::InsufficientFund { address, amount }.into());
        }
        Ok(coins)
    }

    /// Returns a stream of coins for the given address.
    ///
    /// This is a reimplementation of the [`sui_sdk::apis::CoinReadApi:::get_coins_stream`] method
    /// in the `SuiClient` struct. Unlike the original implementation, this version will retry
    /// failed RPC calls.
    fn get_coins_stream_retry(
        &self,
        owner: SuiAddress,
        coin_type: Option<String>,
    ) -> impl Stream<Item = Coin> + '_ {
        stream::unfold(
            (
                vec![],
                /* cursor */ None,
                /* has_next_page */ true,
                coin_type,
            ),
            move |(mut data, cursor, has_next_page, coin_type)| async move {
                if let Some(item) = data.pop() {
                    Some((item, (data, cursor, has_next_page, coin_type)))
                } else if has_next_page {
                    let page = self
                        .failover_sui_client
                        .with_failover(
                            async |client, method| {
                                retry_rpc_errors(
                                    self.get_strategy(),
                                    || async {
                                        client
                                            .coin_read_api()
                                            .get_coins(
                                                owner,
                                                coin_type.clone(),
                                                cursor.clone(),
                                                Some(100),
                                            )
                                            .await
                                            .map_err(SuiClientError::from)
                                    },
                                    self.metrics.clone(),
                                    method,
                                )
                                .await
                                .inspect_err(|error| {
                                    tracing::warn!(%error,
                                            "failed to get coins after retries")
                                })
                            },
                            None,
                            "get_coins",
                        )
                        .await
                        .ok()?;

                    let mut data = page.data;
                    data.reverse();
                    data.pop().map(|item| {
                        (
                            item,
                            (data, page.next_cursor, page.has_next_page, coin_type),
                        )
                    })
                } else {
                    None
                }
            },
        )
    }

    /// Returns the balance for the given coin type owned by address.
    ///
    /// Calls [`sui_sdk::apis::CoinReadApi::get_balance`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn get_balance(
        &self,
        owner: SuiAddress,
        coin_type: Option<String>,
    ) -> SuiClientResult<Balance> {
        self.failover_sui_client
            .with_failover(
                async |client, method| {
                    Ok(retry_rpc_errors(
                        self.get_strategy(),
                        || async {
                            client
                                .coin_read_api()
                                .get_balance(owner, coin_type.clone())
                                .await
                        },
                        self.metrics.clone(),
                        method,
                    )
                    .await?)
                },
                None,
                "get_balance",
            )
            .await
    }

    /// Return a paginated response with the objects owned by the given address.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_owned_objects`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn get_owned_objects(
        &self,
        address: SuiAddress,
        query: Option<SuiObjectResponseQuery>,
        cursor: Option<ObjectID>,
        limit: Option<usize>,
    ) -> SuiClientResult<ObjectsPage> {
        async fn make_request(
            client: Arc<SuiClient>,
            address: SuiAddress,
            query: Option<SuiObjectResponseQuery>,
            cursor: Option<ObjectID>,
            limit: Option<usize>,
        ) -> SuiClientResult<ObjectsPage> {
            Ok(client
                .read_api()
                .get_owned_objects(address, query, cursor, limit)
                .await?)
        }
        let request = |client: Arc<SuiClient>, method: &'static str| {
            let query = query.clone();
            retry_rpc_errors(
                self.get_strategy(),
                move || {
                    let query = query.clone();
                    make_request(client.clone(), address, query, cursor, limit)
                },
                self.metrics.clone(),
                method,
            )
        };

        self.failover_sui_client
            .with_failover(request, None, "get_owned_objects")
            .await
    }

    /// Returns a [`SuiObjectResponse`] based on the provided [`ObjectID`].
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_object_with_options`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn get_object_with_options(
        &self,
        object_id: ObjectID,
        options: SuiObjectDataOptions,
    ) -> SuiClientResult<SuiObjectResponse> {
        async fn make_request(
            client: Arc<SuiClient>,
            object_id: ObjectID,
            options: SuiObjectDataOptions,
        ) -> SuiClientResult<SuiObjectResponse> {
            walrus_utils::crumb!();
            Ok(client
                .read_api()
                .get_object_with_options(object_id, options.clone())
                .await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            let options = options.clone();
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), object_id, options.clone()),
                self.metrics.clone(),
                method,
            )
        };
        walrus_utils::crumb!();
        self.failover_sui_client
            .with_failover(request, None, "get_object_with_options")
            .await
    }

    /// Returns a [`SuiTransactionBlockResponse`] based on the provided [`TransactionDigest`].
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_transaction_with_options`] internally.
    pub async fn get_transaction_with_options(
        &self,
        digest: TransactionDigest,
        options: SuiTransactionBlockResponseOptions,
    ) -> SuiClientResult<SuiTransactionBlockResponse> {
        async fn make_request(
            client: Arc<SuiClient>,
            digest: TransactionDigest,
            options: SuiTransactionBlockResponseOptions,
        ) -> SuiClientResult<SuiTransactionBlockResponse> {
            Ok(client
                .read_api()
                .get_transaction_with_options(digest, options.clone())
                .await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            let options = options.clone();
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), digest, options.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, "get_transaction")
            .await
    }

    /// Return a list of [SuiObjectResponse] from the given vector of [ObjectID]s.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::multi_get_object_with_options`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn multi_get_object_with_options(
        &self,
        object_ids: Vec<ObjectID>,
        options: SuiObjectDataOptions,
    ) -> SuiClientResult<Vec<SuiObjectResponse>> {
        async fn make_request(
            client: Arc<SuiClient>,
            object_ids: Vec<ObjectID>,
            options: SuiObjectDataOptions,
        ) -> SuiClientResult<Vec<SuiObjectResponse>> {
            Ok(client
                .read_api()
                .multi_get_object_with_options(object_ids.clone(), options.clone())
                .await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            let object_ids = object_ids.clone();
            let options = options.clone();
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), object_ids.clone(), options.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, "multi_get_object")
            .await
    }

    /// Returns a map consisting of the move package name and the normalized module.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_normalized_move_modules_by_package`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn get_normalized_move_modules_by_package(
        &self,
        package_id: ObjectID,
    ) -> SuiClientResult<BTreeMap<String, SuiMoveNormalizedModule>> {
        async fn make_request(
            client: Arc<SuiClient>,
            package_id: ObjectID,
        ) -> SuiClientResult<BTreeMap<String, SuiMoveNormalizedModule>> {
            Ok(client
                .read_api()
                .get_normalized_move_modules_by_package(package_id)
                .await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), package_id),
                self.metrics.clone(),
                method,
            )
        };

        self.failover_sui_client
            .with_failover(request, None, "get_normalized_move_modules_by_package")
            .await
    }

    /// Returns the committee information for the given epoch.
    ///
    /// Calls [`sui_sdk::apis::GovernanceApi::get_committee_info`] internally.
    pub async fn get_committee_info(
        &self,
        epoch: Option<BigInt<u64>>,
    ) -> SuiClientResult<SuiCommittee> {
        async fn make_request(
            client: Arc<SuiClient>,
            epoch: Option<BigInt<u64>>,
        ) -> SuiClientResult<SuiCommittee> {
            Ok(client.governance_api().get_committee_info(epoch).await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), epoch),
                self.metrics.clone(),
                method,
            )
        };

        self.failover_sui_client
            .with_failover(request, None, "get_committee_info")
            .await
    }

    /// Returns the reference gas price.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_reference_gas_price`] internally.
    pub async fn get_reference_gas_price(&self) -> SuiClientResult<u64> {
        async fn make_request(client: Arc<SuiClient>) -> SuiClientResult<u64> {
            Ok(client.read_api().get_reference_gas_price().await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, "get_reference_gas_price")
            .await
    }

    /// Executes a transaction dry run.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::dry_run_transaction_block`] internally.
    pub async fn dry_run_transaction_block(
        &self,
        transaction: TransactionData,
    ) -> SuiClientResult<DryRunTransactionBlockResponse> {
        let transaction = Arc::new(transaction);
        async fn make_request(
            client: Arc<SuiClient>,
            transaction: Arc<TransactionData>,
        ) -> SuiClientResult<DryRunTransactionBlockResponse> {
            let tx = TransactionData::clone(&transaction);
            Ok(client.read_api().dry_run_transaction_block(tx).await?)
        }

        let request = move |client: Arc<SuiClient>, method| {
            let transaction = transaction.clone();
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), transaction.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, "dry_run_transaction_block")
            .await
    }

    /// Returns a reference to the current internal [`SuiClient`]. Avoid using this function.
    ///
    // TODO: WAL-778 find callsites to this method, and replace them with implementations that make
    // use of failover, since this call is a cheat to bypass the failover mechanism.
    #[deprecated(
        note = "please implement a full treatment in RetriableSuiClient for your use case"
    )]
    pub async fn get_current_client(&self) -> Arc<SuiClient> {
        self.failover_sui_client.get_current_client().await
    }

    /// Returns a [`SuiObjectResponse`] based on the provided [`ObjectID`].
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_object_with_options`] internally.
    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub async fn get_sui_object<U>(&self, object_id: ObjectID) -> SuiClientResult<U>
    where
        U: AssociatedContractStruct,
    {
        walrus_utils::crumb!();
        let sui_object_response = self
            .get_object_with_options(
                object_id,
                SuiObjectDataOptions::new().with_bcs().with_type(),
            )
            .await?;
        walrus_utils::crumb!();
        get_sui_object_from_object_response(&sui_object_response)
    }

    /// Returns the chain identifier.
    ///
    /// Calls [`sui_sdk::apis::ReadApi::get_chain_identifier`] internally.
    pub async fn get_chain_identifier(&self) -> SuiClientResult<String> {
        async fn make_request(client: Arc<SuiClient>) -> SuiClientResult<String> {
            Ok(client.read_api().get_chain_identifier().await?)
        }
        let request = move |client: Arc<SuiClient>, method| {
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, "get_chain_identifier")
            .await
    }

    // Other wrapper methods.

    #[tracing::instrument(level = Level::DEBUG, skip_all)]
    pub(crate) async fn get_sui_objects<U>(
        &self,
        object_ids: &[ObjectID],
    ) -> SuiClientResult<Vec<U>>
    where
        U: AssociatedContractStruct,
    {
        let mut responses = vec![];
        for obj_id_batch in object_ids.chunks(MULTI_GET_OBJ_LIMIT) {
            responses.extend(
                self.multi_get_object_with_options(
                    obj_id_batch.to_vec(),
                    SuiObjectDataOptions::new().with_bcs().with_type(),
                )
                .await?,
            );
        }

        responses
            .iter()
            .map(|r| get_sui_object_from_object_response(r))
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) async fn get_extended_field<V>(
        &self,
        object_id: ObjectID,
        type_origin_map: &TypeOriginMap,
    ) -> SuiClientResult<V>
    where
        V: DeserializeOwned,
    {
        let key_tag = contracts::extended_field::Key
            .to_move_struct_tag_with_type_map(type_origin_map, &[])?;
        self.get_dynamic_field::<Key, V>(object_id, key_tag.into(), Key { dummy_field: false })
            .await
    }

    #[allow(unused)]
    pub(crate) async fn get_dynamic_field_object<K, V>(
        &self,
        parent: ObjectID,
        key_type: TypeTag,
        key: K,
    ) -> SuiClientResult<V>
    where
        V: AssociatedContractStruct,
        K: DeserializeOwned + Serialize,
    {
        let key_tag = key_type.to_canonical_string(true);
        let key_tag =
            TypeTag::from_str(&format!("0x2::dynamic_object_field::Wrapper<{}>", key_tag))
                .expect("valid type tag");
        let inner_object_id = self.get_dynamic_field(parent, key_tag, key).await?;
        let inner = self.get_sui_object(inner_object_id).await?;
        Ok(inner)
    }

    pub(crate) async fn get_dynamic_field<K, V>(
        &self,
        parent: ObjectID,
        key_type: TypeTag,
        key: K,
    ) -> SuiClientResult<V>
    where
        K: DeserializeOwned + Serialize,
        V: DeserializeOwned,
    {
        let object_id = derive_dynamic_field_id(
            parent,
            &key_type,
            &bcs::to_bytes(&key).expect("key should be serializable"),
        )
        .map_err(|err| SuiClientError::Internal(err.into()))?;

        let field: SuiDynamicField<K, V> = self.get_sui_object(object_id).await?;
        Ok(field.value)
    }

    /// Checks if the Walrus system object exist on chain and returns the Walrus package ID.
    pub(crate) async fn get_system_package_id_from_system_object(
        &self,
        system_object_id: ObjectID,
    ) -> SuiClientResult<ObjectID> {
        let system_object = self
            .get_sui_object::<SystemObjectForDeserialization>(system_object_id)
            .await?;

        let pkg_id = system_object.package_id;
        Ok(pkg_id)
    }

    /// Checks if the Walrus subsidies object exist on chain and returns the subsidies package ID.
    pub(crate) async fn get_subsidies_package_id_from_subsidies_object(
        &self,
        subsidies_object_id: ObjectID,
    ) -> SuiClientResult<ObjectID> {
        let subsidies_object = self
            .get_sui_object::<Subsidies>(subsidies_object_id)
            .await?;

        let pkg_id = subsidies_object.package_id;
        Ok(pkg_id)
    }

    /// Returns the package ID from the type of the given object.
    ///
    /// Note: This returns the package address from the object type, not the newest package ID.
    pub async fn get_package_id_from_object(
        &self,
        object_id: ObjectID,
    ) -> SuiClientResult<ObjectID> {
        let response = self
            .get_object_with_options(
                object_id,
                SuiObjectDataOptions::default().with_type().with_bcs(),
            )
            .await
            .inspect_err(|error| {
                tracing::debug!(%error, %object_id, "unable to get the object");
            })?;

        let pkg_id =
            crate::utils::get_package_id_from_object_response(&response).inspect_err(|error| {
                tracing::debug!(%error, %object_id, "unable to get the package ID from the object");
            })?;
        Ok(pkg_id)
    }

    /// Gets the type origin map for a given package.
    pub(crate) async fn type_origin_map_for_package(
        &self,
        package_id: ObjectID,
    ) -> SuiClientResult<TypeOriginMap> {
        let Ok(Some(SuiRawData::Package(raw_package))) = self
            .get_object_with_options(
                package_id,
                SuiObjectDataOptions::default().with_type().with_bcs(),
            )
            .await?
            .into_object()
            .map(|object| object.bcs)
        else {
            return Err(SuiClientError::WalrusPackageNotFound(package_id));
        };
        Ok(raw_package
            .type_origin_table
            .into_iter()
            .map(|origin| ((origin.module_name, origin.datatype_name), origin.package))
            .collect())
    }

    /// Retrieves the WAL type from the walrus package by getting the type tag of the `Balance`
    /// in the `StakedWal` Move struct.
    #[tracing::instrument(err, skip(self))]
    pub(crate) async fn wal_type_from_package(
        &self,
        package_id: ObjectID,
    ) -> SuiClientResult<String> {
        let normalized_move_modules = self
            .get_normalized_move_modules_by_package(package_id)
            .await?;

        let staked_wal_struct = normalized_move_modules
            .get("staked_wal")
            .and_then(|module| module.structs.get("StakedWal"))
            .ok_or_else(|| SuiClientError::WalTypeNotFound(package_id))?;
        let principal_field_type = staked_wal_struct.fields.iter().find_map(|field| {
            if field.name == "principal" {
                Some(&field.type_)
            } else {
                None
            }
        });
        let Some(SuiMoveNormalizedType::Struct {
            ref type_arguments, ..
        }) = principal_field_type
        else {
            return Err(SuiClientError::WalTypeNotFound(package_id));
        };
        let wal_type = type_arguments
            .first()
            .ok_or_else(|| SuiClientError::WalTypeNotFound(package_id))?;
        let SuiMoveNormalizedType::Struct {
            address,
            module,
            name,
            ..
        } = wal_type
        else {
            return Err(SuiClientError::WalTypeNotFound(package_id));
        };
        ensure!(
            module == "wal" && name == "WAL",
            SuiClientError::WalTypeNotFound(package_id)
        );
        let wal_type = format!("{address}::{module}::{name}");

        tracing::debug!(?wal_type, "WAL type");
        Ok(wal_type)
    }

    /// Calls a dry run with the transaction data to estimate the gas budget.
    ///
    /// This performs the same calculation as the Sui CLI and the TypeScript SDK.
    pub(crate) async fn estimate_gas_budget(
        &self,
        signer: SuiAddress,
        kind: TransactionKind,
        gas_price: u64,
    ) -> SuiClientResult<u64> {
        let dry_run_tx_data = self
            .failover_sui_client
            .get_current_client()
            .await
            .transaction_builder()
            .tx_data_for_dry_run(signer, kind, MAX_GAS_BUDGET, gas_price, None, None)
            .await;
        let effects = self
            .dry_run_transaction_block(dry_run_tx_data)
            .await
            .inspect_err(|error| {
                tracing::debug!(%error, "transaction dry run failed");
            })?
            .effects;
        let gas_cost_summary = effects.gas_cost_summary();

        let safe_overhead = GAS_SAFE_OVERHEAD * gas_price;
        let computation_cost_with_overhead = gas_cost_summary.computation_cost + safe_overhead;
        let gas_usage_with_overhead = gas_cost_summary.net_gas_usage() + safe_overhead as i64;
        Ok(computation_cost_with_overhead.max(gas_usage_with_overhead.max(0) as u64))
    }

    /// Executes a transaction.
    #[tracing::instrument(err, skip(self))]
    pub(crate) async fn execute_transaction(
        &self,
        transaction: Transaction,
        method: &'static str,
    ) -> SuiClientResult<SuiTransactionBlockResponse> {
        async fn make_request(
            client: Arc<SuiClient>,
            transaction: Transaction,
        ) -> SuiClientResult<SuiTransactionBlockResponse> {
            #[cfg(msim)]
            {
                maybe_return_injected_error_in_stake_pool_transaction(&transaction)?;
            }
            Ok(client
                .quorum_driver_api()
                .execute_transaction_block(
                    transaction.clone(),
                    SuiTransactionBlockResponseOptions::new()
                        .with_effects()
                        .with_input()
                        .with_events()
                        .with_object_changes()
                        .with_balance_changes(),
                    Some(WaitForLocalExecution),
                )
                .await?)
        }
        let request = move |client: Arc<SuiClient>, method| {
            let transaction = transaction.clone();
            // Retry here must use the exact same transaction to avoid locked objects.
            retry_rpc_errors(
                self.get_strategy(),
                move || make_request(client.clone(), transaction.clone()),
                self.metrics.clone(),
                method,
            )
        };
        self.failover_sui_client
            .with_failover(request, None, method)
            .await
    }

    /// Gets a backoff strategy, seeded from the internal RNG.
    fn get_strategy(&self) -> ExponentialBackoff<StdRng> {
        self.backoff_config.get_strategy(ThreadRng::default().gen())
    }

    /// Returns the events for the given transaction digest.
    pub async fn get_events(&self, tx_digest: TransactionDigest) -> SuiClientResult<Vec<SuiEvent>> {
        self.failover_sui_client
            .with_failover(
                async |client, method| {
                    retry_rpc_errors(
                        self.get_strategy(),
                        || async { Ok(client.event_api().get_events(tx_digest).await?) },
                        self.metrics.clone(),
                        method,
                    )
                    .await
                },
                None,
                "get_events",
            )
            .await
    }
}

/// Injects a simulated error for testing retry behavior executing sui transactions.
/// We use stake_with_pool as an example here to incorporate with the test logic in
/// `test_ptb_executor_retriable_error` in `test_client.rs`.
#[cfg(msim)]
fn maybe_return_injected_error_in_stake_pool_transaction(
    transaction: &sui_types::transaction::Transaction,
) -> anyhow::Result<()> {
    // Check if this transaction contains a stake_with_pool operation

    use rand::thread_rng;
    use sui_macros::fail_point_if;

    let is_stake_pool_tx =
        transaction
            .transaction_data()
            .move_calls()
            .iter()
            .any(|(_, _, function_name)| {
                *function_name == crate::contracts::staking::stake_with_pool.name
            });

    // Early return if this isn't a stake pool transaction
    if !is_stake_pool_tx {
        return Ok(());
    }

    // Check if we should inject an error via the fail point
    let mut should_inject_error = false;
    fail_point_if!("ptb_executor_stake_pool_retriable_error", || {
        should_inject_error = true;
    });

    if should_inject_error {
        tracing::warn!("injecting a retriable RPC error for stake pool transaction");

        let retriable_error = if thread_rng().gen_bool(0.5) {
            // Simulate a retriable RPC error (502 Bad Gateway).
            jsonrpsee::core::ClientError::Transport(
                "server returned an error status code: 502".into(),
            )
        } else {
            // Simulate a request timeout error.
            jsonrpsee::core::ClientError::RequestTimeout
        };

        Err(sui_sdk::error::Error::RpcError(retriable_error))?;
    }

    Ok(())
}
