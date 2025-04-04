// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Utilities to refill gas for the stress clients.

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use futures::future::try_join_all;
use sui_sdk::types::base_types::SuiAddress;
use tokio::{task::JoinHandle, time::MissedTickBehavior};
use walrus_sui::client::{retry_client::RetriableSuiClient, SuiContractClient};

use super::metrics::ClientMetrics;

/// Refills gas and WAL for the clients.
#[derive(Debug, Clone)]
pub struct Refiller {
    /// The inner implementation of the refiller.
    contract_client: Arc<SuiContractClient>,
    /// The amount of MIST to send at each request.
    gas_refill_size: u64,
    /// The amount of FROST to send at each request.
    wal_refill_size: u64,
    /// The minimum balance the wallet should have before refilling.
    min_balance: u64,
}

impl Refiller {
    /// Creates a new [`Refiller`] from a [`SuiContractClient`].
    pub fn new(
        contract_client: SuiContractClient,
        gas_refill_size: u64,
        wal_refill_size: u64,
        min_balance: u64,
    ) -> Self {
        Self {
            contract_client: Arc::new(contract_client),
            gas_refill_size,
            wal_refill_size,
            min_balance,
        }
    }

    /// Refills gas and WAL for the clients.
    pub fn refill_gas_and_wal(
        &self,
        addresses: Vec<SuiAddress>,
        period: Duration,
        metrics: Arc<ClientMetrics>,
        sui_client: RetriableSuiClient,
    ) -> RefillHandles {
        let _gas_refill_handle = self.refill_gas(
            addresses.clone(),
            period,
            metrics.clone(),
            sui_client.clone(),
        );
        let _wal_refill_handle =
            self.refill_wal(addresses, period, metrics.clone(), sui_client.clone());

        RefillHandles {
            _gas_refill_handle,
            _wal_refill_handle,
        }
    }

    fn refill_gas(
        &self,
        addresses: Vec<SuiAddress>,
        period: Duration,
        metrics: Arc<ClientMetrics>,
        sui_client: RetriableSuiClient,
    ) -> JoinHandle<anyhow::Result<()>> {
        let amount = self.gas_refill_size;
        self.periodic_refill(
            addresses,
            period,
            sui_client,
            None, // Use SUI
            self.min_balance,
            move |refiller, address| {
                let metrics = metrics.clone();
                async move {
                    refiller.send_sui(amount, address).await?;
                    tracing::info!("clients gas coins refilled");
                    metrics.observe_gas_refill();
                    Ok(())
                }
            },
        )
    }

    fn refill_wal(
        &self,
        addresses: Vec<SuiAddress>,
        period: Duration,
        metrics: Arc<ClientMetrics>,
        sui_client: RetriableSuiClient,
    ) -> JoinHandle<anyhow::Result<()>> {
        let amount = self.wal_refill_size;
        self.periodic_refill(
            addresses,
            period,
            sui_client,
            Some(self.wal_coin_type()),
            self.min_balance,
            move |refiller, address| {
                let metrics = metrics.clone();
                async move {
                    refiller.send_wal(amount, address).await?;
                    tracing::info!("clients WAL coins refilled");
                    metrics.observe_wal_refill();
                    Ok(())
                }
            },
        )
    }

    fn periodic_refill<F, Fut>(
        &self,
        addresses: Vec<SuiAddress>,
        period: Duration,
        sui_client: RetriableSuiClient,
        coin_type: Option<String>,
        min_coin_value: u64,
        inner_action: F,
    ) -> JoinHandle<anyhow::Result<()>>
    where
        F: Fn(Arc<SuiContractClient>, SuiAddress) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
    {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let refiller = self.contract_client.clone();
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                let sui_client = &sui_client;
                let _ = try_join_all(addresses.iter().cloned().map(|address| {
                    let coin_type_inner = coin_type.clone();
                    let inner_fut = inner_action(refiller.clone(), address);
                    async move {
                        if should_refill(
                            sui_client,
                            address,
                            coin_type_inner.clone(),
                            min_coin_value,
                        )
                        .await
                        {
                            inner_fut.await.with_context(|| {
                                format!("failed in periodic_refill of address [addres={}]", address)
                            })
                        } else {
                            Ok(())
                        }
                    }
                }))
                .await
                .inspect_err(|error| {
                    tracing::error!(
                        ?error,
                        "error during periodic refill of coin type {:?}",
                        coin_type
                    )
                });
            }
        })
    }

    /// The WAL coin type.
    pub fn wal_coin_type(&self) -> String {
        self.contract_client
            .read_client()
            .wal_coin_type()
            .to_owned()
    }

    /// Sends SUI to the specified address.
    pub async fn send_gas_request(&self, address: SuiAddress) -> Result<()> {
        Ok(self
            .contract_client
            .send_sui(self.gas_refill_size, address)
            .await?)
    }

    /// Sends WAL to the specified address.
    pub async fn send_wal_request(&self, address: SuiAddress) -> Result<()> {
        Ok(self
            .contract_client
            .send_wal(self.wal_refill_size, address)
            .await?)
    }
}

/// Helper struct to hold the handles for the refiller tasks.
#[derive(Debug)]
pub struct RefillHandles {
    /// The handle for the gas refill task.
    pub _gas_refill_handle: JoinHandle<anyhow::Result<()>>,
    /// The handle for the WAL refill task.
    pub _wal_refill_handle: JoinHandle<anyhow::Result<()>>,
}

/// Checks if the wallet should be refilled.
///
/// The wallet should be refilled if it has less than `MIN_BALANCE` in balance. _However_, if the
/// RPC returns an error, we assume that the wallet has enough coins and we do not try to refill
/// it, threfore returning `false`.
pub async fn should_refill(
    sui_client: &RetriableSuiClient,
    address: SuiAddress,
    coin_type: Option<String>,
    min_balance: u64,
) -> bool {
    sui_client
        .get_balance(address, coin_type)
        .await
        .map(|balance| balance.total_balance < min_balance as u128)
        .inspect_err(|error| {
            tracing::debug!(
                ?error,
                %address,
                "failed checking the balance of the address"
            );
        })
        // If the returned value is an error, we assume that the RPC failed, the
        // address has enough coins, and we do not try to refill.
        .unwrap_or(false)
}
