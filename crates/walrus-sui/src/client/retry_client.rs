// Copyright (c) Walrus Foundation
// SPDX-License-Identifier: Apache-2.0

//! Infrastructure for retrying RPC calls with backoff, in case there are network errors.
//!
//! Wraps the [`sui_sdk::SuiClient`] to introduce retries.

use std::{future::Future, sync::Arc, time::Instant};

use retriable_rpc_client::FallbackError;
use walrus_utils::backoff::BackoffStrategy;

pub use self::{
    failover::FailoverWrapper,
    fallible::FallibleRpcClient,
    retriable_rpc_client::{CheckpointRpcError, RetriableClientError, RetriableRpcClient},
    retriable_rpc_error::RetriableRpcError,
    retriable_sui_client::RetriableSuiClient,
};
use super::{SuiClientError, SuiClientResult};
use crate::client::SuiClientMetricSet;

pub mod failover;
pub mod fallible;
pub mod retriable_rpc_client;
pub mod retriable_rpc_error;
pub mod retriable_sui_client;
mod retry_count_guard;

/// The gas overhead to add to the gas budget to ensure that the transaction will succeed.
/// Set based on `GAS_SAFE_OVERHEAD` in the sui CLI. Used for gas budget estimation.
const GAS_SAFE_OVERHEAD: u64 = 1000;

/// The maximum number of objects to get in a single RPC call.
pub(crate) const MULTI_GET_OBJ_LIMIT: usize = 50;

/// Trait to convert an error to a string.
pub trait ToErrorType {
    /// Returns the error type as a string.
    fn to_error_type(&self) -> String;
}

/// Retries the given function while it returns retriable errors.[
async fn retry_rpc_errors<S, F, T, E, Fut>(
    mut strategy: S,
    mut func: F,
    metrics: Option<Arc<SuiClientMetricSet>>,
    method: &'static str,
) -> Result<T, E>
where
    S: BackoffStrategy,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: RetriableRpcError + ToErrorType,
{
    let mut retry_guard = metrics
        .as_ref()
        .map(|m| retry_count_guard::RetryCountGuard::new(m.clone(), method));

    loop {
        let start = Instant::now();
        let value = func().await;

        if let Some(metrics) = &metrics {
            metrics.record_rpc_call(
                method,
                &match value.as_ref() {
                    Ok(_) => "success".to_string(),
                    Err(e) => e.to_error_type(),
                },
                start.elapsed(),
            );
        }

        if let Some(retry_guard) = &mut retry_guard {
            retry_guard.record_result(value.as_ref());
        }

        match value {
            Ok(value) => return Ok(value),
            Err(error) if error.is_retriable_rpc_error() => {
                if let Some(delay) = strategy.next_delay() {
                    tracing::debug!(
                        ?delay,
                        ?error,
                        "attempt failed with retriable RPC error, waiting before retrying"
                    );
                    tokio::time::sleep(delay).await;
                } else {
                    tracing::debug!(
                        "last attempt failed with retriable RPC error, returning last failure value"
                    );
                    return Err(error);
                }
            }
            Err(error) => {
                tracing::debug!("non-retriable error, returning last failure value");
                return Err(error);
            }
        }
    }
}

impl ToErrorType for anyhow::Error {
    fn to_error_type(&self) -> String {
        self.downcast_ref::<sui_sdk::error::Error>()
            .map(|error| error.to_error_type())
            .unwrap_or_else(|| "other".to_string())
    }
}

impl ToErrorType for sui_sdk::error::Error {
    fn to_error_type(&self) -> String {
        match self {
            Self::RpcError(e) => {
                let err_str = e.to_string();
                if err_str.contains("Call") {
                    "rpc_call"
                } else if err_str.contains("Transport") {
                    "rpc_transport"
                } else if err_str.contains("RequestTimeout") {
                    "rpc_timeout"
                } else {
                    "rpc_other"
                }
                .to_string()
            }
            Self::JsonRpcError(_) => "json_rpc".to_string(),
            Self::BcsSerialisationError(_) => "bcs_ser".to_string(),
            Self::JsonSerializationError(_) => "json_ser".to_string(),
            Self::UserInputError(_) => "user_input".to_string(),
            Self::Subscription(_) => "subscription".to_string(),
            Self::FailToConfirmTransactionStatus(_, _) => "tx_confirm".to_string(),
            Self::DataError(_) => "data_error".to_string(),
            Self::ServerVersionMismatch { .. } => "version_mismatch".to_string(),
            Self::InsufficientFund { .. } => "insufficient_fund".to_string(),
            Self::InvalidSignature => "invalid_sig".to_string(),
        }
    }
}

impl ToErrorType for SuiClientError {
    fn to_error_type(&self) -> String {
        match self {
            Self::SuiSdkError(error) => error.to_error_type(),
            _ => "sui_client_error_other".to_string(),
        }
    }
}

impl ToErrorType for tonic::Status {
    fn to_error_type(&self) -> String {
        "other".to_string()
    }
}

impl ToErrorType for RetriableClientError {
    fn to_error_type(&self) -> String {
        match self {
            Self::RpcError(_) => "checkpoint_rpc_error".to_string(),
            Self::FallbackError(_) => "fallback_error".to_string(),
            Self::FailoverError(_) => "failover_error".to_string(),
            Self::Other(_) => "other".to_string(),
            Self::RetryableTimeoutError => "retryable_timeout".to_string(),
            Self::NonRetryableTimeoutError => "non_retryable_timeout".to_string(),
        }
    }
}
