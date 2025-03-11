// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt::{Debug, Display},
    num::NonZeroU16,
    path::PathBuf,
    time::Duration,
};

use eyre::ensure;
use serde::{Deserialize, Serialize};
use walrus_core::ShardIndex;
use walrus_service::{
    node::config,
    testbed::{self, DeployTestbedContractParameters},
};
use walrus_sui::utils::SuiNetwork;

use super::{ProtocolCommands, ProtocolMetrics, ProtocolParameters, BINARY_PATH};
use crate::{benchmark::BenchmarkParameters, client::Instance};

#[derive(Clone, Serialize, Deserialize, Debug)]
enum ShardsAllocation {
    /// Evenly distribute the specified number of shards among the nodes.
    Even(NonZeroU16),
    /// Manually specify the shards for each node.
    Manual(Vec<Vec<ShardIndex>>),
}

impl Default for ShardsAllocation {
    fn default() -> Self {
        Self::Even(NonZeroU16::new(10).unwrap())
    }
}

impl ShardsAllocation {
    fn number_of_shards(&self) -> usize {
        match self {
            Self::Even(n) => n.get() as usize,
            Self::Manual(shards) => shards.iter().map(|s| s.len()).sum::<usize>(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProtocolClientParameters {
    client_config_path: PathBuf,
    wallet_paths: Vec<PathBuf>,
    write_load: u64,
    read_load: u64,
    min_size_log2: usize,
    max_size_log2: usize,
    tasks: usize,
    metrics_port: u16,
}

impl Default for ProtocolClientParameters {
    fn default() -> Self {
        Self {
            client_config_path: PathBuf::from(
                "./crates/walrus-orchestrator/assets/client_config.yaml",
            ),
            wallet_paths: vec![],
            write_load: 60,
            read_load: 60,
            min_size_log2: 23,
            max_size_log2: 24,
            tasks: 10,
            metrics_port: 9584,
        }
    }
}

impl Display for ProtocolClientParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "load: {} writes/min, {} reads/min, 2^{}~2^{} bytes",
            self.write_load, self.read_load, self.min_size_log2, self.max_size_log2
        )
    }
}

impl ProtocolParameters for ProtocolClientParameters {}

pub struct TargetProtocol;

impl ProtocolCommands for TargetProtocol {
    fn protocol_dependencies(&self) -> Vec<&'static str> {
        // Clang is required to compile rocksdb.
        vec!["sudo apt-get -y install clang"]
    }

    fn db_directories(&self) -> Vec<std::path::PathBuf> {
        // The service binary can delete its own storage directory before booting.
        vec![]
    }

    async fn genesis_command<'a, I>(&self, instances: I, parameters: &BenchmarkParameters) -> String
    where
        I: Iterator<Item = &'a Instance>,
    {
        // Generate a command to upload the configs to all instances.
        let client_config_source = &parameters.client_parameters.client_config_path;
        let client_config =
            std::fs::read_to_string(client_config_source).expect("failed to read client config");
        let serialized_client_config =
            serde_yaml::to_string(&client_config).expect("failed to serialize client config");
        let client_config_destination = parameters.settings.working_dir.join("client_config.yaml");
        let upload_client_config_command = format!(
            "echo -e '{serialized_client_config}' > {}",
            client_config_destination.display()
        );

        // Generate a client to upload one wallet per client.
        if instances.collect::<Vec<_>>().len() <= parameters.client_parameters.wallet_paths.len() {
            panic!("Not enough wallets for all instances");
        }

        let mut upload_wallet_commands = Vec::new();
        for (i, wallet_path) in parameters.client_parameters.wallet_paths.iter().enumerate() {
            let wallet = std::fs::read_to_string(wallet_path).expect("failed to read wallet");
            let serialized_wallet =
                serde_yaml::to_string(&wallet).expect("failed to serialize wallet");
            let wallet_destination = parameters
                .settings
                .working_dir
                .join(format!("wallet-{i}.yaml"));
            let upload_wallet_command = format!(
                "echo -e '{serialized_wallet}' > {}",
                wallet_destination.display()
            );
            upload_wallet_commands.push(upload_wallet_command);
        }

        // Output a single command to run on all machines.
        let mut command = vec![
            "source $HOME/.cargo/env".to_string(),
            upload_client_config_command,
        ];
        command.extend(upload_wallet_commands);
        command.join(" && ")
    }

    fn client_command<I>(
        &self,
        instances: I,
        parameters: &BenchmarkParameters,
    ) -> Vec<(Instance, String)>
    where
        I: IntoIterator<Item = Instance>,
    {
        let clients: Vec<_> = instances.into_iter().collect();
        let write_load_per_client =
            (parameters.client_parameters.write_load as usize / clients.len()).max(1);
        let read_load_per_client =
            (parameters.client_parameters.read_load as usize / clients.len()).max(1);

        clients
            .into_iter()
            .enumerate()
            .map(|(i, instance)| {
                let working_dir = &parameters.settings.working_dir;
                let client_config_path = working_dir.clone().join("client_config.yaml");

                let run_command = [
                    format!("./{BINARY_PATH}/walrus-stress"),
                    format!("--write-load {write_load_per_client}"),
                    format!("--read-load {read_load_per_client}"),
                    format!("--config-path {}", client_config_path.display()),
                    format!(
                        "--wallet-path {}",
                        parameters.client_parameters.wallet_paths[i].display()
                    ),
                    format!("--n-clients {}", parameters.client_parameters.tasks),
                    format!(
                        "--metrics-port {}",
                        parameters.client_parameters.metrics_port
                    ),
                    format!(
                        "--min_size_log2 {}",
                        parameters.client_parameters.min_size_log2
                    ),
                    format!(
                        "--max_size_log2 {}",
                        parameters.client_parameters.max_size_log2
                    ),
                ]
                .join(" ");

                let command = ["source $HOME/.cargo/env", &run_command].join(" && ");
                (instance, command)
            })
            .collect()
    }
}

impl ProtocolMetrics for TargetProtocol {
    const BENCHMARK_DURATION: &'static str = "benchmark_duration";
    const TOTAL_TRANSACTIONS: &'static str = "total_transactions";
    const LATENCY_BUCKETS: &'static str = "latency_buckets";
    const LATENCY_SUM: &'static str = "latency_sum";
    const LATENCY_SQUARED_SUM: &'static str = "latency_squared_sum";

    fn clients_metrics_path<I>(
        &self,
        instances: I,
        parameters: &BenchmarkParameters,
    ) -> Vec<(Instance, String)>
    where
        I: IntoIterator<Item = Instance>,
    {
        instances
            .into_iter()
            .map(|instance| {
                let instance_ip = instance.main_ip;
                let metrics_port = parameters.client_parameters.metrics_port;
                let metrics_path = format!("{instance_ip}:{metrics_port}/metrics");
                (instance, metrics_path)
            })
            .collect()
    }
}
