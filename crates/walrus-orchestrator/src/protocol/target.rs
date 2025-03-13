// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt::{Debug, Display},
    fs::File,
    io::BufReader,
    num::NonZeroU16,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use walrus_core::ShardIndex;

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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProtocolClientParameters {
    client_config_path: PathBuf,
    wallets_dir: PathBuf,
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
            wallets_dir: PathBuf::from("./crates/walrus-orchestrator/assets/"),
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

// NOTE: Ugly hack -- the keystore path in every `sui-wallet-{i}.yaml` must match `parameters.settings.working_dir`.
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
        // Generate a command to upload the contract configs to all instances.
        let client_config_source = &parameters.client_parameters.client_config_path;
        let file = File::open(client_config_source).expect("Failed to open client config file");
        let client_config: serde_yaml::Value =
            serde_yaml::from_reader(BufReader::new(file)).expect("Failed to load client config");
        let serialized_client_config =
            serde_yaml::to_string(&client_config).expect("failed to serialize client config");
        let client_config_destination = parameters.settings.working_dir.join("client_config.yaml");
        let upload_client_config_command = format!(
            "echo -e '{serialized_client_config}' > {}",
            client_config_destination.display()
        );

        // Generate a client to upload one wallet per client.
        let mut upload_file_commands = Vec::new();
        for (i, _) in instances.enumerate() {
            for filename in [
                format!("sui-wallet-{i}.yaml"), //SuiClientConfig
                format!("sui-wallet-{i}.aliases"),
                format!("sui-wallet-{i}.keystore"),
            ] {
                let filepath = parameters.client_parameters.wallets_dir.join(&filename);
                let file = File::open(filepath).expect("Failed to to read wallet files");
                let serialized_file = if filename.ends_with(".yaml") {
                    let config: serde_yaml::Value = serde_yaml::from_reader(BufReader::new(file))
                        .expect("Failed to load client config");
                    serde_yaml::to_string(&config).expect("failed to serialize wallet config file")
                } else {
                    let config: serde_json::Value = serde_json::from_reader(BufReader::new(file))
                        .expect("Failed to load client config");
                    serde_json::to_string(&config).expect("failed to serialize client config")
                };
                let file_destination = parameters.settings.working_dir.join(filename);
                let upload_file_command = format!(
                    "echo -e '{serialized_file}' > {}",
                    file_destination.display()
                );
                upload_file_commands.push(upload_file_command);
            }
        }

        // Output a single command to run on all machines.
        let mut command = vec![
            "source $HOME/.cargo/env".to_string(),
            upload_client_config_command,
        ];
        command.extend(upload_file_commands);
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
                let wallet_path = working_dir.clone().join(format!("sui-wallet-{i}.yaml"));

                let run_command = [
                    format!("./{BINARY_PATH}/walrus-stress"),
                    format!("--write-load {write_load_per_client}"),
                    format!("--read-load {read_load_per_client}"),
                    format!("--config-path {}", client_config_path.display()),
                    format!("--wallet-path {}", wallet_path.display()),
                    format!("--n-clients {}", parameters.client_parameters.tasks),
                    format!(
                        "--metrics-port {}",
                        parameters.client_parameters.metrics_port
                    ),
                    format!(
                        "--min-size-log2 {}",
                        parameters.client_parameters.min_size_log2
                    ),
                    format!(
                        "--max-size-log2 {}",
                        parameters.client_parameters.max_size_log2
                    ),
                    format!("--gas_refill_period_millis {}", 1000 * 60 * 60 * 24),
                ]
                .join(" ");

                let command =
                    ["source $HOME/.cargo/env", "ulimit -n 65535", &run_command].join(" && ");
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
