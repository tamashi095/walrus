#!/usr/bin/env bash
# Copyright (c) Walrus Foundation
# SPDX-License-Identifier: Apache-2.0

trap ctrl_c INT

join_by() {
  delim_save="$1"
  delim=""
  shift
  str=""
  for arg in "$@"; do
    str="$str$delim$arg"
    delim="$delim_save"
  done
  echo "$str"
}

kill_tmux_sessions() {
  { tmux ls || true; } | { grep -Eo "stress|staking|dryrun-node-\d*" || true; } | xargs -n1 tmux kill-session -t
}

ctrl_c() {
  kill_tmux_sessions
  exit 0
}

kill_tmux_sessions

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "OPTIONS:"
  echo "  -b <database_url>     Specify a backup database url (ie: postgresql://postgres:postgres@localhost/postgres, default: none)"
  echo "  -c <committee_size>   Number of storage nodes (default: 4)"
  echo "  -d <duration>         Set the length of the epoch (in human readable format, e.g., '60s', default: 1h)"
  echo "  -e                    Use existing config"
  echo "  -f                    Tail the logs of the nodes (default: false)"
  echo "  -h                    Print this usage message"
  echo "  -n <network>          Sui network to generate configs for (default: devnet)"
  echo "  -s <n_shards>         Number of shards (default: 10)"
  echo "  -t                    Use testnet contracts"
}

run_node() {
  cmd="RUST_LOG=$RUST_LOG ./target/release/walrus-node run --config-path $working_dir/$1.yaml ${2:-} \
    |& tee $working_dir/$1.log | tee -a $testbed_log"
  echo "Running within tmux: ($cmd)..." ||:
  tmux new -d -s "$1" "$cmd" || die "failed to invoke tmux with ($cmd)"
}

run_staking() {
  echo "$(date) Running staking client..." > $working_dir/staking.log
  cmd="RUST_BACKTRACE=full RUST_LOG=$RUST_LOG \
    cargo run --release --bin walrus-stress -- \
      --config-path $working_dir/client_config_staking.yaml \
      --sui-network 'http://127.0.0.1:9000;http://127.0.0.1:9123/gas' \
      --metrics-port 9125 \
      staking \
      |& tee -a $working_dir/staking.log | tee -a $testbed_log"
  echo "$(date) Running within tmux: '$cmd'..."
  tmux new -d -s staking "$cmd" || die "failed to invoke tmux with ($cmd)"
}

run_stress() {
  echo "$(date) Running stress client..." > $working_dir/stress.log
  cmd="RUST_BACKTRACE=full RUST_LOG=$RUST_LOG \
    cargo run --release --bin walrus-stress -- \
      --config-path $working_dir/client_config_stress.yaml \
      --sui-network 'http://127.0.0.1:9000;http://127.0.0.1:9123/gas' \
      --metrics-port 9126 \
      stress \
      --write-load 10 \
      --read-load 10 \
      --n-clients 1 \
      --gas-refill-period-millis 60000 |& tee -a $working_dir/stress.log | tee -a $testbed_log"
  echo "Running within tmux: '$cmd'..."
  tmux new -d -s stress "$cmd" || die "failed to invoke tmux with ($cmd)"
}

backup_database_url=
committee_size=4 # Default value of 4 if no argument is provided
epoch_duration=1h
network=devnet
shards=10 # Default value of 4 if no argument is provided
tail_logs=false
use_existing_config=false
contract_dir="./contracts"

while getopts "b:c:d:efhn:s:t" arg; do
  case "${arg}" in
    f)
      tail_logs=true
      ;;
    n)
      network=${OPTARG}
      ;;
    c)
      committee_size=${OPTARG}
      ;;
    s)
      shards=${OPTARG}
      ;;
    d)
      epoch_duration=${OPTARG}
      ;;
    e)
      use_existing_config=true
      ;;
    b)
      backup_database_url=${OPTARG}
      ;;
    t)
      contract_dir="./testnet-contracts"
      ;;
    h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
  esac
done

if ! [ "$committee_size" -gt 0 ] 2>/dev/null; then
  echo "Invalid argument: $committee_size is not a valid positive integer."
  usage
  exit 1
fi

if ! [ "$shards" -ge "$committee_size" ] 2>/dev/null; then
  echo "Invalid argument: $shards is not an integer greater than or equal to 'committee_size'."
  usage
  exit 1
fi

# Set working directory
working_dir="./working_dir"

mkdir -p working_dir
testbed_log="$working_dir"/testbed.log

echo "$0: Using network: $network"
echo "$0: Using committee_size: $committee_size"
echo "$0: Using shards: $shards"
echo "$0: Using epoch_duration: $epoch_duration"
echo "$0: Using backup_database_url: $backup_database_url"


if ! $use_existing_config; then
  if [[ -n "$backup_database_url" ]]; then
    echo "Reverting database migrations to ensure walrus-backup is starting fresh... [backup_database_url=$backup_database_url]"
    diesel migration --database-url "$backup_database_url" revert --all ||: |&
    diesel migration --database-url "$backup_database_url" run

    # shellcheck disable=SC2207
    schema_files=( $(git ls-files '**/schema.rs') )

    # Cleanup the output of the diesel migration. (Annoying by-product of limited diesel support for licenses and formatting.)
    pre-commit run licensesnip --files "${schema_files[@]}" 1>/dev/null 2>&1 ||:
    pre-commit run cargo-fmt --files "${schema_files[@]}" 1>/dev/null 2>&1 ||:
  fi
fi


features=( deploy )
binaries=( walrus walrus-node walrus-deploy )
if [[ -n "$backup_database_url" ]]; then
  features+=( backup )
  binaries+=( walrus-backup )
fi

echo "Building $(join_by ', ' "${binaries[@]}") binaries..."
# shellcheck disable=SC2046
cargo build \
  --release \
  $(printf -- "--bin %s " "${binaries[@]}") \
  --features "$(join_by , "${features[@]}")"

# Derive the ip addresses for the storage nodes
ips=( )
for node_count in $(seq 1 "$committee_size"); do
  ips+=( 127.0.0.1 )
done

# Initialize cleanup to be empty
cleanup=

# Clean up old processes.
pkill -f target/release/walrus ||:
export RUST_LOG=debug,h2=warn,hyper_util=warn,walrus_utils::config=info

if ! $use_existing_config; then
  # Cleanup
  rm -f $working_dir/dryrun-node-*.yaml
  rm -f $working_dir/stress.yaml
  rm -f $working_dir/staking.yaml
  cleanup="--cleanup-storage"

  # Deploy system contract
  echo Deploying system contract...
    ./target/release/walrus-deploy deploy-system-contract \
      --working-dir $working_dir \
      --sui-network "$network" \
      --n-shards "$shards" \
      --host-addresses "${ips[@]}" \
      --storage-price 5 \
      --write-price 1 \
      --epoch-duration "$epoch_duration" \
      --contract-dir "$contract_dir" \
      --with-wal-exchange \
      --with-subsidies \
      |& tee -a "$testbed_log"

  # Generate configs
  generate_dry_run_args=(
    --working-dir "$working_dir"
    --admin-wallet-path "$working_dir"/sui_admin.yaml
    --sui-amount 1000000000
    --extra-client-wallets 'stress,staking'
  )
  if [[ -n "$backup_database_url" ]]; then
    generate_dry_run_args+=( --backup-database-url "$backup_database_url" )
  fi
  echo "Generating configuration [${generate_dry_run_args[*]}]..."
  ./target/release/walrus-deploy generate-dry-run-configs "${generate_dry_run_args[@]}" \
    |& tee -a "$testbed_log"


  echo "
event_processor_config:
  adaptive_downloader_config:
  max_workers: 2
  initial_workers: 2" | \
      tee -a $working_dir/dryrun-node-*[0-9].yaml >/dev/null
fi

node_count=0
#
# shellcheck disable=SC2045
for config in $( ls $working_dir/dryrun-node-*[0-9].yaml ); do
  node_name=$(basename -- "$config")
  node_name="${node_name%.*}"
  run_node "$node_name" "$cleanup" || die "failed to launch node $node_name"
  ((node_count++))
done

run_staking
run_stress

echo "
Spawned $node_count nodes in separate tmux sessions. (See \`tmux ls\` for the list of tmux sessions.)

Client configuration stored at '$working_dir/client_config.yaml'.
See README.md for further information on the Walrus client."

if $tail_logs; then
  tail -F "$testbed_log" | grep --line-buffered --color -Ei "ERROR|CRITICAL|^"
else
  echo "Press Ctrl+C to stop the nodes."
  while (( 1 )); do
    sleep 120
  done
fi
