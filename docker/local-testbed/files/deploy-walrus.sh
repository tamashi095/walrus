#!/bin/bash
# Copyright (c) Walrus Foundation
# SPDX-License-Identifier: Apache-2.0

die() {
  echo "$0: error: $*" >&2
  exit 1
}

# use EPOCH_DURATION to set the epoch duration, default is 2 minutes
# in antithesis test. The exhaustive behavior exploration in antithesis test requires
# epoch duration to be as short as possible in order to explore more epoch change behaviors.
EPOCH_DURATION=${EPOCH_DURATION:-2m}

rm -rf walrus-docs
git clone https://github.com/MystenLabs/walrus-docs.git || die "failed to clone walrus-docs"
cp -r walrus-docs/contracts /opt/walrus/testnet-contracts || die "failed to copy contracts"

cd /opt/walrus || exit 1

rm -rf /opt/walrus/outputs/*

/opt/walrus/bin/walrus-deploy deploy-system-contract \
  --working-dir /opt/walrus/outputs \
  --contract-dir /opt/walrus/contracts \
  --do-not-copy-contracts \
  --sui-network 'http://sui-localnet:9000;http://sui-localnet:9123/gas' \
  --n-shards 100 \
  --host-addresses 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 \
  --storage-price 5 \
  --write-price 1 \
  --epoch-duration "$EPOCH_DURATION" >/opt/walrus/outputs/deploy \
  || die "failed to deploy system contract"

/opt/walrus/bin/walrus-deploy generate-dry-run-configs \
  --sui-amount 1000000000000 \
  --working-dir /opt/walrus/outputs \
  || die "failed to generate dry run configs"
