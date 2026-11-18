# Model-port test fixtures

The JSON inventories in this directory record names, shapes, dtypes, and other
metadata extracted from pinned checkpoints. Model-plan tests expand them locally
and compare them with the real model on the meta device. Where an inventory
contains `tensor_key_ranges`, each row contains a tensor-name pattern, a list of
inclusive index ranges for its placeholders, and the common value for those
keys. For example, `["layers.{}.norm.weight", ["0-2,5"], [32]]` describes four
tensors of shape `[32]`.

`upstream.json` pins release files by revision, byte count, and SHA-256.
Tokenizer and processor tests fetch these through the Hub cache or their
recorded source URL. For a safetensors shard the reader requests only the header
range. The tests never download trained weights through this fixture reader.
When the network is disabled, Hub assets can be read from an existing cache.
Tests that need an unavailable asset report a skip, and a size or hash mismatch
fails the test.

Run the affected pytest modules from a repository checkout. These fixtures
are excluded from both the wheel and the source distribution.
