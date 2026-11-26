# Weightfile reference

A Weightfile declares an image the way a Dockerfile does: one instruction per line, read top to bottom. `wpk validate <Weightfile | dir>` checks one without building it, and `wpk build -t name:tag <Weightfile | dir>` builds it into the local store (see [Building](#building)). The parser lives in `libs/go/weightfile` and the builder in `libs/go/build`.

```dockerfile
FROM registry.example.com/llama-3.1-8b-instruct@sha256:<64 hex>

LAYER french ./adapters/french

MIX domain targets=mlp blocks=8-31 top_k=2
  EXPERT legal    ./adapters/legal
  EXPERT medical  oci://registry.example.com/experts/medical:1.4
  EXPERT finance  hf://acme/llama-fin-full  codec=bit1 eval=./evals/finance.yaml max_drop=0.5
  ROUTER arrow

TEMPLATE ./chat.jinja
LICENSE inherit
```

## Instructions

| Instruction | Form                                                     | Rules                                                             |
| ----------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| `FROM`      | `FROM <source>`                                          | First instruction, exactly once. Becomes the layer `base`.        |
| `LAYER`     | `LAYER <id> <source> [codec=…] [eval=…] [max_drop=…]`    | Stacked on everything above it. Must come before any `MIX`.       |
| `MIX`       | `MIX <id> [targets=…] [blocks=…] [top_k=…]`              | At most one per image, on top of the stack. Opens a block.        |
| `EXPERT`    | `EXPERT <id> <source> [codec=…] [eval=…] [max_drop=…]`   | Belongs to the open `MIX`. Every expert has the mix's parent.     |
| `ROUTER`    | `ROUTER arrow`, `ROUTER train data=<path> [steps=<n>]` or `ROUTER <path>` | At most one per `MIX`. Default `arrow`; `steps` defaults to 2000. A path names a router trained before, such as the `routers/<mix>.safetensors` an export writes. |
| `TEMPLATE`  | `TEMPLATE <path>`                                        | Chat template. At most once.                                      |
| `LICENSE`   | `LICENSE inherit \| <SPDX id> \| <path>`                 | At most once.                                                     |
| `LABEL`     | `LABEL <key>=<value> ...`                                | Manifest annotations, as in a Dockerfile. Each key once.          |

A `MIX` block stays open while `EXPERT` and `ROUTER` lines follow it; any other instruction closes it. Indentation is only for reading. `TRAIN` blocks are reserved and rejected for now.

## Labels

`LABEL org.example.team=nlp "note=two words"` sets annotations on the image's manifest, as a Dockerfile's labels do. Keys are letters, digits, `.`, `_`, `-` and `/`, and each key may be set once per file. The license annotation is set with `LICENSE`, not `LABEL`. `wpk export` writes an image's annotations back as `LABEL` lines (a squashed image's `wpk.squashed` and `wpk.squashed.from` among them), which is how its exported Weightfile rebuilds the same manifest.

## Layer ids

Lowercase letters, digits, `.`, `_` and `-`, starting with a letter or digit, at most 64 characters. Ids are unique across the file. `base` is reserved for the `FROM` layer.

## Sources

| Written as                            | Kind                                               |
| ------------------------------------- | -------------------------------------------------- |
| `.`, `./path`, `../path`, `/abs/path` | Local path, relative to the Weightfile's directory |
| `hf://org/name[/folder][@revision]`   | Hugging Face Hub repository, or a folder in it     |
| `oci://ref`                           | OCI reference                                      |
| anything else                         | OCI reference, as Docker reads `FROM`              |

So `adapters/french` without `./` is read as the registry image `docker.io/adapters/french`. `~` is not expanded.

## Codecs and fidelity

Without `codec=`, or with `codec=lora`, the source is an adapter and is stored as is. Any other codec means the source is a full fine-tune that the build imports as a delta over the layer's parent:

| `codec=` | Kind           | Lossless | Needs                            |
| -------- | -------------- | -------- | -------------------------------- |
| `xor`    | `delta.xor`    | Yes      | —                                |
| `bit1`   | `delta.bit1`   | No       | `max_drop=<n>`; `eval=` and `calibrate=<steps>` optional |
| `sparse` | `delta.sparse` | No       | `max_drop=<n>`; `eval=` and `density=<n>` optional |

`eval=` names the suite the build gates on; without it the build gates on the fine-tune's own replies to general prompts. `max_drop` is in points of next-token accuracy. `density` (in (0, 1], default 0.1) is the fraction of each matrix's delta that `sparse` keeps. `calibrate` (default 200, `0` for none) is how many steps `bit1` spends calibrating its scales. `eval`, `max_drop`, `density` and `calibrate` are rejected where they do not apply.

`codec=xor` is supported: the source is a directory of safetensors weights with exactly the base's tensor names, dtypes and shapes, and the layer must sit on the base or on other `xor` layers (LoRA layers can go on top of it). The image keeps its parent's tokenizer and `config.json`, so the build refuses a fine-tune whose tokenizer or config differs from them in anything but metadata (`_name_or_path`, `transformers_version`, `torch_dtype` and the like): a changed `rope_theta`, say, would otherwise be silently dropped. Build such a fine-tune as a base of its own (`FROM`). The build needs `wpkd`'s build worker. `wpk import --parent <image | dir> [--id name] -t name <dir>` is shorthand for `FROM <parent>` plus one such layer.

`codec=bit1` and `codec=sparse` take the same sources, under the same rules, and store the fine-tune lossily: `bit1` keeps one sign bit per weight plus one scale per matrix (1/16 of BF16), the scales calibrated so the model's next-token distributions match the fine-tune's, and `sparse` keeps the largest `density` fraction of each matrix's change (7% of a BF16 fine-tune at the default 0.1, measured). They need torch and transformers in the build worker (the `serve` extra). The build then:

1. encodes the delta and writes the model the layer gives back;
2. scores the fine-tune and that model on the eval suite by next-token accuracy (the percent of the suite's learnt tokens each model predicts);
3. fails if the fine-tune's score minus the layer's, the drop, exceeds `max_drop`;
4. records both scores, the drop, the suite and the source's digest in the layer's fidelity record, which `wpk history` shows.

An eval suite is a file of JSON Lines in the shapes `ROUTER train` reads (`{"messages": [...]}`, `{"prompt", "completion"}` or `{"text"}`), named after the file, or YAML that names one:

```yaml
name: finance-v2
data: finance.jsonl # relative to this file
```

Without `eval=`, the suite is the fine-tune's greedy replies to a fixed set of general prompts, on which the fine-tune scores about 100: the drop is then about the share of reply tokens where the layer's first choice differs from the fine-tune's. A rebuild reuses the deltas and the eval from the build cache. A lossy layer has no PEFT form, so like an XOR layer it is squashed to export it, and it is baked into the serving base to run it.

## MIX parameters

| Parameter | Values                                                                                        | Default                      |
| --------- | --------------------------------------------------------------------------------------------- | ---------------------------- |
| `targets` | `mlp` (gate, up, down), `attn` (q, k, v, o), `all`, or a comma list of `q,k,v,o,gate,up,down` | `all`                        |
| `blocks`  | `all`, or ascending, non-overlapping ranges such as `0-3,8-31`                                | `all`                        |
| `top_k`   | 1 to the number of experts                                                                    | 2, or 1 with a single expert |

## Syntax

- `#` starts a comment when it begins a word: `./dir#1` is a path, `./dir #1` is a path and a comment.
- A line ending in `\` continues on the next line.
- Double quotes group a value with spaces: `LAYER a "./my adapters/a"`. Inside quotes, `\"` and `\\` are escapes.
- Instruction names are case-insensitive. Parameters are `key=value` with no spaces around `=`.
- All errors in a file are reported at once, each with its line number.

## Building

`wpk build` sends the Weightfile's path to the local `wpkd`, which reads the build context in place and streams progress back. What it accepts so far:

| Instruction       | Accepted source                                                                                               | Reads                                                                                                                                                                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FROM`            | A local directory, a Hugging Face repository (`hf://`), or an image (`oci://ref` or a bare reference)         | From a directory: `*.safetensors` (or the files `model.safetensors.index.json` names), `config.json`, and `tokenizer.json` or `tokenizer.model`; other tokenizer files, `generation_config.json`, `README.md` and license files are kept as meta blobs. From an image: its base and stack of layers, unchanged. |
| `LAYER`, `EXPERT` | A local PEFT LoRA adapter directory, a Hugging Face repository or folder (`hf://`), or an image (`oci://ref`) | From a directory or repository: `adapter_config.json` and `adapter_model.safetensors`, including `modules_to_save` copies. From an image: its top layer.                                                                                                                                                        |
| `ROUTER`          | `arrow`, `train data=<jsonl> [steps=<n>]`, or a local router file (`./…`)                       | Arrow: nothing, its prototypes are a function of the experts' weights, computed when the mix is served. `train`: the data, to train a router with the experts frozen (see Trained routers below). A file: a router trained before, checked against the mix and stored as the mix's `router.v1` blob |
| `TEMPLATE`        | A local file                                                                                                  | Stored as `chat_template.jinja`                                                                                                                                                                                                                                                                                 |
| `LICENSE`         | `inherit`, an SPDX id, or a local file (`./…`)                                                                | A file is stored as `LICENSE`; an SPDX id becomes the manifest's licenses annotation                                                                                                                                                                                                                            |
| `LABEL`           | `key=value` pairs                                                                                             | Each becomes a manifest annotation                                                                                                                                                                                                                                                                              |

The build checks every adapter against the base before storing it: each module's base weight must exist and be a matrix, and `lora_A` must be `[r, in]` and `lora_B` `[out, r]` for a base weight of `[out, in]`. A mismatch fails the build, since it means the adapter was trained on another model. An `EXPERT` with modules outside its `MIX`'s targets or blocks builds with a warning.

### Hugging Face sources

`hf://org/name[/folder][@revision]` names a repository on the Hugging Face Hub, or a folder in it; the revision is a branch, tag or commit and defaults to `main`. `wpkd` resolves it to a commit, downloads only the files the build reads (a model's safetensors weights and meta files; an adapter's config and weights, so no README or tokenizer copies), checks each against the SHA-256 or git blob id the Hub publishes, and caches them by commit in `<root>/cache/hf` (`--hf-cache` or `WPK_HF_CACHE` move it). A rebuild downloads nothing. Repositories with only pickle weights are refused before anything is downloaded. Gated and private repositories need `HF_TOKEN`, or the token `huggingface-cli login` saved; `HF_ENDPOINT` points at a mirror. Building from `hf://` gives the same image digest as building from the downloaded files.

### Adapters with modules_to_save

An adapter trained with `modules_to_save` (a classification head, a resized `lm_head`) carries full copies of those modules beside its LoRA matrices. Each copy is stored in the LoRA layer as a `tensor.v1` blob under the base tensor's name and must match the base tensor's dtype and shape. Squash replaces the base tensor with the copy at that layer's place in the chain, before any LoRA delta of the same layer; export writes the copies back with PEFT's names and the `modules_to_save` list.

### Trained routers

`ROUTER train data=./mix.jsonl steps=2000` trains the mix's router while the build runs, with the base and the experts frozen. It needs `wpkd`'s build worker (`--worker`) with torch, transformers and PEFT, and runs on CUDA, Metal or the CPU.

- **Data.** JSON Lines, one example per line: `{"messages": [...]}`, of which only the last assistant turn is learnt, through the image's chat template; `{"prompt": "...", "completion": "..."}`, of which the completion is learnt; or `{"text": "..."}`, learnt whole. A few thousand examples that mix the experts' domains are enough; examples are cut at 512 tokens.
- **What the build does.** It trains on the base and experts exactly as `wpk run` would serve them, starting from Arrow, and reports the held-out loss (a tenth of the data, at most 200 examples) with Arrow and with the trained router. The router is cached: rebuilding with the same chain, data, steps, template and worker reuses it.
- **What the image holds.** The router's weights, as the mix's `router.v1` blob. `wpk export` writes them to `routers/<mix>.safetensors` and its Weightfile says `ROUTER ./routers/<mix>.safetensors`, so the export rebuilds the same image without training again.

### Images as sources

An image named by `FROM`, `LAYER` or `EXPERT` comes from the local store, or is pulled when the store lacks it (only the missing blobs move; registries on localhost use plain HTTP, and for another insecure registry `wpk pull --plain-http` it first). A bare reference such as `FROM assistant:v1` can also name a local tag.

- **`FROM` an image** starts the new image from that image's base and its whole stack of layers, with their ids, digests and chain IDs unchanged; the Weightfile's `LAYER`s go on top. An image that ends in a `MIX` is refused, since nothing can sit above a mix, and the Weightfile's layer ids must differ from the inherited ones. The image's chat template and license carry over unless the Weightfile sets its own `TEMPLATE` or `LICENSE` (`LICENSE inherit` keeps the image's); its labels do not.
- **`LAYER` or `EXPERT` from an image** takes the image's top layer, under the Weightfile's id. The layer must have been built on exactly the chain it would sit on here, the stack below a `LAYER` or the mix's parent for an `EXPERT`: the build compares chain IDs and refuses a mismatch, naming both chains (`was built on base → french, but here it would sit on base`). That is what keeps an adapter trained on one stack from being silently applied to another. `codec=` does not apply to such a layer.

Building `FROM` a pushed base gives the same image digest as building from the base's directory.

Not yet supported, each with a clear error: pickle weights (`*.bin`, `*.pt`; convert to safetensors), and adapters that use DoRA, `fan_in_fan_out`, trained biases, `rank_pattern` or `alpha_pattern`.

Building the same inputs twice gives the same image digest and stores nothing new; the build cache also lets a rebuild skip reading unchanged source files (`wpk build --no-cache` reads them anyway). Two images on the same base share its blobs, so the second stores only its own layers.
