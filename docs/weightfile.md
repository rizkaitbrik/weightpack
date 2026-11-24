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
| `ROUTER`    | `ROUTER arrow` or `ROUTER train data=<path> [steps=<n>]` | At most one per `MIX`. Default `arrow`; `steps` defaults to 2000. |
| `TEMPLATE`  | `TEMPLATE <path>`                                        | Chat template. At most once.                                      |
| `LICENSE`   | `LICENSE inherit \| <SPDX id> \| <path>`                 | At most once.                                                     |

A `MIX` block stays open while `EXPERT` and `ROUTER` lines follow it; any other instruction closes it. Indentation is only for reading. `TRAIN` blocks are reserved and rejected for now.

## Layer ids

Lowercase letters, digits, `.`, `_` and `-`, starting with a letter or digit, at most 64 characters. Ids are unique across the file. `base` is reserved for the `FROM` layer.

## Sources

| Written as                            | Kind                                               |
| ------------------------------------- | -------------------------------------------------- |
| `.`, `./path`, `../path`, `/abs/path` | Local path, relative to the Weightfile's directory |
| `hf://org/name[@revision]`            | Hugging Face Hub repository                        |
| `oci://ref`                           | OCI reference                                      |
| anything else                         | OCI reference, as Docker reads `FROM`              |

So `adapters/french` without `./` is read as the registry image `docker.io/adapters/french`. `~` is not expanded.

## Codecs and fidelity

Without `codec=`, or with `codec=lora`, the source is an adapter and is stored as is. Any other codec means the source is a full fine-tune that the build imports as a delta over the layer's parent:

| `codec=` | Kind           | Lossless | Needs                            |
| -------- | -------------- | -------- | -------------------------------- |
| `xor`    | `delta.xor`    | Yes      | —                                |
| `bit1`   | `delta.bit1`   | No       | `max_drop=<n>`; `eval=` optional |
| `sparse` | `delta.sparse` | No       | `max_drop=<n>`; `eval=` optional |

`eval=` names the suite the build gates on; without it the domain's shared default suite applies. `eval` and `max_drop` are rejected on lossless codecs.

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

| Instruction       | Accepted source                                | Reads                                                                                                                                                                                                                                 |
| ----------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FROM`            | A local directory                              | `*.safetensors` (or the files `model.safetensors.index.json` names), `config.json`, and `tokenizer.json` or `tokenizer.model`. Other tokenizer files, `generation_config.json`, `README.md` and license files are kept as meta blobs. |
| `LAYER`, `EXPERT` | A local PEFT LoRA adapter directory            | `adapter_config.json` and `adapter_model.safetensors`                                                                                                                                                                                 |
| `ROUTER`          | `arrow`                                        | Nothing: Arrow prototypes are a function of the experts' weights                                                                                                                                                                      |
| `TEMPLATE`        | A local file                                   | Stored as `chat_template.jinja`                                                                                                                                                                                                       |
| `LICENSE`         | `inherit`, an SPDX id, or a local file (`./…`) | A file is stored as `LICENSE`; an SPDX id becomes the manifest's licenses annotation                                                                                                                                                  |

The build checks every adapter against the base before storing it: each module's base weight must exist and be a matrix, and `lora_A` must be `[r, in]` and `lora_B` `[out, r]` for a base weight of `[out, in]`. A mismatch fails the build, since it means the adapter was trained on another model. An `EXPERT` with modules outside its `MIX`'s targets or blocks builds with a warning.

Not yet supported, each with a clear error: `oci://` and `hf://` sources (they come with `wpk pull`), `codec=xor|bit1|sparse` imports (phase 2), `ROUTER train` (phase 3), pickle weights (`*.bin`, `*.pt`; convert to safetensors), and adapters that use DoRA, `fan_in_fan_out`, trained biases, `modules_to_save`, `rank_pattern` or `alpha_pattern`.

Building the same inputs twice gives the same image digest and stores nothing new. Two images on the same base share its blobs, so the second stores only its own layers.
