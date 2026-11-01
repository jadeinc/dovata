# src/layout: an architecture's compute graph, as data

A layout is one model architecture's forward pass written down as a list of ggml ops.
It says which ops one step runs, in which order, on which tensors, and with which
parameters. It is translated from llama.cpp's own graph code, one C++ call to one line,
so the two can be read side by side.

dovata reads a layout together with a model's GGUF header and *binds* them: every op of
every layer is written out with concrete shapes, types and byte sizes. The planner then
decides where each tensor lives, and the Jade compiler generates one kernel per op. A
layout is therefore the contract between dovata and the compiler's kernels, and this
file is its reference.

## Why it exists

The compiler needs to know what to compute, and the planner needs to know what to
place. Both questions have the same answer, and llama.cpp already holds it, but as C++
that builds a ggml graph at run time. A layout is that answer as data: it can be
reviewed, versioned, diffed against llama.cpp when the pin moves, and handed to a
compiler that never links llama.cpp.

Layouts are their own capability, separate from the planner in `src/plan/`. A layout is
the contract with the compiler's kernels, and the planner only decides where a bound
graph's tensors go. Keeping the two apart means the format can be reviewed, versioned
and extended without touching placement, and the planner never needs to know about
architectures.

Each built-in layout is one architecture dovata supports. Supporting a new architecture
means writing its layout; no Rust changes unless it needs an op or an hparam dovata does
not know yet.

## What each file does

| File | What it is |
|---|---|
| `mod.rs` | The module's front door: `builtin`, `SUPPORTED`, `for_model`, and `LayoutError`. |
| `format.rs` | Reads a layout file into a checked `Layout`. Everything that needs no model is checked here, with the place in the file. |
| `expr.rs` | The small expression language dims, params and conditions are written in. |
| `symbols.rs` | The closed set of names expressions may use, and which GGUF key or bind option each one reads. |
| `ops.rs` | The ops a layout may use: their ggml enum, their params, and the shape rule ported from each op's constructor in `ggml.c`. |
| `bind.rs` | `Layout::bind`, and the bound types the planner reads: `BoundGraph`, `BoundOp`, `BoundTensor`, `TensorKind`. |
| `builtin/*.layout.toml` | The built-in layouts: `llama`, `llama4`, `qwen2`, `qwen3`, `qwen3moe`, `deepseek2`, `nemotron`, `deci`. |
| `ggml-ops.txt` | ggml's graph node types, copied from `ggml.h`, which every layout op is checked against. |
| `testutil.rs`, `tests.rs` | Synthetic GGUF headers for each architecture, and the tests that bind them. |

## Who uses it

The planner (`src/plan/`, next) takes a `BoundGraph` and places its tensors. The Jade
compiler takes the same graph, through the plan, and generates kernels from its ops,
dispatching on `ops::ggml_enum`, which gives the ggml enum value each op name stands
for. `gguf-inspect` says whether a file's architecture has a built-in layout. A caller
with an architecture dovata does not support can write a layout and pass it to
`for_model`.

`for_model` takes a passed-in layout only when its `arch` matches the model's
`general.architecture`, because a layout for a different architecture would bind
against tensor names and hparams it was not written for. With no passed-in layout and
no built-in one, the error lists the supported architectures and says how to write a
layout. `builtin` panics if a built-in layout fails to parse, and the
`builtin_layouts_parse` test makes sure none does.

```rust
let model = dovata::load::Model::open(path)?;
let layout = dovata::layout::for_model(&model, None)?;          // the built-in one
let opts = BindOptions { n_ctx: 32768, n_tokens: 512, n_outputs: 1, kv_k: KvType::Q8_0, kv_v: KvType::Q8_0 };
let graph = layout.bind(&model, &opts)?;
```

## The format

A layout is a TOML file with these top-level keys.

| Key | What it holds |
|---|---|
| `schema_version` | `1`. A layout for a version this dovata does not read is refused. |
| `arch` | The `general.architecture` it is for, such as `"qwen3moe"`. |
| `description` | One line, optional. |
| `outputs` | The names the program reads after the step, such as `["result_output"]`. |
| `weights` | The model tensors the layout expects, mirroring `load_arch_tensors`. |
| `unused` | Model tensors llama.cpp loads but one step never reads, each with a reason. |
| `tensors` | The non-weight tensors that persist or are filled from outside: inputs, KV, state. |
| `defaults` | Per-op parameter defaults, such as every `rope`'s frequency base. |
| `prologue`, `block`, `epilogue` | The ops. `block` runs once per layer, with `{L}` the layer index. |

Ops can be written one per line, as the built-ins do:

```toml
block = [
  { op = "rms_norm", in = ["inpL"], out = "norm", eps = "f_norm_rms_eps" },
  { op = "mul", in = ["norm", "blk.{L}.attn_norm.weight"], out = "attn_norm" },
]
```

or as TOML tables, which parse to the same thing:

```toml
[[block]]
op = "rms_norm"
in = ["inpL"]
out = "norm"
eps = "f_norm_rms_eps"
```

### Weights

```toml
weights = [
  { name = "token_embd.weight", dims = ["n_embd", "n_vocab"] },
  { name = "output.weight", dims = ["n_embd", "n_vocab"], fallback = "token_embd.weight" },
  { name = "blk.{L}.attn_q.bias", dims = ["n_embd_head_k * n_head"], optional = true },
  { name = "blk.{L}.ffn_gate_inp.weight", dims = ["n_embd", "n_expert"], when = "n_expert > 0" },
]
```

- `name` is the GGUF tensor name. With `{L}` it is one weight per layer.
- `dims` is optional. When given, bind checks the file's shape against it.
- `optional = true` mirrors llama.cpp's `TENSOR_NOT_REQUIRED`: a file without it binds.
- `fallback` mirrors `TENSOR_DUPLICATED`: when the file lacks the weight, ops that read
  it read the fallback instead. Tied embeddings are this case.
- `when` limits the weight to the layers where the condition holds, the way
  `load_arch_tensors` wraps `create_tensor` in an `if`.

A layout never states a weight's type. The type and size come from the file, so a
Q4_K_M file and a BF16 file of one model bind to the same layout.

### Unused tensors

```toml
unused = [
  { name = "cls.output.weight", why = "rerank head; read only by llama.cpp's rerank pooling" },
]
```

Bind refuses a file with a tensor no op reads, so a layout cannot drop a weight by
accident. llama.cpp is strict in the same way: `done_getting_tensors` refuses a file
with tensors no `create_tensor` call asked for. A few tensors are loaded by llama.cpp
and then read only by features outside one forward step, such as a reranker's
classification head. Those are listed here, with the reason, and stay in the bound
graph as weights no op reads, so the planner can still give every file tensor an
address.

### Declared tensors: inputs, KV and state

```toml
tensors = [
  { name = "inp_tokens", kind = "input", dtype = "i32", dims = ["n_tokens"] },
  { name = "attn_inp_kq_mask", kind = "input", dtype = "f16", dims = ["n_kv", "n_tokens", 1, 1] },
  { name = "attn_inp_k_rot", kind = "input", dtype = "f32", dims = ["n_attn_rot_k", "n_attn_rot_k"], when = "attn_rot_k" },
  { name = "cache_k_l{L}", kind = "kv", dtype = "kv_k", dims = ["n_embd_k_gqa", "n_ctx"] },
]
```

| `kind` | What it is |
|---|---|
| `input` | Filled by the program before each step: token ids, positions, the KQ mask, KV write indices, output row ids, the KV rotation matrices. These are what llama.cpp's `build_inp_*` functions make. Counted as produced when the step starts. |
| `kv` | The KV cache, persistent across steps. `dtype = "kv_k"` or `"kv_v"` takes the type from the bind options. |
| `state` | Recurrent state, persistent across steps. Reserved for the hybrid architectures; it binds, but no built-in uses it yet. |

Activations are not declared. The op that produces one names it with `out`, and its
shape follows from ggml's own rule for that op.

### Ops

```toml
{ op = "view", for = "e in 1..n_expert_used", in = ["ffn_moe_weighted"], out = "cur_experts.{e}",
  ne = ["n_embd", "src.ne2"], nb = ["src.nb2"], offset = "e * src.nb1" }
```

| Key | Meaning |
|---|---|
| `op` | A ggml op, by its graph node type in lower case. See the op vocabulary below. |
| `in` | Input names, in the order the ggml API call takes its tensor arguments. A trailing `?` marks an optional weight that is dropped when the file lacks it; only trailing inputs may be optional, so the rest keep their positions. |
| `out` | The name the result is given. |
| `when` | A condition. When false, the op does not run, and `out` keeps whatever it held. |
| `for` | `"e in a..b"`: run the op once per value, end exclusive, with `{e}` usable in names and `e` in expressions. |
| `dims`, `dtype` | Optional assertions about the result, checked against ggml's rule. |
| anything else | A parameter of the op. |

Names work like variables in llama.cpp's C++, where `cur = ggml_rms_norm(ctx0, cur, eps)`
reassigns `cur`. Assigning a name again makes a new tensor; ops that already read the
old one keep reading it. A name first assigned in the prologue lives for the whole step,
which is how `inpL` carries from one layer to the next. A name first assigned in the
block lives for one layer, so a block can never read the previous layer's value by
mistake, and the epilogue sees only what the prologue created.

In the bound graph, a block activation is named `<name>-<layer>`, such as `ffn_inp-3`,
which is how llama.cpp's `cb()` names graph tensors. A second assignment within a layer
adds `#1`, then `#2`.

`defaults.<op>` sets a parameter for every op of that kind in the layout, and an op's
own value overrides it. The built-ins use it for `rope`, whose nine parameters are
`llm_graph_context` members every call passes:

```toml
defaults.rope.n_dims = "n_rot"
defaults.rope.mode = "neox"
defaults.rope.freq_base = "freq_base"
```

### Expressions

Dims, parameters and conditions are integers, floats, or expressions in a small
language: `+ - * / %`, comparisons, `&& || !`, `c ? a : b`, parentheses, and three
functions. Integer division truncates as in C, so `n_ff / n_expert_used` means what it
means in llama.cpp.

Any float operand makes the result a float. Division by zero, overflow, and a float
where an integer is needed are errors, never a guessed value. A dim must come out as an
integer: a float is refused even when it is whole, because a dimension that came out of
float arithmetic is a layout bug. A condition must be a comparison or a truth value; a
number is not taken as true or false, so a layout writes `x != 0`.

The grammar, loosest first:

```text
expr    = or ( "?" expr ":" expr )?
or      = and ( "||" and )*
and     = cmp ( "&&" cmp )*
cmp     = sum ( ( "==" | "!=" | "<" | "<=" | ">" | ">=" ) sum )?
sum     = product ( ( "+" | "-" ) product )*
product = unary ( ( "*" | "/" | "%" ) unary )*
unary   = ( "!" | "-" ) unary | primary
primary = number | name | name "(" expr ")" | "has(" tensor-name ")" | "(" expr ")"
```

Comparisons do not chain, so `a < b < c` is an error. The parser in `expr.rs` has one
function per line of this grammar, and each calls the next tighter line for its
operands. The text inside `has(...)` is kept as one token, since a tensor name holds
dots and braces, and `src.ne0` is one name.

- `sqrt(x)`, as in `scale = "1 / sqrt(n_embd_head_v)"`.
- `row_size(n)`: `ggml_row_size` of the op's first input type over `n` elements. Views
  are written in bytes, as in ggml, so a view of a quantized cache is written exactly as
  llama.cpp writes it.
- `has(name)`: whether the model file holds that tensor. The name must be a declared
  weight, so a typo is an error instead of a condition that is always false.

Inside an op's params, `src.ne0` to `src.ne3` and `src.nb0` to `src.nb3` are the first
input's shape and byte strides, the way llama.cpp writes `cur->ne[1]` and `k->nb[2]`. `L`
is the layer index inside the block, and a `for` variable is in scope in its op.

Arithmetic is allowed because llama.cpp's graph code is written in it: shapes such as
`n_embd_head_k * n_head`, views such as `i*experts->nb[1]`, conditions such as
`il == n_layer - 1`. Without it, each would need a named symbol in Rust, and the model
knowledge would move out of the layout into code a reviewer cannot read next to
llama.cpp.

The three uses come straight from llama.cpp's code. `create_tensor` writes shapes as
products, such as `{n_embd_head_k * n_head, n_embd}` for the attention output weight.
Every `ggml_view_*` call computes its strides and offset from the tensor it views, as in
`ggml_row_size(k->type, n_embd_head_k)` and `cur->ne[0]*cur->ne[1]`. Conditions pick
layers and features: `(il + 1) % step` interleaves llama4's layers, and
`ggml_is_quantized(type_k) && n_embd_head_k % 64 == 0` turns on llama.cpp's KV
rotation.

An expression is parsed once, when the layout is read, and evaluated once per layer
when the layout is bound. The evaluator in `expr.rs` knows nothing about models. It asks
a scope for each name, for `has()` and for `row_size()`, and the binder supplies that
scope: it reads symbols from the model's header and the bind options, and it fills the
`{L}` and loop placeholders of a `has()` name, since only the binder knows the current
layer and loop value.

### Symbols

Every other name is one of a closed set, read from the model's header by reference. A
layout can never name a GGUF key or hardcode one model's hparam. The names are
llama.cpp's own.

| Symbol | Where it comes from |
|---|---|
| `n_layer` | `<arch>.block_count` |
| `n_embd` | `<arch>.embedding_length` |
| `n_head` | `<arch>.attention.head_count`, per layer when the key is an array |
| `n_head_kv` | `<arch>.attention.head_count_kv`, per layer; `n_head` when absent |
| `n_embd_head_k` | `<arch>.attention.key_length`; `n_embd / n_head` when absent |
| `n_embd_head_v` | `<arch>.attention.value_length`; `n_embd / n_head` when absent |
| `n_embd_k_gqa` | `n_embd_head_k * n_head_kv`, per layer |
| `n_embd_v_gqa` | `n_embd_head_v * n_head_kv`, per layer |
| `n_rot` | `<arch>.rope.dimension_count`; `n_embd_head_k` when absent |
| `n_ff` | `<arch>.feed_forward_length`, per layer when an array |
| `n_expert` | `<arch>.expert_count`; 0 when absent |
| `n_expert_used` | `<arch>.expert_used_count`; 0 when absent |
| `n_ff_exp` | `<arch>.expert_feed_forward_length`; 0 when absent |
| `n_vocab` | the length of `tokenizer.ggml.tokens` |
| `n_ctx_train` | `<arch>.context_length` |
| `f_norm_rms_eps` | `<arch>.attention.layer_norm_rms_epsilon`; required when used |
| `f_norm_eps` | `<arch>.attention.layer_norm_epsilon`; required when used |
| `freq_base` | `<arch>.rope.freq_base`; 10000 when absent |
| `freq_scale` | 1 over `<arch>.rope.scaling.factor`; 1 when absent or when scaling is none |
| `ext_factor` | 1 when `<arch>.rope.scaling.type` is yarn, else 0 |
| `attn_factor` | llama-context.cpp's YaRN magnitude correction, times `<arch>.rope.scaling.attn_factor` |
| `beta_fast`, `beta_slow` | 32 and 1, llama_hparams' defaults |
| `n_ctx_orig` | `<arch>.rope.scaling.original_context_length`; `n_ctx_train` when absent |
| `n_ctx` | the bind option: KV cache length |
| `n_kv` | the KV cells one step attends over, planned at its maximum, `n_ctx` |
| `n_tokens` | the bind option: tokens in one step |
| `n_outputs` | the bind option: tokens whose logits the step returns |
| `kv_k_quantized`, `kv_v_quantized` | whether the K or V cache type is block-quantized |
| `kv_k_f32`, `kv_v_f32` | whether the K or V cache type is f32 |
| `attn_rot_k`, `attn_rot_v` | whether llama.cpp rotates K (and Q) or V by a Hadamard matrix: the cache is quantized and the head width divides by 64 |
| `n_attn_rot_k`, `n_attn_rot_v` | the rotation sizes `build_input_k_rot` and `build_input_v_rot` use |
| `n_moe_layer_step` | `<arch>.interleave_moe_layer_step`; required when used |
| `n_no_rope_layer_step` | llama4: `n_layer` when `<arch>.attention.sliding_window` is 0, else 4, llama_hparams' default |
| `n_swa` | llama4: 8192, which llama4 sets over the file's value; 0 when `<arch>.attention.sliding_window` is 0 |
| `is_swa` | per layer; llama4: `L % p < p - 1`, with p from `<arch>.attention.sliding_window_pattern`, 4 when absent; false when `n_swa` is 0 |
| `freq_base_swa`, `freq_scale_swa` | llama4: a chunked layer's rope base and scale, llama_hparams' `rope_freq_base_train_swa` and `rope_freq_scale_train_swa` |
| `n_ctx_swa` | the SWA cache's length: `n_ctx`, the size llama.cpp gives it with `swa_full` |
| `n_kv_swa` | the SWA cache cells one step attends over, planned at its maximum, `n_ctx_swa` |
| `n_layer_dense_lead` | `<arch>.leading_dense_block_count`; 0 when absent |
| `q_lora_rank` | `<arch>.attention.q_lora_rank`, required; 0 for deepseek2's lite variants (26 or 27 layers, or 48 with a vocabulary of 128256), which never read it |
| `kv_lora_rank` | `<arch>.attention.kv_lora_rank`; required when used |
| `is_mla` | whether `<arch>.attention.key_length_mla` and `value_length_mla` are both set and not 0, as `llama_hparams::is_mla` decides |
| `n_embd_head_k_mla`, `n_embd_head_v_mla` | `<arch>.attention.key_length_mla` and `value_length_mla` when `is_mla`; else `n_embd_head_k` and `n_embd_head_v` |
| `n_expert_shared` | `<arch>.expert_shared_count`; required when used |
| `n_expert_groups`, `n_group_used` | `<arch>.expert_group_count` and `expert_group_used_count`; 0 when absent |
| `expert_gating_func` | `<arch>.expert_gating_func`, as `llama_expert_gating_func_type`: 1 softmax, 2 sigmoid; deepseek2's default when absent or 0 |
| `expert_weights_norm` | `<arch>.expert_weights_norm`; false when absent |
| `expert_weights_scale` | `<arch>.expert_weights_scale`; 0 when absent |
| `f_attn_temp_scale` | `<arch>.attention.temperature_scale`; 0 when absent |
| `mscale` | deepseek2.cpp's YaRN factor for its `kq_scale`, from `attn_factor`, `freq_scale` and `<arch>.rope.scaling.yarn_log_multiplier` |

Outside a block, a per-layer symbol reads layer 0, as llama.cpp's `n_head()` does.
Hparams only some architectures read, such as `attention.scale`, have no symbol yet,
because a symbol would give a value llama.cpp never reads for the others. One is added
with the first layout for an architecture that reads it, and its row in `SYMBOLS` says
which architecture that is.

The set is closed so that every hparam is read by reference from the model's own
header. The same layout then serves every model of its architecture, and a model with an
unusual epsilon or rope base gets its own value without anyone editing the layout. An
open set, where a layout names GGUF keys directly, would let one layout quietly hardcode
one model's numbers.

The names come from llama.cpp: the `LLAMA_LOAD_LOCALS` macro in `src/llama-model.h`,
such as `n_embd`, `n_embd_k_gqa` and `n_rot`; the `llama_hparams` fields, such as
`f_norm_rms_eps` and `n_ff_exp`; and the `llm_graph_context` members the graph code
passes to `ggml_rope_ext`, such as `freq_base`, `ext_factor` and `n_ctx_orig`. A layout
line then reads the way the C++ call it came from reads.

`symbols.rs` reads each symbol the way llama.cpp reads it at commit 52b3df0: the same
key, the same default when the key is absent, and per layer where llama.cpp keeps a
per-layer array through `get_key_or_arr`. A few details follow llama.cpp closely.

- `n_head` is 0 when its key is absent. The head widths default to `n_embd / n_head` at
  layer 0, as `llama_model_base::load_hparams` does, and to 0 when the model has no
  heads.
- `n_vocab` falls back to `<arch>.vocab_size` when the file has no
  `tokenizer.ggml.tokens`.
- The rope scaling type is read as `llama_rope_scaling_type_from_string` reads it, and
  is `linear` when the key is absent.
- `freq_scale` is `cparams.rope_freq_scale`, computed in f32 as llama.cpp computes it,
  from `rope.scaling.factor` or the older `rope.scale_linear`.
- `attn_factor` is `cparams.yarn_attn_factor`, from the block in `llama_context`'s
  constructor. That block has a `rope_yarn_log_mul` branch, which is taken for
  deepseek2, the one built-in architecture that reads `rope.scaling.yarn_log_multiplier`.
  deepseek2's `load_arch_hparams` divides the key by 0.1 to undo the convert script.
- The KV rotation symbols repeat the rule in `llama_kv_cache`'s constructor that turns
  on a Hadamard rotation of Q, K and V when the cache is quantized. `n_attn_rot_k`
  follows `llama_kv_cache::build_input_k_rot`, which doubles from 64 while the result
  still divides the head width.
- `n_no_rope_layer_step`, `n_swa`, `is_swa`, `freq_base_swa` and `freq_scale_swa`
  follow llama4's `load_arch_hparams`, and refuse other architectures, which set them
  their own way. `freq_base_swa` is `rope.freq_base_swa`, else `freq_base`.
  `freq_scale_swa` is `rope_freq_scale_train`, the scale before llama_context resets it
  to 1 for scaling type none. `attention.sliding_window_pattern` is read as a scalar,
  as `get_key_or_arr(kid, uint32_t &)` reads it, so an array leaves the default of 4.
- The deepseek2 rows follow `llama_model_deepseek2::load_arch_hparams`, including its
  defaults for older files: `expert_gating_func` is softmax when the key is absent, or
  sigmoid for GLM 4.7 Lite (47 or 48 layers and a vocabulary of 154880), and a lite
  variant's `q_lora_rank` is 0 even when the file has the key. The vocabulary those
  rules compare is `vocab_size` first and the token list second, the reverse of
  `n_vocab`, because that is the order deepseek2 reads them in.
- `mscale` is a symbol rather than an expression because it takes a logarithm, which
  the expression language does not have: deepseek2.cpp computes `attn_factor * (1 +
  0.1 ln(1/freq_scale))`, times `1 + 0.1 * rope_yarn_log_mul * ln(1/freq_scale)`, in f32,
  and the layout writes `kq_scale` as `mscale * mscale / sqrt(n_embd_head_k_mla)`.

### The op vocabulary

An op name is a ggml graph node type: its `ggml_op` enum value in lower case
(`GGML_OP_MUL_MAT` is `mul_mat`), or for `GGML_OP_UNARY` and `GGML_OP_GLU` nodes, the
sub-op (`GGML_UNARY_OP_SILU` is `silu`). That makes each name exactly one kind of node
the compiler must generate a kernel for, and `every_op_name_exists_in_ggml` checks
each against `ggml-ops.txt`, a copy of the three enums in ggml's `ggml.h` taken at
llama.cpp commit 52b3df0023659b142ce29f75c7a82cf437769c33. API function names would not do: `ggml_rope`,
`ggml_rope_ext` and `ggml_rope_ext_inplace` build the same node, `ggml_argsort_top_k`
builds two (`argsort` and `view`), and `ggml_cast` builds a `cpy`. `ggml_view_1d` to
`ggml_view_4d` likewise all build a `view`. A layout written in function names would
hide nodes the kernels must still implement.

| Op | Inputs | Params |
|---|---|---|
| `add`, `sub`, `mul`, `div` | a, b (b broadcasts over a) | |
| `sqr`, `sqrt` | a | |
| `scale` | a | `s`, `b` (0) |
| `clamp` | a | `min`, `max` |
| `sum_rows` | a | |
| `norm`, `rms_norm` | a | `eps` |
| `mul_mat` | a, b | `prec` (`default`, `f32`), `hint` (`none`, `src0_is_hadamard`) |
| `mul_mat_id` | experts, b, ids | |
| `cpy` | a, or a and a destination | `type`, with one input: this is `ggml_cast` |
| `cont` | a | `ne`, optional |
| `reshape` | a | `ne` |
| `view` | a | `ne`, `nb` (bytes, nb1 and up; packed when left out), `offset` (bytes, 0) |
| `permute` | a | `axes` |
| `transpose` | a | |
| `get_rows` | rows, ids | |
| `set_rows` | destination, rows, ids | |
| `soft_max` | a, mask optional | `scale` (1), `max_bias` (0) |
| `rope` | a, positions, frequency factors optional | `n_dims`, `mode` (`norm`, `neox`, `mrope`, `vision`, `imrope`), `n_ctx_orig`, `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow` |
| `flash_attn_ext` | q, k, v, mask optional | `scale`, `max_bias` (0), `logit_softcap` (0), `prec` |
| `argsort` | a | `order` (`asc`, `desc`) |
| `top_k` | a | `k` |
| `silu`, `gelu`, `relu`, `sigmoid`, `tanh`, `exp`, `softplus` | a | |
| `swiglu`, `geglu`, `reglu` | gate and up (the `_split` form), or one input split in half | |
| `concat` | a, b (same type; every dim but `dim` equal) | `dim` |
| `fill` | a (f32 or f16, contiguous) | `c`, the value of every element of a new tensor shaped like a |
| `repeat` | a | `ne`, the result's shape: a whole number of copies of a in every dim |

Each op carries the shape rule from its constructor in `ggml/src/ggml.c`, asserts
included. A rule sees each input's shape, byte strides and type, the same three things
ggml's constructors read from `ne`, `nb` and `type`. So a layout never states an activation's shape: it follows from the weights'
real shapes and the bind options. A weight wired into the wrong matmul fails at bind
time with the assert ggml would have tripped. An op that is not in the table is refused
with a pointer to `ops.rs`; adding one means porting its shape rule, because dovata
cannot plan memory for a node whose output size it does not know. Running the rules at
bind time is what lets activation shapes follow from the weights, as they do in
llama.cpp, and what makes a mistake fail with ggml's assert instead of planning a buffer
of the wrong size.

The size helpers and some shape rules in `ops.rs` are ports of named functions in
`ggml/src/ggml.c`:

| `ops.rs` | ggml |
|---|---|
| `row_size` | `ggml_row_size`; a partial block is refused, since it cannot be addressed |
| `packed_nb` | the strides `ggml_new_tensor_impl` sets for a contiguous tensor |
| `nbytes` | `ggml_nbytes`, the span a tensor covers, contiguous or not |
| `contiguous_n`, `is_contiguous` | `ggml_is_contiguous_n`: dims above `n` must be packed, dims up to `n` may stride |
| `dims4` | the padding with ones that the `_1d` to `_4d` constructors do |
| `broadcast` | `ggml_add_impl` and its siblings, which require `ggml_can_repeat` |
| `cpy` | `ggml_cast(a, type)`, a one-input `cpy` into a new tensor, and `ggml_cpy(a, b)`, which writes into `b` and returns a view of it |
| `view` | `ggml_view_1d` to `_4d`, with `nb` from nb1 up and `offset` both in bytes, as the C call takes them; a view with no `nb` is packed |
| `permute` | `ggml_permute`: input dim `d` goes to position `axes[d]` |
| `set_rows` | `ggml_set_rows(a, b, c)`, which writes the rows of `b` into `a` at the ids in `c` and returns a view of `a` |
| `repeat` | `ggml_repeat(a, b)`, which reads only `b`'s shape; the layout states that shape in `ne`, as `ggml_repeat_4d` takes it, so the node's one input is the one ggml reads |

The inputs of `set_rows` are in that call order, while ggml stores them internally as
`src = [b, c, a]`. `is_view_op` names the ops that only reinterpret their input's
memory and compute nothing: `view`, `reshape`, `permute` and `transpose`. These are the
nodes ggml's backends skip. `set_rows` and a two-input `cpy` also return a view, but they
write into it, so they are compute nodes.

### An example: the qwen3moe block, abridged

This is the MoE half of `builtin/qwen3moe.layout.toml`'s block, which follows
`llm_graph_context::build_moe_ffn` call by call.

```toml
  # build_moe_ffn with softmax gating. Shapes use src.ne*, because the last layer runs
  # it over n_outputs rows rather than n_tokens.
  { op = "mul_mat", in = ["blk.{L}.ffn_gate_inp.weight", "ffn_norm"], out = "ffn_moe_logits" },
  { op = "soft_max", in = ["ffn_moe_logits"], out = "ffn_moe_probs" },
  # ggml_argsort_top_k: a descending argsort, viewed down to the first n_expert_used.
  { op = "argsort", in = ["ffn_moe_probs"], out = "ffn_moe_argsort", order = "desc" },
  { op = "view", in = ["ffn_moe_argsort"], out = "ffn_moe_topk", ne = ["n_expert_used", "src.ne1", "src.ne2", "src.ne3"], nb = ["src.nb1", "src.nb2", "src.nb3"] },
  { op = "reshape", in = ["ffn_moe_probs"], out = "probs", ne = [1, "n_expert", "src.ne1"] },
  { op = "get_rows", in = ["probs", "ffn_moe_topk"], out = "ffn_moe_weights" },
  # qwen3moe.cpp passes norm_w = true: the chosen weights are scaled to sum to 1.
  { op = "reshape", in = ["ffn_moe_weights"], out = "weights", ne = ["n_expert_used", "src.ne2"] },
  { op = "sum_rows", in = ["weights"], out = "ffn_moe_weights_sum" },
  { op = "clamp", in = ["ffn_moe_weights_sum"], out = "ffn_moe_weights_sum_clamped", min = 6.103515625e-5, max = inf },
  { op = "div", in = ["weights", "ffn_moe_weights_sum_clamped"], out = "ffn_moe_weights_norm" },
  { op = "reshape", in = ["ffn_moe_weights_norm"], out = "weights", ne = [1, "n_expert_used", "src.ne1"] },
  { op = "reshape", in = ["ffn_norm"], out = "moe_in", ne = ["n_embd", 1, "src.ne1"] },
  { op = "mul_mat_id", in = ["blk.{L}.ffn_up_exps.weight", "moe_in", "ffn_moe_topk"], out = "ffn_moe_up" },
  { op = "mul_mat_id", in = ["blk.{L}.ffn_gate_exps.weight", "moe_in", "ffn_moe_topk"], out = "ffn_moe_gate" },
  { op = "swiglu", in = ["ffn_moe_gate", "ffn_moe_up"], out = "ffn_moe_swiglu" },
  { op = "mul_mat_id", in = ["blk.{L}.ffn_down_exps.weight", "ffn_moe_swiglu", "ffn_moe_topk"], out = "ffn_moe_down" },
  { op = "mul", in = ["ffn_moe_down", "weights"], out = "ffn_moe_weighted" },
  # Sum over the chosen experts: a view of each, all views first, then an add chain.
  { op = "view", in = ["ffn_moe_weighted"], out = "ffn_moe_out", ne = ["n_embd", "src.ne2"], nb = ["src.nb2"], offset = 0 },
  { op = "view", for = "e in 1..n_expert_used", in = ["ffn_moe_weighted"], out = "cur_experts.{e}", ne = ["n_embd", "src.ne2"], nb = ["src.nb2"], offset = "e * src.nb1" },
  { op = "add", for = "e in 1..n_expert_used", in = ["ffn_moe_out", "cur_experts.{e}"], out = "ffn_moe_out" },
  { op = "cont", when = "n_expert_used == 1", in = ["ffn_moe_out"], out = "ffn_moe_out" },
  { op = "add", in = ["ffn_moe_out", "ffn_inp"], out = "inpL" },
```

The attention half writes the KV cache with `set_rows` and reads it back through views,
as `llama_kv_cache::cpy_k` and `get_k` do, and attends with `flash_attn_ext`. The
built-ins always take llama.cpp's flash-attention path, because it is the one that can
read a quantized V cache, and with it the V cache is not transposed.

## Variants: one mechanism, `when`

Real architectures vary inside one graph, and the format has to say so. The cases
llama.cpp's model code handles are:

- optional tensors with a stand-in: `output.weight` falls back to `token_embd.weight`;
- optional biases: qwen2's Q, K and V biases, llama's output and FFN biases;
- alternative tensor sets: a fused `attn_qkv.weight`, or separate Q, K and V;
- per-layer variants: deepseek2's leading dense blocks then MoE blocks, llama4's
  interleaved MoE and NoPE layers, deci's per-layer head counts and attention-free
  layers, qwen35moe's recurrent and attention layers;
- variants chosen by the bind options: the KV rotation for a quantized cache, the f16
  cast for an f32 cache.

The format uses one mechanism for all of them: a `when` condition on an op or a weight,
over tensor presence (`has(...)`), hparams resolved per layer, the layer index `L`, and
the bind options. A skipped op leaves its `out` name as it was, which is exactly what
llama.cpp's `if (layer.wq_b) Qcur = ggml_add(ctx0, Qcur, layer.wq_b);` does. Two small
helpers cover what a condition would make clumsy: `fallback` for `TENSOR_DUPLICATED`,
and a trailing `?` for an optional source argument such as rope's frequency factors.

The alternatives were weaker. Variants selected by tensor presence alone cover biases
and fused QKV, but not llama4's NoPE layers (no tensor marks them) or the KV rotation
(which depends on the bind options, not the file). Named variant blocks, one full op
list per kind of layer, duplicate everything the variants share (deepseek2's dense and
MoE layers have identical attention), and they cannot express a single optional bias
without multiplying the variants. Per-op conditions mirror how llama.cpp itself is
written: the variation is a scattering of `if`s inside one graph function, and a layout
written the same way can be read against it line by line.

The cost is that a branch of many ops repeats its condition on each line; llama's MoE
path carries `when = "n_expert > 0"` on 18 lines. Bind guards against the two ways a
condition can go wrong. A condition that skips an op reading a weight leaves that weight
unread, which is refused. A condition that skips the op producing a name a later op
reads is a use-before-produce error.

`builtin/deci.layout.toml` shows the costs at their largest. Each layer is one of four
kinds, chosen by its own `n_head`, `n_head_kv` and `n_ff`, so its self-attention branch
carries `n_head > 0 && n_head_kv > 0` on 42 lines and its FFN carries `n_ff > 0` on 14.
It also needs the alias the deepseek2 sketch below asks about: an attention-free layer
sets `cur = inpL` and then `ffn_inp = cur`, so the two ops that read `ffn_inp` are each
written twice, once reading `ffn_inp` and once reading `inpL`. And its output projection
is created in two branches with two shapes; a weight is declared once, so its `dims`
pick the shape with `n_head_kv == 0 ? n_embd : n_embd_head_k * n_head`.

## When a layout is checked

A layout is checked in two stages, so each mistake is reported as early as it can be
found.

`Layout::from_toml` in `format.rs` checks everything that needs no model. A typo then
fails when the layout is read, with its place in the file, and not on the first model
someone binds it to. This stage checks op names, parameter names and kinds, input
counts, the syntax of every expression, the names expressions read, the placeholders in
tensor names, and whether every tensor an op reads is declared or produced somewhere in
the layout. Every `has()` must name a declared weight, since a misspelled one would be
false for every model and quietly drop an op.

`Layout::bind` in `bind.rs` checks what needs a model: presence, shapes and order. The
list of what it refuses ends the next section. Reading a name before the op that
produces it has run is its own error, `UseBeforeProduce`, so a misordered layout is told
apart from a misspelled one. `bind` also refuses a layout for another architecture, for
code that skips `for_model`.

## What binding produces

`Layout::bind(&Model, &BindOptions) -> Result<BoundGraph, LayoutError>`.

A bound graph is one forward step over `n_tokens` tokens, written out in full, and
nothing in it refers back to the layout or the model's metadata. Because a block
activation is named the way llama.cpp's `cb()` names graph tensors, a bound graph can be
compared with a llama.cpp graph dump name for name. The bound graphs of a Q4_K_M file
and a BF16 file of one model differ only in each weight's `ggml_type` and `bytes`.

- `ops` is every op of the step in execution order, each a `BoundOp` with its inputs and
  outputs as `TensorId`s, its layer, and its parameters as concrete values. Enum
  parameters are ggml's integer constants; a view's `nb` and `offset` are bytes. Float
  parameters are kept as f64, and ggml stores them as f32.
- `tensors` is every tensor, each a `BoundTensor` with a name, `dims` (ggml's `ne`,
  trailing 1s dropped), `ggml_type`, `bytes`, layer, `output` flag and kind. The file's
  tensors come first, in file order, so a weight's id is its index in
  `Model::tensors`.
- `TensorKind` is `Weight { part, file_offset }`, `Input`, `Activation`, `Kv`, `State`,
  or `View { of, byte_offset, nb }`. A view's `of` always owns memory, so a view needs
  none of its own and is addressed as `of` plus `byte_offset`. `reshape`, `view`,
  `permute` and `transpose` give views, and so do `set_rows` and a two-input `cpy`,
  which write into their destination and return it. A tensor's kind decides where the
  planner puts it.

The planner derives lifetimes from op order; a use of a view counts as a use of its
`of`. Activation types are what each op's ggml rule gives, which is f32 except where
ggml says otherwise (`argsort` gives i32, a view keeps its source's type, `cpy` gives
its `type`). KV types come from the bind options, and every byte size from
`gguf::ggml_type_layout`.

Bind refuses: a layout for another architecture; a missing required weight; a weight
whose shape differs from its `dims`; an input that resolves to nothing, or to an
activation not yet produced; an op whose inputs break ggml's shape rule; and a file
tensor no op reads that `unused` does not list.

## Two KV caches: llama4's chunked attention

llama4's rope layers attend in chunks of `n_swa` tokens, and llama.cpp keeps their KV in
a second cache, the SWA half of `llama_kv_cache_iswa`. That cache has its own cells, so
it has its own write indices, its own mask and its own rotation matrices. The layout
writes this with what the format already has.

- The second cache's inputs are declared again with a `_swa` suffix, under
  `when = "n_swa > 0"`, since a file that turns chunked attention off has one cache.
- The KV tensors stay one per layer. A chunked layer's is `n_ctx_swa` cells long, and
  its read views span `n_kv_swa` cells: `dims = ["n_embd_k_gqa", "is_swa ? n_ctx_swa : n_ctx"]`.
- Each op that reads one of the two sets is written twice, under `!is_swa` and
  `is_swa`, because `build_attn` picks one tensor or the other per layer:

```toml
  { op = "set_rows", when = "!is_swa", in = ["cache_k_l{L}", "k_cur", "inp_k_idxs"], out = "k_cache" },
  { op = "set_rows", when = "is_swa", in = ["cache_k_l{L}", "k_cur", "inp_k_idxs_swa"], out = "k_cache" },
```

llama4 has seven such pairs: the four Hadamard rotations, the two KV writes and the
attention. This is the pattern the deepseek2 sketch below calls awkward. A skipped op
cannot make one name stand for another, so each consumer is written once per choice.

`n_ctx_swa` is `n_ctx`, the size llama.cpp gives the SWA cache with `swa_full`, the
default of `llama_context_default_params`. llama-cli and llama-server default to
`swa_full = false`, which sizes it `GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256)`: for
Scout at `n_ctx` 32768 with 512-token batches, 8704 cells instead of 32768 in 36 of the
48 layers. That size depends on `n_ubatch`, the largest batch the context was made for.
The bind options have only `n_tokens`, this step's batch, and a persistent cache sized
from it would change size between a prefill step and a decode step. When the bind
options carry the batch size llama.cpp sizes the cache with, only `n_ctx_swa` in
`symbols.rs` needs to change.

What the program writes into the inputs is not part of a layout. For llama4 that is the
chunked mask, where a token at position p sees the cells from `floor(p / n_swa) * n_swa`
to p, and `inp_attn_scale`, the NoPE layers' temperature, `log(floor((p + 1) / 8192) +
1) * 0.1 + 1`. Both use constants that llama4's `load_arch_hparams` sets in code rather
than reads from the file. The layout's header comment records them.

## deepseek2: MLA attention and a K-only cache

`builtin/deepseek2.layout.toml` covers every file llama.cpp loads as deepseek2: DeepSeek
V2, V2-Lite, V3 and R1, Kimi K2, Moonlight, GLM-4.7-Flash, GigaChat3 and Mistral Large 3.
It needed the deepseek2 symbols in the table above, a deepseek2 branch in `attn_factor`,
and the ops `concat`, `fill` and `repeat`. Everything else is the format as it was.

- *Two attention paths, chosen by `is_mla`.* A file converted since llama.cpp split
  `attn_kv_b` in two takes MLA with absorption. Q's non-rope part is multiplied by
  `attn_k_b`, `concat` joins it to the rope part, and K is the compressed KV joined to
  K's one rope head. The cache is `cache_k_l{L}` alone, `n_embd_k_gqa` wide, which is
  `kv_lora_rank + n_rot` with one KV head. V is a view of that cache's first
  `kv_lora_rank` values, as `build_attn` makes it for `llm_graph_input_attn_k`. After
  `flash_attn_ext`, a permute, a `mul_mat` by `attn_v_b`, a permute and a `cont` turn
  the result back into `n_head` heads. That path builds no Hadamard rotation, so the
  rotation inputs and ops carry `!is_mla`. An older file with `attn_kv_b` whole takes
  the legacy path: K and V are expanded before the cache, `repeat` copies K's rope head
  to every head, and the cache, the rotation and the attention are the ordinary ones.
- *Low-rank Q.* `q_lora_rank > 0` selects `attn_q_a`, its norm and `attn_q_b`, and
  `q_lora_rank == 0` selects the one `attn_q`, on the weights and the ops alike.
- *Leading dense blocks.* `L < n_layer_dense_lead` and its complement split the FFN
  weights and ops. An MoE layer adds its shared experts, one dense FFN
  `n_ff_exp * n_expert_shared` wide, to the routed experts' output.
- *Routing.* `expert_gating_func` picks `soft_max` or `sigmoid`. `expert_weights_norm`
  and `expert_weights_scale` turn on the normalization and the scale.
  `n_expert_groups > 1` adds DeepSeek V3's group masking with `fill` and `set_rows`. A
  fused `ffn_gate_up_exps` is split into its gate and up halves by views.
- *Scales.* `kq_scale` is `mscale * mscale / sqrt(n_embd_head_k_mla)`. Mistral Large
  3's `attention.temperature_scale` adds the `inp_attn_scale` input and one `mul` into Q.

The layout binds real headers of DeepSeek-Coder-V2-Lite in both its legacy and its MLA
form, Moonlight-16B-A3B, GigaChat3-10B-A1.8B, GLM-4.7-Flash, DeepSeek-R1-0528, Mistral
Large 3 and Kimi K2, whose five split parts open as one model. No published deepseek2
file has more than one expert group, so only the synthetic header checks that path.

Four places were awkward to write, and are open for review:

- *An alias.* In C++, `selection_probs = probs` makes one pointer stand for another, and
  a layout name cannot stand for another without a node. Each consumer of
  `selection_probs` is written once per case under complementary conditions: the group
  reshape twice, and the expert argsort twice, once reading `selection_probs` and once
  reading `ffn_moe_probs`. A pure alias, one name for another with no node, would
  remove both pairs.
- *A branch that reassigns `cur`.* After the MLA decompression the C++ goes on with
  `cur`, whichever path ran. The two paths' `cur` have different names here,
  `fattn_mla` and `__fattn__`, so the reshape that follows is written twice. Giving
  every reassignment one name avoids it, which is what the routing does for
  `ffn_moe_weights`.
- *A shape-only input.* `ggml_repeat(a, b)` reads only `b`'s shape. The layout states
  that shape in `repeat`'s `ne`, so the node's inputs match ggml's `src`, but the line
  no longer says the shape is `q_pe`'s.
- *A logarithm.* The expression language has none, so `mscale` is a symbol computed in
  Rust instead of a formula in the layout.

## Sketch: qwen35moe (Qwen3.6), delta-net layers with recurrent state

qwen35moe mixes linear-attention layers, which carry recurrent state, with ordinary
attention layers. llama.cpp marks each layer from `attention.recurrent_layers`, or by
default every fourth layer is attention: `is_recr(i) = (i + 1) % full_attention_interval
!= 0`. That is one per-layer symbol, `is_recurrent`, read the way `load_arch_hparams`
reads it, and every variant hangs off it.

The state is declared like the KV cache, per layer, with `kind = "state"`, and only for
the layers that have it:

```toml
tensors = [
  { name = "cache_r_l{L}", kind = "state", dtype = "f32", dims = ["n_embd_r", "n_seq_max"], when = "is_recurrent" },
  { name = "cache_s_l{L}", kind = "state", dtype = "f32", dims = ["n_embd_s", "n_seq_max"], when = "is_recurrent" },
  { name = "cache_k_l{L}", kind = "kv", dtype = "kv_k", dims = ["n_embd_k_gqa", "n_ctx"], when = "!is_recurrent" },
  { name = "cache_v_l{L}", kind = "kv", dtype = "kv_v", dims = ["n_embd_v_gqa", "n_ctx"], when = "!is_recurrent" },
]
```

The block has two attention branches under complementary conditions, then the shared
MoE FFN:

```toml
  # build_layer_attn_linear
  { op = "mul_mat", when = "is_recurrent", in = ["blk.{L}.attn_qkv.weight", "attn_norm"], out = "qkv_mixed" },
  { op = "softplus", when = "is_recurrent", in = ["alpha_biased"], out = "alpha_softplus" },
  # build_conv_state: the previous conv window and this step's input, then the conv
  { op = "ssm_conv", when = "is_recurrent", in = ["conv_input", "blk.{L}.ssm_conv1d.weight"], out = "conv_out" },
  { op = "l2_norm", when = "is_recurrent", in = ["q_conv"], out = "q_conv", eps = "f_norm_rms_eps" },
  { op = "gated_delta_net", when = "is_recurrent", in = ["q_conv", "k_conv", "v_conv", "gate", "beta", "state"], out = "delta_out" },
  # write the new state back, as build_rs and the conv state store do
  { op = "cpy", when = "is_recurrent", in = ["new_state", "cache_s_l{L}"], out = "s_stored" },
  # build_layer_attn, with a gated output and imrope sections
  { op = "mul_mat", when = "!is_recurrent", in = ["blk.{L}.attn_q.weight", "attn_norm"], out = "Qcur" },
```

What it needs beyond round 1: the `is_recurrent` symbol and the SSM hparams
(`ssm_d_conv`, `ssm_d_inner`, `ssm_d_state`, `ssm_dt_rank`, `ssm_n_group`, and the
state widths `n_embd_r`, `n_embd_s`); a bind option for the number of sequences the
state is kept for (`n_seq_max`), since recurrent state is per sequence rather than per
token; the ops `ssm_conv`, `gated_delta_net`, `l2_norm` and `concat` with their shape
rules; and rope's `sections` parameter for `imrope`. The `state` kind already binds,
and the planner already treats it like KV: persistent, in its own region.

Two questions are open for review. About half the block's ops are under `is_recurrent`
and half under its negation, so a grouping construct, a `when` shared by a run of ops,
may be worth adding once this layout exists to measure the repetition. And llama.cpp
updates recurrent state with an in-graph copy whose source rows depend on which
sequences are in the batch (`s_copy`); a compiled program that serves one sequence may
want a simpler form, which is the compiler's call to make with the layout's ops in
front of it.

## Tests

`tests.rs` holds the tests that span the capability: reading layouts, the built-in
layouts, and binding, as code outside the module sees them. Each other file's own tests
cover its internals.

The tests bind against synthetic headers from `testutil.rs`. A layout binds against a
`Model`, and a model is a GGUF header: metadata plus a tensor index. `testutil.rs` writes
one in memory, with the keys preflight requires and a complete tensor index whose
offsets are aligned to 32 bytes, the way gguf.cpp writes the data section. Every
built-in layout can then be bound on any machine, with no model file. The shapes are
small but follow each architecture's `load_arch_tensors` exactly. They are large
enough that every block-quantized type the tests use divides them: rows of 64 and 128,
and a head width of 64, so the KV rotation applies. The builders follow
`llama_model_qwen3moe`, `llama_model_qwen3`, `llama_model_llama` and
`llama_model_llama4`, with the attention weights `create_tensor_qkv` makes. The llama
builder can also write the Mixtral form, llama with experts `n_ff` wide. The llama4
builder writes experts on every `interleave_moe_layer_step`-th layer, which is every
layer for Scout and every second one for Maverick, and can write a fused QKV weight. The
tensor types are chosen per test, which is how one layout is bound to an f16 file and a
quantized one. The deepseek2 builder follows `llama_model_deepseek2` and takes a
`Deepseek2` variant: MLA or a legacy `attn_kv_b`, with or without the low-rank Q, and
the gating, bias, normalization, scale, expert groups, fused gate and up experts and
temperature scale each choosable. Its K cache row is 128 values, so the rule for
rotating a quantized cache holds, and the tests can show that the MLA path still builds
no rotation.

The qwen2, nemotron and deci builders follow `llama_model_qwen2`, `llama_model_nemotron`
and `llama_model_deci`. qwen2 and nemotron create Q as `n_embd` wide, so their tests use
an `n_embd` of 128, two heads of 64. The deci builder takes each layer's `n_head`,
`n_head_kv` and `n_ff` and writes them as per-layer int32 arrays, as
`conversion/deci.py` does, with `Header::i32s`; its test binds one layer of each kind.

`a_real_header_binds_its_builtin_layout` binds a real model when `DOVATA_TEST_HEADER`
names a GGUF file, or the first tens of megabytes of one; step 6 below shows the
command. For a split model, name the first part, with the start of every other part
beside it under llama.cpp's part names, since `Model::open` reads them all. The test
checks the bound KV cache against `ModelDims`, which reads one head count for every
layer. A file whose head counts are per-layer arrays, such as deci's, is priced layer by
layer from the header instead, and an MLA model, which keeps no V cache, is priced
without one.

## Adding a layout

1. Find the architecture's files in a checkout of llama.cpp, from
   https://github.com/ggml-org/llama.cpp, and record its commit in the layout's header
   comment: `src/models/<arch>.cpp` for `load_arch_hparams`, `load_arch_tensors` and
   the graph, and the `build_*` helpers it calls in `src/llama-graph.cpp`. The built-in
   layouts were translated at commit 52b3df0023659b142ce29f75c7a82cf437769c33. A new op
   name also goes into `ggml-ops.txt`, copied from that release's `ggml.h`.
2. Write `weights` from `load_arch_tensors`: one line per `create_tensor`, with
   `optional` for `TENSOR_NOT_REQUIRED`, `fallback` for `TENSOR_DUPLICATED`, and `when`
   for the `if` around it. List in `unused` anything it creates that the graph never
   reads, and say why.
3. Write `tensors` from the `build_inp_*` calls and the memory the graph uses: KV per
   layer, state per recurrent layer.
4. Write the ops in the order the C++ creates them, one line per ggml call, with a
   comment above each group naming the llama.cpp function it comes from. Reference
   hparams through symbols; when a value is a per-architecture constant in llama.cpp,
   such as the rope type from `llama_model_rope_type`, write it with a comment citing
   where llama.cpp sets it.
5. If it needs an op or a symbol dovata lacks, add the op to `ops.rs` with its shape
   rule ported from `ggml.c`, or the symbol to `symbols.rs` reading the key the way
   `load_arch_hparams` does, and add it to the tables in this README.
6. Add the file to `builtin_source` and `SUPPORTED` in `mod.rs`, add a synthetic header
   to `testutil.rs`, and bind it in a test. Then bind a real header: range-fetch the
   first 25 MB of a GGUF and run
   `DOVATA_TEST_HEADER=<file> cargo test -p dovata --lib a_real_header -- --nocapture`.

## Not covered yet

- NVFP4 sidecar scales (`*.scale`, `*.input_scale`). A file with them is refused by the
  unread-tensor check, which names them.
- LongRoPE factor tensors (`rope_factors_long`, `_short`) for llama. Refused the same way.
- LoRA adapters and control vectors, which are run-time additions in llama.cpp.
- The embeddings input path of `build_inp_embd`, for multimodal inputs. Layouts take
  token ids.
- Multiple KV streams. The layouts plan one sequence stream, as `kv_unified` does.
- The smaller SWA cache llama.cpp makes with `swa_full = false`. See "Two KV caches".
