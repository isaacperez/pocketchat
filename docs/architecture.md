# PocketChat Architecture (Cortex)

## Purpose

`Cortex` consumes character IDs and produces:
- iterative hypothesis-update state,
- next-character logits for `N` prediction steps.

The model is tokenizer-free and attention-free. Instead of attention, it uses:
- differentiable window sampling (`Glimpse`),
- cosine-similarity matching (`Matcher`),
- iterative hidden-state refinement.

## Tensor Notation

- `B`: batch size
- `L`: input sequence length
- `V`: vocabulary size
- `D`: char embedding dim (`embedding_dim`)
- `H`: number of hypotheses (`num_hypotheses`)
- `K`: number of parts per hypothesis (`num_parts`)
- `W`: glimpse window size (`window_size`)
- `I`: number of internal iterations (`num_iterations`)
- `N`: number of predicted next chars (`num_next_chars`)

Internal dims:
- `D_hidden`: hidden state dim (`hidden_state_dim`)
- `D_h`: hypothesis dim (`hypothesis_dim`)
- `D_o`: observation dim (`observation_dim`)
- `D_u`: updated hypothesis dim (`updated_hypothesis_dim`)

## Component Catalog

### 1. Input Embeddings

- Module: `char_embeddings` (`nn.Embedding`)
- Input: `token_ids` `[B, L]` (or `[L]`)
- Output: `token_embeddings` `[B, L, D]`
- Meaning:
  - This is the model's lexical space. It turns symbolic characters into continuous features that can be compared and transformed.
- Functionality:
  - Learns one vector per character ID.
  - Handles `padding_idx` when configured.
  - Provides the embedding matrix reused for output projection in weight tying.
- Notes:
  - Its embedding matrix is reused by `NextCharPredictor` for weight tying.

### 2. Base Hidden State

- Parameter: `base_hidden_state` (`nn.Parameter`) `[D_hidden]`
- Runtime shape: expanded to `[B, D_hidden]`
- Meaning:
  - This is the starting internal belief before reading evidence from the current input.
- Functionality:
  - Acts as the initial state for each sample.
  - Is refined iteratively through predicted deltas.

### 3. HypothesisGenerator

- Input: hidden state `[B, D_hidden]`
- Output: hypotheses `[B, H, D_h]`
- Architecture: `RMSNorm -> Linear -> GeLU -> Linear`
- Meaning:
  - Produces multiple parallel candidate interpretations of what matters in the current sequence.
- Functionality:
  - Expands one hidden state into `H` hypothesis embeddings.
  - Normalizes and transforms hidden features before splitting into hypotheses.

### 4. HypothesisDecomposer

- Input: hypotheses `[B, H, D_h]`
- Outputs:
  - `center_logits` `[B, H, K]`
  - `zoom_logits` `[B, H, K]`
  - `references` `[B, H, K, D]`
  - `adapters` `[B, H, K, D]`
- Meaning:
  - Breaks each hypothesis into smaller sub-queries ("parts"), where each part decides where to look and what to match.
- Functionality:
  - Predicts `center` and `zoom` controls for Glimpse.
  - Predicts `reference` and `adapter` vectors for Matcher.
  - Enables part-level evidence extraction instead of one monolithic comparison.

### 5. Glimpse

- Inputs:
  - `token_embeddings` `[B, L, D]`
  - flattened controls (`center_logits`, `zoom_logits`) as `[B, H*K]`
  - optional `lengths` `[B]`
- Output:
  - flat windows `[B, H*K, W, D]`, reshaped to `[B, H, K, W, D]`
- Meaning:
  - This is the model's differentiable "where to read" mechanism over the input timeline.
- Functionality:
  - Samples fixed-size windows at continuous positions.
  - Uses linear interpolation so positions are not limited to integer indices.
  - Applies border clamping to keep sampling valid.
  - Keeps gradients through center/zoom controls.

### 6. Matcher

- Inputs:
  - part windows `[B, H, K, W, D]`
  - `references` `[B, H, K, D]`
  - `adapters` `[B, H, K, D]`
- Output: heatmaps `[B, H, K, W]` in `[-1, 1]`
- Meaning:
  - This is the model's "does this looked-at content fit my part hypothesis?" scorer.
- Functionality:
  - Modulates window embeddings with `adapter`.
  - Computes cosine similarity against `reference`.
  - Returns a position-wise evidence heatmap for each part.

### 7. HypothesisObservationComposer

- Input: part heatmaps `[B, H, K, W]`
- Output: observations `[B, H, D_o]`
- Architecture: flatten `(K*W)` then `Linear -> GeLU -> Linear`
- Meaning:
  - Converts distributed local evidence into one compact observation per hypothesis.
- Functionality:
  - Aggregates all part-level heatmaps.
  - Produces one observation embedding for each hypothesis.

### 8. HypothesisUpdater

- Inputs:
  - hypotheses `[B, H, D_h]`
  - observations `[B, H, D_o]`
- Output: updated hypotheses `[B, H, D_u]`
- Architecture: concat then `Linear -> GeLU -> Linear`
- Meaning:
  - Revises each hypothesis after seeing evidence from the sequence.
- Functionality:
  - Fuses prior hypothesis embedding with observation embedding.
  - Produces an updated hypothesis representation.

### 9. HiddenStateDeltaPredictor

- Input: updated hypotheses `[B, H, D_u]`
- Output: hidden-state delta `[B, D_hidden]`
- Architecture: flatten `(H*D_u)` then `Linear -> GeLU -> Linear`
- State update rule:
  - `hidden_state = hidden_state + hidden_state_delta`
- Meaning:
  - Summarizes all updated hypotheses into a single global correction to model state.
- Functionality:
  - Aggregates per-hypothesis updates.
  - Predicts the additive delta used to refine hidden state each iteration.

### 10. NextCharPredictor

- Input: hypothesis update history `[B, H, I, D_u]`
- Output: logits `[B, N, V]`
- Architecture: flatten then `Linear -> GeLU -> Linear`
- Meaning:
  - Translates iterative internal reasoning history into concrete next-character predictions.
- Functionality:
  - Consumes the full hypothesis history across iterations.
  - Produces logits for `N` future character positions.
- Weight tying:
  - In `Cortex`, predictor uses `tie_with_embedding=True`.
  - Final token logits are projected using `char_embeddings.weight`.

## End-to-End Forward Flow

Inputs:
- `token_ids`: `[B, L]` or `[L]`
- `lengths` (optional): `[B]`

Pipeline:
1. Character embedding lookup -> `[B, L, D]`
2. Initialize hidden state from learned base parameter -> `[B, D_hidden]`
3. Repeat `I` times:
   1. Generate hypotheses
   2. Decompose into part controls (`center/zoom`) and matcher signals (`reference/adapter`)
   3. Extract differentiable windows via `Glimpse`
   4. Produce part heatmaps via `Matcher`
   5. Compose heatmaps into observation embeddings
   6. Update hypotheses
   7. Predict hidden-state delta and update hidden state
4. Stack histories:
   - `hypothesis_update_history`: `[B, H, I, D_u]`
   - `hidden_state_delta_history`: `[B, D_hidden, I]`
5. Predict next-char logits with tied projection -> `[B, N, V]`
6. Derive greedy predictions (`argmax`) -> `[B, N]`

## Internal Cortex Diagram

```mermaid
flowchart TD
  A[token_ids BxL] --> B[char_embeddings]
  B --> C[token_embeddings BxLxD]

  D[base_hidden_state D_hidden] --> E[expand to BxD_hidden]
  E --> F{{Iterate I times}}

  C --> G[Glimpse]

  F --> H[HypothesisGenerator\nBxD_hidden -> BxHxD_h]
  H --> I[HypothesisDecomposer\n-> center/zoom/ref/adapter]

  I --> J[center_logits, zoom_logits\nBxHxK]
  I --> K[references, adapters\nBxHxKxD]

  J --> G
  G --> L[part_windows\nBxHxKxWxD]

  L --> M[Matcher]
  K --> M
  M --> N[part_heatmaps\nBxHxKxW]

  N --> O[HypothesisObservationComposer\n-> BxHxD_o]
  H --> P[HypothesisUpdater\n(hypotheses + observations)]
  O --> P

  P --> Q[updated_hypotheses\nBxHxD_u]
  Q --> R[HiddenStateDeltaPredictor\n-> BxD_hidden]
  R --> S[hidden_state += delta]
  S --> F

  Q --> T[stack over iterations\nBxHxIxD_u]
  T --> U[NextCharPredictor\n(weight tied to embeddings)]
  U --> V[next_char_logits\nBxNxV]
```

## Training-Time System Context

Cortex is used inside a broader training pipeline:
- `CharCorpus` for bounded character vocabulary and encoding.
- `NextCharSequenceDataset` for fixed windows + next-`N` targets.
- Variable-length collate policy for train, fixed-length policy for validation.
- Cross-entropy loss selecting per-sample last valid position using `lengths`.

### Training Components: Meaning and Functionality

#### `CharCorpus`

- Meaning:
  - Defines the symbol universe the model can understand and generate.
- Functionality:
  - Reads raw text, builds bounded vocab, encodes/decodes character IDs.

#### `NextCharSequenceDataset`

- Meaning:
  - Converts one long stream into supervised next-step prediction examples.
- Functionality:
  - Produces input windows and next-`N` targets per position.

#### Variable-Length Collate

- Meaning:
  - Forces robustness to varying context lengths during training.
- Functionality:
  - Samples per-example lengths in `[min_seq_len, max_seq_len]`, pads batch, outputs `lengths`.

#### `next_char_cross_entropy_loss`

- Meaning:
  - Trains the model to predict future characters from its internal reasoning state.
- Functionality:
  - Computes CE on logits `[B, N, V]`.
  - Selects the last valid target position per sample using `lengths`.

## Training Data/Model/Loss Diagram

```mermaid
flowchart LR
  A[Raw text file] --> B[CharCorpus.from_file]
  B --> C[token_ids T]

  C --> D[NextCharSequenceDataset\nfixed seq_len=max_seq_len]
  D --> E[Train DataLoader\nvariable lengths + random offset]
  D --> F[Val DataLoader\nfixed length]

  E --> G[batch_inputs BxLmax\nbatch_targets BxLmaxxN\nlengths B]
  F --> H[batch_inputs BxL\nbatch_targets BxLxN\nlengths B]

  G --> I[Cortex.forward(inputs, lengths)]
  H --> I

  I --> J[next_char_logits BxNxV]
  G --> K[next_char_cross_entropy_loss\nwith lengths]
  H --> K
  J --> K

  K --> L[optimizer + scheduler\ncheckpoint/resume]
```

## Forward Outputs (`Cortex.forward`)

Returned dictionary keys:
- `token_embeddings`
- `base_hidden_state`
- `final_hidden_state`
- `hidden_state_delta_history`
- `hypothesis_update_history`
- `last_updated_hypotheses`
- `next_char_logits`
- `next_char_predictions`
