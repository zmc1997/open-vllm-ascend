# FSA device planning and graph replay

The optional A3 device planner replaces the external CPU LRU planning work.
It preserves the FSA plan layout, cache slot assignment, request ownership,
miss order and cross-stream dependencies. It does not replace attention,
change TopK selection or freeze the plan captured for the first input.

## Configuration and inputs

Set `VLLM_ASCEND_FSA_PLANNER_MODE` consistently on the fused-overlap Decode workers:

| Value | Planning and TP transport |
| --- | --- |
| `cpu` (default) | CPU LRU planning and the existing TP broadcast |
| `hccl` | A3 NPU planning and HCCL broadcast |
| `notify` | A3 NPU planning and shared DDR with native notifications for a host-local TP group |

NPU mode loads the planner library from the installed `vllm_ascend` package.
A missing or incompatible library fails at initialization. Existing launch
scripts can still specify an absolute `VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY`
override. When no mode is set, the legacy `VLLM_ASCEND_FSA_PLAN_IPC=1`
selects `notify`. Otherwise, an explicit library override selects `hccl`.

The planner derives TopK width, selection-cache capacity, maximum simultaneous
rows and maximum token position from the existing model/cache configuration.
It obtains vector-core count and available UB memory from the device resource
query. These values determine allocation sizes and vector tiles at initialization.
There is no required deployment tuple of TopK 2048, capacity 4096, 32 rows,
1M context or TP1/TP2. The TP group already supplied by the framework is used
for plan publication.

The original FSA selection-cache contract remains in force:

- TopK is positive and at most 2048; selection-cache capacity covers TopK.
- The existing FSA dtype, NoPE/RoPE dimensions, layouts, block alignment and
  effective-length rules are unchanged.
- Slot/physical-row fields use the existing signed-int16 plan representation;
  token positions and lengths use signed int32. Their representation limits
  are not model-specific deployment limits.
- The original platform's DBO policy is unchanged. Removing an extra planner
  initialization check does not enable overlapping microbatches in a platform
  that disables them.

TopK 2048 is the original full selection-cache contract, documented when FSA
was introduced in commit `963e2b8f89`. Host shape validation alone does not
prove larger TopK support: the original kernel disables selection-cache plan
consumption and updates outside that contract. This change does not widen the
attention kernel's supported domain.

## Work on each step

```text
initialize from config, tensor geometry and device resources
    -> allocate persistent row state, staging and plan buffers
    -> capture the existing model graph for each batch descriptor

update input tensors, TopK values, request IDs and effective lengths
    -> replay the matching graph
    -> compute this step's new plan from current inputs and prior KV state
    -> publish the plan to the TP group
    -> consume it only after its producer and KV writeback complete
```

Graph replay keeps task shapes and buffer addresses stable. It does not keep
TopK values, lengths, request-to-row assignments or cache decisions constant.
Different batch descriptors use the framework's existing capture entries;
persistent request state is shared through the planner's row mapping.

Capacity changes alter the number of vector tiles in the same planning
algorithm. They do not select a deployment-specific implementation or a
separate scalar paged-sort algorithm. No CPU planner fallback is taken after
explicitly enabling the device library.

`VLLM_ASCEND_FSA_ASYNC_PLAN=0` keeps eager planning on the current compute
stream. With async planning enabled, or during graph capture, the planner uses
the existing auxiliary stream and completion dependencies. CPU KV writeback,
TP publication and owner-layer plan reuse retain their original ordering.

## KV registration and binary compatibility

KV registration is an initialization operation. Re-registering after execution
has stopped drains outstanding work, invalidates graphs and cached attention
parameters referring to the old storage, and rebuilds planner state. The
normal model capture lifecycle must then capture the replacement buffers.
It is not an in-flight cache hot-swap, and old captured graphs must not be
replayed with new storage.

The runtime-geometry implementation uses versioned C entry points. Rebuild
`libfsa_mapped_recurrent_planner.so` together with this checkout; a library
exporting only the old fixed-geometry entry points cannot be loaded with the
new Python caller. See the optional planner build in [installation](../installation.md).

## Validation boundary

CPU tests execute the production planner source through an AscendC host shim
and compare plan outputs and recurrent state with the original CPU planner.
They also cover Python allocation, dispatch and registration lifecycle logic.
A shim checks algorithmic behavior; it does not validate AscendC compilation,
DMA/event behavior, TP communication or real graph replay.

Hardware compilation, same-input eager/graph comparisons and performance
measurements remain separate validation steps. No device or model performance
result is implied by passing the local tests.
