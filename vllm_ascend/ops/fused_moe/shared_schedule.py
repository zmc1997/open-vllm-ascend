# SPDX-License-Identifier: Apache-2.0
"""Scheme 3 stream dependencies, independent of the tensor/operator backend."""


def run_scheme3(npu, shared_stream, inputs, shared_steps, gate, route):
    main = npu.current_stream()
    ready = main.record_event()
    with npu.stream(shared_stream):
        shared_stream.wait_event(ready)
        inputs.record_stream(shared_stream)
        next(shared_steps)  # quant

    logits = gate()
    gated = main.record_event()
    with npu.stream(shared_stream):
        shared_stream.wait_event(gated)
        next(shared_steps)  # Gateup

    activated = False

    def before_fused_experts():
        nonlocal activated
        if activated:
            raise RuntimeError("scheme 3 routing callback called twice")
        topk_done = main.record_event()
        with npu.stream(shared_stream):
            shared_stream.wait_event(topk_done)
            next(shared_steps)  # SwiGLU
        activated = True

    routed_out = route(logits, before_fused_experts)
    if not activated:
        raise RuntimeError("scheme 3 routing callback was not called")
    # Includes routed finalization/TP reduction, a conservative stronger wait
    # than waiting only for dispatch_ffn_combine.
    routed_done = main.record_event()
    with npu.stream(shared_stream):
        shared_stream.wait_event(routed_done)
        shared_out = next(shared_steps)  # Down
    main.wait_stream(shared_stream)
    shared_out.record_stream(main)
    return shared_out, routed_out
