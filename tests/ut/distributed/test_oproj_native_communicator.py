"""CPU-only contract checks of the actual communicator class; no torch import.

python test_oproj_native_communicator.py --source-root /path/to/vllm-ascend -v
"""

import argparse
import ast
import ctypes
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[3]


class Device:
    def __init__(self, value):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        else:
            pieces = value.split(":")
            self.type, self.index = pieces[0], int(pieces[1]) if len(pieces) == 2 else None

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    def __init__(self, shape=(8192,), dtype="int8", device="npu:0", contiguous=True, pointer=0x100000):
        self.shape, self.dtype, self.device = shape, dtype, Device(device)
        self.ndim, self.contiguous, self.pointer = len(shape), contiguous, pointer

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous


class TestOProjNativeCommunicator(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.state = {"available": True, "disabled": False, "symbol": True,
                      "a2a_result": 0, "ar_none": False, "stream": 17,
                      "destroy_errors": (), "shutdown_error": None}
        self.exchange = SimpleNamespace(world_size=2, cpu_group=SimpleNamespace(size=2, backend="gloo", name="exchange"))
        self.total = SimpleNamespace(world_size=4, cpu_group=SimpleNamespace(size=4, backend="gloo", name="total"))
        owner = self

        class Communicator:
            def __init__(self, group, device):
                self.available, self.disabled = owner.state["available"], owner.state["disabled"]
                self.device, self.comm = device, group.name
                self.hccl = SimpleNamespace(lib=object(), HCCL_CHECK=self.check, hcclCommDestroy=self.destroy)
                owner.calls.append(("init", group.name, device.index))

            def check(self, code):
                if code:
                    raise RuntimeError(f"HCCL error {code}")

            def destroy(self, comm):
                owner.calls.append(("destroy", comm))
                if comm in owner.state["destroy_errors"]:
                    raise RuntimeError(f"destroy failed: {comm}")

            def all_reduce(self, partial, stream):
                owner.calls.append(("ar", self.comm, partial, stream.npu_stream))
                return None if owner.state["ar_none"] else Tensor(partial.shape, "bfloat16", pointer=0x300000)

        def signature(*types):
            self.calls.append(("signature", types))

            def bind(symbol):
                if not self.state["symbol"]:
                    raise AttributeError("HcclAlltoAll missing")
                self.calls.append(("bind", symbol[0]))

                def collective(*args):
                    self.calls.append(("a2a", args))
                    return self.state["a2a_result"]

                return collective

            return bind

        def synchronize(device):
            self.calls.append(("sync", device.index))
            if self.state["shutdown_error"] == "sync":
                raise RuntimeError("sync failed")

        def barrier(group):
            self.calls.append(("barrier", group.name))
            if self.state["shutdown_error"] == group.name:
                raise RuntimeError("barrier failed")

        torch = SimpleNamespace(device=Device, Tensor=Tensor, int8="int8", bfloat16="bfloat16",
                                npu=SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=self.state["stream"]),
                                                    synchronize=synchronize))
        dist = SimpleNamespace(Backend=SimpleNamespace(GLOO="gloo"), get_world_size=lambda group: group.size,
                               get_backend=lambda group: group.backend,
                               barrier=barrier)
        path = SOURCE_ROOT / "vllm_ascend/distributed/oproj_communicator.py"
        source = path.read_text()
        compile(source, str(path), "exec")
        tree = ast.parse(source)
        body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OProjNativeCommunicator"]
        self.assertEqual(len(body), 1)
        namespace = {"torch": torch, "dist": dist, "PyHcclCommunicator": Communicator,
                     "ctypes": SimpleNamespace(CFUNCTYPE=signature, c_uint64=ctypes.c_uint64),
                     "hcclResult_t": ctypes.c_int, "hcclDataType_t": ctypes.c_int,
                     "hcclDataTypeEnum": SimpleNamespace(from_torch=lambda dtype: 0),
                     "buffer_type": ctypes.c_void_p, "hcclComm_t": ctypes.c_void_p, "aclrtStream_t": ctypes.c_void_p}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
        self.Communicator = namespace["OProjNativeCommunicator"]

    def make(self):
        return self.Communicator(self.exchange, self.total, "npu:0")

    def test_group_initialization_and_private_abi(self):
        self.make()
        self.assertEqual([c for c in self.calls if c[0] == "init"], [("init", "total", 0), ("init", "exchange", 0)])
        types = next(c[1] for c in self.calls if c[0] == "signature")
        self.assertEqual(types, (ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p))
        self.assertIn(("bind", "HcclAlltoAll"), self.calls)

    def test_invalid_groups_devices_and_unavailable_fail_closed(self):
        for group in (self.exchange, self.total):
            for field in ("world_size", "cpu_size", "backend"):
                with self.subTest(group=group.cpu_group.name, field=field):
                    target, attr = (group, field) if field == "world_size" else (group.cpu_group, "size" if field == "cpu_size" else "backend")
                    old = getattr(target, attr)
                    setattr(target, attr, "hccl" if field == "backend" else 1)
                    with self.assertRaises(ValueError):
                        self.make()
                    setattr(target, attr, old)
        for device in ("cpu:0", "npu"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.Communicator(self.exchange, self.total, device)
        for key, value, error in (("available", False, RuntimeError), ("disabled", True, RuntimeError), ("symbol", False, AttributeError)):
            old = self.state[key]
            self.state[key] = value
            with self.subTest(key=key), self.assertRaises(error):
                self.make()
            self.state[key] = old

    def test_a2a_counts_buffers_and_dynamic_current_stream(self):
        comm = self.make()
        for n in (1, 6):
            self.state["stream"] = 17 + n
            send, recv = Tensor((n * 8192,)), Tensor((n * 8192,), pointer=0x200000)
            self.assertIsNone(comm.all_to_all(send, recv))
            args = self.calls[-1][1]
            self.assertEqual((args[0].value, args[1], args[2], args[3].value, args[4], args[5], args[6], args[7].value),
                             (send.pointer, n * 4096, 0, recv.pointer, n * 4096, 0, "exchange", 17 + n))
        self.assertEqual(sum(c[0] == "init" for c in self.calls), 2)

    def test_a2a_invalid_inputs_and_hccl_error(self):
        comm = self.make()
        cases = [Tensor(dtype="bfloat16"), Tensor(device="npu:1"), Tensor(shape=(2, 4096)), Tensor(contiguous=False),
                 Tensor(shape=(0,)), Tensor(shape=(3,)), Tensor(shape=(8190,))]
        for bad in cases:
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                comm.all_to_all(bad, Tensor(pointer=0x200000))
        for ptr in (0x100000, 0x100001):
            with self.subTest(pointer=ptr), self.assertRaises(ValueError):
                comm.all_to_all(Tensor(), Tensor(pointer=ptr))
        self.state["a2a_result"] = 7
        with self.assertRaisesRegex(RuntimeError, "HCCL error 7"):
            comm.all_to_all(Tensor(), Tensor(pointer=0x200000))

    def test_ar_dtype_current_stream_and_missing_output(self):
        comm = self.make()
        partial = Tensor((12, 8192), "bfloat16")
        self.state["stream"] = 49
        self.assertEqual(comm.all_reduce(partial).shape, partial.shape)
        self.assertEqual(self.calls[-1], ("ar", "total", partial, 49))
        for bad in (Tensor(), Tensor((12, 8192), "bfloat16", "npu:1"), Tensor((12, 8192), "bfloat16", contiguous=False), Tensor((0, 8192), "bfloat16")):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                comm.all_reduce(bad)
        self.state["ar_none"] = True
        with self.assertRaisesRegex(RuntimeError, "no output"):
            comm.all_reduce(partial)

    def test_close_is_collective_idempotent_and_disables_methods(self):
        comm = self.make()
        self.calls.clear()
        comm.close()
        comm.close()
        self.assertTrue(comm._closed)
        self.assertEqual(self.calls, [("sync", 0), ("barrier", "exchange"), ("barrier", "total"),
                                     ("destroy", "exchange"), ("destroy", "total")])
        with self.assertRaises(RuntimeError):
            comm.all_to_all(Tensor(), Tensor(pointer=0x200000))
        with self.assertRaises(RuntimeError):
            comm.all_reduce(Tensor((12, 8192), "bfloat16"))

    def test_destroy_failure_remains_visible_without_retrying_handles(self):
        for failures in (("exchange",), ("total",), ("exchange", "total")):
            with self.subTest(failures=failures):
                self.state["destroy_errors"] = failures
                comm = self.make()
                self.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "destroy failed") as first:
                    comm.close()
                self.assertEqual(str(first.exception), f"destroy failed: {failures[-1]}")
                if len(failures) == 2:
                    self.assertEqual(str(first.exception.__context__), "destroy failed: exchange")
                self.assertFalse(comm._closed)
                self.assertTrue(comm._exchange.disabled and comm._total.disabled)
                self.assertEqual([c for c in self.calls if c[0] == "destroy"],
                                 [("destroy", "exchange"), ("destroy", "total")])
                calls_after_failure = list(self.calls)
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, "previously failed.*recreate the process") as retry:
                        comm.close()
                    self.assertIs(retry.exception.__cause__, first.exception)
                with self.assertRaises(RuntimeError):
                    comm.all_to_all(Tensor(), Tensor(pointer=0x200000))
                with self.assertRaises(RuntimeError):
                    comm.all_reduce(Tensor((12, 8192), "bfloat16"))
                self.assertEqual(self.calls, calls_after_failure)

    def test_shutdown_prerequisite_failure_remains_visible_without_retry(self):
        for failure in ("sync", "exchange", "total"):
            with self.subTest(failure=failure):
                comm = self.make()
                self.state["shutdown_error"] = failure
                self.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "sync failed|barrier failed") as first:
                    comm.close()
                self.assertFalse(comm._closed)
                self.assertTrue(comm._exchange.disabled and comm._total.disabled)
                self.assertFalse(any(c[0] == "destroy" for c in self.calls))
                calls_after_failure = list(self.calls)
                with self.assertRaisesRegex(RuntimeError, "previously failed.*recreate the process") as retry:
                    comm.close()
                self.assertIs(retry.exception.__cause__, first.exception)
                with self.assertRaises(RuntimeError):
                    comm.all_to_all(Tensor(), Tensor(pointer=0x200000))
                with self.assertRaises(RuntimeError):
                    comm.all_reduce(Tensor((12, 8192), "bfloat16"))
                self.assertEqual(self.calls, calls_after_failure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args, remaining = parser.parse_known_args()
    SOURCE_ROOT = args.source_root.resolve()
    unittest.main(argv=[__file__, *remaining])
