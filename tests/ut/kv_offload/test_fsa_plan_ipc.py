# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle checks; native Notify execution needs device validation."""

import ast
import ctypes
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


SOURCE = Path(__file__).resolve().parents[3] / (
    "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/fsa_plan_ipc.py"
)


def load_holder():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    # Run the unchanged holder with only its distributed import substituted.
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.Import)
                and any(alias.name == "torch.distributed" for alias in node.names))
    ]
    dist = Mock()
    namespace = {"dist": dist}
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace["FsaPlanIpc"], dist


def pending_holder(rank=0):
    cls, _ = load_holder()
    holder = cls.__new__(cls)
    holder.rank, holder.size, holder._failed = rank, 2, False
    peer = 1 - rank
    holder.owned = {peer: ctypes.c_void_p(11)}
    holder.imported = {peer: ctypes.c_void_p(22)}
    holder.owner_payload = object()
    holder._wait = Mock(return_value=0)
    holder._record = Mock(return_value=0)
    holder._destroy = Mock(return_value=0)
    return holder


class TestFsaPlanIpc(unittest.TestCase):
    def test_destroy_failure_keeps_only_failed_handle_and_can_retry(self):
        holder = pending_holder()
        payload = holder.owner_payload
        imported, owned = holder.imported[1], holder.owned[1]
        holder._destroy.side_effect = [23, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "imported TP1.*23"):
            holder.close()
        self.assertEqual(holder.imported, {1: imported})
        self.assertEqual(holder.owned, {})
        self.assertIs(holder.owner_payload, payload)
        with self.assertRaises(RuntimeError):
            holder.ensure_usable()

        holder.close()
        self.assertEqual(holder.imported, {})
        self.assertEqual(holder.owned, {})
        self.assertIsNone(holder.owner_payload)
        self.assertEqual(holder._destroy.call_args_list,
                         [call(imported), call(owned), call(imported)])
        holder.close()
        self.assertEqual(holder._destroy.call_count, 3)

    def test_ready_and_ack_waits_are_finite_and_rank_ordered(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                holder = pending_holder(rank)
                sequence = Mock()
                sequence.attach_mock(holder._record, "record")
                sequence.attach_mock(holder._wait, "wait")
                holder.before_publish(99)
                holder.after_publish(99)
                timeout = holder._wait.call_args.args[-1]
                self.assertGreater(timeout, 0)
                self.assertLess(timeout, 0xFFFFFFFF)
                peer = 1 - rank
                wait = call.wait(holder.owned[peer], 99, timeout)
                record = call.record(holder.imported[peer], 99)
                self.assertEqual(sequence.mock_calls,
                                 [record, wait] if rank == 0 else [wait, record])

    def test_any_submission_failure_rejects_later_work(self):
        for rank in (0, 1):
            for phase in ("before_publish", "after_publish"):
                for failure in (7, RuntimeError("SDK call failed")):
                    with self.subTest(rank=rank, phase=phase, failure=str(failure)):
                        holder = pending_holder(rank)
                        uses_record = (rank == 0) == (phase == "before_publish")
                        operation = holder._record if uses_record else holder._wait
                        if isinstance(failure, Exception):
                            operation.side_effect = failure
                        else:
                            operation.return_value = failure
                        with self.assertRaises(RuntimeError):
                            getattr(holder, phase)(99)
                        counts = (holder._record.call_count, holder._wait.call_count)
                        for next_phase in ("before_publish", "after_publish"):
                            with self.assertRaisesRegex(RuntimeError, "cannot be used"):
                                getattr(holder, next_phase)(99)
                        self.assertEqual(counts, (holder._record.call_count, holder._wait.call_count))

    def test_failure_mode_is_checked_before_allocation(self):
        for mode, status, expected in (
            (0, 0, "ACL_STOP_ON_FAILURE"),
            (1, 5, "failure mode.*5"),
            (1, 0, "allocation reached"),
        ):
            with self.subTest(mode=mode, status=status):
                cls, dist = load_holder()
                dist.get_rank.return_value = 0
                dist.get_world_size.return_value = 2

                def gather(results, local, **_):
                    results[:] = [local, (2002, None)]

                def get_attribute(stream, attr, output):
                    self.assertEqual((stream, attr), (99, 1))
                    output._obj.failure_mode = mode
                    return status

                def get_pid(output):
                    output._obj.value = 2001
                    return 0

                functions = (
                    "CreateNotify", "DestroyNotify", "NotifyGetExportKey",
                    "NotifySetImportPid", "NotifyImportByKey",
                    "RecordNotify", "WaitAndResetNotify",
                )
                library = SimpleNamespace(**{
                    "aclrt" + name: Mock(return_value=0) for name in functions
                })
                library.aclrtGetStreamAttribute = Mock(side_effect=get_attribute)
                library.aclrtDeviceGetBareTgid = Mock(side_effect=get_pid)
                dist.all_gather_object.side_effect = gather
                allocate = Mock(side_effect=RuntimeError("allocation reached"))
                with patch.object(ctypes, "CDLL", return_value=library):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        cls(object(), allocate, 99)
                self.assertEqual(allocate.call_count, int(mode == 1 and status == 0))
                library.aclrtCreateNotify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
