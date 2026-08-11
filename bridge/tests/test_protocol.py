"""Unit tests for bridge/protocol.py's frame + payload (de)serialization."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import protocol as proto  # noqa: E402


class FrameTests(unittest.TestCase):
    def test_round_trip(self):
        frame = proto.encode_frame(proto.LOOKUP, request_id=42, payload=b"hello")
        f = proto.decode_frame(frame)
        self.assertEqual(f.msg_type, proto.LOOKUP)
        self.assertEqual(f.request_id, 42)
        self.assertEqual(f.payload, b"hello")
        self.assertEqual(f.version, proto.PROTOCOL_VERSION)

    def test_empty_payload(self):
        f = proto.decode_frame(proto.encode_frame(proto.PING, 1, b""))
        self.assertEqual(f.payload, b"")

    def test_too_short_raises(self):
        with self.assertRaises(proto.ProtocolError):
            proto.decode_frame(b"\x00" * 4)

    def test_bad_magic_raises(self):
        raw = bytearray(proto.encode_frame(proto.PING, 1, b""))
        raw[0] ^= 0xFF
        with self.assertRaises(proto.ProtocolError):
            proto.decode_frame(bytes(raw))

    def test_payload_len_mismatch_raises(self):
        raw = proto.encode_frame(proto.PING, 1, b"") + b"\x00"  # trailing garbage byte
        with self.assertRaises(proto.ProtocolError):
            proto.decode_frame(raw)

    def test_oversized_payload_rejected_at_encode(self):
        with self.assertRaises(proto.ProtocolError):
            proto.encode_frame(proto.STORE, 1, b"\x00" * (proto.MAX_FRAME_BYTES + 1))


class ReaderWriterTests(unittest.TestCase):
    def test_all_field_types_round_trip(self):
        w = proto.Writer().u8(200).u32(0xDEADBEEF).u64(0x0123456789ABCDEF).i32(-12345).str("héllo")
        r = proto.Reader(w.bytes())
        self.assertEqual(r.u8(), 200)
        self.assertEqual(r.u32(), 0xDEADBEEF)
        self.assertEqual(r.u64(), 0x0123456789ABCDEF)
        self.assertEqual(r.i32(), -12345)
        self.assertEqual(r.str(), "héllo")

    def test_i32_array(self):
        w = proto.Writer()
        for v in (-5, 0, 100, 99999):
            w.i32(v)
        r = proto.Reader(w.bytes())
        self.assertEqual(r.i32_array(4), [-5, 0, 100, 99999])

    def test_underflow_raises_not_crashes(self):
        r = proto.Reader(b"\x01\x02")
        with self.assertRaises(proto.ProtocolError):
            r.u32()

    def test_truncated_string_raises(self):
        # length prefix claims more bytes than are actually present
        raw = proto.Writer().u32(100).bytes()  # no string bytes follow
        r = proto.Reader(raw)
        with self.assertRaises(proto.ProtocolError):
            r.str()


class MessageHelperTests(unittest.TestCase):
    def test_hello_request_round_trip(self):
        payload = (
            proto.Writer()
            .u32(52).u32(2).u32(128).u32(16).u8(1).u32(1).str("muse-glimmer-30B")
            .bytes()
        )
        req = proto.HelloRequest.decode(payload)
        self.assertEqual(req.num_layers, 52)
        self.assertEqual(req.num_kv_heads, 2)
        self.assertEqual(req.head_dim, 128)
        self.assertEqual(req.block_size, 16)
        self.assertTrue(req.int8_kv)
        self.assertEqual(req.elem_bytes, 1)
        self.assertEqual(req.model_name, "muse-glimmer-30B")

    def test_hello_ack_round_trip(self):
        payload = proto.encode_hello_ack(True, "0.5.3", 256)
        r = proto.Reader(payload)
        self.assertEqual(r.u8(), 1)
        self.assertEqual(r.str(), "0.5.3")
        self.assertEqual(r.u32(), 256)
        self.assertEqual(r.str(), "")

    def test_lookup_request_round_trip(self):
        payload = proto.Writer().u32(3).i32(1).i32(2).i32(3).bytes()
        req = proto.LookupRequest.decode(payload)
        self.assertEqual(req.token_ids, [1, 2, 3])

    def test_lookup_resp_round_trip(self):
        chunks = [proto.LookupChunk(0, 256, 0), proto.LookupChunk(256, 256, 65536)]
        payload = proto.encode_lookup_resp(512, "/sparkinfer_kv_1_p", chunks)
        r = proto.Reader(payload)
        self.assertEqual(r.i32(), 512)
        self.assertEqual(r.str(), "/sparkinfer_kv_1_p")
        n = r.u32()
        self.assertEqual(n, 2)
        got = [(r.i32(), r.i32(), r.u64()) for _ in range(n)]
        self.assertEqual(got, [(0, 256, 0), (256, 256, 65536)])

    def test_store_request_round_trip(self):
        payload = (
            proto.Writer().u32(4).i32(10).i32(11).i32(12).i32(13)
            .u32(0).u32(4).str("/sparkinfer_kv_2_c")
            .bytes()
        )
        req = proto.StoreRequest.decode(payload)
        self.assertEqual(req.token_ids, [10, 11, 12, 13])
        self.assertEqual(req.new_start_tok, 0)
        self.assertEqual(req.new_end_tok, 4)
        self.assertEqual(req.shm_name, "/sparkinfer_kv_2_c")

    def test_store_ack_round_trip(self):
        payload = proto.encode_store_ack(False, "layout mismatch")
        r = proto.Reader(payload)
        self.assertEqual(r.u8(), 0)
        self.assertEqual(r.str(), "layout mismatch")


if __name__ == "__main__":
    unittest.main()
