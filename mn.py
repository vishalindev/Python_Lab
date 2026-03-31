"""DeepStream RTSP ingestion helper.

This module provides :class:`DeepStreamRTSPIngestion` for receiving camera
frames from an RTSP source using NVIDIA DeepStream/GStreamer bindings.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
except (ImportError, ValueError) as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "GStreamer Python bindings are required. Install PyGObject/GStreamer."
    ) from exc


@dataclass
class FramePacket:
    """Container for frame data received from the RTSP stream."""

    frame: np.ndarray
    pts_ns: Optional[int] = None


class DeepStreamRTSPIngestion:
    """Ingest RTSP video and expose decoded frames.

    The class creates a GStreamer pipeline that decodes H264/H265 RTSP input
    and publishes raw BGR frames through a thread-safe queue.

    Parameters
    ----------
    rtsp_url:
        RTSP camera URL.
    latency_ms:
        Network jitter-buffer latency in milliseconds.
    queue_size:
        Max number of queued frames (oldest frames are dropped when full).
    """

    def __init__(self, rtsp_url: str, latency_ms: int = 200, queue_size: int = 30) -> None:
        if not rtsp_url:
            raise ValueError("rtsp_url must be a non-empty string.")

        Gst.init(None)

        self.rtsp_url = rtsp_url
        self.latency_ms = latency_ms
        self._frame_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=queue_size)

        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._pipeline = Gst.parse_launch(self._build_pipeline_description())
        self._appsink = self._pipeline.get_by_name("ingest_sink")
        if self._appsink is None:
            raise RuntimeError("Failed to create appsink element in pipeline.")

        self._appsink.connect("new-sample", self._on_new_sample)

    def _build_pipeline_description(self) -> str:
        """Build a DeepStream-compatible RTSP ingest pipeline."""
        # If DeepStream decoder plugins are present, decodebin typically picks
        # NVIDIA accelerated paths automatically.
        return (
            f'rtspsrc location="{self.rtsp_url}" latency={self.latency_ms} ! '
            "rtph264depay ! h264parse ! decodebin ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=ingest_sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

    def start(self) -> None:
        """Start the pipeline and begin collecting frames."""
        if self._loop is not None:
            return

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        state_ret = self._pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set GStreamer pipeline to PLAYING state.")

        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Stop streaming and release resources."""
        self._pipeline.set_state(Gst.State.NULL)

        if self._loop is not None:
            self._loop.quit()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)

        self._loop = None
        self._loop_thread = None

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return Gst.FlowReturn.ERROR

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
            packet = FramePacket(frame=frame, pts_ns=buffer.pts)
            if self._frame_queue.full():
                _ = self._frame_queue.get_nowait()
            self._frame_queue.put_nowait(packet)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            self.stop()
            raise RuntimeError(f"GStreamer error: {err}; debug={dbg}")
        if msg_type == Gst.MessageType.EOS:
            self.stop()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return one frame as a NumPy array in BGR format.

        Parameters
        ----------
        timeout:
            Seconds to wait for a frame before returning ``None``.
        """
        try:
            packet = self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return packet.frame

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield frames continuously until the ingestion is stopped."""
        while True:
            frame = self.get_frame(timeout=timeout)
            if frame is not None:
                yield frame
            elif self._loop is None:
                break


if __name__ == "__main__":
    # Example usage
    SAMPLE_RTSP = "rtsp://username:password@camera_ip:554/stream"
    ingest = DeepStreamRTSPIngestion(SAMPLE_RTSP)
    ingest.start()

    try:
        for idx, frame in enumerate(ingest.frames()):
            print(f"Frame {idx}: shape={frame.shape}")
            if idx >= 5:
                break
    finally:
        ingest.stop()
