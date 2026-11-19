#!/usr/bin/env python3
"""ANE power from the SMC rail PP0b, without sudo (issue #289).

powermetrics' `ANE Power` is a model estimate, and on Apple M6 it is missing. The
SMC key PP0b is a measured rail that rises with ANE load, but it also feeds the CPU
cluster IOReport calls PACC (M4: the P cores; M5: the Super cores; M6: the Super and
Performance cores). IOReport `PMP / Energy` reports that cluster's power, so

    ANE power ~= (PP0b - PACC) loaded - (PP0b - PACC) idle

Both are read without root: the SMC through the AppleSMC user client, IOReport
through libIOReport, with ctypes only (no native build). RailSampler.create()
returns None on a machine without them, e.g. the off-device CI runners.

The SMC updates PP0b about once a second, so a window needs many seconds. The idle
offset holds while only the calling thread runs and shifts 1-2 W when other threads
load the same cluster.

Run standalone to print a second of readings:
  python3 bench/_ane_rail.py
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import threading

RAIL_KEY = "PP0b"
SAMPLE_S = 0.25   # sampler period; PP0b itself updates about once a second
# fewer distinct PP0b readings than this is flagged: on an M4 under a steady ANE GEMM,
# an 8-reading window's median was within 3-6% of a 90 s window's (2.5 s: up to 76%)
MIN_UPDATES = 8
# the rail-minus-cluster term may dip 1-2 W below its idle value when other threads load
# the cluster; further below, PP0b is not feeding that cluster on this machine (a Mac16,9
# read -12 to -29 W, #304) and the ANE term is not a measurement
RAIL_DEFICIT_W = 3.0
# the IOReport group holding the per-cluster Energy histograms: "PMP" through macOS 27.0,
# "PMP0" on macOS 27.2
PMP_GROUPS = ("PMP", "PMP0")


# estimator (pure, off-device testable)
def _median(v: list[float]) -> float:
    return float(statistics.median(v)) if v else float("nan")


def rail_updates(pp0b: list[float]) -> int:
    """Distinct readings: the SMC holds PP0b between updates, so the sampler sees runs."""
    return sum(1 for i, x in enumerate(pp0b) if i == 0 or x != pp0b[i - 1])


def summarize(loaded: list[dict], idle: list[dict]) -> dict:
    """Idle-subtracted ACTIVE ANE and CPU-cluster power from samples ({pp0b, pacc, eacc} W).

    Each term is the window median (robust to held readings and spikes) minus the idle
    median, clipped at 0. The ANE term is the rail minus its cluster, so the cluster's
    own load (the calling thread) cancels."""
    def ane(s):
        return s["pp0b"] - s["pacc"]

    def cpu(s):
        return s["pacc"] + s["eacc"]

    out: dict = {"n_samples": len(loaded), "n_rail_updates": rail_updates([s["pp0b"] for s in loaded])}
    for name, f in (("ane", ane), ("cpu", cpu)):
        lo, idl = _median([f(s) for s in loaded]), _median([f(s) for s in idle])
        out[f"{name}_loaded_mW"] = lo * 1e3
        out[f"{name}_active_mW"] = max(0.0, lo - idl) * 1e3 if lo == lo and idl == idl else float("nan")
    raw = [ane(s) for s in loaded]
    mean = sum(raw) / len(raw) if raw else float("nan")
    out["ane_cv_pct"] = statistics.pstdev(raw) / mean * 100.0 if len(raw) > 1 and mean > 0 else float("nan")
    flags = []
    if out["n_rail_updates"] < MIN_UPDATES:
        flags.append(f"only {out['n_rail_updates']} PP0b updates - short window, treat as indicative")
    if out["ane_cv_pct"] > 35.0:
        flags.append(f"rail CV {out['ane_cv_pct']:.0f}% (>35%) - low confidence")
    deficit = _median([ane(s) for s in idle]) - _median(raw)
    if deficit > RAIL_DEFICIT_W:
        out["ane_active_mW"] = float("nan")
        flags.append(f"PP0b reads {deficit:.1f} W below its CPU cluster vs idle - the rail does "
                     "not carry that cluster on this machine, no ANE power from it")
    out["flags"] = flags
    return out


# SMC (AppleSMC user client); the struct layout must match the kernel's (80 bytes)
class _SmcVers(ctypes.Structure):
    _fields_ = [("major", ctypes.c_char), ("minor", ctypes.c_char), ("build", ctypes.c_char),
                ("reserved", ctypes.c_char), ("release", ctypes.c_uint16)]


class _SmcPLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16),
                ("cpu", ctypes.c_uint32), ("gpu", ctypes.c_uint32), ("mem", ctypes.c_uint32)]


class _SmcInfo(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("type", ctypes.c_uint32), ("attr", ctypes.c_uint8)]


class _SmcMsg(ctypes.Structure):
    _fields_ = [("key", ctypes.c_uint32), ("vers", _SmcVers), ("plimit", _SmcPLimit),
                ("info", _SmcInfo), ("result", ctypes.c_uint8), ("status", ctypes.c_uint8),
                ("cmd", ctypes.c_uint8), ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]


_SMC_SELECTOR, _SMC_READ_BYTES, _SMC_READ_INFO = 2, 5, 9


def _fourcc(k: str) -> int:
    return int.from_bytes(k.encode("ascii"), "big")


class _Smc:
    def __init__(self):
        assert ctypes.sizeof(_SmcMsg) == 80
        iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_uint32)]
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        iokit.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                                    ctypes.c_size_t, ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_size_t)]
        self._iokit = iokit
        svc = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
        if not svc:
            raise OSError("no AppleSMC service")
        task = ctypes.c_uint32.in_dll(ctypes.CDLL("/usr/lib/libSystem.B.dylib"), "mach_task_self_")
        conn = ctypes.c_uint32(0)
        kr = iokit.IOServiceOpen(svc, task.value, 0, ctypes.byref(conn))
        iokit.IOObjectRelease(svc)
        if kr != 0 or not conn.value:
            raise OSError(f"IOServiceOpen(AppleSMC) failed: {kr}")
        self._conn = conn.value

    def _call(self, msg: _SmcMsg) -> _SmcMsg | None:
        out = _SmcMsg()
        size = ctypes.c_size_t(ctypes.sizeof(out))
        kr = self._iokit.IOConnectCallStructMethod(self._conn, _SMC_SELECTOR, ctypes.byref(msg),
                                                   ctypes.sizeof(msg), ctypes.byref(out), ctypes.byref(size))
        return out if kr == 0 and out.result == 0 else None

    def read_float(self, key: str) -> float | None:
        msg = _SmcMsg(key=_fourcc(key), cmd=_SMC_READ_INFO)
        info = self._call(msg)
        if info is None or info.info.size != 4 or info.info.type != _fourcc("flt "):
            return None
        msg.info.size, msg.cmd = 4, _SMC_READ_BYTES
        out = self._call(msg)
        return ctypes.c_float.from_buffer_copy(bytes(out.bytes[:4])).value if out is not None else None


# IOReport PMP / Energy: per-cluster histograms whose states are watt bins (" 0.250W", "1W")
class _Energy:
    def __init__(self):
        vp = ctypes.c_void_p
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [vp]
        cf.CFArrayGetValueAtIndex.restype = vp
        cf.CFArrayGetValueAtIndex.argtypes = [vp, ctypes.c_long]
        cf.CFRelease.argtypes = [vp]
        ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        ior.IOReportCopyChannelsInGroup.restype = vp
        ior.IOReportCopyChannelsInGroup.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        ior.IOReportCreateSubscription.restype = vp
        ior.IOReportCreateSubscription.argtypes = [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64, vp]
        ior.IOReportCreateSamples.restype = vp
        ior.IOReportCreateSamples.argtypes = [vp, vp, vp]
        ior.IOReportCreateSamplesDelta.restype = vp
        ior.IOReportCreateSamplesDelta.argtypes = [vp, vp, vp]
        ior.IOReportChannelGetChannelName.restype = vp
        ior.IOReportChannelGetChannelName.argtypes = [vp]
        ior.IOReportChannelGetFormat.restype = ctypes.c_int32
        ior.IOReportChannelGetFormat.argtypes = [vp]
        ior.IOReportStateGetCount.restype = ctypes.c_int32
        ior.IOReportStateGetCount.argtypes = [vp]
        ior.IOReportStateGetNameForIndex.restype = vp
        ior.IOReportStateGetNameForIndex.argtypes = [vp, ctypes.c_int32]
        ior.IOReportStateGetResidency.restype = ctypes.c_int64
        ior.IOReportStateGetResidency.argtypes = [vp, ctypes.c_int32]
        self._cf, self._ior = cf, ior
        self._channels_key = self._cfstr("IOReportChannels")
        self._sub = None
        for group in PMP_GROUPS:   # an absent group still returns a (channel-less) dict; subscribing fails
            chans = ior.IOReportCopyChannelsInGroup(self._cfstr(group), self._cfstr("Energy"), 0, 0, 0)
            if not chans:
                continue
            self._subbed = vp()
            self._sub = ior.IOReportCreateSubscription(None, chans, ctypes.byref(self._subbed), 0, None)
            cf.CFRelease(chans)
            if self._sub and self._subbed:
                break
        else:
            raise OSError("no IOReport PMP/Energy group, or its subscription was refused")
        self._prev = ior.IOReportCreateSamples(self._sub, self._subbed, None)

    def _cfstr(self, s: str):
        return self._cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)   # UTF-8

    def _pystr(self, ref) -> str:
        buf = ctypes.create_string_buffer(128)
        return buf.value.decode() if ref and self._cf.CFStringGetCString(ref, buf, 128, 0x08000100) else ""

    def sample(self) -> dict[str, float]:
        """Count-weighted mean W per channel since the previous call."""
        cf, ior = self._cf, self._ior
        cur = ior.IOReportCreateSamples(self._sub, self._subbed, None)
        if not cur:
            return {}
        d = ior.IOReportCreateSamplesDelta(self._prev, cur, None)
        cf.CFRelease(self._prev)
        self._prev = cur
        if not d:
            return {}
        out = {}
        arr = cf.CFDictionaryGetValue(d, self._channels_key)
        for i in range(cf.CFArrayGetCount(arr) if arr else 0):
            ch = cf.CFArrayGetValueAtIndex(arr, i)
            if ior.IOReportChannelGetFormat(ch) != 2:   # 2 = state histogram
                continue
            n = w = 0.0
            for j in range(ior.IOReportStateGetCount(ch)):
                res = ior.IOReportStateGetResidency(ch, j)
                if res > 0:
                    w += float(self._pystr(ior.IOReportStateGetNameForIndex(ch, j)).strip().rstrip("W")) * res
                    n += res
            if n:
                out[self._pystr(ior.IOReportChannelGetChannelName(ch))] = w / n
        cf.CFRelease(d)
        return out


class RailSampler:
    """PP0b + the CPU-cluster power, sampled every SAMPLE_S on a background thread:
    start() before the window, stop() after it returns the samples."""

    def __init__(self, smc: _Smc, energy: _Energy):
        self._smc, self._energy = smc, energy
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def create(cls) -> RailSampler | None:
        """A sampler, or None when this machine has no SMC rail / IOReport (e.g. CI)."""
        if sys.platform != "darwin":
            return None
        try:
            smc = _Smc()
            return cls(smc, _Energy()) if smc.read_float(RAIL_KEY) is not None else None
        except Exception:   # noqa: BLE001 - no rail is "no rail", whatever the cause
            return None

    def _read(self) -> dict | None:
        w = self._smc.read_float(RAIL_KEY)
        if w is None:
            return None
        e = self._energy.sample()
        return {"pp0b": w,
                "pacc": sum(v for k, v in e.items() if k.startswith("PACC")),   # cluster + its SRAM
                "eacc": sum(v for k, v in e.items() if k.startswith("EACC"))}

    def _loop(self):
        while not self._stop.wait(SAMPLE_S):
            s = self._read()
            if s is not None:
                self._samples.append(s)

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._energy.sample()   # restart the histogram delta now
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return list(self._samples)


def main():
    s = RailSampler.create()
    if s is None:
        print("no SMC rail PP0b / IOReport PMP Energy on this machine")
        return
    s.start()
    threading.Event().wait(1.0)
    for x in s.stop():
        print("  ".join(f"{k} {v:.3f}" for k, v in x.items()))


if __name__ == "__main__":
    main()
