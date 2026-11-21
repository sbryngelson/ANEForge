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

The other CPU clusters (EACC; MACC0/MACC1 on an M5 Pro) are not on PP0b: with every
core of an M5 Pro busy, PP0b read 43 W beside 43 W of PACC and 57 W of MACC. They
count toward CPU power only.

On an M4, M5 or M6 (SRAM_OFF_RAIL_CHIPS) nor is the cluster's SRAM (the "PACC0 SRAM"
channel). The CPU clusters' DVFS tables in the device tree (pmgr voltage-states1/5) each
come with a separate -sram voltage table, and IOReport SoC Stats / Events has
VDD_CPU_SRAM_VOLTAGE_CHANGE_COMPLETION: the CPU SRAM is its own voltage domain. Its rail
is SMC PP7b, which follows the PACC0 SRAM and EACC0 SRAM channels across ALU-, L1-, L2-
and DRAM-bound loads on 1 to all cores (dPP7b = a dPACC0_SRAM + c dEACC0_SRAM + b:
a 0.68, c 1.09 on an M4 (Mac16,12); 0.67, 1.04 on an M5 (Mac17,3); 0.42, 0.54 on an M6
(Mac18,5); rms 0.06-0.13 W) and not the ANE (+0.33 W under a GEMM that adds 9 W to PP0b,
M6). So on those chips only the cluster's cores are subtracted from PP0b and the SRAM
counts toward CPU power. The ANE's own DVFS tables (voltage-states8/29) have no -sram
table, so its SRAM is taken to be on PP0b with it. On an M5 Pro PP7b reads 0 under load
and PP0b follows PACC plus its SRAM, so there, and on any chip not yet checked, the SRAM
channel stays in the PP0b term.

PACC0 is a model estimate and its error depends on the load, not only on its size: with
1-6 busy threads it read 2-5 W above PP0b on an L1-resident loop and 0.5-1.8 W below it
on DRAM streaming (M4, M6). With only the calling thread busy that stays within ~1 W.

The histograms are coarse and their state names are the bins' upper edges: on an M5 Pro
(macOS 27.2) a cluster has 2 W bins and its SRAM 0.25 W bins, and an idle cluster sits
wholly in the first, "2W". Against PP0b with the ANE idle, PACC read 2.8 W high at idle,
1.8-2.5 W high with one or two busy cores and 0.6 W low with four to six, so the
idle-subtracted ANE term of a CPU-only load came out 0.2-3.3 W, not 0. Read a
rail-sourced ANE power on such a chip as good to a couple of watts. On an M4 (Mac16,12)
and an M6 (Mac18,5), both macOS 27.2.1, and an M5 (Mac17,3, macOS 27.2) the bins are
finer: PACC0 1 W, PACC0 SRAM 0.125 W, EACC0 0.25 W. An idle PACC0 still sits in its first bin, reading 1 W.

The rail is used only on a chip with exactly one PACC cluster (its SRAM channel aside),
where the PACC term is the one cluster PP0b feeds. A chip with several (e.g. the two P
clusters of a Max) gets no sampler, and the benchmark falls back to powermetrics, until
the standalone printout below shows which cluster shares PP0b there.

Checked on (the PACC cluster on PP0b):
  M4      Mac16,12  macOS 27.0.1 and 27.2.1, group PMP, PACC0 (the 4 P cores)
  M5      Mac17,3   macOS 27.2 (and an M5 on 27.0.0), group PMP, PACC0 (the 4 Super cores)
  M6      Mac18,5   macOS 27.0.1 and 27.2.1, group PMP, PACC0 (2 Super + 4 Performance)
  M5 Pro  Mac17,8   group PMP0 (macOS 27.2.1; PMP has no channels, also on 27.0.1),
                    PACC; PP0b does not carry MACC0/MACC1
  M4 Max  Mac16,9   PP0b stayed near 1.4 W under up to 29 W of PACC* (#304)
The IOReport `Energy Model` group (364 channels on an M5 Pro) was tried as a fallback: on
an M4 every channel in it (PCPU, ECPU, CPU Energy, ...) reads 0 without root, so it would
give up the no-sudo advantage. Not worth retrying for that purpose.

Run standalone to list the IOReport channels and their bin widths, then print PP0b and
every channel each second; with --load the second half of the run keeps every CPU core
busy (ANE idle), and the summary shows which channel rises together with PP0b:
  python3 bench/_ane_rail.py [--seconds 6] [--load]
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing
import os
import statistics
import sys
import threading

RAIL_KEY = "PP0b"
# sampler period: PP0b updates every 1.00 s (M4 and M6, under load), so a faster sampler only
# repeats readings and costs GIL time in the timed loop. Just under 1 s, with ~13 ms of
# work per wake-up, every update is still seen and a 10 s window has ~11 samples; at
# exactly 1 s the work pushes the period past the update interval (~9 samples per 10 s).
# A 2 s --quick window (rail beside powermetrics) gets only 2 samples and is flagged by
# MIN_UPDATES; at 0.25 s its 8 samples held only 2-3 distinct PP0b readings anyway
SAMPLE_S = 0.9
# fewer distinct PP0b readings than this is flagged: on an M4 under a steady ANE GEMM,
# an 8-reading window's median was within 3-6% of a 90 s window's (2.5 s: up to 76%)
MIN_UPDATES = 8
# the rail-minus-cluster term may dip 1-2 W below its idle value when other threads load
# the cluster; further below, PP0b is not feeding that cluster on this machine (a Mac16,9
# read -12 to -29 W, #304) and the ANE term is not a measurement
RAIL_DEFICIT_W = 3.0
# the IOReport group holding the per-cluster Energy histograms: "PMP" on M4, M5 and M6
# (macOS 27.0); on an M5 Pro "PMP" has no channels (macOS 27.0.1, #295) and the group is
# "PMP0" (seen on macOS 27.2.1)
PMP_GROUPS = ("PMP", "PMP0")
# chips (machdep.cpu.brand_string, matched exactly) whose CPU SRAM is on SMC PP7b, so PP0b
# carries only the PACC cores (see above). Elsewhere, e.g. the M5 Pro, where PP7b reads 0
# and PP0b follows PACC plus its SRAM, the SRAM channel stays in the PP0b term
SRAM_OFF_RAIL_CHIPS = ("Apple M4", "Apple M5", "Apple M6")


# estimator (pure, off-device testable)
def _median(v: list[float]) -> float:
    return float(statistics.median(v)) if v else float("nan")


def rail_updates(pp0b: list[float]) -> int:
    """Distinct readings: the SMC holds PP0b between updates, so the sampler sees runs."""
    return sum(1 for i, x in enumerate(pp0b) if i == 0 or x != pp0b[i - 1])


def split_clusters(e: dict[str, float], sram_off_rail: bool = False) -> tuple[float, float]:
    """(pacc, eacc) W from IOReport channel -> W: the cluster PP0b feeds, and the CPU power
    it does not carry (EACC*, and MACC* where the chip has them). With sram_off_rail (a chip
    in SRAM_OFF_RAIL_CHIPS) the PACC SRAM channel is on PP7b, so it moves to the second term."""
    def on_rail(k):
        return k.startswith("PACC") and not (sram_off_rail and k.endswith(" SRAM"))
    return (sum(v for k, v in e.items() if on_rail(k)),
            sum(v for k, v in e.items() if k.startswith(("PACC", "EACC", "MACC")) and not on_rail(k)))


def rail_clusters(channels) -> list[str]:
    """The PACC clusters among IOReport channel names, each SRAM channel folded into its
    cluster: ["PACC0"] on M4/M5/M6, ["PACC"] on an M5 Pro, more than one on a Max."""
    return sorted({c.removesuffix(" SRAM") for c in channels if c.startswith("PACC")})


def _channel_medians(samples: list[dict]) -> dict[str, float]:
    """Window median W per channel. A powered-down cluster has no residency, so its channel
    is missing from that sample: count it as 0 W, or a mostly-off cluster reads high."""
    chs = [s["ch"] for s in samples if "ch" in s]
    names = sorted({k for c in chs for k in c})
    return {k: round(_median([c.get(k, 0.0) for c in chs]), 4) for k in names}


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
    # each IOReport channel on its own (window median, W), so a submission shows which
    # cluster moved with PP0b rather than only their sum
    if any("ch" in s for s in loaded):
        out["channels_loaded_W"] = _channel_medians(loaded)
        out["channels_idle_W"] = _channel_medians(idle)
    out["flags"] = flags
    return out


def _chip() -> str:
    """machdep.cpu.brand_string, e.g. "Apple M6" or "Apple M5 Pro"; "" if unreadable."""
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.c_void_p, ctypes.c_size_t]
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(b"machdep.cpu.brand_string", None, ctypes.byref(size), None, 0) != 0:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(b"machdep.cpu.brand_string", buf, ctypes.byref(size), None, 0) != 0:
            return ""
        return buf.value.decode(errors="replace")
    except (OSError, AttributeError):
        return ""


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
        self.group, self.channels = "", []
        self.states: dict[str, list[str]] = {}   # channel -> state (bin) names, filled by sample()
        for group in PMP_GROUPS:   # an absent group still returns a (channel-less) dict; subscribing fails
            chans = ior.IOReportCopyChannelsInGroup(self._cfstr(group), self._cfstr("Energy"), 0, 0, 0)
            if not chans:
                continue
            arr = cf.CFDictionaryGetValue(chans, self._channels_key)
            names = [self._pystr(ior.IOReportChannelGetChannelName(cf.CFArrayGetValueAtIndex(arr, i)))
                     for i in range(cf.CFArrayGetCount(arr) if arr else 0)]
            self._subbed = vp()
            self._sub = ior.IOReportCreateSubscription(None, chans, ctypes.byref(self._subbed), 0, None)
            cf.CFRelease(chans)
            if self._sub and self._subbed:
                self.group, self.channels = group, names
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
            name = self._pystr(ior.IOReportChannelGetChannelName(ch))
            if name not in self.states:
                self.states[name] = [self._pystr(ior.IOReportStateGetNameForIndex(ch, j)).strip()
                                     for j in range(ior.IOReportStateGetCount(ch))]
            n = w = 0.0
            for j in range(ior.IOReportStateGetCount(ch)):
                res = ior.IOReportStateGetResidency(ch, j)
                if res > 0:
                    w += float(self.states[name][j].rstrip("W")) * res
                    n += res
            if n:
                out[name] = w / n
        cf.CFRelease(d)
        return out


class RailSampler:
    """PP0b + the CPU-cluster power, sampled every SAMPLE_S on a background thread:
    start() before the window, stop() after it returns the samples."""

    def __init__(self, smc: _Smc, energy: _Energy, sram_off_rail: bool = False):
        self._smc, self._energy, self.sram_off_rail = smc, energy, sram_off_rail
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def channels(self) -> list[str]:
        return self._energy.channels

    @classmethod
    def create(cls, any_clusters: bool = False) -> RailSampler | None:
        """A sampler, or None when this machine has no SMC rail / IOReport (e.g. CI), or
        when it has more than one PACC cluster: which of them PP0b feeds is not known
        there, and no number beats an unvalidated one. any_clusters=True skips that check
        (the standalone printout, which is how a multi-cluster chip gets checked)."""
        if sys.platform != "darwin":
            return None
        try:
            smc = _Smc()
            if smc.read_float(RAIL_KEY) is None:
                return None
            energy = _Energy()
        except Exception:   # noqa: BLE001 - no rail is "no rail", whatever the cause
            return None
        if not any_clusters and len(rail_clusters(energy.channels)) != 1:
            return None
        return cls(smc, energy, _chip() in SRAM_OFF_RAIL_CHIPS)

    def _read(self) -> dict | None:
        w = self._smc.read_float(RAIL_KEY)
        if w is None:
            return None
        ch = self._energy.sample()
        pacc, eacc = split_clusters(ch, self.sram_off_rail)
        return {"pp0b": w, "pacc": pacc, "eacc": eacc, "ch": ch}

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


def _bins(states: list[str]) -> str:
    """'32 x 1 W (to 32W)': state names are the bins' upper edges, evenly spaced."""
    try:
        edges = [float(x.rstrip("W")) for x in states]
    except ValueError:
        return f"{len(states)} states: {states[:3]}..."
    width = edges[1] - edges[0] if len(edges) > 1 else edges[0]
    return f"{len(edges)} x {width:g} W (to {states[-1]})" if edges else "no states"


def _burn():   # one busy CPU core, run in a child process (no GIL sharing with the sampler)
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=6.0, help="run length (default 6)")
    ap.add_argument("--load", action="store_true",
                    help="keep every CPU core busy for the second half (leave the ANE idle)")
    a = ap.parse_args(argv)
    s = RailSampler.create(any_clusters=True)
    if s is None:
        print("no SMC rail PP0b / IOReport PMP Energy on this machine")
        return
    clusters = rail_clusters(s.channels)
    used = len(clusters) == 1
    print(f"IOReport {s._energy.group}/Energy: {', '.join(s.channels)}")
    print(f"PACC clusters: {', '.join(clusters) or 'none'} -> the benchmark "
          + ("uses the rail" if used else "does not use the rail (needs exactly one)"))
    print(f"{_chip() or 'unknown chip'}: PACC SRAM "
          + ("on PP7b, not subtracted from PP0b" if s.sram_off_rail else "counted as on PP0b (not a checked chip)"))
    s.start()
    half = a.seconds / 2 if a.load else a.seconds
    threading.Event().wait(half)
    procs = []
    if a.load:
        n = os.cpu_count() or 1
        procs = [multiprocessing.Process(target=_burn, daemon=True) for _ in range(n)]
        for p in procs:
            p.start()
        print(f"loading {n} CPU cores (ANE idle) from t={half:.1f} s")
        threading.Event().wait(a.seconds - half)
    samples = s.stop()
    for p in procs:
        p.terminate()
    print("bins (state names are upper edges):")
    for c in s.channels:
        print(f"  {c:16} {_bins(s._energy.states.get(c, []))}")
    if not samples:
        print("no samples")
        return
    cols = sorted({k for x in samples for k in x["ch"]})
    print(f"{'t':>5} {'PP0b':>7} " + " ".join(f"{c:>11}" for c in cols))
    for i, x in enumerate(samples, 1):
        print(f"{i * SAMPLE_S:5.1f} {x['pp0b']:7.2f} "
              + " ".join(f"{x['ch'].get(c, float('nan')):11.2f}" for c in cols))
    if a.load:
        k = max(1, int(half / SAMPLE_S))   # skip the sample that straddles the switch
        idle, busy = samples[:k - 1] or samples[:1], samples[k + 1:] or samples[-1:]
        print("median rise under CPU load (W); the PACC channel(s) that rise like PP0b share the rail:")
        rows = [("PP0b", [x["pp0b"] for x in idle], [x["pp0b"] for x in busy])]
        rows += [(c, [x["ch"].get(c, 0.0) for x in idle], [x["ch"].get(c, 0.0) for x in busy]) for c in cols]
        for name, lo, hi in rows:
            print(f"  {name:16} {_median(lo):7.2f} -> {_median(hi):7.2f}  ({_median(hi) - _median(lo):+.2f})")


if __name__ == "__main__":
    main()
