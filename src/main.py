# -*- coding: utf-8 -*-
"""
CPU Scheduling Visualizer (PyQt + Matplotlib) — Dark UI (Final++)
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ---- PyQt imports (PyQt5 preferred, PyQt6 fallback) ----
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QIcon, QColor, QBrush
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QCheckBox,
        QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QTabWidget, QTextBrowser,
        QFrame, QHeaderView, QAbstractItemView, QScrollArea, QRadioButton, QButtonGroup
    )
    QT_API = "PyQt5"
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont, QIcon, QColor, QBrush
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QCheckBox,
        QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QTabWidget, QTextBrowser,
        QFrame, QHeaderView, QAbstractItemView, QScrollArea, QRadioButton, QButtonGroup
    )
    QT_API = "PyQt6"

# ---- Matplotlib embeds ----
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# --------------------------
#  Compat helpers
# --------------------------
def set_header_stretch(header: QHeaderView):
    try:
        header.setSectionResizeMode(QHeaderView.Stretch)  # PyQt5
    except Exception:
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # PyQt6

def set_select_rows(table: QTableWidget):
    try:
        table.setSelectionBehavior(QAbstractItemView.SelectRows)  # PyQt5
    except Exception:
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # PyQt6

def make_bold_font(family="Arial", size=10):
    f = QFont(family, size)
    f.setBold(True)
    return f

# --------------------------
#  Algorithm meta (for PDF + UI)
# --------------------------
ALGO_META = {
    "FIFO / FCFS": {
        "title": "FIFO / FCFS (First-Come, First-Served)",
        "summary": "Non-preemptive. Executes processes strictly in arrival order.",
        "how": "Maintain a single ready queue sorted by arrival time. When CPU becomes free, pick the earliest-arrived process and run it to completion (no preemption).",
        "pros": ["Very simple, low overhead", "Fair by arrival order", "Good baseline for comparison"],
        "cons": ["Convoy effect (long job blocks many short jobs)", "Poor average waiting time on mixed workloads"],
        "notes": ["Convoy effect demo is included in Preset Scenarios."]
    },
    "SJF": {
        "title": "SJF (Shortest Job First) / SPN",
        "summary": "Non-preemptive. Selects the ready process with the smallest burst time.",
        "how": "At each CPU decision, among arrived processes choose the one with minimum burst. Run it to completion.",
        "pros": ["Often minimizes average waiting time", "Great for batch workloads"],
        "cons": ["Starvation risk for long jobs", "Requires burst prediction in real OSes"],
        "notes": ["This tool uses the same implementation for SJF/SPN."]
    },
    "SPN": {
        "title": "SPN (Shortest Process Next) / SJF",
        "summary": "Non-preemptive SJF variant. Chooses smallest burst among ready processes.",
        "how": "Same as SJF: choose the smallest burst among ready jobs, then run to completion.",
        "pros": ["Low average waiting time", "Easy to explain in class"],
        "cons": ["Starvation possible", "Needs burst estimation"],
        "notes": ["We keep both SJF and SPN as separate menu names for educational clarity."]
    },
    "HRRN": {
        "title": "HRRN (Highest Response Ratio Next)",
        "summary": "Non-preemptive. Balances waiting and service time using response ratio.",
        "how": "Compute R = (W + S) / S for each ready process (W=waiting, S=burst). Pick the highest R and run to completion.",
        "pros": ["Reduces starvation compared to SJF", "Good compromise of fairness and efficiency"],
        "cons": ["Still non-preemptive; interactive response can be worse than RR", "Recomputing ratio at each decision"],
        "notes": ["Great for teaching fairness vs efficiency trade-offs."]
    },
    "RR": {
        "title": "RR (Round Robin)",
        "summary": "Preemptive time-sharing scheduler using a fixed quantum.",
        "how": "Use a FIFO ready queue. Run head process for at most quantum. If unfinished, re-queue it at the tail. Decisions happen every quantum or completion.",
        "pros": ["Fair CPU sharing", "Good response time for interactive workloads"],
        "cons": ["Too small quantum → high overhead; too large → behaves like FCFS", "Waiting time may increase for long jobs"],
        "notes": ["Decision Timeline logs arrivals, queue state, requeue/finish, idle, and CS."]
    },
    "SRTF (continuous)": {
        "title": "SRTF (continuous)",
        "summary": "Preemptive SJF. Re-evaluates immediately on every arrival event.",
        "how": "Whenever a new process arrives, compare its remaining time with the currently running one. If the new one is shorter, preempt instantly.",
        "pros": ["Excellent average waiting time", "Short jobs finish quickly"],
        "cons": ["More preemptions → more overhead", "Starvation risk for long jobs"],
        "notes": ["Event-driven implementation. Tie-break: remaining → arrival → PID."]
    },
    "SRTF (quantum-check)": {
        "title": "SRTF (quantum-check)",
        "summary": "Preemptive-ish variant. Re-evaluates only at quantum boundaries (or completion).",
        "how": "Choose the smallest remaining time among ready processes, then run it for quantum (or until completion). Arrivals during the slice do not preempt until the next decision point.",
        "pros": ["More stable than continuous SRTF", "Fewer context switches than continuous SRTF"],
        "cons": ["Less responsive than continuous SRTF", "Still can starve long jobs without aging"],
        "notes": ["Uses the same Quantum input as RR."]
    },
    "MLQ": {
        "title": "MLQ (Multi-Level Queue) — 4 Queues",
        "summary": "Processes are assigned to fixed priority queues. Higher queues have strict priority.",
        "how": "Pick the highest non-empty queue. Schedule within it using that queue’s policy. Lower queues run only when higher queues are empty.",
        "pros": ["Clear class separation", "Easy to demonstrate strict priority"],
        "cons": ["Lower queues can starve", "Rigid: no movement between queues"],
        "notes": ["In this UI, each RR queue can have its own quantum (Q1/Q2/Q3)."]
    },
    "MLFQ": {
        "title": "MLFQ (Multi-Level Feedback Queue) — 4 Queues",
        "summary": "Dynamic priority queues; processes can be demoted/promoted based on behavior.",
        "how": "Jobs start high. If a job uses full quantum, it is demoted. Interactive jobs stay high; CPU-bound jobs drift down. Optional aging/promotion can reduce starvation.",
        "pros": ["Great responsiveness + fairness", "Adapts to behavior"],
        "cons": ["More parameters, harder to tune", "Starvation possible without aging"],
        "notes": ["Enable 'Realistic mode' to activate aging/promotion."]
    },
}

# --------------------------
#  Educational HTML (UI)
# --------------------------
def algo_html(title: str, summary: str, how: str, pros: List[str], cons: List[str], notes: Optional[List[str]] = None) -> str:
    pros_li = "".join([f"<li>{p}</li>" for p in pros])
    cons_li = "".join([f"<li>{c}</li>" for c in cons])
    notes_li = "".join([f"<li>{n}</li>" for n in (notes or [])])

    notes_block = ""
    if notes:
        notes_block = f"""
        <div style="margin-top:10px;
                    background:rgba(90,120,255,0.10);
                    border:1px solid rgba(90,120,255,0.30);
                    border-radius:12px;
                    padding:10px;
                    border-top:3px solid rgba(90,120,255,0.75);">
          <div style="font-weight:900; color:#b7c7ff; margin-bottom:6px;">Notes</div>
          <ul style="margin:0 0 0 18px; padding:0;">{notes_li}</ul>
        </div>
        """

    return f"""
    <div style="font-family:Segoe UI, Arial; font-size:12px; line-height:1.45;">
      <div style="font-size:14px; font-weight:800; margin-bottom:6px;">{title}</div>
      <div style="color:#cfd6e6; margin-bottom:10px;">{summary}</div>

      <div style="margin-bottom:10px;">
        <div style="font-weight:800; margin-bottom:4px;">How it works</div>
        <div style="background:#101217; border:1px solid #2a2f3a; border-radius:10px; padding:8px;">
          {how}
        </div>
      </div>

      <!-- 2-column Pros/Cons table -->
      <table style="width:100%; border-collapse:separate; border-spacing:0; overflow:hidden;
                    border:1px solid #2a2f3a; border-radius:12px; margin-top:6px;">
        <tr>
          <th style="width:50%; text-align:left; padding:10px;
                     background:rgba(40,140,80,0.16);
                     border-bottom:1px solid rgba(40,140,80,0.35);
                     color:#9ff2b8; font-weight:900;">
            Pros
          </th>
          <th style="width:50%; text-align:left; padding:10px;
                     background:rgba(180,70,70,0.16);
                     border-bottom:1px solid rgba(180,70,70,0.35);
                     color:#ffb3b3; font-weight:900;">
            Cons
          </th>
        </tr>
        <tr>
          <td style="vertical-align:top; padding:10px; background:rgba(40,140,80,0.08);
                     border-right:1px solid #2a2f3a;">
            <ul style="margin:0 0 0 18px; padding:0;">{pros_li}</ul>
          </td>
          <td style="vertical-align:top; padding:10px; background:rgba(180,70,70,0.08);">
            <ul style="margin:0 0 0 18px; padding:0;">{cons_li}</ul>
          </td>
        </tr>
      </table>

      {notes_block}
    </div>
    """

ALGO_DESCRIPTIONS = {
    k: algo_html(v["title"], v["summary"], v["how"], v["pros"], v["cons"], v.get("notes"))
    for k, v in ALGO_META.items()
}

# --------------------------
#  Preset Scenarios
# --------------------------
PRESET_SCENARIOS = {
    "None (Manual)": {"description": "Manual input.", "suggested_algo": None, "processes": []},
    "Interactive workload": {
        "description": "Short bursts arriving frequently → RR shines (good response time).",
        "suggested_algo": "RR",
        "processes": [
            {"pid":"P1","arrival":0,"burst":2},
            {"pid":"P2","arrival":1,"burst":1},
            {"pid":"P3","arrival":2,"burst":2},
            {"pid":"P4","arrival":3,"burst":1},
            {"pid":"P5","arrival":4,"burst":2},
        ]
    },
    "CPU-bound workload": {
        "description": "All arrive together, long CPU bursts → SJF minimizes waiting time.",
        "suggested_algo": "SJF",
        "processes": [
            {"pid":"P1","arrival":0,"burst":20},
            {"pid":"P2","arrival":0,"burst":18},
            {"pid":"P3","arrival":0,"burst":16},
            {"pid":"P4","arrival":0,"burst":14},
        ]
    },
    "Mixed workload": {
        "description": "Mixture of short/long bursts → compare FCFS vs SJF vs RR.",
        "suggested_algo": "SJF",
        "processes": [
            {"pid":"P1","arrival":0,"burst":10},
            {"pid":"P2","arrival":1,"burst":2},
            {"pid":"P3","arrival":2,"burst":8},
            {"pid":"P4","arrival":3,"burst":1},
            {"pid":"P5","arrival":4,"burst":4},
        ]
    },
    "Convoy effect (FCFS worst case)": {
        "description": "One long job arrives first, many short jobs stuck behind it → FCFS convoy effect.",
        "suggested_algo": "FIFO / FCFS",
        "processes": [
            {"pid":"P1","arrival":0,"burst":25},
            {"pid":"P2","arrival":1,"burst":2},
            {"pid":"P3","arrival":2,"burst":2},
            {"pid":"P4","arrival":3,"burst":2},
        ]
    },
    "Starvation scenario": {
        "description": "Long job at t=0, short jobs keep arriving → SJF/SRTF can starve long job.",
        "suggested_algo": "SRTF (continuous)",
        "processes": [
            {"pid":"P1","arrival":0,"burst":30},
            {"pid":"P2","arrival":1,"burst":1},
            {"pid":"P3","arrival":2,"burst":1},
            {"pid":"P4","arrival":3,"burst":1},
            {"pid":"P5","arrival":4,"burst":1},
            {"pid":"P6","arrival":5,"burst":1},
        ]
    },
    "Real-time / short jobs burst": {
        "description": "Many short jobs, arrival spread → SRTF finishes them quickly; RR shows fairness.",
        "suggested_algo": "SRTF (continuous)",
        "processes": [
            {"pid":"P1","arrival":0,"burst":3},
            {"pid":"P2","arrival":1,"burst":2},
            {"pid":"P3","arrival":2,"burst":1},
            {"pid":"P4","arrival":4,"burst":2},
            {"pid":"P5","arrival":6,"burst":1},
        ]
    },
    "MLQ class separation demo": {
        "description": "Queue 1 dominates CPU; low queue runs only when high queues empty.",
        "suggested_algo": "MLQ",
        "processes": [
            {"pid":"SYS","arrival":0,"burst":4,"mlq":1},
            {"pid":"UI","arrival":1,"burst":3,"mlq":1},
            {"pid":"BG1","arrival":0,"burst":20,"mlq":3},
            {"pid":"BG2","arrival":5,"burst":15,"mlq":3},
        ]
    },
}

# --------------------------
#  Data models
# --------------------------
@dataclass
class Process:
    pid: str
    arrival: float
    burst: float
    mlq_class: int = 1  # 1..4

@dataclass
class Segment:
    label: str
    start: float
    end: float
    kind: str           # RUN | CS | IDLE
    qlevel: int = 0     # 1..4
    owner: Optional[str] = None  # for CS attached to owner PID


# --------------------------
#  Utils
# --------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lighten_color(color, amount=0.25):
    c = mcolors.to_rgb(color)
    return tuple((1 - amount) * x + amount * 1.0 for x in c)


# --------------------------
#  Scheduler
# --------------------------
class Scheduler:
    def __init__(self, context_switch: float, count_cs_with_idle: bool):
        self.cs = max(0.0, float(context_switch))
        self.count_cs_with_idle = bool(count_cs_with_idle)

    def _idle(self, segs: List[Segment], t0: float, t1: float):
        if t1 > t0:
            segs.append(Segment("IDLE", t0, t1, "IDLE"))

    # 🔧 CHANGED: allow forcing CS even when prev == nxt (for quantum/time-slice decision boundaries)
    def _cs(self, segs: List[Segment], t: float, prev: Optional[str], nxt: Optional[str], force_same_pid: bool = False) -> float:
        if self.cs <= 0:
            return t
        if prev is None or nxt is None:
            return t

        # Only skip "same pid" if not forced
        if (prev == nxt) and (not force_same_pid):
            return t

        if (prev == "IDLE" or nxt == "IDLE") and not self.count_cs_with_idle:
            return t

        segs.append(Segment("CS", t, t + self.cs, "CS", owner=prev))
        return t + self.cs

    def compute_metrics(self, procs: List[Process], segs: List[Segment], force_start_at_zero: bool = True) -> Dict:
        pid_list = [p.pid for p in procs]
        at = {p.pid: p.arrival for p in procs}
        burst = {p.pid: p.burst for p in procs}

        first_start = {pid: None for pid in pid_list}
        completion = {pid: None for pid in pid_list}

        run_only = 0.0
        cs_time = 0.0
        idle_time = 0.0
        cs_count = 0

        if not segs:
            total_time = 0.0
        else:
            start_min = 0.0 if force_start_at_zero else min(s.start for s in segs)
            end_max = max(s.end for s in segs)
            total_time = max(0.0, end_max - start_min)

        for s in segs:
            dur = s.end - s.start
            if s.kind == "RUN":
                run_only += dur
                if s.label in first_start and first_start[s.label] is None:
                    first_start[s.label] = s.start
                if s.label in completion:
                    completion[s.label] = s.end
            elif s.kind == "CS":
                cs_count += 1
                cs_time += dur
            elif s.kind == "IDLE":
                idle_time += dur

        WT, TT, RT = {}, {}, {}
        for pid in pid_list:
            if completion[pid] is None or first_start[pid] is None:
                WT[pid] = TT[pid] = RT[pid] = 0.0
            else:
                TT[pid] = completion[pid] - at[pid]
                RT[pid] = first_start[pid] - at[pid]
                WT[pid] = TT[pid] - burst[pid]

        def avg(d):
            vals = list(d.values())
            return sum(vals) / len(vals) if vals else 0.0

        util_run = (run_only / total_time) * 100.0 if total_time > 0 else 0.0
        util_run_cs = ((run_only + cs_time) / total_time) * 100.0 if total_time > 0 else 0.0
        throughput = (len(pid_list) / total_time) if total_time > 0 else 0.0

        return {
            "WT": WT, "TT": TT, "RT": RT,
            "avg_WT": avg(WT), "avg_TT": avg(TT), "avg_RT": avg(RT),
            "total_time": total_time,
            "idle_time": idle_time,
            "cs_time": cs_time,
            "cs_count": cs_count,
            "cpu_util_run_only": util_run,
            "cpu_util_including_cs": util_run_cs,
            "throughput": throughput,
        }

    # -------- FCFS --------
    def fcfs(self, procs: List[Process]) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []
        t = 0.0
        prev = None
        for p in sorted(procs, key=lambda x: (x.arrival, x.pid)):
            if t < p.arrival:
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, p.arrival)
                log.append(f"t={t:.2f} → IDLE until arrival at {p.arrival:.2f}")
                t = p.arrival
                prev = "IDLE"

            if prev is not None:
                t = self._cs(segs, t, prev, p.pid)

            segs.append(Segment(p.pid, t, t + p.burst, "RUN"))
            log.append(f"t={t:.2f} → Pick {p.pid} (FCFS: earliest arrival). Run to completion.")
            t += p.burst
            prev = p.pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- SJF/SPN --------
    def spn(self, procs: List[Process]) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []
        rem = {p.pid: p.burst for p in procs}
        done = set()
        t = 0.0
        prev = None
        while len(done) < len(procs):
            ready = [p for p in procs if p.pid not in done and p.arrival <= t]
            if not ready:
                nxt = min(p.arrival for p in procs if p.pid not in done)
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                continue

            chosen = min(ready, key=lambda p: (rem[p.pid], p.arrival, p.pid))
            if prev is not None:
                t = self._cs(segs, t, prev, chosen.pid)
            segs.append(Segment(chosen.pid, t, t + rem[chosen.pid], "RUN"))
            log.append(f"t={t:.2f} → Pick {chosen.pid} (SJF/SPN: shortest among ready). Run to completion.")
            t += rem[chosen.pid]
            rem[chosen.pid] = 0.0
            done.add(chosen.pid)
            prev = chosen.pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- HRRN --------
    def hrrn(self, procs: List[Process]) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []
        rem = {p.pid: p.burst for p in procs}
        done = set()
        t = 0.0
        prev = None

        while len(done) < len(procs):
            ready = [p for p in procs if p.pid not in done and p.arrival <= t]
            if not ready:
                nxt = min(p.arrival for p in procs if p.pid not in done)
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                continue

            def ratio(p):
                W = t - p.arrival
                S = rem[p.pid]
                # W+S/S is correct
                return W / S if S > 0 else -1

            chosen = max(ready, key=lambda p: (ratio(p), -p.arrival, p.pid))
            if prev is not None:
                t = self._cs(segs, t, prev, chosen.pid)
            segs.append(Segment(chosen.pid, t, t + rem[chosen.pid], "RUN"))
            log.append(f"t={t:.2f} → Pick {chosen.pid} (HRRN: max (W+S)/S). Run to completion.")
            t += rem[chosen.pid]
            rem[chosen.pid] = 0.0
            done.add(chosen.pid)
            prev = chosen.pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- RR --------
    def rr(self, procs: List[Process], quantum: float) -> Tuple[List[Segment], List[str], Dict]:
        q = max(0.0001, float(quantum))
        segs: List[Segment] = []
        log: List[str] = []
        rem = {p.pid: p.burst for p in procs}
        arrivals = sorted(procs, key=lambda p: (p.arrival, p.pid))
        i = 0
        rq: List[str] = []
        t = 0.0
        prev = None

        def push(now):
            nonlocal i
            added = []
            while i < len(arrivals) and arrivals[i].arrival <= now:
                rq.append(arrivals[i].pid)
                added.append(arrivals[i].pid)
                i += 1
            if added:
                log.append(f"t={now:.2f} → Arrival: +{added} | ReadyQueue={rq}")

        push(t)

        while rq or i < len(arrivals):
            if not rq:
                nxt = arrivals[i].arrival
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                push(t)
                continue

            pid = rq.pop(0)
            if rem[pid] <= 0:
                continue

            log.append(f"t={t:.2f} → Pick {pid} (head of ReadyQueue).")

            if prev is not None:
                # 🔧 CHANGED: force CS at quantum decision boundaries even if prev == pid
                t2 = self._cs(segs, t, prev, pid, force_same_pid=True)
                if t2 != t:
                    log.append(f"t={t:.2f} → CS {prev}→{pid} ({self.cs:.2f})")
                t = t2

            run = min(q, rem[pid])
            segs.append(Segment(pid, t, t + run, "RUN"))
            log.append(f"t={t:.2f} → Run {pid} for {run:.2f} (q={q:.2f}) | remaining {rem[pid]:.2f}→{rem[pid]-run:.2f}")
            t += run
            rem[pid] -= run

            push(t)

            if rem[pid] > 0:
                rq.append(pid)
                log.append(f"t={t:.2f} → Requeue {pid} | ReadyQueue={rq}")
            else:
                log.append(f"t={t:.2f} → Finish {pid} | ReadyQueue={rq}")

            prev = pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- SRTF continuous (FIXED: no segment split unless real preempt/finish) --------
    def srtf_continuous(self, procs: List[Process]) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []

        rem = {p.pid: p.burst for p in procs}
        done = set()

        arrivals = sorted(procs, key=lambda p: (p.arrival, p.pid))
        idx = 0

        def next_arrival_time():
            return arrivals[idx].arrival if idx < len(arrivals) else None

        def ready(now):
            return [p for p in procs if p.pid not in done and p.arrival <= now and rem[p.pid] > 0]

        def pick(now):
            r = ready(now)
            if not r:
                return None
            return min(r, key=lambda p: (rem[p.pid], p.arrival, p.pid)).pid

        # initial idle
        t = 0.0
        prev = None
        first_arr = min((p.arrival for p in procs), default=0.0)
        if first_arr > 0:
            self._idle(segs, 0.0, first_arr)
            log.append(f"t=0.00 → IDLE until first arrival at {first_arr:.2f}")
            t = first_arr
            prev = "IDLE"

        # consume arrivals up to t
        while idx < len(arrivals) and arrivals[idx].arrival <= t:
            log.append(f"t={arrivals[idx].arrival:.2f} → Arrival: +[{arrivals[idx].pid}]")
            idx += 1

        current: Optional[str] = None
        cur_start: Optional[float] = None

        while len(done) < len(procs):
            if current is None:
                pid = pick(t)
                if pid is None:
                    nxt = next_arrival_time()
                    if nxt is None:
                        break
                    if prev is not None:
                        t = self._cs(segs, t, prev, "IDLE")
                    self._idle(segs, t, nxt)
                    log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                    t = nxt
                    prev = "IDLE"
                    while idx < len(arrivals) and arrivals[idx].arrival <= t:
                        log.append(f"t={arrivals[idx].arrival:.2f} → Arrival: +[{arrivals[idx].pid}]")
                        idx += 1
                    continue

                # CS only if real switch (continuous SRTF should NOT force same-pid CS)
                if prev is not None and prev != pid:
                    t2 = self._cs(segs, t, prev, pid)
                    if t2 != t:
                        log.append(f"t={t:.2f} → CS {prev}→{pid} ({self.cs:.2f})")
                    t = t2

                current = pid
                cur_start = t
                log.append(f"t={t:.2f} → Pick {pid} (SRTF: smallest remaining)")

            # run current until next arrival or finish
            nxt_arr = next_arrival_time()
            time_to_finish = rem[current]

            if nxt_arr is None:
                # finish directly
                t_end = t + time_to_finish
                rem[current] = 0.0
                segs.append(Segment(current, cur_start, t_end, "RUN"))
                t = t_end
                done.add(current)
                log.append(f"t={t:.2f} → Finish {current}")
                prev = current
                current = None
                cur_start = None
                continue

            # next event time
            t_event = min(t + time_to_finish, nxt_arr)

            # advance time
            run = max(0.0, t_event - t)
            rem[current] -= run
            t = t_event

            # arrival happened?
            if abs(t - nxt_arr) <= 1e-12:
                # log arrivals at this instant
                while idx < len(arrivals) and abs(arrivals[idx].arrival - t) <= 1e-12:
                    log.append(f"t={arrivals[idx].arrival:.2f} → Arrival: +[{arrivals[idx].pid}]")
                    idx += 1

                # re-evaluate: only preempt if different pid wins
                nxt_pid = pick(t)
                if nxt_pid is not None and nxt_pid != current:
                    # close current run segment at preempt time
                    segs.append(Segment(current, cur_start, t, "RUN"))
                    log.append(f"t={t:.2f} → Preempt {current} (new shorter ready: {nxt_pid})")
                    prev = current
                    current = None
                    cur_start = None
                    continue

                # else: same pid continues -> DO NOT close segment, DO NOT add CS, DO NOT restart
                log.append(f"t={t:.2f} → Arrival event; continue {current}")
                continue

            # finish happened
            if rem[current] <= 1e-12:
                segs.append(Segment(current, cur_start, t, "RUN"))
                done.add(current)
                log.append(f"t={t:.2f} → Finish {current}")
                prev = current
                current = None
                cur_start = None
                continue

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- SRTF quantum-check --------
    def srtf_quantum_check(self, procs: List[Process], quantum: float) -> Tuple[List[Segment], List[str], Dict]:
        q = max(0.0001, float(quantum))
        segs: List[Segment] = []
        log: List[str] = []

        rem = {p.pid: p.burst for p in procs}
        done = set()
        t = 0.0
        prev = None

        arrivals = sorted(procs, key=lambda p: (p.arrival, p.pid))
        i = 0
        ready_list: List[str] = []

        pid_to_arrival = {p.pid: p.arrival for p in procs}

        def push(now):
            nonlocal i
            added = []
            while i < len(arrivals) and arrivals[i].arrival <= now:
                ready_list.append(arrivals[i].pid)
                added.append(arrivals[i].pid)
                i += 1
            if added:
                log.append(f"t={now:.2f} → Arrival: +{added} | Ready={ready_list}")

        def choose(now):
            candidates = [pid for pid in ready_list if pid not in done and rem[pid] > 0 and pid_to_arrival[pid] <= now]
            if not candidates:
                return None
            return min(candidates, key=lambda pid: (rem[pid], pid_to_arrival[pid], pid))

        first_arr = min((p.arrival for p in procs), default=0.0)
        if first_arr > 0:
            self._idle(segs, 0.0, first_arr)
            log.append(f"t=0.00 → IDLE until first arrival at {first_arr:.2f}")
            t = first_arr
            prev = "IDLE"

        push(t)

        while len(done) < len(procs):
            pid = choose(t)
            if pid is None:
                if i >= len(arrivals):
                    break
                nxt = arrivals[i].arrival
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                push(t)
                continue

            if prev is not None:
                # 🔧 CHANGED: force CS at each quantum-check decision boundary, even if same pid
                t2 = self._cs(segs, t, prev, pid, force_same_pid=True)
                if t2 != t:
                    log.append(f"t={t:.2f} → CS {prev}→{pid} ({self.cs:.2f})")
                t = t2

            run = min(q, rem[pid])
            segs.append(Segment(pid, t, t + run, "RUN"))
            log.append(f"t={t:.2f} → Pick {pid} (min remaining). Run slice {run:.2f} (q-check={q:.2f}).")
            t += run
            rem[pid] -= run

            push(t)

            if rem[pid] <= 1e-9:
                done.add(pid)
                log.append(f"t={t:.2f} → Finish {pid}")
            prev = pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- MLQ --------
    def mlq(
        self,
        procs: List[Process],
        q_algos: List[str],
        q_quantums: List[float],
    ) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []
        rem = {p.pid: p.burst for p in procs}
        done = set()
        t = 0.0
        prev = None

        byq: Dict[int, List[Process]] = {1: [], 2: [], 3: [], 4: []}
        for p in procs:
            byq[clamp(int(p.mlq_class), 1, 4)].append(p)

        rr_q: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}

        def next_arrival_any():
            cand = [p.arrival for p in procs if p.pid not in done and rem[p.pid] > 0]
            return min(cand) if cand else None

        def merge_rr_queue(chosen_q: int, ready_list: List[Process]):
            existing = set(rr_q[chosen_q])
            newly = [p.pid for p in sorted(ready_list, key=lambda p: (p.arrival, p.pid)) if p.pid not in existing]
            rr_q[chosen_q].extend(newly)
            rr_q[chosen_q] = [pid for pid in rr_q[chosen_q] if pid not in done and rem.get(pid, 0.0) > 0]

        while len(done) < len(procs):
            chosen_q = None
            ready_list = None
            for qn in [1, 2, 3, 4]:
                ready = [p for p in byq[qn] if p.pid not in done and p.arrival <= t and rem[p.pid] > 0]
                if ready:
                    chosen_q = qn
                    ready_list = ready
                    break

            if chosen_q is None:
                nxt = next_arrival_any()
                if nxt is None:
                    break
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                continue

            algo = q_algos[chosen_q - 1]
            pid = None
            run = 0.0
            reason = ""

            if algo in ("SJF", "SPN"):
                pid = min(ready_list, key=lambda p: (rem[p.pid], p.arrival, p.pid)).pid
                run = rem[pid]
                reason = "SPN"
            elif algo == "FCFS":
                pid = min(ready_list, key=lambda p: (p.arrival, p.pid)).pid
                run = rem[pid]
                reason = "FCFS"
            elif algo == "HRRN":
                def ratio(p):
                    W = t - p.arrival
                    S = rem[p.pid]
                    return (W + S) / S if S > 0 else -1
                pid = max(ready_list, key=lambda p: (ratio(p), -p.arrival, p.pid)).pid
                run = rem[pid]
                reason = "HRRN"
            elif algo == "RR":
                qidx = max(0, min(2, chosen_q - 1))
                q = max(0.0001, float(q_quantums[qidx]))

                merge_rr_queue(chosen_q, ready_list)
                if not rr_q[chosen_q]:
                    rr_q[chosen_q] = [p.pid for p in sorted(ready_list, key=lambda p: (p.arrival, p.pid))]
                pid = rr_q[chosen_q].pop(0)
                run = min(q, rem[pid])
                reason = f"RR q{chosen_q}={q:.2f}"
            else:
                pid = min(ready_list, key=lambda p: (p.arrival, p.pid)).pid
                run = rem[pid]
                reason = "default"

            if prev is not None:
                # 🔧 CHANGED: only force same-pid CS for RR slices in MLQ
                force_same = (algo == "RR")
                t2 = self._cs(segs, t, prev, pid, force_same_pid=force_same)
                if t2 != t:
                    log.append(f"t={t:.2f} → CS {prev}→{pid} ({self.cs:.2f})")
                t = t2

            segs.append(Segment(pid, t, t + run, "RUN", qlevel=chosen_q))
            log.append(f"t={t:.2f} → Queue {chosen_q} selected (strict). Run {pid} ({reason}).")
            t += run
            rem[pid] -= run

            if rem[pid] <= 1e-9:
                done.add(pid)
                log.append(f"t={t:.2f} → Finish {pid}")
            else:
                if algo == "RR":
                    rr_q[chosen_q].append(pid)

            prev = pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m

    # -------- MLFQ --------
    def mlfq(self, procs: List[Process], q_algos: List[str], q1: float, q2: float, q3: float, realistic_aging: bool) -> Tuple[List[Segment], List[str], Dict]:
        segs: List[Segment] = []
        log: List[str] = []
        rem = {p.pid: p.burst for p in procs}
        at = {p.pid: p.arrival for p in procs}
        done = set()
        t = 0.0
        prev = None

        quanta = [max(0.0001,float(q1)), max(0.0001,float(q2)), max(0.0001,float(q3)), None]
        level = {p.pid: 1 for p in procs}
        queues = {1: [],2: [],3: [],4: []}

        arrivals = sorted(procs, key=lambda p:(p.arrival,p.pid))
        i = 0
        aging_interval = 10.0
        last_cpu = {p.pid: None for p in procs}

        def push(now):
            nonlocal i
            added=[]
            while i < len(arrivals) and arrivals[i].arrival <= now:
                pid = arrivals[i].pid
                if pid not in queues[1]:
                    queues[1].append(pid)
                added.append(pid)
                i += 1
            if added:
                log.append(f"t={now:.2f} → Arrival: +{added}")

        def next_arrival():
            return arrivals[i].arrival if i < len(arrivals) else None

        def apply_aging(now):
            if not realistic_aging:
                return
            for pid in list(level.keys()):
                if pid in done or at[pid] > now or level[pid] <= 1:
                    continue
                last = last_cpu[pid] if last_cpu[pid] is not None else at[pid]
                if now - last >= aging_interval:
                    old = level[pid]
                    new = old - 1
                    level[pid] = new
                    if pid in queues[old]:
                        queues[old].remove(pid)
                    if pid not in queues[new]:
                        queues[new].append(pid)
                    last_cpu[pid] = now
                    log.append(f"t={now:.2f} → Aging promote {pid}: Q{old}→Q{new}")

        first_arr = min((p.arrival for p in procs), default=0.0)
        if first_arr > 0:
            self._idle(segs, 0.0, first_arr)
            log.append(f"t=0.00 → IDLE until first arrival at {first_arr:.2f}")
            t = first_arr
            prev = "IDLE"

        push(t)

        while len(done) < len(procs):
            apply_aging(t)

            chosen_q = None
            pid = None
            run = 0.0

            for qn in [1,2,3,4]:
                ready = [x for x in queues[qn] if x not in done and rem[x] > 0 and at[x] <= t]
                if not ready:
                    continue
                algo = q_algos[qn-1]
                if qn <= 3:
                    quantum = quanta[qn-1]
                    if algo == "SRTF":
                        pid = min(ready, key=lambda x:(rem[x], at[x], x))
                    else:
                        pid = ready[0]
                    run = min(quantum, rem[pid])
                    chosen_q = qn
                    break
                else:
                    if algo in ("SJF","SPN"):
                        pid = min(ready, key=lambda x:(rem[x], at[x], x))
                    elif algo == "HRRN":
                        def ratio(x):
                            W = t - at[x]
                            S = rem[x]
                            return (W+S)/S if S>0 else -1
                        pid = max(ready, key=lambda x:(ratio(x), -at[x], x))
                    else:
                        pid = min(ready, key=lambda x:(at[x], x))
                    run = rem[pid]
                    chosen_q = qn
                    break

            if pid is None:
                nxt = next_arrival()
                if nxt is None:
                    break
                if prev is not None:
                    t = self._cs(segs, t, prev, "IDLE")
                self._idle(segs, t, nxt)
                log.append(f"t={t:.2f} → IDLE until arrival at {nxt:.2f}")
                t = nxt
                prev = "IDLE"
                push(t)
                continue

            if prev is not None:
                # 🔧 CHANGED: MLFQ queues 1..3 are time-sliced => force CS even if same pid
                force_same = (chosen_q is not None and chosen_q <= 3)
                t2 = self._cs(segs, t, prev, pid, force_same_pid=force_same)
                if t2 != t:
                    log.append(f"t={t:.2f} → CS {prev}→{pid} ({self.cs:.2f})")
                t = t2

            segs.append(Segment(pid, t, t+run, "RUN", qlevel=chosen_q))
            log.append(f"t={t:.2f} → Pick {pid} from Q{chosen_q}. Run {run:.2f}.")
            t += run
            rem[pid] -= run
            last_cpu[pid] = t

            if pid in queues[chosen_q]:
                queues[chosen_q].remove(pid)

            push(t)

            if rem[pid] <= 1e-9:
                done.add(pid)
                log.append(f"t={t:.2f} → Finish {pid}")
            else:
                if chosen_q <= 3:
                    used_full = run >= quanta[chosen_q-1] - 1e-9
                    if used_full:
                        newq = min(4, chosen_q+1)
                        level[pid] = newq
                        queues[newq].append(pid)
                        log.append(f"t={t:.2f} → Demote {pid} to Q{newq}")
                    else:
                        queues[chosen_q].append(pid)
                else:
                    queues[4].append(pid)

            prev = pid

        m = self.compute_metrics(procs, segs, True)
        return segs, log, m


# --------------------------
#  Gantt Canvas
# --------------------------
class GanttCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def draw_gantt(
        self,
        processes: List[Process],
        segments: List[Segment],
        title: str,
        pid_colors: Dict[str, Tuple[float, float, float]],
        y_order: List[str],
        show_queue_styles: bool = False,
        show_arrival_markers: bool = True,
    ):
        ax = self.ax
        ax.clear()

        y_index = {pid: i for i, pid in enumerate(y_order)}
        bar_h = 0.70

        idle_color = (0.60, 0.60, 0.60)
        cs_edge = (1.0, 0.65, 0.15)

        hatch_map = {1: None, 2: "//", 3: "xx", 4: ".."}
        alpha_map = {1: 1.00, 2: 0.82, 3: 0.68, 4: 0.55}

        for s in segments:
            if s.kind == "IDLE":
                ax.axvspan(s.start, s.end, facecolor=idle_color, alpha=0.12, edgecolor=None, zorder=0)

        total_end = 0.0

        for s in segments:
            total_end = max(total_end, s.end)

            if s.kind == "RUN":
                pid = s.label
                if pid not in y_index:
                    continue
                y = y_index[pid]
                base = pid_colors.get(pid, (0.2, 0.5, 0.9))

                qn = s.qlevel if show_queue_styles else 0
                hatch = hatch_map.get(qn, None)
                alpha = alpha_map.get(qn, 1.0) if show_queue_styles else 1.0

                ax.barh(
                    y, s.end - s.start, left=s.start, height=bar_h,
                    color=base, edgecolor="black", linewidth=0.8,
                    alpha=alpha, hatch=hatch, zorder=3
                )
                ax.text(
                    s.start + (s.end - s.start)/2.0, y, pid,
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=4
                )

            elif s.kind == "CS":
                owner = s.owner
                if owner is None or owner not in y_index:
                    continue
                y = y_index[owner]
                base = pid_colors.get(owner, (0.2, 0.5, 0.9))
                cs_fill = lighten_color(base, 0.28)

                ax.barh(
                    y, s.end - s.start, left=s.start, height=bar_h,
                    color=cs_fill, edgecolor=cs_edge, linewidth=1.2,
                    alpha=0.95, zorder=5
                )
                if (s.end - s.start) >= 0.35:
                    ax.text(
                        s.start + (s.end - s.start)/2.0, y, "CS",
                        ha="center", va="center", color="black",
                        fontsize=8, fontweight="bold", zorder=6
                    )

        if show_arrival_markers:
            trans = ax.get_xaxis_transform()
            for p in processes:
                if p.pid in y_index:
                    y = y_index[p.pid]
                    ax.plot([p.arrival], [y], marker="v", markersize=6, color="black", zorder=7)
                    ax.text(p.arrival, -0.10, f"{p.arrival:g}", transform=trans,
                            ha="center", va="top", fontsize=8, color="#e6e6e6")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Process")

        ax.set_yticks(list(range(len(y_order))))
        ax.set_yticklabels(y_order)

        end = max(1.0, total_end)
        end_ceil = math.ceil(end * 1.0001)
        ax.set_xlim(0, end_ceil)

        span = end_ceil
        step = 1.0
        if span > 50:
            step = 5.0
        elif span > 20:
            step = 2.0

        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_minor_locator(MultipleLocator(step/2.0))

        ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.35)
        ax.grid(True, axis="x", which="minor", linestyle=":", alpha=0.20)

        ax.set_ylim(-0.8, len(y_order) - 0.2)

        idle_patch = mpatches.Patch(color=idle_color, alpha=0.12, label="IDLE")
        cs_patch = mpatches.Patch(facecolor=lighten_color((0.2,0.5,0.9),0.28), edgecolor=(1.0,0.65,0.15), label="CS (attached)")
        arr_handle = Line2D([0],[0], marker='v', linestyle='None', markersize=6, color='black', label='Arrival')
        handles = [idle_patch, cs_patch, arr_handle]
        if show_queue_styles:
            for qn in [1,2,3,4]:
                handles.append(mpatches.Patch(facecolor=(0.2,0.5,0.9), alpha=alpha_map[qn], hatch=hatch_map[qn], label=f"Queue {qn} style"))

        ax.legend(handles=handles, loc="lower right", framealpha=0.75, fontsize=8)

        self.fig.tight_layout()
        self.draw()


# --------------------------
#  PDF Report (ReportLab) — REAL landscape template for Gantt
# --------------------------
def export_pdf_report(path: str, algo: str, procs: List[Process], metrics: Dict, gantt_png_path: str):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame,
        Paragraph, Spacer, Table, TableStyle, Image,
        PageBreak, NextPageTemplate
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=8)
    h_style = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10.5, leading=13)

    # Frames
    portrait_pagesize = A4
    landscape_pagesize = landscape(A4)

    p_left = 1.5*cm; p_right = 1.5*cm; p_top = 1.2*cm; p_bottom = 1.2*cm
    l_left = 1.2*cm; l_right = 1.2*cm; l_top = 1.0*cm; l_bottom = 1.0*cm

    portrait_frame = Frame(
        p_left, p_bottom,
        portrait_pagesize[0]-p_left-p_right,
        portrait_pagesize[1]-p_top-p_bottom,
        id="portrait_frame"
    )

    landscape_frame = Frame(
        l_left, l_bottom,
        landscape_pagesize[0]-l_left-l_right,
        landscape_pagesize[1]-l_top-l_bottom,
        id="landscape_frame"
    )

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        w, h = canvas._pagesize
        canvas.drawRightString(w - 1.2*cm, 0.85*cm, f"CPU Scheduling Visualizer  •  Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(path, pagesize=portrait_pagesize)

    doc.addPageTemplates([
        PageTemplate(id="PORT", frames=[portrait_frame], pagesize=portrait_pagesize, onPage=on_page),
        PageTemplate(id="LAND", frames=[landscape_frame], pagesize=landscape_pagesize, onPage=on_page),
    ])

    elems = []

    elems.append(Paragraph("CPU Scheduling Visualizer — Report", title_style))
    elems.append(Paragraph(f"<b>Algorithm:</b> {algo}", body_style))
    elems.append(Spacer(1, 8))

    summary_data = [
        ["Metric", "Value"],
        ["CPU Utilization (RUN)", f"{metrics.get('cpu_util_run_only',0):.2f}%"],
        ["CPU Utilization (RUN+CS)", f"{metrics.get('cpu_util_including_cs',0):.2f}%"],
        ["Throughput", f"{metrics.get('throughput',0):.4f} / time unit"],
        ["Total Time", f"{metrics.get('total_time',0):.2f}"],
        ["Total Context Switches", f"{metrics.get('cs_count',0)}"],
        ["Avg Waiting Time", f"{metrics.get('avg_WT',0):.2f}"],
        ["Avg Turnaround Time", f"{metrics.get('avg_TT',0):.2f}"],
        ["Avg Response Time", f"{metrics.get('avg_RT',0):.2f}"],
    ]
    summary_tbl = Table(summary_data, colWidths=[7*cm, 7*cm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9edf5")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c7cfdd")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    elems.append(summary_tbl)

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Processes", h_style))

    show_mlq = (algo == "MLQ")
    if show_mlq:
        proc_data = [["PID", "Arrival", "Burst", "MLQ Queue"]]
        colw = [3.2*cm, 3.3*cm, 3.3*cm, 4.2*cm]
        for p in procs:
            proc_data.append([p.pid, f"{p.arrival:g}", f"{p.burst:g}", str(p.mlq_class)])
    else:
        proc_data = [["PID", "Arrival", "Burst"]]
        colw = [4.5*cm, 5.0*cm, 5.0*cm]
        for p in procs:
            proc_data.append([p.pid, f"{p.arrival:g}", f"{p.burst:g}"])

    proc_tbl = Table(proc_data, colWidths=colw)
    proc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9edf5")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c7cfdd")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    elems.append(proc_tbl)

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Per-process Metrics", h_style))
    WT, TT, RT = metrics.get("WT",{}), metrics.get("TT",{}), metrics.get("RT",{})
    met_data = [["PID", "RT", "WT", "TT"]]
    for p in procs:
        met_data.append([p.pid, f"{RT.get(p.pid,0):.2f}", f"{WT.get(p.pid,0):.2f}", f"{TT.get(p.pid,0):.2f}"])
    met_data.append(["AVG", f"{metrics.get('avg_RT',0):.2f}", f"{metrics.get('avg_WT',0):.2f}", f"{metrics.get('avg_TT',0):.2f}"])
    met_tbl = Table(met_data, colWidths=[3.2*cm, 4.0*cm, 4.0*cm, 4.6*cm])
    met_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9edf5")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c7cfdd")),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#f7f9fc")]),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#eef2ff")),
        ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    elems.append(met_tbl)

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Algorithm Explanation", h_style))

    meta = ALGO_META.get(algo, None)
    if meta:
        elems.append(Paragraph(f"<b>{meta['title']}</b><br/>{meta['summary']}", body_style))
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(f"<b>How it works:</b> {meta['how']}", body_style))
        elems.append(Spacer(1, 8))

        pros = meta.get("pros", [])
        cons = meta.get("cons", [])
        pros_html = "<br/>".join([f"• {p}" for p in pros]) if pros else "—"
        cons_html = "<br/>".join([f"• {c}" for c in cons]) if cons else "—"

        pc_tbl = Table(
            [["Pros", "Cons"], [Paragraph(pros_html, body_style), Paragraph(cons_html, body_style)]],
            colWidths=[8.0*cm, 8.0*cm]
        )
        pc_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#e9f7ee")),
            ("BACKGROUND", (1,0), (1,0), colors.HexColor("#fdecec")),
            ("TEXTCOLOR", (0,0), (0,0), colors.HexColor("#145a32")),
            ("TEXTCOLOR", (1,0), (1,0), colors.HexColor("#922b21")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c7cfdd")),
            ("VALIGN", (0,1), (-1,1), "TOP"),
            ("BACKGROUND", (0,1), (0,1), colors.HexColor("#f3fbf6")),
            ("BACKGROUND", (1,1), (1,1), colors.HexColor("#fff5f5")),
            ("LINEABOVE", (0,1), (0,1), 2.0, colors.HexColor("#2ecc71")),
            ("LINEABOVE", (1,1), (1,1), 2.0, colors.HexColor("#e74c3c")),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        elems.append(pc_tbl)

    # Switch to real landscape for gantt page
    elems.append(NextPageTemplate("LAND"))
    elems.append(PageBreak())
    elems.append(Paragraph("Gantt Chart", title_style))
    elems.append(Spacer(1, 8))

    if gantt_png_path and os.path.exists(gantt_png_path):
        img = Image(gantt_png_path, width=26.8*cm, height=15.2*cm)
        elems.append(img)
    else:
        elems.append(Paragraph("Gantt image not found.", body_style))

    doc.build(elems)
# taskbara icon
if sys.platform.startswith("win"):
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("yourcompany.cpu_scheduling_visualizer")

def enable_dark_titlebar(qt_window):
    """
    Enable Windows 10/11 dark title bar for a given Qt window.
    Works only on Windows. Safe no-op on other OS.
    """
    if not sys.platform.startswith("win"):
        return

    try:
        import ctypes
        from ctypes import wintypes

        hwnd = int(qt_window.winId())

        # Windows builds use either 19 or 20 for immersive dark mode
        DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20

        value = ctypes.c_int(1)
        dwmapi = ctypes.windll.dwmapi

        # Try attribute 20 first
        res = dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_int(DWMWA_USE_IMMERSIVE_DARK_MODE),
            ctypes.byref(value),
            ctypes.sizeof(value)
        )

        # Fallback to 19 if 20 didn't work
        if res != 0:
            dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                ctypes.c_int(DWMWA_USE_IMMERSIVE_DARK_MODE_OLD),
                ctypes.byref(value),
                ctypes.sizeof(value)
            )
    except Exception:
        # If anything fails, don't crash the app
        pass

# --------------------------
#  Main Window
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPU Scheduling Visualizer")
        self.resize(1500, 760)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "resources", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.processes: List[Process] = []
        self.last_segments: List[Segment] = []
        self.last_metrics: Dict = {}
        self.last_log_debug: List[str] = []  # debug log kept
        self.pid_colors: Dict[str, Tuple[float, float, float]] = {}

        self._in_validation = False
        self._last_valid_cell: Dict[Tuple[int,int], str] = {}  # (row,col)->last valid text

        self._timeline_mode = "Story"  # Story | Explained | Debug

        self._build_ui()
        self._apply_dark_theme()
        self._connect_signals()
        self._update_dynamic_ui()
        self._update_algo_info()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Left panel
        left = QVBoxLayout()
        left.setSpacing(10)
        root.addLayout(left, 1)

        # Processes group
        grp_proc = QGroupBox("Processes")
        left.addWidget(grp_proc)
        proc_layout = QVBoxLayout(grp_proc)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["PID", "Arrival", "Burst", "MLQ Queue (1-4)"])
        set_select_rows(self.table)
        self.table.setAlternatingRowColors(True)
        set_header_stretch(self.table.horizontalHeader())

        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 140)

        proc_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        proc_layout.addLayout(btn_row)
        self.btn_add = QPushButton("Add Row")
        self.btn_del = QPushButton("Delete Selected")
        self.btn_rand = QPushButton("Add Random")
        self.btn_clear = QPushButton("Clear")

        self.combo_scenario = QComboBox()
        self.combo_scenario.addItems(list(PRESET_SCENARIOS.keys()))
        self.combo_scenario.setToolTip("Preset Scenario: auto-fill Processes for educational demos")

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
        btn_row.addWidget(self.btn_rand)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.combo_scenario)

        # Controls group
        grp_ctrl = QGroupBox("Simulation Controls")
        left.addWidget(grp_ctrl)
        ctrl = QGridLayout(grp_ctrl)
        ctrl.setHorizontalSpacing(10)
        ctrl.setVerticalSpacing(8)

        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "FIFO / FCFS",
            "SJF",
            "SPN",
            "HRRN",
            "RR",
            "SRTF (continuous)",
            "SRTF (quantum-check)",
            "MLQ",
            "MLFQ"
        ])

        self.spin_cs = QDoubleSpinBox()
        self.spin_cs.setRange(0.0, 1000.0)
        self.spin_cs.setDecimals(2)
        self.spin_cs.setSingleStep(0.1)
        self.spin_cs.setValue(1.00)

        self.chk_cs_idle = QCheckBox("Count CS when switching to/from IDLE")
        self.chk_cs_idle.setChecked(False)

        self.spin_quantum = QDoubleSpinBox()
        self.spin_quantum.setRange(0.1, 1000.0)
        self.spin_quantum.setDecimals(2)
        self.spin_quantum.setSingleStep(0.1)
        self.spin_quantum.setValue(2.00)

        r = 0
        ctrl.addWidget(QLabel("Algorithm:"), r, 0)
        ctrl.addWidget(self.combo_algo, r, 1, 1, 2); r += 1

        ctrl.addWidget(QLabel("Context switch time:"), r, 0)
        ctrl.addWidget(self.spin_cs, r, 1)
        ctrl.addWidget(self.chk_cs_idle, r, 2); r += 1

        ctrl.addWidget(QLabel("Quantum (RR / SRTF q-check):"), r, 0)
        ctrl.addWidget(self.spin_quantum, r, 1, 1, 2); r += 1

        # MLQ/MLFQ settings
        self.grp_ml_outer = QGroupBox("MLQ / MLFQ Settings (4 Queues)")
        left.addWidget(self.grp_ml_outer)

        outer_v = QVBoxLayout(self.grp_ml_outer)
        outer_v.setContentsMargins(8, 10, 8, 8)

        self.ml_scroll = QScrollArea()
        self.ml_scroll.setWidgetResizable(True)
        self.ml_scroll.setFixedHeight(220)
        outer_v.addWidget(self.ml_scroll)

        self.ml_inner = QWidget()
        self.ml_scroll.setWidget(self.ml_inner)
        ml = QGridLayout(self.ml_inner)
        ml.setHorizontalSpacing(10)
        ml.setVerticalSpacing(10)

        self.combo_q1 = QComboBox(); self.combo_q2 = QComboBox(); self.combo_q3 = QComboBox(); self.combo_q4 = QComboBox()
        preemptive_choices = ["RR", "SRTF"]
        nonpreemptive_choices = ["FCFS", "SPN", "HRRN"]
        self.combo_q1.addItems(preemptive_choices)
        self.combo_q2.addItems(preemptive_choices)
        self.combo_q3.addItems(preemptive_choices)
        self.combo_q4.addItems(nonpreemptive_choices)
        self.combo_q1.setCurrentText("RR")
        self.combo_q2.setCurrentText("RR")
        self.combo_q3.setCurrentText("RR")
        self.combo_q4.setCurrentText("FCFS")

        self.lbl_q1 = QLabel("Quantum Q1:")
        self.lbl_q2 = QLabel("Quantum Q2:")
        self.lbl_q3 = QLabel("Quantum Q3:")

        self.spin_q1 = QDoubleSpinBox(); self.spin_q2 = QDoubleSpinBox(); self.spin_q3 = QDoubleSpinBox()
        for sp in (self.spin_q1, self.spin_q2, self.spin_q3):
            sp.setRange(0.1, 1000.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.1)
        self.spin_q1.setValue(1.10)
        self.spin_q2.setValue(4.00)
        self.spin_q3.setValue(8.00)

        self.chk_mlfq_real = QCheckBox("MLFQ realistic mode (Aging/Promotion anti-starvation)")
        self.chk_mlfq_real.setChecked(False)
        self.chk_mlfq_real.setToolTip("Fixed defaults: aging interval=10 time units, promote one level if waited ≥ 10.")

        ml.addWidget(QLabel("Queue 1 Algo:"), 0, 0); ml.addWidget(self.combo_q1, 0, 1)
        ml.addWidget(QLabel("Queue 2 Algo:"), 1, 0); ml.addWidget(self.combo_q2, 1, 1)
        ml.addWidget(QLabel("Queue 3 Algo:"), 2, 0); ml.addWidget(self.combo_q3, 2, 1)
        ml.addWidget(QLabel("Queue 4 Algo (Non-preemptive):"), 3, 0); ml.addWidget(self.combo_q4, 3, 1)

        ml.addWidget(self.lbl_q1, 4, 0); ml.addWidget(self.spin_q1, 4, 1)
        ml.addWidget(self.lbl_q2, 5, 0); ml.addWidget(self.spin_q2, 5, 1)
        ml.addWidget(self.lbl_q3, 6, 0); ml.addWidget(self.spin_q3, 6, 1)
        ml.addWidget(self.chk_mlfq_real, 7, 0, 1, 2)

        # Algorithm description
        left.addWidget(QLabel("Algorithm explanation (English):"))
        self.algo_info = QTextBrowser()
        self.algo_info.setMinimumHeight(170)
        left.addWidget(self.algo_info)

        # Run & file buttons
        run_row = QHBoxLayout()
        left.addLayout(run_row)

        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("PrimaryButton")

        self.btn_import = QPushButton("Import CSV")
        self.btn_export = QPushButton("Export CSV")
        self.btn_save_png = QPushButton("Save Gantt PNG")

        self.btn_report_pdf = QPushButton("Export PDF Report")
        self.btn_report_pdf.setObjectName("AccentButton")

        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_import)
        run_row.addWidget(self.btn_export)
        run_row.addWidget(self.btn_save_png)
        run_row.addWidget(self.btn_report_pdf)

        left.addStretch(1)

        # Right panel
        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, 2)

        grp_tl = QGroupBox("Decision Timeline (Educational)")
        right.addWidget(grp_tl)
        tlv = QVBoxLayout(grp_tl)

        # Timeline controls (Story/Explained/Debug)
        tl_controls = QHBoxLayout()
        tlv.addLayout(tl_controls)
        tl_controls.addWidget(QLabel("View:"))

        self.rb_story = QRadioButton("Story")
        self.rb_expl = QRadioButton("Explained")
        self.rb_debug = QRadioButton("Debug")
        self.rb_story.setChecked(True)

        self.timeline_group = QButtonGroup(self)
        self.timeline_group.addButton(self.rb_story)
        self.timeline_group.addButton(self.rb_expl)
        self.timeline_group.addButton(self.rb_debug)

        tl_controls.addWidget(self.rb_story)
        tl_controls.addWidget(self.rb_expl)
        tl_controls.addWidget(self.rb_debug)
        tl_controls.addStretch(1)

        self.timeline = QTextBrowser()
        self.timeline.setFixedHeight(140)
        tlv.addWidget(self.timeline)

        # Tabs
        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)

        # Gantt tab
        tab_gantt = QWidget()
        self.tabs.addTab(tab_gantt, "Gantt")
        gl = QVBoxLayout(tab_gantt)
        self.canvas = GanttCanvas(tab_gantt)
        gl.addWidget(self.canvas)

        # Metrics tab
        tab_metrics = QWidget()
        self.tabs.addTab(tab_metrics, "Metrics")
        mlayout = QVBoxLayout(tab_metrics)
        mlayout.setSpacing(8)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)
        mlayout.addLayout(cards_row)

        # hierarchy: System -> Overhead -> Output -> Averages
        self.card_util = self._make_card("CPU Utilization", accent="cpu")
        self.card_cs = self._make_card("Context Switches", accent="cs")
        self.card_thr = self._make_card("Throughput", accent="thr")
        self.card_avg = self._make_card("Averages (RT/WT/TT)", accent="avg")
        for c in (self.card_util, self.card_cs, self.card_thr, self.card_avg):
            c.setFixedHeight(92)
            cards_row.addWidget(c)

        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["PID", "RT", "WT", "TT"])
        set_header_stretch(self.metrics_table.horizontalHeader())
        self.metrics_table.setAlternatingRowColors(True)
        mlayout.addWidget(self.metrics_table, 1)

    def _make_card(self, title: str, accent: str) -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        card.setProperty("accent", accent)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        t = QLabel(title)
        t.setObjectName("MetricTitle")
        v = QLabel("--")
        v.setObjectName("MetricValue")
        bf = QFont("Arial", 14)
        bf.setBold(True)
        v.setFont(bf)
        v.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(v)
        card._value_label = v
        return card

    def _apply_dark_theme(self):
        qss = """
        QWidget {
            background-color: #14161a;
            color: #e6e6e6;
            font-size: 12px;
        }

        QGroupBox {
            border: 1px solid #2a2f3a;
            border-radius: 10px;
            margin-top: 10px;
            padding: 8px;
            background-color: rgba(20,22,26, 0.75);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: #cfd6e6;
        }

        QTableWidget {
            background-color: #101217;
            alternate-background-color: #0d0f14;
            gridline-color: #2a2f3a;
            border: 1px solid #2a2f3a;
            border-radius: 10px;
            selection-background-color: #2a2f3a;
            selection-color: #ffffff;
        }
        QTableWidget::item {
            color: #e6e6e6;
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #2a2f3a;
            color: #ffffff;
        }

        QHeaderView::section {
            background-color: #1b1f27;
            color: #e6e6e6;
            padding: 6px;
            border: 1px solid #2a2f3a;
        }
        QTableCornerButton::section {
            background-color: #1b1f27;
            border: 1px solid #2a2f3a;
        }

        QTabWidget::pane {
            border: 1px solid #2a2f3a;
            border-radius: 10px;
            background-color: #101217;
        }
        QTabBar::tab {
            background-color: #1b1f27;
            color: #e6e6e6;
            padding: 8px 14px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background-color: #2a2f3a;
            color: #ffffff;
        }

        QScrollArea {
            border: none;
            background-color: transparent;
        }

        /* Modern thin scrollbar */
        QScrollBar:vertical {
            width: 8px;
            background: #0f1116;
            border: 1px solid #2a2f3a;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #39404e;
            border-radius: 5px;
            min-height: 24px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4a5366;
        }
        QScrollBar:horizontal {
            height: 8px;
            background: #0f1116;
            border: 1px solid #2a2f3a;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #39404e;
            border-radius: 5px;
            min-width: 24px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #4a5366;
        }

        QComboBox, QDoubleSpinBox {
            background-color: #101217;
            border: 1px solid #2a2f3a;
            border-radius: 8px;
            padding: 6px;
        }
        QComboBox:disabled, QDoubleSpinBox:disabled {
            color: #9aa3b2;
            background-color: #0d0f14;
            border: 1px solid #232833;
        }

        QPushButton {
            background-color: #1b1f27;
            border: 1px solid #2a2f3a;
            border-radius: 10px;
            padding: 10px 12px;
        }
        QPushButton:hover { background-color: #242a35; }
        QPushButton:pressed { background-color: #2a2f3a; }

        QPushButton#PrimaryButton {
            background-color: #2b6cff;
            border: 1px solid #2b6cff;
            color: #ffffff;
            font-weight: 800;
        }
        QPushButton#PrimaryButton:hover { background-color: #3a7bff; }

        QPushButton#AccentButton {
            background-color: #1f8f5a;
            border: 1px solid #1f8f5a;
            color: #ffffff;
            font-weight: 800;
        }
        QPushButton#AccentButton:hover { background-color: #27a46a; }

        QTextBrowser {
            background-color: #101217;
            border: 1px solid #2a2f3a;
            border-radius: 10px;
            padding: 8px;
        }

        QFrame#MetricCard {
            background-color: rgba(27,31,39, 0.85);
            border: 1px solid #2a2f3a;
            border-radius: 14px;
        }

        /* accents */
        QFrame#MetricCard[accent="cpu"] { border-top: 3px solid rgba(70,130,255,0.95); }
        QFrame#MetricCard[accent="cs"]  { border-top: 3px solid rgba(255,165,70,0.95); }
        QFrame#MetricCard[accent="thr"] { border-top: 3px solid rgba(70,210,130,0.95); }
        QFrame#MetricCard[accent="avg"] { border-top: 3px solid rgba(170,120,255,0.95); }

        QLabel#MetricTitle { color: #b8c0d6; }
        QLabel#MetricValue { color: #ffffff; }
        """
        self.setStyleSheet(qss)

    def _connect_signals(self):
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_rand.clicked.connect(self.add_random)
        self.btn_clear.clicked.connect(self.clear_all)

        self.combo_scenario.currentIndexChanged.connect(self.apply_scenario)

        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_save_png.clicked.connect(self.save_gantt_png)
        self.btn_report_pdf.clicked.connect(self.export_pdf)

        self.combo_algo.currentIndexChanged.connect(self._update_dynamic_ui)
        self.combo_algo.currentIndexChanged.connect(self._update_algo_info)

        self.table.itemChanged.connect(self._live_validate_cell)

        # timeline view mode
        self.rb_story.toggled.connect(lambda: self._set_timeline_mode("Story"))
        self.rb_expl.toggled.connect(lambda: self._set_timeline_mode("Explained"))
        self.rb_debug.toggled.connect(lambda: self._set_timeline_mode("Debug"))

    def _set_timeline_mode(self, mode: str):
        if mode == "Story" and not self.rb_story.isChecked():
            return
        if mode == "Explained" and not self.rb_expl.isChecked():
            return
        if mode == "Debug" and not self.rb_debug.isChecked():
            return
        self._timeline_mode = mode
        self._render_timeline()

    def _update_dynamic_ui(self):
        algo = self.combo_algo.currentText()

        non_preemptive = algo in ("FIFO / FCFS", "SJF", "SPN", "HRRN")
        quantum_needed = algo in ("RR", "SRTF (quantum-check)")
        self.spin_quantum.setEnabled(quantum_needed and not non_preemptive)

        self.grp_ml_outer.setVisible(algo in ("MLQ", "MLFQ"))

        is_mlq = (algo == "MLQ")
        is_mlfq = (algo == "MLFQ")
        for w in (self.spin_q1, self.spin_q2, self.spin_q3):
            w.setEnabled(is_mlq or is_mlfq)
        self.chk_mlfq_real.setEnabled(is_mlfq)

        show_mlq_col = algo in ("MLQ", "MLFQ")
        self.table.setColumnHidden(3, not show_mlq_col)

    def _update_algo_info(self):
        algo = self.combo_algo.currentText()
        self.algo_info.setHtml(ALGO_DESCRIPTIONS.get(algo, "<b>Select an algorithm</b>"))

    # ---------- Live validation (hard prevention) ----------
    def _live_validate_cell(self, item: QTableWidgetItem):
        if self._in_validation:
            return
        self._in_validation = True
        try:
            r = item.row()
            c = item.column()
            if c in (1, 2):  # Arrival / Burst
                text = item.text().strip()

                val = safe_float(text, None)
                bad = False
                tip = ""

                if val is None:
                    bad = True
                    tip = "Must be a number."
                else:
                    if c == 1 and val < 0:
                        bad = True
                        tip = "Arrival must be ≥ 0"
                    if c == 2 and val <= 0:
                        bad = True
                        tip = "Burst must be > 0"

                key = (r, c)
                if not bad:
                    self._last_valid_cell[key] = text
                    item.setBackground(QBrush())
                    item.setToolTip("")
                else:
                    prev = self._last_valid_cell.get(key, "0" if c == 1 else "1")
                    item.setText(prev)
                    item.setBackground(QBrush(QColor("#3a1f1f")))
                    item.setToolTip(tip)

        finally:
            self._in_validation = False

    # ---------- Timeline rendering ----------
    def _render_timeline(self):
        if not self.last_log_debug:
            self.timeline.setHtml("")
            return

        if self._timeline_mode == "Debug":
            self.timeline.setHtml("<br>".join(self.last_log_debug))
            return

        lines = []
        for ln in self.last_log_debug:
            keep = any(k in ln for k in ("Pick", "Run", "Finish", "IDLE", "Preempt", "Requeue", "Arrival", "Queue", "CS "))
            if not keep:
                continue
            if "remaining" in ln:
                ln = ln.split("|")[0].strip()
            lines.append(ln)

        if self._timeline_mode == "Explained":
            self.timeline.setHtml("<br>".join(lines))
            return

        story = []
        for ln in lines:
            if "CS " in ln:
                continue
            if "Pick " in ln:
                story.append("▶️ " + ln.split("→", 1)[1].strip().replace("Pick ", "Select "))
            elif "Run " in ln:
                story.append("⏱️ " + ln.split("→", 1)[1].strip())
            elif "Finish " in ln:
                story.append("✅ " + ln.split("→", 1)[1].strip())
            elif "Preempt " in ln:
                story.append("🔀 " + ln.split("→", 1)[1].strip())
            elif "Requeue " in ln:
                story.append("🔄 " + ln.split("→", 1)[1].strip())
            elif "IDLE" in ln:
                story.append("… " + ln.split("→", 1)[1].strip())
            elif "Arrival:" in ln:
                story.append("📌 " + ln.split("→", 1)[1].strip())
            elif "Queue " in ln and "selected" in ln:
                story.append("📋 " + ln.split("→", 1)[1].strip())
        self.timeline.setHtml("<br>".join(story))

    # ---------- Processes table ops ----------
    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        defaults = [f"P{r+1}", "0", "1", "1"]
        for c, val in enumerate(defaults):
            it = QTableWidgetItem(val)
            self.table.setItem(r, c, it)
            if c in (1, 2):
                self._last_valid_cell[(r, c)] = val

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def clear_all(self):
        self.table.setRowCount(0)
        self.processes = []
        self.last_segments = []
        self.last_metrics = {}
        self.last_log_debug = []
        self.pid_colors = {}
        self._last_valid_cell = {}

        self.canvas.ax.clear()
        self.canvas.draw()
        self.metrics_table.setRowCount(0)
        self.timeline.setHtml("")

        for card in (self.card_util, self.card_cs, self.card_thr, self.card_avg):
            card._value_label.setText("--")

        self.combo_scenario.blockSignals(True)
        self.combo_scenario.setCurrentText("None (Manual)")
        self.combo_scenario.blockSignals(False)

    def add_random(self):
        import random
        r = self.table.rowCount()
        self.table.insertRow(r)
        pid = f"P{r+1}"
        arrival = random.randint(0, 10)
        burst = random.randint(1, 20)
        mlq = random.randint(1, 4)
        vals = [pid, str(arrival), str(burst), str(mlq)]
        for c, val in enumerate(vals):
            it = QTableWidgetItem(val)
            self.table.setItem(r, c, it)
            if c in (1, 2):
                self._last_valid_cell[(r, c)] = val

    def _read_processes_from_table(self) -> List[Process]:
        procs: List[Process] = []
        for r in range(self.table.rowCount()):
            pid = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip() or f"P{r+1}"
            arrival = safe_float(self.table.item(r, 1).text() if self.table.item(r, 1) else "0", 0.0)
            burst = safe_float(self.table.item(r, 2).text() if self.table.item(r, 2) else "1", 1.0)
            mlq = int(safe_float(self.table.item(r, 3).text() if self.table.item(r, 3) else "1", 1.0))

            if burst <= 0:
                raise ValueError(f"Burst must be > 0 for {pid}")
            if arrival < 0:
                raise ValueError(f"Arrival must be >= 0 for {pid}")

            procs.append(Process(pid=pid, arrival=arrival, burst=burst, mlq_class=clamp(mlq, 1, 4)))

        seen = set()
        for p in procs:
            if p.pid in seen:
                raise ValueError(f"Duplicate PID: {p.pid}")
            seen.add(p.pid)
        return procs

    def _y_order_from_table(self, procs: List[Process]) -> List[str]:
        order = []
        for r in range(self.table.rowCount()):
            pid = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            if pid:
                order.append(pid)
        valid = {p.pid for p in procs}
        order = [pid for pid in order if pid in valid]
        for p in procs:
            if p.pid not in order:
                order.append(p.pid)
        return order

    # ---------- Preset scenarios ----------
    def apply_scenario(self):
        name = self.combo_scenario.currentText()
        sc = PRESET_SCENARIOS.get(name, PRESET_SCENARIOS["None (Manual)"])

        self.table.setRowCount(0)
        self._last_valid_cell = {}

        procs = sc.get("processes", [])
        for row in procs:
            r = self.table.rowCount()
            self.table.insertRow(r)
            pid = row.get("pid", f"P{r+1}")
            arr = row.get("arrival", 0)
            bur = row.get("burst", 1)
            mlq = row.get("mlq", 1)
            vals = [pid, str(arr), str(bur), str(mlq)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                self.table.setItem(r, c, it)
                if c in (1, 2):
                    self._last_valid_cell[(r, c)] = str(v)

        suggested = sc.get("suggested_algo")
        if suggested:
            self.combo_algo.setCurrentText(suggested)

        if suggested:
            extra = f"<hr><b>Scenario:</b> {name}<br><span style='color:#cfd6e6'>{sc.get('description','')}</span>"
            self.algo_info.setHtml(ALGO_DESCRIPTIONS.get(self.combo_algo.currentText(), "") + extra)

    # ---------- Import/Export ----------
    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.table.setRowCount(0)
                self._last_valid_cell = {}
                for row in reader:
                    r = self.table.rowCount()
                    self.table.insertRow(r)
                    vals = [
                        row.get("PID", f"P{r+1}"),
                        row.get("Arrival", "0"),
                        row.get("Burst", "1"),
                        row.get("MLQ", "1"),
                    ]
                    for c, v in enumerate(vals):
                        it = QTableWidgetItem(str(v))
                        self.table.setItem(r, c, it)
                        if c in (1, 2):
                            self._last_valid_cell[(r, c)] = str(v)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "processes.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            procs = self._read_processes_from_table()
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["PID", "Arrival", "Burst", "MLQ"])
                for p in procs:
                    w.writerow([p.pid, p.arrival, p.burst, p.mlq_class])
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def save_gantt_png(self):
        if not self.last_segments:
            QMessageBox.information(self, "Nothing to save", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Gantt PNG", "gantt.png", "PNG Files (*.png)")
        if not path:
            return
        try:
            self.canvas.fig.savefig(path, dpi=160, bbox_inches="tight")
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def export_pdf(self):
        if not self.last_segments or not self.processes or not self.last_metrics:
            QMessageBox.information(self, "Nothing to export", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Report", "report.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        try:
            tmp_png = os.path.join(os.path.dirname(path), "_gantt_tmp.png")
            self.canvas.fig.savefig(tmp_png, dpi=240, bbox_inches="tight")

            export_pdf_report(
                path=path,
                algo=self.combo_algo.currentText(),
                procs=self.processes,
                metrics=self.last_metrics,
                gantt_png_path=tmp_png
            )

            try:
                if os.path.exists(tmp_png):
                    os.remove(tmp_png)
            except Exception:
                pass

            QMessageBox.information(self, "Exported", f"PDF report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # ---------- Run ----------
    def run_simulation(self):
        try:
            procs = self._read_processes_from_table()
            if not procs:
                QMessageBox.warning(self, "No processes", "Please add at least one process.")
                return
            self.processes = procs

            algo = self.combo_algo.currentText()
            cs = float(self.spin_cs.value())
            count_cs_idle = bool(self.chk_cs_idle.isChecked())
            quantum = float(self.spin_quantum.value())

            scheduler = Scheduler(context_switch=cs, count_cs_with_idle=count_cs_idle)
            self.pid_colors = self._assign_pid_colors(self.processes)

            show_queue = False

            if algo == "FIFO / FCFS":
                segs, log, metrics = scheduler.fcfs(self.processes)
            elif algo in ("SJF", "SPN"):
                segs, log, metrics = scheduler.spn(self.processes)
            elif algo == "HRRN":
                segs, log, metrics = scheduler.hrrn(self.processes)
            elif algo == "RR":
                segs, log, metrics = scheduler.rr(self.processes, quantum)
            elif algo == "SRTF (continuous)":
                segs, log, metrics = scheduler.srtf_continuous(self.processes)
            elif algo == "SRTF (quantum-check)":
                segs, log, metrics = scheduler.srtf_quantum_check(self.processes, quantum)
            elif algo == "MLQ":
                q_algos = [self.combo_q1.currentText(), self.combo_q2.currentText(), self.combo_q3.currentText(), self.combo_q4.currentText()]
                q_quanta = [float(self.spin_q1.value()), float(self.spin_q2.value()), float(self.spin_q3.value())]
                segs, log, metrics = scheduler.mlq(self.processes, q_algos=q_algos, q_quantums=q_quanta)
                show_queue = True
            elif algo == "MLFQ":
                q_algos = [self.combo_q1.currentText(), self.combo_q2.currentText(), self.combo_q3.currentText(), self.combo_q4.currentText()]
                segs, log, metrics = scheduler.mlfq(
                    self.processes, q_algos=q_algos,
                    q1=float(self.spin_q1.value()), q2=float(self.spin_q2.value()), q3=float(self.spin_q3.value()),
                    realistic_aging=bool(self.chk_mlfq_real.isChecked())
                )
                show_queue = True
            else:
                raise ValueError("Unknown algorithm")

            self.last_segments = segs
            self.last_metrics = metrics
            self.last_log_debug = log

            self._render_timeline()

            y_order = self._y_order_from_table(self.processes)
            self.canvas.draw_gantt(
                processes=self.processes,
                segments=segs,
                title=algo,
                pid_colors=self.pid_colors,
                y_order=y_order,
                show_queue_styles=show_queue,
                show_arrival_markers=True
            )

            self._fill_metrics(metrics, self.processes, algo)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _assign_pid_colors(self, processes: List[Process]) -> Dict[str, Tuple[float, float, float]]:
        colors = list(mcolors.TABLEAU_COLORS.values())
        order = self._y_order_from_table(processes)
        cmap = {}
        for i, pid in enumerate(order):
            cmap[pid] = mcolors.to_rgb(colors[i % len(colors)])
        return cmap

    def _fill_metrics(self, metrics: Dict, processes: List[Process], algo: str):
        util_run = metrics.get("cpu_util_run_only", 0.0)
        util_run_cs = metrics.get("cpu_util_including_cs", 0.0)
        thr = metrics.get("throughput", 0.0)
        cs_count = metrics.get("cs_count", 0)

        self.card_util._value_label.setText(f"Effective CPU: {util_run:.1f}%\nIncluding overhead: {util_run_cs:.1f}%")

        suffix = ""
        non_preemptive = algo in ("FIFO / FCFS", "SJF", "SPN", "HRRN")
        if cs_count == 0:
            if non_preemptive:
                suffix = " (Non-preemptive)"
            elif float(self.spin_cs.value()) <= 1e-12:
                suffix = " (CS disabled)"
        self.card_cs._value_label.setText(f"{cs_count}{suffix}")

        self.card_thr._value_label.setText(f"{thr:.4f} / time unit")

        self.card_avg._value_label.setText(
            f"RT: {metrics.get('avg_RT',0.0):.2f} ★\nWT: {metrics.get('avg_WT',0.0):.2f} | TT: {metrics.get('avg_TT',0.0):.2f}"
        )

        WT = metrics.get("WT", {})
        TT = metrics.get("TT", {})
        RT = metrics.get("RT", {})

        self.metrics_table.setRowCount(0)
        order = self._y_order_from_table(processes)
        for pid in order:
            r = self.metrics_table.rowCount()
            self.metrics_table.insertRow(r)
            self.metrics_table.setItem(r, 0, QTableWidgetItem(pid))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(f"{RT.get(pid,0.0):.2f}"))
            self.metrics_table.setItem(r, 2, QTableWidgetItem(f"{WT.get(pid,0.0):.2f}"))
            self.metrics_table.setItem(r, 3, QTableWidgetItem(f"{TT.get(pid,0.0):.2f}"))

        r = self.metrics_table.rowCount()
        self.metrics_table.insertRow(r)
        avg_bg = QColor("#1f2633")
        for c, txt in enumerate([
            "AVG",
            f"{metrics.get('avg_RT',0.0):.2f}",
            f"{metrics.get('avg_WT',0.0):.2f}",
            f"{metrics.get('avg_TT',0.0):.2f}",
        ]):
            item = QTableWidgetItem(txt)
            f = QFont("Arial", 10)
            f.setBold(True)
            item.setFont(f)
            item.setBackground(avg_bg)
            item.setForeground(QColor("#ffffff"))
            self.metrics_table.setItem(r, c, item)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    enable_dark_titlebar(w)
    w.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()
