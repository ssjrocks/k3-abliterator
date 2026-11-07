"""The window: engine controls, then Prompts -> Directions -> Test -> Bake."""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from . import APP_ID, APP_NAME, __version__, appreports, bake, engine, paths, updater, workflow
from .prompts import DEFAULT_NORMAL, parse_lines
from .store import Store, model_shape

LOG = logging.getLogger(__name__)
RECOMMENDED = 24


class BackgroundTask(QThread):
    """Run one blocking call off the GUI thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task):
        super().__init__()
        self._task = task

    def run(self):
        try:
            self.succeeded.emit(self._task())
        except Exception as exc:
            LOG.warning("background task failed: %s", exc)
            self.failed.emit(str(exc))


class JobRunner(QThread, workflow.Job):
    """Runs one workflow function; its calls come back to the window as signals."""

    progressed = pyqtSignal(int, int, str)
    logged = pyqtSignal(str)
    streamed = pyqtSignal(str, str, str)
    rowed = pyqtSignal(dict)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self._cancel = False

    def run(self):
        try:
            self.done.emit(self._fn(self))
        except (workflow.Cancelled, InterruptedError):
            self.stopped.emit()
        except Exception as exc:
            if self._cancel:
                self.stopped.emit()
                return
            LOG.exception("job failed")
            appreports.report_error(exc, "running a job")
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancel = True

    # workflow.Job
    def progress(self, done, total, text): self.progressed.emit(done, total, text)
    def log(self, text): self.logged.emit(text)
    def stream(self, pane, kind, chunk): self.streamed.emit(pane, kind, chunk)
    def row(self, data): self.rowed.emit(data)
    def cancelled(self): return self._cancel


class LayerChart(QWidget):
    """Separation score per layer; click a bar to pick that layer."""

    picked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.scores: np.ndarray | None = None
        self.selected = -1
        self.setMinimumHeight(170)
        self.setToolTip("Separation between refused and normal prompts along each layer's direction, "
                        "measured on held-out prompts. Higher is a cleaner refusal direction.")

    def set_scores(self, scores, selected):
        self.scores, self.selected = scores, selected
        self.update()

    def _bar_rect(self, i, n, top, height, peak):
        w = self.width() / n
        h = max(0.0, float(self.scores[i])) / peak * (height - 4) if peak > 0 else 0
        return QRectF(i * w + 1, top + height - h, max(w - 2, 1), h)

    def paintEvent(self, _event):
        p = QPainter(self)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QPalette.ColorRole.Base))
        if self.scores is None or not len(self.scores):
            p.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Capture the prompts to see each layer's score")
            return
        n = len(self.scores)
        peak = float(np.nanmax(self.scores)) if np.isfinite(np.nanmax(self.scores)) else 1.0
        top, height = 6, self.height() - 22
        accent = pal.color(QPalette.ColorRole.Highlight)
        muted = QColor(accent)
        muted.setAlpha(110)
        for i in range(n):
            p.fillRect(self._bar_rect(i, n, top, height, peak), accent if i == self.selected else muted)
        p.setPen(pal.color(QPalette.ColorRole.Text))
        p.setFont(QFont(self.font().family(), 8))
        for i in range(0, n, 10):
            p.drawText(QRectF(i * self.width() / n - 10, self.height() - 16, 30, 14),
                       Qt.AlignmentFlag.AlignLeft, str(i))

    def mousePressEvent(self, event):
        if self.scores is not None and len(self.scores):
            i = int(event.position().x() / self.width() * len(self.scores))
            self.picked.emit(max(0, min(len(self.scores) - 1, i)))


class Window(QMainWindow):
    report_finished = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 820)
        self.store = Store()
        s = self.store.state
        self.hidden, self.n_layers = self._shape(s.model)
        self.server = engine.Server(Path(s.model), self.store.control)
        self.job: JobRunner | None = None
        self.stats = None
        self.picked_layer = -1
        self._tasks: set[BackgroundTask] = set()
        self._update = None
        self._log_pos = 0

        self._build_menu()
        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(self._build_engine_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_prompts(), "1  Prompts")
        self.tabs.addTab(self._build_directions(), "2  Directions")
        self.tabs.addTab(self._build_test(), "3  Test")
        self.tabs.addTab(self._build_bake(), "4  Bake")
        self.tabs.addTab(self._build_log(), "Engine log")
        root.addWidget(self.tabs, 1)
        root.addLayout(self._build_job_bar())
        self.setCentralWidget(central)

        self._refresh_rounds()
        self._refresh_counts()
        self._engine_timer = QTimer(self, interval=3000, timeout=self._poll_engine)
        self._engine_timer.start()
        self._poll_engine()
        self._save_timer = QTimer(self, singleShot=True, interval=600, timeout=self._save_prompts)
        self.report_finished.connect(self._on_report_finished)
        QTimer.singleShot(2000, self.start_update_check)

    # ------------------------------------------------------------ layout

    def _shape(self, model):
        try:
            return model_shape(model)
        except (OSError, KeyError, ValueError):
            return 7168, 93

    def _build_menu(self):
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Check for updates...", self.check_for_updates)
        help_menu.addAction("Report a problem...", self.report_problem)
        help_menu.addAction("About", self.show_about)
        self.update_action = QAction("Update available", self, triggered=self.install_update)
        self.update_action.setVisible(False)
        self.menuBar().addAction(self.update_action)

    def _build_engine_bar(self):
        bar = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_edit = QLineEdit(self.store.state.model)
        self.model_edit.editingFinished.connect(self._model_changed)
        row.addWidget(self.model_edit, 1)
        browse = QPushButton("Browse...", clicked=self._browse_model)
        row.addWidget(browse)
        self.engine_status = QLabel()
        self.engine_status.setMinimumWidth(260)
        row.addWidget(self.engine_status)
        self.engine_btn = QPushButton("Start engine", clicked=self._toggle_engine)
        row.addWidget(self.engine_btn)
        bar.addLayout(row)
        self.other_row = QWidget()
        other = QHBoxLayout(self.other_row)
        other.setContentsMargins(0, 0, 0, 0)
        self.other_label = QLabel()
        self.other_label.setWordWrap(True)
        other.addWidget(self.other_label, 1)
        self.other_btn = QPushButton("Stop that server", clicked=self._stop_other)
        other.addWidget(self.other_btn)
        self.other_row.setVisible(False)
        bar.addWidget(self.other_row)
        return bar

    def _build_job_bar(self):
        row = QHBoxLayout()
        self.job_label = QLabel("Ready")
        row.addWidget(self.job_label, 1)
        self.job_bar = QProgressBar()
        self.job_bar.setMaximumWidth(320)
        self.job_bar.setVisible(False)
        row.addWidget(self.job_bar)
        self.cancel_btn = QPushButton("Cancel", clicked=self._cancel_job)
        self.cancel_btn.setVisible(False)
        row.addWidget(self.cancel_btn)
        return row

    def _build_prompts(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        intro = QLabel(
            "Put the prompts this model <b>still refuses</b> on the left, one per line, and everyday "
            "prompts it answers on the right. The refusal direction is the difference between the two "
            f"sets, so use at least {RECOMMENDED} of each, worded in different ways.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        split = QSplitter()
        left = QGroupBox("Prompts it still refuses")
        ll = QVBoxLayout(left)
        self.refused_edit = QPlainTextEdit("\n".join(self.store.state.refused))
        self.refused_edit.setPlaceholderText("One prompt per line")
        self.refused_edit.textChanged.connect(self._save_timer_start)
        ll.addWidget(self.refused_edit)
        self.refused_count = QLabel()
        ll.addWidget(self.refused_count)
        right = QGroupBox("Normal prompts")
        rl = QVBoxLayout(right)
        self.normal_edit = QPlainTextEdit("\n".join(self.store.state.normal))
        self.normal_edit.textChanged.connect(self._save_timer_start)
        rl.addWidget(self.normal_edit)
        nrow = QHBoxLayout()
        self.normal_count = QLabel()
        nrow.addWidget(self.normal_count, 1)
        nrow.addWidget(QPushButton("Reset to defaults", clicked=self._reset_normal))
        rl.addLayout(nrow)
        split.addWidget(left)
        split.addWidget(right)
        lay.addWidget(split, 1)
        self.thinking_check = QCheckBox("Thinking mode (use the mode you chat in; it's much slower to test)")
        self.thinking_check.setChecked(self.store.state.thinking)
        self.thinking_check.toggled.connect(self._thinking_changed)
        lay.addWidget(self.thinking_check)
        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        lay.addWidget(self.estimate_label)
        return page

    def _build_directions(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        find = QGroupBox("Find a new direction")
        fl = QVBoxLayout(find)
        note = QLabel(
            "Reads every prompt through the model with the current ablation on, then compares the "
            "two sets at each layer. Run it again after adding a direction to find what's left.")
        note.setWordWrap(True)
        fl.addWidget(note)
        self.find_btn = QPushButton("Capture prompts and compute", clicked=self._find)
        fl.addWidget(self.find_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.chart = LayerChart()
        self.chart.picked.connect(self._pick_layer)
        fl.addWidget(self.chart)
        prow = QHBoxLayout()
        self.pick_label = QLabel("No direction computed yet")
        prow.addWidget(self.pick_label, 1)
        self.add_btn = QPushButton("Add this direction", clicked=self._add_direction)
        self.add_btn.setEnabled(False)
        prow.addWidget(self.add_btn)
        fl.addLayout(prow)
        lay.addWidget(find)

        active = QGroupBox("Active ablation (applies to the next request)")
        al = QVBoxLayout(active)
        self.rounds_list = QListWidget()
        self.rounds_list.setMaximumHeight(110)
        al.addWidget(self.rounds_list)
        al.addWidget(QPushButton("Remove selected", clicked=self._remove_round), alignment=Qt.AlignmentFlag.AlignLeft)
        form = QFormLayout()
        srow = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=150)
        self.scale_slider.setValue(int(round(self.store.state.scale * 100)))
        self.scale_label = QLabel()
        self.scale_slider.valueChanged.connect(self._scale_changed)
        srow.addWidget(self.scale_slider, 1)
        srow.addWidget(self.scale_label)
        form.addRow("Strength:", srow)
        lrow = QHBoxLayout()
        self.lo_spin = QSpinBox(minimum=0, maximum=self.n_layers - 1, value=self.store.state.lo)
        self.hi_spin = QSpinBox(minimum=0, maximum=self.n_layers - 1, value=min(self.store.state.hi, self.n_layers - 1))
        self.lo_spin.valueChanged.connect(self._range_changed)
        self.hi_spin.valueChanged.connect(self._range_changed)
        lrow.addWidget(QLabel("from"))
        lrow.addWidget(self.lo_spin)
        lrow.addWidget(QLabel("to"))
        lrow.addWidget(self.hi_spin)
        lrow.addStretch(1)
        form.addRow("Layers:", lrow)
        for w in (self.lo_spin, self.hi_spin):
            w.setToolTip("Layers whose outputs stop writing the direction. What earlier layers wrote still "
                         "reaches later ones, so keep the full range (and the embeddings) to remove it everywhere.")
        self.embed_check = QCheckBox("Also the token embeddings", checked=self.store.state.embed)
        self.embed_check.toggled.connect(self._range_changed)
        form.addRow("", self.embed_check)
        al.addLayout(form)
        self._scale_changed(self.scale_slider.value(), save=False)
        lay.addWidget(active)
        lay.addStretch(1)
        return page

    def _build_test(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.test_prompt = QPlainTextEdit()
        self.test_prompt.setPlaceholderText("A prompt to try")
        self.test_prompt.setMaximumHeight(80)
        lay.addWidget(self.test_prompt)
        row = QHBoxLayout()
        row.addWidget(QLabel("Max tokens:"))
        self.tokens_spin = QSpinBox(minimum=1, maximum=4096, value=120)
        row.addWidget(self.tokens_spin)
        self.ask_orig = QPushButton("Ask original", clicked=lambda: self._ask(["original"]))
        self.ask_abl = QPushButton("Ask ablated", clicked=lambda: self._ask(["ablated"]))
        self.ask_both = QPushButton("Ask both", clicked=lambda: self._ask(["original", "ablated"]))
        for b in (self.ask_orig, self.ask_abl, self.ask_both):
            row.addWidget(b)
        self.ask_estimate = QLabel()
        row.addWidget(self.ask_estimate, 1)
        lay.addLayout(row)
        panes = QSplitter()
        self.out = {}
        for key, title in (("original", "Original"), ("ablated", "Ablated")):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            view = QTextBrowser()
            bl.addWidget(view)
            self.out[key] = view
            panes.addWidget(box)
        lay.addWidget(panes, 2)

        check = QGroupBox("Refusal check: run every refused prompt")
        cl = QVBoxLayout(check)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Tokens each:"))
        self.check_tokens = QSpinBox(minimum=8, maximum=512, value=40)
        crow.addWidget(self.check_tokens)
        self.check_original = QCheckBox("Also the original (twice as long)")
        crow.addWidget(self.check_original)
        self.check_btn = QPushButton("Run refusal check", clicked=self._refusal_check)
        crow.addWidget(self.check_btn)
        self.check_summary = QLabel()
        crow.addWidget(self.check_summary, 1)
        cl.addLayout(crow)
        self.check_table = QTableWidget(0, 3)
        self.check_table.setHorizontalHeaderLabels(["Prompt", "Original", "Ablated"])
        self.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.setWordWrap(True)
        cl.addWidget(self.check_table)
        lay.addWidget(check, 2)
        return page

    def _build_bake(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        intro = QLabel(
            "Writes the active ablation into a new copy of the model, so it works without this app "
            "(in <code>coli chat</code> / <code>coli web</code>, or anything that loads safetensors). "
            "Only the tensors that write to the hidden state are rewritten into one new file; "
            "every other file is a link to the original, which isn't changed.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("New model folder:"))
        self.bake_out = QLineEdit(str(Path(self.store.state.model).with_name(Path(self.store.state.model).name + "-plus")))
        self.bake_out.editingFinished.connect(self._bake_plan)
        row.addWidget(self.bake_out, 1)
        row.addWidget(QPushButton("Browse...", clicked=self._browse_bake))
        lay.addLayout(row)
        self.bake_info = QLabel()
        self.bake_info.setWordWrap(True)
        lay.addWidget(self.bake_info)
        self.bake_btn = QPushButton("Bake", clicked=self._bake)
        lay.addWidget(self.bake_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.bake_result = QLabel()
        self.bake_result.setWordWrap(True)
        self.bake_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.bake_result)
        lay.addStretch(1)
        return page

    def _build_log(self):
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setFont(QFont("monospace", 9))
        return self.log_view

    # ------------------------------------------------------------ engine

    def _browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Kimi K3 model folder", self.model_edit.text())
        if path:
            self.model_edit.setText(path)
            self._model_changed()

    def _model_changed(self):
        path = self.model_edit.text().strip()
        if path == self.store.state.model:
            return
        if self.server.running():
            QMessageBox.information(self, "Model", "Stop the engine first, then change the model.")
            self.model_edit.setText(self.store.state.model)
            return
        if not (Path(path) / "config.json").exists():
            QMessageBox.warning(self, "Model", f"No config.json in {path}")
            self.model_edit.setText(self.store.state.model)
            return
        self.store.state.model = path
        self.store.save()
        self.hidden, self.n_layers = self._shape(path)
        self.server = engine.Server(Path(path), self.store.control)

    def _toggle_engine(self):
        if self.server.running():
            if self.job and self.job.isRunning():
                QMessageBox.information(self, "Engine", "Cancel the running job first.")
                return
            self.server.stop()
            self._poll_engine()
            return
        if engine.other_servers():
            QMessageBox.warning(self, "Engine", "Another Kimi K3 server is running. Two don't fit in "
                                "memory together: stop it first (the button under the model path).")
            return
        self.engine_btn.setEnabled(False)
        self.engine_status.setText("Building the engine...")

        def started(_):
            self.engine_btn.setEnabled(True)
            self._poll_engine()

        def failed(msg):
            self.engine_btn.setEnabled(True)
            self._poll_engine()
            QMessageBox.critical(self, "Engine", msg)

        self._run_task(self.server.start, started, failed)

    def _poll_engine(self):
        running = self.server.running()
        foreign = running and not self.server.control_matches()
        healthy = running and not foreign and self.server.healthy()
        mem = self.server.memory_gb() if running else None
        mem_s = f", {mem:.0f} GB" if mem else ""
        if foreign:
            self.engine_status.setText("⚠ Engine started elsewhere: stop it and start it here")
        elif healthy:
            self.engine_status.setText(f"● Engine ready{mem_s}")
        elif running:
            self.engine_status.setText(f"◌ Engine loading the model{mem_s}...")
        else:
            self.engine_status.setText("○ Engine stopped")
        self.engine_btn.setText("Stop engine" if running else "Start engine")
        self.engine_ready = healthy
        others = [] if running else engine.other_servers()
        if others:
            pid, cmd = others[0]
            self.other_label.setText(f"Another Kimi K3 server is running and holds the memory this needs: "
                                     f"<code>{cmd[:160]}</code> (PID {pid})")
        self.other_row.setVisible(bool(others))
        self._other_pid = others[0][0] if others else None
        self._update_buttons()
        self._tail_log()

    def _stop_other(self):
        if not self._other_pid:
            return
        if QMessageBox.question(self, "Stop server", f"Stop the colibri server with PID {self._other_pid}? "
                                "Any chat it's answering stops too.") == QMessageBox.StandardButton.Yes:
            engine.stop_pid(self._other_pid)
            QTimer.singleShot(2000, self._poll_engine)

    def _tail_log(self):
        log = paths.engine_log()
        try:
            size = log.stat().st_size
        except OSError:
            return
        if size < self._log_pos:
            self._log_pos = 0
        if size == self._log_pos:
            return
        with open(log, "rb") as f:
            if self._log_pos == 0 and size > 200_000:
                self._log_pos = size - 200_000
            f.seek(self._log_pos)
            text = f.read().decode(errors="replace")
        self._log_pos = size
        lines = [ln for ln in text.splitlines() if '"GET /health' not in ln]   # this window's own polling
        if lines:
            self.log_view.appendPlainText("\n".join(lines))

    # ------------------------------------------------------------ jobs

    def _update_buttons(self):
        busy = bool(self.job and self.job.isRunning())
        ready = getattr(self, "engine_ready", False) and not busy
        self.find_btn.setEnabled(ready and bool(self.store.state.refused) and bool(self.store.state.normal))
        for b in (self.ask_orig, self.ask_both, self.check_btn):
            b.setEnabled(ready)
        self.ask_abl.setEnabled(ready and bool(self.store.state.rounds))
        self.ask_both.setEnabled(ready and bool(self.store.state.rounds))
        self.check_btn.setEnabled(ready and bool(self.store.state.refused))
        self.bake_btn.setEnabled(not busy and bool(self.store.state.rounds))
        self.add_btn.setEnabled(not busy and self.stats is not None and self.picked_layer >= 0)
        self.model_edit.setEnabled(not busy)

    def _start_job(self, fn, on_done, label):
        if self.job and self.job.isRunning():
            return
        self._save_prompts()
        self.job = JobRunner(fn)
        self.job.progressed.connect(self._job_progress)
        self.job.logged.connect(lambda t: self.job_label.setText(t))
        self.job.streamed.connect(self._job_stream)
        self.job.rowed.connect(self._job_row)
        self.job.done.connect(on_done)
        self.job.failed.connect(self._job_failed)
        self.job.stopped.connect(lambda: self.job_label.setText("Cancelled"))
        self.job.finished.connect(self._job_finished)
        self.job_label.setText(label)
        self.job_bar.setRange(0, 0)
        self.job_bar.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(True)
        self.job.start()
        self._update_buttons()

    def _job_progress(self, done, total, text):
        self.job_label.setText(text)
        if total:
            self.job_bar.setRange(0, total)
            self.job_bar.setValue(done)
        else:
            self.job_bar.setRange(0, 0)

    def _job_failed(self, msg):
        self.job_label.setText("Failed")
        QMessageBox.critical(self, APP_NAME, msg)

    def _job_finished(self):
        self.job_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._update_buttons()

    def _cancel_job(self):
        if self.job:
            self.job.cancel()
            self.cancel_btn.setEnabled(False)
            self.job_label.setText("Cancelling after the current step...")

    def _job_stream(self, pane, kind, chunk):
        view = self.out[pane]
        cursor = view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = cursor.charFormat()
        fmt.setForeground(view.palette().color(
            QPalette.ColorRole.PlaceholderText if kind == "reasoning" else QPalette.ColorRole.Text))
        cursor.setCharFormat(fmt)
        cursor.insertText(chunk)
        view.setTextCursor(cursor)
        view.ensureCursorVisible()

    def _job_row(self, data):
        t = self.check_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(self.store.state.refused[data["index"]]))
        for col, mode in ((1, "original"), (2, "ablated")):
            if mode in data:
                refused, text = data[mode]
                item = QTableWidgetItem(("REFUSED  " if refused else "answered  ") + text[:300])
                item.setToolTip(text)
                item.setForeground(QColor("#c0392b") if refused else QColor("#2e8b57"))
                t.setItem(r, col, item)
        t.resizeRowToContents(r)

    # ------------------------------------------------------------ prompts

    def _save_timer_start(self):
        self._save_timer.start()
        self._refresh_counts()

    def _save_prompts(self):
        s = self.store.state
        s.refused = parse_lines(self.refused_edit.toPlainText())
        s.normal = parse_lines(self.normal_edit.toPlainText())
        self.store.save()
        self._update_buttons()

    def _reset_normal(self):
        self.normal_edit.setPlainText("\n".join(DEFAULT_NORMAL))

    def _thinking_changed(self, on):
        self.store.state.thinking = on
        self.store.save()
        self._refresh_counts()

    def _refresh_counts(self):
        refused = parse_lines(self.refused_edit.toPlainText())
        normal = parse_lines(self.normal_edit.toPlainText())
        for label, n in ((self.refused_count, len(refused)), (self.normal_count, len(normal))):
            label.setText(f"{n} prompts" + ("" if n >= RECOMMENDED else f" (use at least {RECOMMENDED})"))
        s = self.store.state
        if s.seconds_per_prompt:
            per = s.seconds_per_prompt
            basis = "measured"
        else:
            chars = sum(len(p) for p in refused + normal) / max(1, len(refused + normal))
            per = (chars / 3.5 + 20) * 4.0     # tokens incl. chat template; a 28-character prompt took ~115 s here
            basis = "estimated from a first test on this PC"
        total = per * (len(refused) + len(normal))
        self.estimate_label.setText(
            f"Capturing both sets takes about {workflow._fmt(total)} ({workflow._fmt(per)} a prompt, {basis}). "
            "Captures are kept, so unchanged prompts aren't read again until the ablation changes.")
        if s.seconds_per_token:
            self.ask_estimate.setText(f"≈ {s.seconds_per_token:.1f} s a token, plus reading the prompt")

    # ------------------------------------------------------------ directions

    def _find(self):
        self._start_job(lambda job: workflow.find_direction(job, self.store, self.server),
                        self._found, "Capturing...")

    def _found(self, stats):
        self.stats = stats
        self._pick_layer(stats.best())
        self.job_label.setText("Pick a layer (the best one is selected) and add its direction")
        self._refresh_counts()

    def _pick_layer(self, layer):
        if self.stats is None:
            return
        self.picked_layer = layer
        self.chart.set_scores(self.stats.score, layer)
        where = "the head's input" if layer == len(self.stats.score) - 1 else f"layer {layer}'s input"
        self.pick_label.setText(f"Direction from {where}: separation {self.stats.score[layer]:.2f} "
                                f"({self.stats.n_refused} refused, {self.stats.n_normal} normal prompts)")
        self._update_buttons()

    def _add_direction(self):
        st, layer = self.stats, self.picked_layer
        self.store.add_round(st.dirs[layer], layer, st.score[layer], st.n_refused, st.n_normal)
        self.stats = None
        self.picked_layer = -1
        self.chart.set_scores(None, -1)
        self.pick_label.setText("Added. Test it, or capture again to look for what's left.")
        self._refresh_rounds()

    def _remove_round(self):
        row = self.rounds_list.currentRow()
        if row >= 0:
            self.store.remove_round(row)
            self._refresh_rounds()

    def _refresh_rounds(self):
        self.rounds_list.clear()
        for i, r in enumerate(self.store.state.rounds, 1):
            self.rounds_list.addItem(f"Direction {i}: from layer {r.layer}, separation {r.score:.2f}, "
                                     f"{r.n_refused} refused / {r.n_normal} normal prompts")
        if not self.store.state.rounds:
            self.rounds_list.addItem("None yet: the engine runs the model unchanged")
        self._update_buttons()
        self._bake_plan()

    def _scale_changed(self, value, save=True):
        self.scale_label.setText(f"{value}%")
        if save:
            self.store.state.scale = value / 100
            self.store.save()
            self._bake_plan()

    def _range_changed(self):
        s = self.store.state
        s.lo, s.hi = self.lo_spin.value(), self.hi_spin.value()
        if s.lo > s.hi:
            s.lo, s.hi = s.hi, s.lo
        s.embed = self.embed_check.isChecked()
        self.store.save()
        self._bake_plan()

    # ------------------------------------------------------------ test

    def _ask(self, which):
        prompt = self.test_prompt.toPlainText().strip()
        if not prompt:
            return
        for pane in which:
            self.out[pane].clear()

        def done(replies):
            for pane, r in replies.items():
                tps = f", {r.seconds / r.tokens:.1f} s a token" if r.tokens else ""
                self.out[pane].append(f"\n\n— {r.tokens} tokens in {workflow._fmt(r.seconds)}{tps}")
            self.job_label.setText("Done")
            self._refresh_counts()

        self._start_job(lambda job: workflow.ask(job, self.store, self.server, prompt,
                                                 self.tokens_spin.value(), which), done, "Asking...")

    def _refusal_check(self):
        self.check_table.setRowCount(0)
        self.check_summary.clear()
        original = self.check_original.isChecked()

        def done(counts):
            n = len(self.store.state.refused)
            self.check_summary.setText("   ".join(f"{m}: {c} of {n} refused" for m, c in counts.items()))
            self.job_label.setText("Refusal check done (a keyword check: read the answers)")

        self._start_job(lambda job: workflow.refusal_check(job, self.store, self.server,
                                                           self.check_tokens.value(), original),
                        done, "Checking...")

    # ------------------------------------------------------------ bake

    def _browse_bake(self):
        path, _ = QFileDialog.getSaveFileName(self, "New model folder", self.bake_out.text())
        if path:
            self.bake_out.setText(path)
            self._bake_plan()

    def _bake_plan(self):
        if not hasattr(self, "bake_info"):
            return
        s = self.store.state
        if not s.rounds:
            self.bake_info.setText("Add a direction first.")
            return
        try:
            targets = bake.plan(Path(s.model), s.lo, s.hi, s.embed)
        except Exception as exc:
            self.bake_info.setText(f"Can't plan the bake: {exc}")
            return
        need = sum(t.nbytes for t in targets)
        out = Path(self.bake_out.text())
        parent = next((p for p in [out, *out.parents] if p.exists()), Path("/"))
        free = shutil.disk_usage(parent).free
        self.bake_info.setText(
            f"{len(s.rounds)} direction(s) at {s.scale:.0%}, layers {s.lo}–{s.hi}"
            f"{', embeddings' if s.embed else ''}: rewrites {len(targets)} tensors, "
            f"{need / 2**30:.1f} GB. {free / 2**30:.0f} GB free there.")

    def _bake(self):
        s = self.store.state
        out = Path(self.bake_out.text()).expanduser()
        if out.exists():
            QMessageBox.warning(self, "Bake", f"{out} already exists. Pick a new folder.")
            return
        dirs = self.store.directions()

        def run(job):
            def progress(done, total, name):
                job.check()
                job.progress(done >> 20, max(1, total >> 20), f"Writing {name.rsplit('.', 2)[0] if name else 'index'}")
            return bake.bake(Path(s.model), out, dirs, s.scale, s.lo, s.hi, s.embed, progress, job.cancelled)

        def done(targets):
            self.bake_result.setText(
                f"Baked {len(targets)} tensors into {out}.\n\nUse it like any colibri model, without this app:\n"
                f"cd ~/colibri && ./c/coli web --model {out} --gpu auto --auto-tier")
            self.job_label.setText("Bake done")

        self._start_job(run, done, "Baking...")

    # ------------------------------------------------------------ updates, reports

    def _run_task(self, task, on_success, on_failure):
        worker = BackgroundTask(task)
        worker.succeeded.connect(on_success)
        worker.failed.connect(on_failure)
        worker.finished.connect(lambda: self._tasks.discard(worker))
        self._tasks.add(worker)  # dropping a running QThread aborts the process
        worker.start()

    def start_update_check(self):
        if updater.due():
            updater.mark_checked()
            self._run_task(updater.check, self._on_update_checked, lambda _msg: None)

    def check_for_updates(self):
        def done(update):
            self._on_update_checked(update)
            if update is None:
                QMessageBox.information(self, "Updates", f"{APP_NAME} {__version__} is up to date.")

        self._run_task(updater.check, done,
                       lambda msg: QMessageBox.warning(self, "Updates", f"Could not check:\n\n{msg}"))

    def _on_update_checked(self, update):
        if update is not None:
            self._update = update
            self.update_action.setText(f"Update to {update.version}")
            self.update_action.setVisible(True)

    def install_update(self):
        if self.job and self.job.isRunning():
            QMessageBox.information(self, "Update", "Wait for the running job to finish, or cancel it.")
            return
        update = self._update
        self.update_action.setEnabled(False)
        self.job_label.setText(f"Installing {update.version}...")

        def failed(msg):
            self.update_action.setEnabled(True)
            self.job_label.setText("Ready")
            QMessageBox.warning(self, "Update failed", msg)

        self._run_task(lambda: updater.install(update), lambda _r: updater.restart(), failed)

    def report_problem(self):
        note, ok = QInputDialog.getMultiLineText(
            self, "Report a problem",
            "What went wrong? This is sent with the app's log and the end of the engine log, "
            "never your prompts or the model's answers.")
        if ok:
            s = self.store.state
            extra = {"directions": str(len(s.rounds)), "scale": str(s.scale), "layers": f"{s.lo}-{s.hi}",
                     "thinking": str(s.thinking), "engine": "running" if self.server.running() else "stopped"}
            appreports.send_report(note, extra, on_done=self.report_finished.emit)

    def _on_report_finished(self, sent):
        if sent:
            QMessageBox.information(self, "Report a problem", "Report sent. Thanks.")
        else:
            QMessageBox.warning(self, "Report a problem", "The report could not be sent.")

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME}</b> {__version__}<br><br>Finds the directions behind the refusals "
                          "a Kimi K3 model still makes and removes them, live in colibri or baked into a copy "
                          "of the weights.")

    # ------------------------------------------------------------ closing

    def closeEvent(self, event):
        if self.job and self.job.isRunning():
            if QMessageBox.question(self, APP_NAME, "A job is running. Cancel it and quit?") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.job.cancel()
            self.job.wait(5000)
        if self.server.running():
            answer = QMessageBox.question(
                self, APP_NAME, "Stop the engine too? It holds about 45 GB of RAM. "
                "If you keep it running, the app picks it up again next time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Yes:
                self.server.stop()
        self._save_prompts()
        event.accept()


def run_gui():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)
    win = Window()
    win.show()
    sys.exit(app.exec())
