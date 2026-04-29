"""ASL Translation TUI — Claude Code-style evaluator for MMSLT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)

_ROOT = Path(__file__).parent.parent


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
Screen {
    background: #1a1a1a;
    color: #ddd8d0;
}

Header {
    background: #1a1a1a;
    color: #ddd8d0;
    border-bottom: solid #2e2e2e;
}

Footer {
    background: #141414;
    border-top: solid #2e2e2e;
    color: #555050;
}

#main {
    height: 1fr;
}

/* ── Sidebar ── */
#sidebar {
    width: 40;
    border-right: solid #2e2e2e;
    background: #141414;
}

#sidebar-title {
    background: #141414;
    color: #554e48;
    text-style: bold;
    padding: 0 2;
    height: 1;
    border-bottom: solid #2e2e2e;
}

#search {
    border: blank;
    border-bottom: solid #2e2e2e;
    background: #141414;
    color: #ddd8d0;
    margin: 0;
    padding: 0 2;
    height: 3;
}

#search:focus {
    border: blank;
    border-bottom: solid #c96442;
    background: #1e1814;
    color: #ddd8d0;
}

VimListView {
    background: #141414;
    height: 1fr;
    scrollbar-size: 1 1;
    scrollbar-color: #2e2e2e;
    scrollbar-color-active: #c96442;
}

VimListView > ListItem {
    padding: 0 2;
    color: #666460;
    background: #141414;
}

VimListView > ListItem:hover {
    background: #1e1e1e;
    color: #ddd8d0;
}

VimListView > ListItem.--highlight {
    background: #2d1a12;
    color: #e08060;
}

VimListView > ListItem.evaluated {
    color: #3d7a5a;
}

VimListView > ListItem.evaluated.--highlight {
    background: #0d2018;
    color: #4ec994;
}

/* ── Content area ── */
#content {
    padding: 0 2;
}

/* ── Result panel ── */
#result-panel {
    height: 1fr;
    border: solid #2e2e2e;
    margin: 1 0 0 0;
    padding: 1 2;
    background: #1e1e1e;
}

#result-header {
    color: #554e48;
    text-style: bold;
    height: 1;
    margin-bottom: 1;
    border-bottom: solid #2e2e2e;
}

#video-title {
    color: #c96442;
    text-style: bold;
    height: 1;
    margin-bottom: 1;
}

#translations {
    height: 1fr;
}

#pred-block {
    width: 1fr;
    padding: 0 2 0 0;
    border-right: solid #2e2e2e;
}

#ref-block {
    width: 1fr;
    padding: 0 0 0 2;
}

.trans-label {
    text-style: bold;
    height: 1;
    margin-bottom: 1;
}

#pred-label {
    color: #4ec994;
}

#ref-label {
    color: #568fd4;
}

#pred-text {
    color: #a0d4ba;
    height: auto;
}

#ref-text {
    color: #88aed4;
    height: auto;
}

#video-metrics {
    height: 1;
    color: #443e38;
    margin-top: 1;
    border-top: solid #2e2e2e;
    padding-top: 1;
}

/* ── Stats panel ── */
#stats-panel {
    height: 6;
    border: solid #2e2e2e;
    margin: 1 0 1 0;
    padding: 1 2;
    background: #1e1e1e;
}

#stats-header {
    color: #554e48;
    text-style: bold;
    height: 1;
    border-bottom: solid #2e2e2e;
    margin-bottom: 1;
}

#stats-content {
    color: #ddd8d0;
    height: auto;
}

/* ── Status bar ── */
#status-bar {
    height: 1;
    background: #141414;
    border-top: solid #2e2e2e;
    padding: 0 2;
}

#status-text {
    color: #554e48;
}

#status-text.loading {
    color: #c96442;
}

#status-text.running {
    color: #c9a242;
}

#status-text.ready {
    color: #4ec994;
}

#status-text.error {
    color: #e05a5a;
}

/* ── Empty state ── */
#empty-state {
    color: #3a3530;
    text-align: center;
    padding: 4 0;
    height: auto;
}
"""


# ── Messages ─────────────────────────────────────────────────────────────────

class VideosLoaded(Message):
    pass


class ModelLoaded(Message):
    pass


class InferenceDone(Message):
    def __init__(self, result) -> None:
        super().__init__()
        self.result = result


class InferenceError(Message):
    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class StatusUpdate(Message):
    def __init__(self, text: str, style: str = "loading") -> None:
        super().__init__()
        self.text = text
        self.style = style


# ── Widgets ───────────────────────────────────────────────────────────────────

class VideoListItem(ListItem):
    def __init__(self, video_info, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_info = video_info
        self._evaluated = False

    def compose(self) -> ComposeResult:
        name = self.video_info.video_name
        if len(name) > 34:
            name = name[:31] + "…"
        yield Label(name)

    def mark_evaluated(self) -> None:
        if not self._evaluated:
            self._evaluated = True
            self.add_class("evaluated")


class VimListView(ListView):
    """ListView with j/k/g/G vim navigation."""

    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
    ]


# ── Main app ─────────────────────────────────────────────────────────────────

class MMSLTApp(App):
    """Claude Code-styled TUI for evaluating MMSLT sign language translation."""

    TITLE = "ASL Translation Evaluator"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run_all", "Run All", show=True),
        Binding("/", "focus_search", "Filter", show=True),
        Binding("escape", "blur_search", "Back to list", show=False),
        Binding("j", "noop", "↓", show=True),
        Binding("k", "noop", "↑", show=True),
    ]

    def __init__(self, engine_kwargs: dict) -> None:
        super().__init__()
        self._engine_kwargs = engine_kwargs
        self._engine = None
        self._stats = None
        self._current_result = None
        self._all_videos: list = []
        self._filtered_videos: list = []
        self._running_all = False
        self._list_items: dict = {}          # video_name → VideoListItem
        self._evaluated_names: set = set()   # video_names already evaluated

    # ── Composition ───────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Label("TEST VIDEOS", id="sidebar-title")
                yield Input(placeholder="  Filter…", id="search")
                yield VimListView(id="video-list")
            # Content
            with Vertical(id="content"):
                with Vertical(id="result-panel"):
                    yield Label("TRANSLATION RESULT", id="result-header")
                    yield Static("Select a video from the list to run inference.", id="empty-state")
                    yield Label("", id="video-title")
                    with Horizontal(id="translations"):
                        with Vertical(id="pred-block"):
                            yield Label("PREDICTED", id="pred-label", classes="trans-label")
                            yield Static("", id="pred-text")
                        with Vertical(id="ref-block"):
                            yield Label("REFERENCE", id="ref-label", classes="trans-label")
                            yield Static("", id="ref-text")
                    yield Static("", id="video-metrics")
                with Vertical(id="stats-panel"):
                    yield Label("CUMULATIVE STATS", id="stats-header")
                    yield Static(_format_stats_empty(), id="stats-content")
        with Horizontal(id="status-bar"):
            yield Static("Scanning test videos…", id="status-text", classes="loading")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#video-title").display = False
        self.query_one("#translations").display = False
        self.query_one("#video-metrics").display = False
        self._load_videos_immediately()
        self._load_model()

    @on(VideosLoaded)
    def _focus_list_after_load(self, _: VideosLoaded) -> None:
        self.query_one("#video-list", VimListView).focus()

    # ── Video list (immediate, no model needed) ───────────────────────────

    @work(thread=True)
    def _load_videos_immediately(self) -> None:
        """Scan the test directory and populate the list before the model loads."""
        try:
            from tui.inference import load_test_videos
            config_path = self._engine_kwargs.get("config_path", "src/configs/config_mmslt_phoenix.yaml")
            videos = load_test_videos(config_path)
            self._all_videos = videos
            self._filtered_videos = list(videos)
            self.post_message(VideosLoaded())
        except Exception as exc:
            self.post_message(StatusUpdate(f"Could not scan test dir: {exc}", "error"))

    # ── Model loading (background thread) ────────────────────────────────

    @work(thread=True)
    def _load_model(self) -> None:
        try:
            from tui.inference import InferenceEngine, CumulativeStats

            def on_status(msg: str) -> None:
                self.post_message(StatusUpdate(msg, "loading"))

            kwargs = dict(self._engine_kwargs)
            kwargs["on_status"] = on_status

            engine = InferenceEngine(**kwargs)
            self._engine = engine
            self._stats = CumulativeStats()
            # Sync the engine's video list with whatever we already loaded
            if not self._all_videos:
                self._all_videos = engine.videos
                self._filtered_videos = list(self._all_videos)
                self._populate_list(self._filtered_videos)
            self.post_message(ModelLoaded())
        except Exception as exc:
            self.post_message(InferenceError(str(exc)))

    @on(VideosLoaded)
    def _on_videos_loaded(self, _: VideosLoaded) -> None:
        self._populate_list(self._filtered_videos)
        n = len(self._all_videos)
        self._set_status(f"{n} videos  ·  j/k navigate  ·  / filter  ·  model loading…", "loading")

    @on(ModelLoaded)
    def _on_model_loaded(self, _: ModelLoaded) -> None:
        self._set_status("Ready — press Enter to run inference on a video.", "ready")

    def _populate_list(self, videos: list) -> None:
        lv = self.query_one("#video-list", VimListView)
        lv.clear()
        self._list_items = {}
        for video in videos:
            item = VideoListItem(video)
            if video.video_name in self._evaluated_names:
                item.mark_evaluated()
            lv.append(item)
            self._list_items[video.video_name] = item

    # ── Search / filter ───────────────────────────────────────────────────

    @on(Input.Changed, "#search")
    def _on_search(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        if not query:
            self._filtered_videos = list(self._all_videos)
        else:
            self._filtered_videos = [
                v for v in self._all_videos if query in v.video_name.lower()
            ]
        self._populate_list(self._filtered_videos)

    def action_focus_search(self) -> None:
        inp = self.query_one("#search", Input)
        inp.focus()
        inp.cursor_position = len(inp.value)

    def action_blur_search(self) -> None:
        self.query_one("#video-list", VimListView).focus()

    def action_noop(self) -> None:
        pass  # j/k hints in footer are handled by VimListView itself

    # ── Video selection → inference ───────────────────────────────────────

    def on_key(self, event) -> None:
        """Handle Enter on the video list directly — more reliable than ListView.Selected."""
        if event.key != "enter":
            return
        focused = self.focused
        if not isinstance(focused, VimListView):
            return
        event.stop()
        item = focused.highlighted_child
        if not isinstance(item, VideoListItem):
            return
        if self._engine is None:
            self._set_status("Model still loading — please wait…", "loading")
            return
        self._run_inference(item.video_info)

    @work(thread=True)
    def _run_inference(self, video) -> None:
        try:
            self.post_message(StatusUpdate(f"Running inference on {video.video_name}…", "running"))
            result = self._engine.run_single(video)
            self.post_message(InferenceDone(result))
        except Exception as exc:
            self.post_message(InferenceError(str(exc)))

    @on(InferenceDone)
    def _on_inference_done(self, event: InferenceDone) -> None:
        result = event.result
        self._current_result = result
        self._stats.add(result.prediction, result.reference, result.inference_time)

        # Mark the list item as evaluated
        self._evaluated_names.add(result.video_info.video_name)
        item = self._list_items.get(result.video_info.video_name)
        if item:
            item.mark_evaluated()

        # Update result panel
        self.query_one("#empty-state").display = False
        self.query_one("#video-title").display = True
        self.query_one("#translations").display = True
        self.query_one("#video-metrics").display = True

        title = result.video_info.video_name
        if len(title) > 70:
            title = title[:67] + "…"
        self.query_one("#video-title", Label).update(title)
        self.query_one("#pred-text", Static).update(result.prediction or "[dim](empty)[/dim]")
        self.query_one("#ref-text", Static).update(result.reference or "[dim](empty)[/dim]")

        # Per-video metrics
        from tui.inference import _rouge_l_sentence
        from sacrebleu.metrics import BLEU
        b4 = BLEU(max_ngram_order=4, effective_order=True).sentence_score(
            result.prediction, [result.reference]
        ).score
        rl = _rouge_l_sentence(result.prediction, result.reference) * 100
        self.query_one("#video-metrics", Static).update(
            f"[dim]This video:[/dim]  "
            f"BLEU-4 [bold]{b4:.1f}[/bold]  "
            f"ROUGE-L [bold]{rl:.1f}[/bold]  "
            f"Time [bold]{result.inference_time:.2f}s[/bold]  "
            f"Frames [bold]{result.video_info.num_frames}[/bold]"
        )

        # Update cumulative stats
        self._refresh_stats()
        self._set_status(
            f"Done  ({self._stats.count} evaluated) — {result.inference_time:.2f}s", "ready"
        )

    @on(InferenceError)
    def _on_inference_error(self, event: InferenceError) -> None:
        short = event.error.splitlines()[-1] if event.error else "Unknown error"
        self._set_status(f"Error: {short}", "error")
        self.query_one("#pred-text", Static).update(
            f"[bold red]Error:[/bold red] {event.error}"
        )
        self._running_all = False

    # ── Run all ───────────────────────────────────────────────────────────

    def action_run_all(self) -> None:
        if self._engine is None or self._running_all:
            return
        self._running_all = True
        self._run_all_worker()

    @work(thread=True)
    def _run_all_worker(self) -> None:
        videos = list(self._filtered_videos)
        for i, video in enumerate(videos):
            if not self._running_all:
                break
            self.post_message(StatusUpdate(
                f"Running all: {i + 1}/{len(videos)} — {video.video_name}…", "running"
            ))
            try:
                result = self._engine.run_single(video)
                self.post_message(InferenceDone(result))
            except Exception as exc:
                self.post_message(InferenceError(str(exc)))
        self._running_all = False

    # ── Stats ─────────────────────────────────────────────────────────────

    def _refresh_stats(self) -> None:
        s = self._stats
        self.query_one("#stats-content", Static).update(_format_stats(s))

    # ── Status bar ────────────────────────────────────────────────────────

    def _set_status(self, text: str, style: str = "ready") -> None:
        bar = self.query_one("#status-text", Static)
        bar.remove_class("loading", "running", "ready", "error")
        bar.add_class(style)
        bar.update(text)

    @on(StatusUpdate)
    def _on_status_update(self, event: StatusUpdate) -> None:
        self._set_status(event.text, event.style)

    # ── Quit ─────────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        self._running_all = False
        self.exit()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_stats_empty() -> str:
    return (
        "[dim]No videos evaluated yet.[/dim]\n"
        "[dim]Select a video and press Enter, or press [bold]r[/bold] to run all.[/dim]"
    )


def _format_stats(s) -> str:
    if s.count == 0:
        return _format_stats_empty()
    b1 = s.bleu(1)
    b2 = s.bleu(2)
    b3 = s.bleu(3)
    b4 = s.bleu(4)
    rl = s.rouge_l()
    avg_t = s.avg_inference_time()
    return (
        f"[dim]Evaluated:[/dim] [bold]{s.count}[/bold] videos   "
        f"[dim]Avg time:[/dim] [bold]{avg_t:.2f}s[/bold]\n"
        f"[dim]BLEU-1:[/dim] [bold]{b1:.2f}[/bold]  "
        f"[dim]BLEU-2:[/dim] [bold]{b2:.2f}[/bold]  "
        f"[dim]BLEU-3:[/dim] [bold]{b3:.2f}[/bold]  "
        f"[dim]BLEU-4:[/dim] [bold]{b4:.2f}[/bold]  "
        f"[dim]ROUGE-L:[/dim] [bold]{rl:.2f}[/bold]"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ASL Translation TUI — evaluates an MMSLT checkpoint on the Phoenix test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to MMSLT checkpoint (.pth)")
    p.add_argument(
        "--config",
        default="src/configs/config_mmslt_phoenix.yaml",
        help="YAML config path",
    )
    p.add_argument("--language_decoder", default="gemma4", choices=["mbart", "gemma4"])
    p.add_argument("--gemma4_model_id", default="google/gemma-4-E2B-it")
    p.add_argument("--vision_backbone", default="resnet18")
    p.add_argument("--gmmlp_checkpoint", default="")
    p.add_argument("--gmmlp_model_id", default="google/gemma-4-E2B-it")
    p.add_argument("--gmmlp_model_family", default="gemma4", choices=["llava", "gemma4"])
    p.add_argument("--gmmlp_lora_r", type=int, default=16)
    p.add_argument("--gmmlp_lora_alpha", type=int, default=32)
    p.add_argument("--gmmlp_lora_dropout", type=float, default=0.05)
    p.add_argument("--gmmlp_num_latents", type=int, default=64)
    p.add_argument("--gmmlp_num_media_embeds", type=int, default=512)
    p.add_argument("--gmmlp_vision_chunk_size", type=int, default=8)
    p.add_argument("--gmmlp_feat_cache", default="")
    p.add_argument("--eval_max_new_tokens", type=int, default=80)
    p.add_argument("--eval_num_beams", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    # Change to project root so relative paths in config work
    import os
    os.chdir(_ROOT)
    sys.path.insert(0, str(_ROOT / "src"))

    args = _build_arg_parser().parse_args()

    engine_kwargs = {
        "checkpoint_path": args.checkpoint,
        "config_path": args.config,
        "language_decoder": args.language_decoder,
        "gemma4_model_id": args.gemma4_model_id,
        "vision_backbone": args.vision_backbone,
        "gmmlp_checkpoint": args.gmmlp_checkpoint,
        "gmmlp_model_id": args.gmmlp_model_id,
        "gmmlp_model_family": args.gmmlp_model_family,
        "gmmlp_lora_r": args.gmmlp_lora_r,
        "gmmlp_lora_alpha": args.gmmlp_lora_alpha,
        "gmmlp_lora_dropout": args.gmmlp_lora_dropout,
        "gmmlp_num_latents": args.gmmlp_num_latents,
        "gmmlp_num_media_embeds": args.gmmlp_num_media_embeds,
        "gmmlp_vision_chunk_size": args.gmmlp_vision_chunk_size,
        "gmmlp_feat_cache": args.gmmlp_feat_cache,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "eval_num_beams": args.eval_num_beams,
        "device": args.device,
    }

    app = MMSLTApp(engine_kwargs=engine_kwargs)
    app.run()


if __name__ == "__main__":
    main()
