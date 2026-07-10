"""
theme.py
--------
Sistema de temas (oscuro / claro) para la aplicación.

Uso:
    from app.ui.theme import ThemeManager, DARK, LIGHT, Theme

    ThemeManager.toggle()          # alterna el tema
    ThemeManager.on_change(cb)     # registra un listener
    t = ThemeManager.current()     # obtiene el tema activo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Theme:
    name: str
    # ── Fondos ──────────────────────────────────────────────────────
    bg: str           # fondo principal de las pantallas
    surface: str      # fondo de tarjetas / paneles
    sidebar: str      # fondo del sidebar
    # ── Acento ──────────────────────────────────────────────────────
    accent: str
    accent_h: str     # acento hover
    accent_alpha: str # acento semi-transparente (estado activo)
    # ── Texto ───────────────────────────────────────────────────────
    text: str
    subtext: str
    # ── Bordes / separadores ─────────────────────────────────────────
    divider: str
    # ── Controles ────────────────────────────────────────────────────
    input_bg: str
    btn_neutral_bg: str
    btn_neutral_hover: str
    btn_disabled_bg: str
    btn_disabled_text: str
    bar_track: str          # fondo de QProgressBar
    card_dark: str          # tarjeta algo más oscura (ej. XAI cards)
    # ── Tabla ────────────────────────────────────────────────────────
    table_header_bg: str
    scrollbar_bg: str
    scrollbar_handle: str
    # ── Sidebar ──────────────────────────────────────────────────────
    sidebar_hover_bg: str
    # ── Matplotlib ───────────────────────────────────────────────────
    mpl_spine: str
    mpl_grid: str


DARK = Theme(
    name="dark",
    bg="#121212", surface="#1E1E1E", sidebar="#1A1A1A",
    accent="#2979FF", accent_h="#5499FF",
    accent_alpha="rgba(41,121,255,0.18)",
    text="#E0E0E0", subtext="#9E9E9E", divider="#2A2A2A",
    input_bg="#252525",
    btn_neutral_bg="#333333", btn_neutral_hover="#444444",
    btn_disabled_bg="#424242", btn_disabled_text="#757575",
    bar_track="#252525",
    card_dark="#282828",
    table_header_bg="#252525",
    scrollbar_bg="#1A1A1A", scrollbar_handle="#444444",
    sidebar_hover_bg="rgba(255,255,255,0.05)",
    mpl_spine="#333333", mpl_grid="#2A2A2A",
)

LIGHT = Theme(
    name="light",
    bg="#F5F5F5", surface="#FFFFFF", sidebar="#E8E8E8",
    accent="#1565C0", accent_h="#1976D2",
    accent_alpha="rgba(21,101,192,0.12)",
    text="#212121", subtext="#757575", divider="#CFCFCF",
    input_bg="#FFFFFF",
    btn_neutral_bg="#DCDCDC", btn_neutral_hover="#C4C4C4",
    btn_disabled_bg="#E0E0E0", btn_disabled_text="#A0A0A0",
    bar_track="#E0E0E0",
    card_dark="#EEEEEE",
    table_header_bg="#EEEEEE",
    scrollbar_bg="#DDDDDD", scrollbar_handle="#AAAAAA",
    sidebar_hover_bg="rgba(0,0,0,0.06)",
    mpl_spine="#CFCFCF", mpl_grid="#E8E8E8",
)


class ThemeManager:
    """Singleton que gestiona el tema activo y notifica a los listeners."""

    _current: Theme = DARK
    _listeners: list[Callable[[Theme], None]] = []

    @classmethod
    def current(cls) -> Theme:
        return cls._current

    @classmethod
    def is_dark(cls) -> bool:
        return cls._current is DARK

    @classmethod
    def set_theme(cls, theme: Theme) -> None:
        cls._current = theme
        for cb in list(cls._listeners):
            cb(theme)

    @classmethod
    def toggle(cls) -> Theme:
        new = LIGHT if cls._current is DARK else DARK
        cls.set_theme(new)
        return new

    @classmethod
    def on_change(cls, callback: Callable[[Theme], None]) -> None:
        if callback not in cls._listeners:
            cls._listeners.append(callback)
