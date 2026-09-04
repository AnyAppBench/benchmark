"""Cross-app music task ports for the app-generalization suite.

Each port mirrors a canonical Retro Music-style task (create playlist, add to
playing queue, save the queue as a playlist, view playlist duration, create
two playlists, add a single song to the queue) and adds four playback-control
templates (play song, pause song, shuffle queue, skip to next track).

Only playlist-creation ports are active for DB-backed evaluation. Those
verifiers inspect Android MediaProvider playlist tables; playback/queue ports
remain scaffolded but are de-scoped from active runs because they do not
produce a durable database state.
"""

from __future__ import annotations

import dataclasses
import os
import random
import tempfile
from typing import Any, Final
import wave

from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


_PLAYLIST_NAMES: Final[tuple[str, ...]] = (
    "Morning Mix",
    "Workout Beats",
    "Chill Vibes",
    "Road Trip",
    "Focus",
    "Throwbacks",
    "Weekend",
    "Favorites",
    "Discover",
    "Late Night",
)

_SONG_TITLES: Final[tuple[str, ...]] = (
    "Sunset Drive",
    "Neon Lights",
    "Echoes",
    "Heartlines",
    "Open Road",
    "Wildflower",
    "Glass House",
    "Slow Burn",
    "Paper Moon",
    "Riptide",
)

_ARTISTS: Final[tuple[str, ...]] = (
    "Aurora Lane",
    "The Wavelengths",
    "Junior Bloom",
    "Echo Lake",
    "Velvet Hours",
    "Polar Drift",
)
_MEDIA_PROVIDER_DB_PATH: Final[str] = (
    "/data/data/com.google.android.providers.media.module/databases/external.db"
)


@dataclasses.dataclass(frozen=True)
class _MediaPlaylistRow(sqlite_schema_utils.SQLiteRow):
  name: str | None = None
  title: str | None = None
  _display_name: str | None = None
  _id: int = -1


def _list_media_playlists(
    env: interface.AsyncEnv,
) -> list[_MediaPlaylistRow]:
  try:
    with env.controller.pull_file(_MEDIA_PROVIDER_DB_PATH) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          "external.db",
      )
      return sqlite_utils.execute_query(
          "SELECT _id, name, title, _display_name FROM audio_playlists;",
          local_db_path,
          _MediaPlaylistRow,
      )
  except FileNotFoundError:
    return []


def _media_playlist_exists(env: interface.AsyncEnv, playlist_name: str) -> bool:
  expected = playlist_name.strip().lower()
  for row in _list_media_playlists(env):
    for value in (row.name, row.title, row._display_name):
      if value and value.strip().lower().removesuffix(".m3u") == expected:
        return True
  return False


def _seed_music_library(env: interface.AsyncEnv) -> None:
  """Creates a small local library so playback/queue tasks are feasible."""
  user_data_generation.clear_internal_storage(env)
  for index, title in enumerate(_SONG_TITLES):
    _write_wav_file_to_device(
        file_utils.convert_to_posix_path(
            device_constants.MUSIC_DATA, f"{title}.wav"
        ),
        env,
        duration_seconds=3 * 60 + index,
    )
  try:
    adb_utils.issue_generic_request(
        [
            "shell",
            "sh",
            "-c",
            "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            "-d file:///storage/emulated/0/Music >/dev/null 2>&1 &",
        ],
        env.controller,
        timeout_sec=2,
    )
  except Exception:  # The broadcast often succeeds but does not return quickly.
    pass


def _grant_music_permissions(
    env: interface.AsyncEnv, package_name: str
) -> None:
  """Best-effort permission grants for first-run music app setup."""
  for permission in (
      "android.permission.READ_MEDIA_AUDIO",
      "android.permission.POST_NOTIFICATIONS",
  ):
    try:
      adb_utils.issue_generic_request(
          ["shell", "pm", "grant", package_name, permission],
          env.controller,
          timeout_sec=2,
      )
    except Exception:
      pass


def _write_wav_file_to_device(
    remote_path: str,
    env: interface.AsyncEnv,
    duration_seconds: int,
) -> None:
  """Writes a silent WAV file without relying on ffmpeg/pydub."""
  sample_rate = 8000
  samples = b"\x00\x00" * sample_rate * duration_seconds
  local_path = ""
  try:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
      local_path = handle.name
    with wave.open(local_path, "wb") as wav_file:
      wav_file.setnchannels(1)
      wav_file.setsampwidth(2)
      wav_file.setframerate(sample_rate)
      wav_file.writeframes(samples)
    file_utils.copy_data_to_device(local_path, remote_path, env.controller)
  finally:
    if local_path:
      try:
        os.remove(local_path)
      except FileNotFoundError:
        pass


class _MusicPackageAppEval(base.PackageAppEval):
  """Package-launched music task with deterministic media preloading."""

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _seed_music_library(env)
    _grant_music_permissions(env, self.package_name)
    adb_utils.issue_generic_request(
        ["shell", "am", "force-stop", self.package_name],
        env.controller,
        timeout_sec=5,
    )
    adb_utils.launch_app(self.package_name, env.controller)


# -----------------------------------------------------------------------------
# Param generators (one per template).
# -----------------------------------------------------------------------------


def _generate_create_playlist_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "playlist_name": random.choice(_PLAYLIST_NAMES),
  }


def _generate_playing_queue_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "song_title": random.choice(_SONG_TITLES),
      "artist": random.choice(_ARTISTS),
  }


def _generate_save_playlist_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "playlist_name": random.choice(_PLAYLIST_NAMES),
  }


def _generate_playlist_duration_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "playlist_name": random.choice(_PLAYLIST_NAMES),
  }


def _generate_create_two_playlists_params() -> dict[str, Any]:
  name_a, name_b = random.sample(_PLAYLIST_NAMES, 2)
  return {
      "seed": random.randint(0, 1_000_000),
      "playlist_name_a": name_a,
      "playlist_name_b": name_b,
  }


def _generate_add_to_queue_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "song_title": random.choice(_SONG_TITLES),
  }


def _generate_play_song_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "song_title": random.choice(_SONG_TITLES),
  }


def _generate_pause_song_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
  }


def _generate_shuffle_queue_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
  }


def _generate_next_track_params() -> dict[str, Any]:
  current_index = random.randrange(len(_SONG_TITLES))
  current_song = _SONG_TITLES[current_index]
  expected_next_song = _SONG_TITLES[(current_index + 1) % len(_SONG_TITLES)]
  return {
      "seed": random.randint(0, 1_000_000),
      "current_song": current_song,
      "expected_next_song": expected_next_song,
  }


def _generate_rename_playlist_params() -> dict[str, Any]:
  old_name, new_name = random.sample(_PLAYLIST_NAMES, 2)
  return {
      "seed": random.randint(0, 1_000_000),
      "old_playlist_name": old_name,
      "new_playlist_name": new_name,
  }


def _generate_playlist_song_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "playlist_name": random.choice(_PLAYLIST_NAMES),
      "song_title": random.choice(_SONG_TITLES),
  }


# -----------------------------------------------------------------------------
# Base evaluators (shared by every app port).
# -----------------------------------------------------------------------------


class _RetroCreatePlaylistBase(_MusicPackageAppEval):
  """Base port: create a playlist with the requested name."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "playlist_name": {"type": "string"},
      },
      "required": ["playlist_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if _media_playlist_exists(env, self._params["playlist_name"]) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_create_playlist_params()


class _RetroPlayingQueueBase(_MusicPackageAppEval):
  """Base port: add a specific song by an artist to the playing queue."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "song_title": {"type": "string"},
          "artist": {"type": "string"},
      },
      "required": ["song_title", "artist"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    song_ok = base.element_text_contains(
        ui_elements, (self._params["song_title"],)
    )
    queue_ok = base.element_text_contains(
        ui_elements, ("queue", "now playing", "up next")
    )
    return 1.0 if song_ok and queue_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_playing_queue_params()


class _RetroSavePlaylistBase(_MusicPackageAppEval):
  """Base port: save the current playing queue as a named playlist."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "playlist_name": {"type": "string"},
      },
      "required": ["playlist_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    name_ok = base.element_text_contains(
        ui_elements, (self._params["playlist_name"],)
    )
    saved_ok = base.element_text_contains(ui_elements, ("saved", "playlist"))
    return 1.0 if name_ok and saved_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_save_playlist_params()


class _RetroPlaylistDurationBase(_MusicPackageAppEval):
  """Base port: open a playlist and surface its total duration."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "playlist_name": {"type": "string"},
      },
      "required": ["playlist_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    duration_markers = (
        "duration",
        "total",
        "mins",
        "minutes",
        "hours",
        "h ",
    )
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements, duration_markers
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_playlist_duration_params()


class _RetroCreateTwoPlaylistsBase(_MusicPackageAppEval):
  """Base port: create two distinct playlists in a row."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "playlist_name_a": {"type": "string"},
          "playlist_name_b": {"type": "string"},
      },
      "required": ["playlist_name_a", "playlist_name_b"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    a_ok = _media_playlist_exists(env, self._params["playlist_name_a"])
    b_ok = _media_playlist_exists(env, self._params["playlist_name_b"])
    return 1.0 if a_ok and b_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_create_two_playlists_params()


class _RetroAddToQueueBase(_MusicPackageAppEval):
  """Base port: add a specific song to the playing queue."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "song_title": {"type": "string"},
      },
      "required": ["song_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    song_ok = base.element_text_contains(
        ui_elements, (self._params["song_title"],)
    )
    queue_ok = base.element_text_contains(
        ui_elements, ("queue", "added", "added to queue")
    )
    return 1.0 if song_ok and queue_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_to_queue_params()


class _RetroPlaySongBase(_MusicPackageAppEval):
  """Base port: start playback of a specific song."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "song_title": {"type": "string"},
      },
      "required": ["song_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    song_ok = base.element_text_contains(
        ui_elements, (self._params["song_title"],)
    )
    playing_ok = base.element_text_contains(
        ui_elements, ("playing", "now playing", "pause")
    )
    return 1.0 if song_ok and playing_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_play_song_params()


class _RetroPauseSongBase(_MusicPackageAppEval):
  """Base port: pause the currently playing song."""

  complexity = 1.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
      },
      "required": [],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    paused_markers = ("paused", "resume")
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements, paused_markers
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_pause_song_params()


class _RetroShuffleQueueBase(_MusicPackageAppEval):
  """Base port: shuffle the current playing queue."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
      },
      "required": [],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    shuffle_markers = ("shuffle", "shuffled", "random")
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements, shuffle_markers
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_shuffle_queue_params()


class _RetroNextTrackBase(_MusicPackageAppEval):
  """Base port: skip to the next track in the playing queue."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "current_song": {"type": "string"},
          "expected_next_song": {"type": "string"},
      },
      "required": ["current_song", "expected_next_song"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["expected_next_song"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_next_track_params()


class _RetroRenamePlaylistBase(_MusicPackageAppEval):
  """Base port: rename an existing playlist."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "old_playlist_name": {"type": "string"},
          "new_playlist_name": {"type": "string"},
      },
      "required": ["old_playlist_name", "new_playlist_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    new_ok = _media_playlist_exists(env, self._params["new_playlist_name"])
    old_ok = _media_playlist_exists(env, self._params["old_playlist_name"])
    return 1.0 if new_ok and not old_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_rename_playlist_params()


class _RetroAddToPlaylistBase(_MusicPackageAppEval):
  """Base port: add a song to a named playlist."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "playlist_name": {"type": "string"},
          "song_title": {"type": "string"},
      },
      "required": ["playlist_name", "song_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    song_ok = base.element_text_contains(ui_elements, (self._params["song_title"],))
    playlist_ok = base.element_text_contains(
        ui_elements, (self._params["playlist_name"], "playlist")
    )
    return 1.0 if song_ok and playlist_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_playlist_song_params()


class _RetroRemoveFromPlaylistBase(_RetroAddToPlaylistBase):
  """Base port: remove a song from a playlist."""

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    playlist_ok = base.element_text_contains(
        ui_elements, (self._params["playlist_name"], "playlist")
    )
    removed_ok = not base.element_text_contains(
        ui_elements, (self._params["song_title"],)
    )
    return 1.0 if playlist_ok and removed_ok else 0.0


class _RetroReorderQueueBase(_MusicPackageAppEval):
  """Base port: reorder the current queue."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {"seed": {"type": "integer"}},
      "required": [],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markers = ("queue", "up next", "reorder", "drag")
    return (
        1.0
        if base.element_text_contains(env.get_state().ui_elements, markers)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"seed": random.randint(0, 1_000_000)}


class _RetroSleepTimerBase(_MusicPackageAppEval):
  """Base port: set a sleep timer."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {"seed": {"type": "integer"}},
      "required": [],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markers = ("sleep", "timer", "minutes", "turn off")
    return (
        1.0
        if base.element_text_contains(env.get_state().ui_elements, markers)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"seed": random.randint(0, 1_000_000)}


class _RetroSearchAndPlayBase(_RetroPlaySongBase):
  """Base port: search for a song and play it."""

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    song_ok = base.element_text_contains(ui_elements, (self._params["song_title"],))
    play_ok = base.element_text_contains(
        ui_elements, ("playing", "pause", "now playing")
    )
    return 1.0 if song_ok and play_ok else 0.0


# -----------------------------------------------------------------------------
# Per-app package constants.
# -----------------------------------------------------------------------------

_RETROMUSIC_PACKAGE: Final[str] = "code.name.monkey.retromusic"
_FOSSIFYMUSIC_PACKAGE: Final[str] = "org.fossify.musicplayer"
_APOLLO_PACKAGE: Final[str] = "org.nuclearfog.apollo"
_SICMU_PACKAGE: Final[str] = "xyz.mordorx.sicmu"
_PHONOGRAPH_PACKAGE: Final[str] = "player.phonograph.plus"
_MONSTER_PACKAGE: Final[str] = "com.ztftrue.music"


# ----------- Retro Music ----------


class RetroCreatePlaylistForRetroMusic(_RetroCreatePlaylistBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, create a new playlist named '{playlist_name}'."
  )


class RetroPlayingQueueForRetroMusic(_RetroPlayingQueueBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, add the song '{song_title}' by {artist} to the"
      " playing queue, then open the playing queue and leave the song visible."
  )


class RetroSavePlaylistForRetroMusic(_RetroSavePlaylistBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, save the current playing queue as a playlist"
      " named '{playlist_name}'."
  )


class RetroPlaylistDurationForRetroMusic(_RetroPlaylistDurationBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, open the playlist '{playlist_name}' and view"
      " its total duration."
  )


class RetroCreateTwoPlaylistsForRetroMusic(_RetroCreateTwoPlaylistsBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, create two new playlists named"
      " '{playlist_name_a}' and '{playlist_name_b}'."
  )


class RetroAddToQueueForRetroMusic(_RetroAddToQueueBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, add the song '{song_title}' to the playing"
      " queue, then open the playing queue and leave the song visible."
  )


class RetroPlaySongForRetroMusic(_RetroPlaySongBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = "In the Retro Music app, play the song titled '{song_title}'."


class RetroPauseSongForRetroMusic(_RetroPauseSongBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = "In the Retro Music app, pause the currently playing song."


class RetroShuffleQueueForRetroMusic(_RetroShuffleQueueBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = "In the Retro Music app, shuffle the current playing queue."


class RetroNextTrackForRetroMusic(_RetroNextTrackBase):
  app_names = (_RETROMUSIC_PACKAGE,)
  package_name = _RETROMUSIC_PACKAGE
  template = (
      "In the Retro Music app, play '{current_song}', then skip to the next"
      " track so that '{expected_next_song}' is playing."
  )


# ----------- Fossify Music ----------


class RetroCreatePlaylistForFossifyMusic(_RetroCreatePlaylistBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, create a new playlist named"
      " '{playlist_name}'."
  )


class RetroPlayingQueueForFossifyMusic(_RetroPlayingQueueBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, add the song '{song_title}' by {artist} to"
      " the playing queue, then open the playing queue and leave the song"
      " visible."
  )


class RetroSavePlaylistForFossifyMusic(_RetroSavePlaylistBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, save the current playing queue as a playlist"
      " named '{playlist_name}'."
  )


class RetroPlaylistDurationForFossifyMusic(_RetroPlaylistDurationBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, open the playlist '{playlist_name}' and view"
      " its total duration."
  )


class RetroCreateTwoPlaylistsForFossifyMusic(_RetroCreateTwoPlaylistsBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, create two new playlists named"
      " '{playlist_name_a}' and '{playlist_name_b}'."
  )


class RetroAddToQueueForFossifyMusic(_RetroAddToQueueBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, add the song '{song_title}' to the playing"
      " queue, then open the playing queue and leave the song visible."
  )


class RetroPlaySongForFossifyMusic(_RetroPlaySongBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = "In the Fossify Music app, play the song titled '{song_title}'."


class RetroPauseSongForFossifyMusic(_RetroPauseSongBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = "In the Fossify Music app, pause the currently playing song."


class RetroShuffleQueueForFossifyMusic(_RetroShuffleQueueBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = "In the Fossify Music app, shuffle the current playing queue."


class RetroNextTrackForFossifyMusic(_RetroNextTrackBase):
  app_names = (_FOSSIFYMUSIC_PACKAGE,)
  package_name = _FOSSIFYMUSIC_PACKAGE
  template = (
      "In the Fossify Music app, play '{current_song}', then skip to the next"
      " track so that '{expected_next_song}' is playing."
  )


# ----------- Apollo ----------


class RetroCreatePlaylistForApollo(_RetroCreatePlaylistBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, create a new playlist named '{playlist_name}'."
  )


class RetroPlayingQueueForApollo(_RetroPlayingQueueBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, add the song '{song_title}' by {artist} to the"
      " playing queue, then open the playing queue and leave the song visible."
  )


class RetroSavePlaylistForApollo(_RetroSavePlaylistBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, save the current playing queue as a playlist named"
      " '{playlist_name}'."
  )


class RetroPlaylistDurationForApollo(_RetroPlaylistDurationBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, open the playlist '{playlist_name}' and view its"
      " total duration."
  )


class RetroCreateTwoPlaylistsForApollo(_RetroCreateTwoPlaylistsBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, create two new playlists named '{playlist_name_a}'"
      " and '{playlist_name_b}'."
  )


class RetroAddToQueueForApollo(_RetroAddToQueueBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, add the song '{song_title}' to the playing queue,"
      " then open the playing queue and leave the song visible."
  )


class RetroPlaySongForApollo(_RetroPlaySongBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = "In the Apollo app, play the song titled '{song_title}'."


class RetroPauseSongForApollo(_RetroPauseSongBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = "In the Apollo app, pause the currently playing song."


class RetroShuffleQueueForApollo(_RetroShuffleQueueBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = "In the Apollo app, shuffle the current playing queue."


class RetroNextTrackForApollo(_RetroNextTrackBase):
  app_names = (_APOLLO_PACKAGE,)
  package_name = _APOLLO_PACKAGE
  template = (
      "In the Apollo app, play '{current_song}', then skip to the next track"
      " so that '{expected_next_song}' is playing."
  )


# ----------- SicMu Neo ----------


class RetroCreatePlaylistForSicMuNeo(_RetroCreatePlaylistBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, create a new playlist named '{playlist_name}'."
  )


class RetroPlayingQueueForSicMuNeo(_RetroPlayingQueueBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, add the song '{song_title}' by {artist} to the"
      " playing queue, then open the playing queue and leave the song visible."
  )


class RetroSavePlaylistForSicMuNeo(_RetroSavePlaylistBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, save the current playing queue as a playlist"
      " named '{playlist_name}'."
  )


class RetroPlaylistDurationForSicMuNeo(_RetroPlaylistDurationBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, open the playlist '{playlist_name}' and view its"
      " total duration."
  )


class RetroCreateTwoPlaylistsForSicMuNeo(_RetroCreateTwoPlaylistsBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, create two new playlists named"
      " '{playlist_name_a}' and '{playlist_name_b}'."
  )


class RetroAddToQueueForSicMuNeo(_RetroAddToQueueBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, add the song '{song_title}' to the playing"
      " queue, then open the playing queue and leave the song visible."
  )


class RetroPlaySongForSicMuNeo(_RetroPlaySongBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = "In the SicMu Neo app, play the song titled '{song_title}'."


class RetroPauseSongForSicMuNeo(_RetroPauseSongBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = "In the SicMu Neo app, pause the currently playing song."


class RetroShuffleQueueForSicMuNeo(_RetroShuffleQueueBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = "In the SicMu Neo app, shuffle the current playing queue."


class RetroNextTrackForSicMuNeo(_RetroNextTrackBase):
  app_names = (_SICMU_PACKAGE,)
  package_name = _SICMU_PACKAGE
  template = (
      "In the SicMu Neo app, play '{current_song}', then skip to the next"
      " track so that '{expected_next_song}' is playing."
  )


# ----------- Phonograph Plus ----------


class RetroCreatePlaylistForPhonographPlus(_RetroCreatePlaylistBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, create a new playlist named"
      " '{playlist_name}'."
  )


class RetroPlayingQueueForPhonographPlus(_RetroPlayingQueueBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, add the song '{song_title}' by {artist} to"
      " the playing queue, then open the playing queue and leave the song"
      " visible."
  )


class RetroSavePlaylistForPhonographPlus(_RetroSavePlaylistBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, save the current playing queue as a"
      " playlist named '{playlist_name}'."
  )


class RetroPlaylistDurationForPhonographPlus(_RetroPlaylistDurationBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, open the playlist '{playlist_name}' and"
      " view its total duration."
  )


class RetroCreateTwoPlaylistsForPhonographPlus(_RetroCreateTwoPlaylistsBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, create two new playlists named"
      " '{playlist_name_a}' and '{playlist_name_b}'."
  )


class RetroAddToQueueForPhonographPlus(_RetroAddToQueueBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, add the song '{song_title}' to the playing"
      " queue, then open the playing queue and leave the song visible."
  )


class RetroPlaySongForPhonographPlus(_RetroPlaySongBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, play the song titled '{song_title}'."
  )


class RetroPauseSongForPhonographPlus(_RetroPauseSongBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = "In the Phonograph Plus app, pause the currently playing song."


class RetroShuffleQueueForPhonographPlus(_RetroShuffleQueueBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, shuffle the current playing queue."
  )


class RetroNextTrackForPhonographPlus(_RetroNextTrackBase):
  app_names = (_PHONOGRAPH_PACKAGE,)
  package_name = _PHONOGRAPH_PACKAGE
  template = (
      "In the Phonograph Plus app, play '{current_song}', then skip to the next"
      " track so that '{expected_next_song}' is playing."
  )


# ----------- MonsterMusic ----------


class RetroCreatePlaylistForMonsterMusic(_RetroCreatePlaylistBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, create a new playlist named"
      " '{playlist_name}'."
  )


class RetroPlayingQueueForMonsterMusic(_RetroPlayingQueueBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, add the song '{song_title}' by {artist} to"
      " the playing queue, then open the playing queue and leave the song"
      " visible."
  )


class RetroSavePlaylistForMonsterMusic(_RetroSavePlaylistBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, save the current playing queue as a playlist"
      " named '{playlist_name}'."
  )


class RetroPlaylistDurationForMonsterMusic(_RetroPlaylistDurationBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, open the playlist '{playlist_name}' and view"
      " its total duration."
  )


class RetroCreateTwoPlaylistsForMonsterMusic(_RetroCreateTwoPlaylistsBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, create two new playlists named"
      " '{playlist_name_a}' and '{playlist_name_b}'."
  )


class RetroAddToQueueForMonsterMusic(_RetroAddToQueueBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, add the song '{song_title}' to the playing"
      " queue, then open the playing queue and leave the song visible."
  )


class RetroPlaySongForMonsterMusic(_RetroPlaySongBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = "In the MonsterMusic app, play the song titled '{song_title}'."


class RetroPauseSongForMonsterMusic(_RetroPauseSongBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = "In the MonsterMusic app, pause the currently playing song."


class RetroShuffleQueueForMonsterMusic(_RetroShuffleQueueBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = "In the MonsterMusic app, shuffle the current playing queue."


class RetroNextTrackForMonsterMusic(_RetroNextTrackBase):
  app_names = (_MONSTER_PACKAGE,)
  package_name = _MONSTER_PACKAGE
  template = (
      "In the MonsterMusic app, play '{current_song}', then skip to the next"
      " track so that '{expected_next_song}' is playing."
  )


def _make_music_table1_task(
    class_name: str,
    base_class: type[_MusicPackageAppEval],
    package_name: str,
    template: str,
) -> type[_MusicPackageAppEval]:
  return type(
      class_name,
      (base_class,),
      {
          "__module__": __name__,
          "app_names": (package_name,),
          "package_name": package_name,
          "template": template,
      },
  )


_MUSIC_TABLE1_MISSING_SPECS: Final[
    tuple[tuple[str, type[_MusicPackageAppEval], str], ...]
] = (
    (
        "RenamePlaylist",
        _RetroRenamePlaylistBase,
        "In the {app} app, rename playlist '{old_playlist_name}' to "
        "'{new_playlist_name}'.",
    ),
    (
        "AddToPlaylist",
        _RetroAddToPlaylistBase,
        "In the {app} app, add song '{song_title}' to playlist "
        "'{playlist_name}'.",
    ),
    (
        "RemoveFromPlaylist",
        _RetroRemoveFromPlaylistBase,
        "In the {app} app, remove song '{song_title}' from playlist "
        "'{playlist_name}'.",
    ),
    (
        "ReorderQueue",
        _RetroReorderQueueBase,
        "In the {app} app, reorder the current playing queue.",
    ),
    (
        "SleepTimer",
        _RetroSleepTimerBase,
        "In the {app} app, set a sleep timer for the currently playing music.",
    ),
    (
        "SearchAndPlay",
        _RetroSearchAndPlayBase,
        "In the {app} app, search for and play the song '{song_title}'.",
    ),
)

_MUSIC_TABLE1_APPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("RetroMusic", _RETROMUSIC_PACKAGE, "Retro Music"),
    ("FossifyMusic", _FOSSIFYMUSIC_PACKAGE, "Fossify Music"),
    ("Apollo", _APOLLO_PACKAGE, "Apollo"),
    ("SicMuNeo", _SICMU_PACKAGE, "SicMu Neo"),
    ("PhonographPlus", _PHONOGRAPH_PACKAGE, "Phonograph Plus"),
    ("MonsterMusic", _MONSTER_PACKAGE, "MonsterMusic"),
)

for _suffix, _package, _display_name in _MUSIC_TABLE1_APPS:
  for _task_name, _base_class, _template in _MUSIC_TABLE1_MISSING_SPECS:
    globals()[f"Retro{_task_name}For{_suffix}"] = _make_music_table1_task(
        f"Retro{_task_name}For{_suffix}",
        _base_class,
        _package,
        _template.replace("{app}", _display_name),
    )
