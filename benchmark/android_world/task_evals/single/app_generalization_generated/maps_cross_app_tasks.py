"""Cross-app maps task ports for the app-generalization suite.

Map apps are hard to validate uniformly because map renderers render very
differently across apps. For durable favorite/marker/track tasks, this module
follows ``docs/tasks_guide.md`` and reads each app's real storage where it is
stable:

  * OsmAnd: ``favorites.gpx``, ``map_markers_db``, and GPX track files.
  * Organic Maps / CoMaps: ``My Places.kml`` bookmarks.
  * MAPS.ME: transient UI tasks only; user saved-place state lives in binary
    ``My Places.kmb``, not the downloaded-guides SQLite catalog.

Google Maps does not expose a reliable local saved-place table in this test
image, and MAPS.ME does not expose reliable local saved-place or GPX/KML
artifacts in ours. Generated adapters remain importable for compatibility,
but the frozen ten-task profile fully deschedules Google Maps and MAPS.ME so
every included app has the same semantic grid. Validation mode is reported
separately by the live markdown reporter.

The UI-only tasks use a combination of:

  * UI-text checks for visible side-panel content (favorite name, marker
    state, track list entry).
  * Content-description checks for ``pin`` / ``marker`` / ``star`` icons on
    the main map surface.

These are intentionally loose and should be treated as lower-bound signals;
papers that need stricter validation should extend the ``is_successful``
method per app.

Tasks in this module:

  * ``MapsSearchPlace`` -- search for a named place.
  * ``MapsViewPlaceInfo`` -- open the place-details panel.
  * ``MapsAddFavorite`` -- search, then add to favorites.
  * ``MapsRemoveFavorite`` -- add, then remove a favorite.
  * ``MapsAddMarker`` -- search, then drop a marker.
  * ``MapsDeleteMarker`` -- drop, then delete a marker.
  * ``MapsGetDirections`` -- pull up a route between two named places.
  * ``MapsSearchNearbyPlace`` -- search for a category near a place.
  * ``MapsExportLocation`` -- export/share a place as GPX/KML/link artifact.
  * ``MapsShareLocation`` -- open the share-sheet for a location.
"""

from __future__ import annotations

import base64
import random
import re
import shlex
import time
from xml.etree import ElementTree
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)


# Keep these aligned with AndroidWorld's upstream OsmAnd task locations. The
# emulator image ships with this offline map data; global landmarks often need
# network or extra map downloads and turn otherwise valid tasks infeasible.
_PLACES: Final[tuple[str, ...]] = (
    "Balzers, Liechtenstein",
    "Bendern, Liechtenstein",
    "Malbun, Liechtenstein",
    "Nendeln, Liechtenstein",
    "Oberplanken, Liechtenstein",
    "Planken, Liechtenstein",
    "Rotenboden, Liechtenstein",
    "Ruggell, Liechtenstein",
    "Schaan, Liechtenstein",
    "Schaanwald, Liechtenstein",
    "Schönberg, Liechtenstein",
    "Triesen, Liechtenstein",
)

_TRACK_NAMES: Final[tuple[str, ...]] = (
    "Morning Run",
    "Forest Hike",
    "City Walk",
    "Sunset Loop",
    "Bike Ride",
)

_PLACE_COORDS: Final[dict[str, tuple[float, float]]] = {
    "Balzers, Liechtenstein": (47.0688832, 9.5061564),
    "Bendern, Liechtenstein": (47.2122151, 9.5062101),
    "Malbun, Liechtenstein": (47.1026191, 9.6083057),
    "Nendeln, Liechtenstein": (47.1973857, 9.5430636),
    "Oberplanken, Liechtenstein": (47.1784977, 9.5450163),
    "Planken, Liechtenstein": (47.1858882, 9.5452201),
    "Rotenboden, Liechtenstein": (47.1275785, 9.5387131),
    "Ruggell, Liechtenstein": (47.23976, 9.5262837),
    "Schaan, Liechtenstein": (47.1663432, 9.5103085),
    "Schaanwald, Liechtenstein": (47.2165476, 9.5699984),
    "Schönberg, Liechtenstein": (47.1303814, 9.5930117),
    "Triesen, Liechtenstein": (47.106997, 9.5274854),
}
_COORD_TOLERANCE_DEG: Final[float] = 0.001

_OSMAND_PACKAGE_NAME: Final[str] = "net.osmand.plus"
_ORGANIC_MAPS_PACKAGE_NAME: Final[str] = "app.organicmaps"
_GOOGLE_MAPS_PACKAGE_NAME: Final[str] = "com.google.android.apps.maps"
_COMAPS_PACKAGE_NAME: Final[str] = "app.comaps.fdroid"
_MAPS_ME_PACKAGE_NAME: Final[str] = "com.mapswithme.maps.pro"
_CHOOSER_PACKAGES: Final[frozenset[str]] = frozenset((
    "com.android.intentresolver",
    # Framework-bundled ResolverActivity builds expose package ``android``.
    "android",
))
_CHOOSER_TEXT_PREVIEW_RESOURCE_ID: Final[str] = "content_preview_text"
_EXPORT_CLIPBOARD_SENTINEL: Final[str] = "CATBENCH_MAP_EXPORT_CLIPBOARD_RESET"
_SHELL_STATUS_MARKER: Final[str] = "__CATBENCH_MAPS_SHELL_STATUS__:"

_ROUTE_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:min(?:ute)?s?|h(?:ou)?rs?|hrs?)(?!\w)",
    re.IGNORECASE,
)
_ROUTE_DISTANCE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:km|mi(?:les?)?|m(?:eters?)?|ft)(?!\w)",
    re.IGNORECASE,
)
_ROUTE_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:route|trip|distance|via|fastest|recommended)\b",
    re.IGNORECASE,
)
_MAP_RESULT_RESOURCE_TOKENS: Final[tuple[str, ...]] = (
    "result",
    "suggestion",
    "search_item",
    "place",
    "poi",
)
_EMPTY_RESULT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:no|0)\s+(?:search\s+)?results?\b|"
    r"\bnothing\s+(?:was\s+)?found\b|\bcould\s+not\s+find\b",
    re.IGNORECASE,
)

_OSMAND_FAVORITES_PATH: Final[str] = (
    "/data/media/0/Android/data/net.osmand.plus/files/favorites/favorites.gpx"
)
_OSMAND_MARKERS_DB: Final[str] = (
    "/data/data/net.osmand.plus/databases/map_markers_db"
)
_OSMAND_TRACKS_DIR: Final[str] = (
    "/data/media/0/Android/data/net.osmand.plus/files/tracks"
)
_SHARED_TRACK_EXPORT_DIRS: Final[tuple[str, ...]] = (
    "/data/media/0/Download",
    "/data/media/0/Documents",
)
_SHARED_LINK_EXPORT_DIRS: Final[tuple[str, ...]] = (
    "/data/media/0/Download",
    "/data/media/0/Documents",
)
_ORGANIC_KML_PATH: Final[str] = (
    "/data/data/app.organicmaps/files/bookmarks/My Places.kml"
)
_COMAPS_KML_PATH: Final[str] = (
    "/data/data/app.comaps.fdroid/files/bookmarks/My Places.kml"
)
_MAPS_ME_GUIDES_DB: Final[str] = (
    "/data/data/com.mapswithme.maps.pro/databases/guides"
)
_KML_PACKAGES_TO_PATHS: Final[dict[str, str]] = {
    _ORGANIC_MAPS_PACKAGE_NAME: _ORGANIC_KML_PATH,
    _COMAPS_PACKAGE_NAME: _COMAPS_KML_PATH,
}
# Storage-validated maps apps, per the AW durable-state guideline. MAPS.ME is
# deliberately NOT here: its `guides` SQLite Bookmark table is the
# downloaded-guides catalog (user bookmarks live in the binary
# `My Places.kmb`), so DB-based seeding/validation never reflects what the
# agent or the app actually does. Like Google Maps, its generated adapters are
# retained for compatibility but it is excluded from the frozen profile (see
# app_generalization_profiles.py).
_STORAGE_FAVORITE_PACKAGES: Final[frozenset[str]] = frozenset((
    _OSMAND_PACKAGE_NAME,
    _ORGANIC_MAPS_PACKAGE_NAME,
    _COMAPS_PACKAGE_NAME,
))
_STORAGE_MARKER_PACKAGES: Final[frozenset[str]] = _STORAGE_FAVORITE_PACKAGES
_TRACK_STORAGE_PATHS: Final[dict[str, tuple[str, ...]]] = {
    _OSMAND_PACKAGE_NAME: (_OSMAND_TRACKS_DIR, *_SHARED_TRACK_EXPORT_DIRS),
    _ORGANIC_MAPS_PACKAGE_NAME: (
        "/data/data/app.organicmaps/files",
        "/data/media/0/Android/data/app.organicmaps",
        *_SHARED_TRACK_EXPORT_DIRS,
    ),
    _COMAPS_PACKAGE_NAME: (
        "/data/data/app.comaps.fdroid/files",
        "/data/media/0/Android/data/app.comaps.fdroid",
        *_SHARED_TRACK_EXPORT_DIRS,
    ),
    _MAPS_ME_PACKAGE_NAME: (
        "/data/data/com.mapswithme.maps.pro/files",
        "/data/media/0/Android/data/com.mapswithme.maps.pro",
        *_SHARED_TRACK_EXPORT_DIRS,
    ),
    _GOOGLE_MAPS_PACKAGE_NAME: (
        "/data/data/com.google.android.apps.maps/files",
        "/data/media/0/Android/data/com.google.android.apps.maps",
        *_SHARED_TRACK_EXPORT_DIRS,
    ),
}
_EXPORT_STORAGE_PATHS: Final[dict[str, tuple[str, ...]]] = _TRACK_STORAGE_PATHS
_CATBENCH_MARKER_LABELS: Final[tuple[str, ...]] = (
    "Meetup",
    "Picnic",
    "Viewpoint",
    "Hotel",
    "Trail",
    "Lookout",
)
_CITY_NAMES: Final[tuple[str, ...]] = tuple(
    place.split(",", 1)[0].strip() for place in _PLACES
)
_CATBENCH_STORAGE_NAMES: Final[tuple[str, ...]] = (
    *_PLACES,
    *_CITY_NAMES,
    *_CATBENCH_MARKER_LABELS,
)


def _choose_two_places() -> tuple[str, str]:
  a, b = random.sample(_PLACES, 2)
  return a, b


def _parse_coords(location: str) -> tuple[float, float] | None:
  try:
    coords = tuple(float(value) for value in re.findall(r"-?\d*\.?\d+", location))
  except ValueError:
    return None
  return coords if len(coords) == 2 else None


def _random_location_str(
    *,
    names_only: bool = False,
    num_locations: int = 1,
) -> list[str]:
  """Matches AndroidWorld OsmAnd's name-or-coordinate parameter style."""
  locations = random.sample(list(_PLACES), num_locations)
  if names_only:
    return locations
  coords = [_PLACE_COORDS[location] for location in locations]
  if random.getrandbits(1):
    return locations
  return [f"{lat}, {lon}" for lat, lon in coords]


def _adb_shell(env: interface.AsyncEnv, cmd: str) -> str:
  out = adb_utils.issue_generic_request(["shell", cmd], env.controller)
  return out.generic.output.decode("utf-8", errors="ignore") if out else ""


def _su_shell(env: interface.AsyncEnv, cmd: str) -> str:
  return _adb_shell(env, f"su 0 sh -c {shlex.quote(f'{cmd} || true')}")


def _su_shell_checked(
    env: interface.AsyncEnv,
    cmd: str,
    *,
    operation: str,
) -> str:
  """Runs a read command and distinguishes empty state from read failure."""
  status_var = "_catbench_maps_status"
  script = (
      f"{cmd}\n"
      f"{status_var}=$?\n"
      f"printf '\\n{_SHELL_STATUS_MARKER}%s\\n' \"${status_var}\""
  )
  try:
    output = _adb_shell(env, f"su 0 sh -c {shlex.quote(script)}")
  except Exception as error:
    raise base.VerifierStateReadError(
        f"Could not {operation}: ADB read failed"
    ) from error

  separator = f"\n{_SHELL_STATUS_MARKER}"
  payload, found, status_output = output.rpartition(separator)
  if not found:
    raise base.VerifierStateReadError(
        f"Could not {operation}: missing shell status marker"
    )
  try:
    status = int(status_output.splitlines()[0].strip())
  except (IndexError, ValueError) as error:
    raise base.VerifierStateReadError(
        f"Could not {operation}: malformed shell status"
    ) from error
  if status != 0:
    raise base.VerifierStateReadError(
        f"Could not {operation}: shell command exited with status {status}"
    )
  return payload


def _sqlite_exec(env: interface.AsyncEnv, db_path: str, sql: str) -> str:
  quoted_db_path = shlex.quote(db_path)
  return _su_shell_checked(
      env,
      (
          f"if [ -e {quoted_db_path} ]; then "
          f"sqlite3 {quoted_db_path} {shlex.quote(sql)} 2>/dev/null; fi"
      ),
      operation=f"read SQLite state from {db_path}",
  )


def _sql_quote(value: str) -> str:
  return "'" + value.replace("'", "''") + "'"


def _sql_in(values: tuple[str, ...]) -> str:
  return "(" + ",".join(_sql_quote(value) for value in values) + ")"


def _sqlite_coords_filter(
    lat_column: str,
    lon_column: str,
    places: tuple[str, ...],
) -> str:
  clauses = []
  for place in places:
    coords = _coords(place)
    if coords is None:
      continue
    lat, lon = coords
    clauses.append(
        f"(ABS({lat_column} - {lat}) <= {_COORD_TOLERANCE_DEG}"
        f" AND ABS({lon_column} - {lon}) <= {_COORD_TOLERANCE_DEG})"
    )
  return "(" + " OR ".join(clauses) + ")" if clauses else "0"


def _read_root_file(env: interface.AsyncEnv, path: str) -> str:
  quoted_path = shlex.quote(path)
  return _su_shell_checked(
      env,
      (
          f"if [ -e {quoted_path} ]; then "
          f"cat {quoted_path} 2>/dev/null; fi"
      ),
      operation=f"read file state from {path}",
  )


def _list_root_files(
    env: interface.AsyncEnv,
    path: str,
    *,
    name_pattern: str = "*.gpx",
) -> tuple[str, ...]:
  quoted_path = shlex.quote(path)
  out = _su_shell_checked(
      env,
      (
          f"if [ -d {quoted_path} ]; then "
          f"find {quoted_path} -type f"
          f" -name {shlex.quote(name_pattern)} 2>/dev/null; fi"
      ),
      operation=f"list files under {path}",
  )
  return tuple(line.strip() for line in out.splitlines() if line.strip())


def _list_root_files_by_extensions(
    env: interface.AsyncEnv,
    path: str,
    extensions: tuple[str, ...],
) -> tuple[str, ...]:
  patterns = " -o ".join(
      f"-iname {shlex.quote('*.' + extension.lstrip('.'))}"
      for extension in extensions
  )
  quoted_path = shlex.quote(path)
  out = _su_shell_checked(
      env,
      (
          f"if [ -d {quoted_path} ]; then "
          f"find {quoted_path} -type f"
          f" \\( {patterns} \\) 2>/dev/null; fi"
      ),
      operation=f"list files under {path}",
  )
  return tuple(line.strip() for line in out.splitlines() if line.strip())


def _write_root_file(env: interface.AsyncEnv, path: str, content: str) -> None:
  encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
  parent = path.rsplit("/", 1)[0]
  _su_shell(
      env,
      (
          f"mkdir -p {shlex.quote(parent)} && "
          f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
      ),
  )


def _force_stop_and_launch(package_name: str, env: interface.AsyncEnv) -> None:
  _adb_shell(env, f"am force-stop {shlex.quote(package_name)}")
  time.sleep(0.5)
  adb_utils.launch_app(package_name, env.controller)


def _ui_text_contains_word(
    ui_elements: list[Any],
    candidates: tuple[str, ...],
) -> bool:
  patterns = tuple(
      re.compile(rf"\b{re.escape(candidate.casefold())}\b")
      for candidate in candidates
  )
  for element in ui_elements:
    for field in (element.text, element.content_description):
      if not field:
        continue
      lowered = field.casefold()
      if any(pattern.search(lowered) for pattern in patterns):
        return True
  return False


def _place_candidates(place: str) -> tuple[str, ...]:
  city = place.split(",", 1)[0].strip()
  if city and city.casefold() != place.casefold():
    return (place, city)
  return (place,)


def _text_contains_place_candidate(text: str | None, candidate: str) -> bool:
  if not text:
    return False
  return (
      re.search(
          rf"(?<!\w){re.escape(candidate.casefold())}(?!\w)",
          text.casefold(),
      )
      is not None
  )


def _place_visible(ui_elements: list[Any], place: str) -> bool:
  candidates = _place_candidates(place)
  for element in ui_elements:
    for field in (element.text, element.content_description):
      if any(
          _text_contains_place_candidate(field, candidate)
          for candidate in candidates
      ):
        return True
  return False


def _element_is_editable(element: Any) -> bool:
  """Whether an accessibility element is an input rather than result text."""
  if getattr(element, "is_editable", False) is True:
    return True
  class_name = (getattr(element, "class_name", None) or "").casefold()
  return class_name.endswith("edittext")


def _element_resource(element: Any) -> str:
  return (
      getattr(element, "resource_id", None)
      or getattr(element, "resource_name", None)
      or ""
  ).casefold()


def _element_text(element: Any) -> str:
  return "\n".join(
      field
      for field in (element.text, element.content_description)
      if field
  )


def _element_has_result_semantics(element: Any) -> bool:
  resource = _element_resource(element)
  return getattr(element, "is_clickable", False) is True or any(
      token in resource for token in _MAP_RESULT_RESOURCE_TOKENS
  )


def _place_search_result_visible(ui_elements: list[Any], place: str) -> bool:
  """Requires the target on a result/detail surface, not in the query box."""
  candidates = _place_candidates(place)
  has_noneditable_place = False
  for element in ui_elements:
    if _element_is_editable(element):
      continue
    if not any(
        _text_contains_place_candidate(field, candidate)
        for field in (element.text, element.content_description)
        for candidate in candidates
    ):
      continue
    has_noneditable_place = True
    if _element_has_result_semantics(element):
      return True

  # Selected-place panels often expose their title as a plain TextView. Bind
  # that title to detail actions rather than accepting the title by itself.
  return has_noneditable_place and base.element_text_contains_word(
      ui_elements,
      ("address", "directions", "route", "save", "share", "favorite"),
  )


def _computed_route_visible(ui_elements: list[Any]) -> bool:
  """Requires a computed-route readout, not a Directions/Start control."""
  for element in ui_elements:
    if _element_is_editable(element):
      continue
    resource = _element_resource(element)
    for field in (element.text, element.content_description):
      if not field:
        continue
      # A duration is route-result evidence by itself. A distance alone could
      # be the map scale, so it additionally needs route context in the same
      # accessible field or in the element's resource name.
      if _ROUTE_DURATION_RE.search(field):
        return True
      if _ROUTE_DISTANCE_RE.search(field) and (
          _ROUTE_CONTEXT_RE.search(field)
          or any(token in resource for token in ("route", "trip", "distance"))
      ):
        return True
  return False


def _populated_nearby_result_visible(
    ui_elements: list[Any],
    *,
    place: str,
    category: str,
) -> bool:
  """Whether at least one concrete, non-input nearby result is exposed."""
  excluded_labels = {
      candidate.casefold().strip() for candidate in _place_candidates(place)
  }
  excluded_labels.add(category.casefold().strip())
  excluded_labels.update(("search", "nearby", "results", "search results"))

  for element in ui_elements:
    if _element_is_editable(element) or not _element_has_result_semantics(
        element
    ):
      continue
    text = _element_text(element).strip()
    if not text or _EMPTY_RESULT_RE.search(text):
      continue
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if normalized in excluded_labels:
      continue
    return True
  return False


def _chooser_payload_contains_place(
    ui_elements: list[Any],
    place: str,
) -> bool:
  """Whether IntentResolver's ``EXTRA_TEXT`` preview binds to ``place``.

  Android's chooser renders shared text in ``content_preview_text``.  Checking
  that resolver-owned field avoids combining a generic chooser with a place
  name left visible in the source map app.  We deliberately do not accept the
  optional chooser title: ``Intent.EXTRA_TITLE`` can differ from the actual
  shared ``Intent.EXTRA_TEXT`` payload.
  """
  for element in ui_elements:
    package = (getattr(element, "package_name", None) or "").casefold()
    if package not in _CHOOSER_PACKAGES:
      continue
    resource = (
        getattr(element, "resource_id", None)
        or getattr(element, "resource_name", None)
        or ""
    )
    if resource.rsplit("/", 1)[-1].casefold() != (
        _CHOOSER_TEXT_PREVIEW_RESOURCE_ID
    ):
      continue
    payload = "\n".join(
        field
        for field in (element.text, element.content_description)
        if field
    )
    if _link_text_contains_place(payload, place):
      return True
  return False


def _coords(place: str) -> tuple[float, float] | None:
  return _PLACE_COORDS.get(place) or _parse_coords(place)


def _coords_match(
    actual: tuple[float, float],
    expected: tuple[float, float],
    *,
    delta_deg: float = _COORD_TOLERANCE_DEG,
) -> bool:
  return all(abs(a - e) <= delta_deg for a, e in zip(actual, expected))


def _parse_kml_coordinates(text: str | None) -> tuple[float, float] | None:
  if not text:
    return None
  stripped = text.strip()
  if not stripped:
    return None
  parts = stripped.split()[0].split(",")
  if len(parts) < 2:
    return None
  try:
    lon = float(parts[0])
    lat = float(parts[1])
  except ValueError:
    return None
  return lat, lon


def _entry_name(element: ElementTree.Element) -> str:
  for child in element.iter():
    local_name = child.tag.rsplit("}", 1)[-1].casefold()
    if local_name == "name":
      return child.text or ""
  return ""


def _entry_coords(element: ElementTree.Element) -> tuple[float, float] | None:
  local_name = element.tag.rsplit("}", 1)[-1].casefold()
  if local_name == "wpt":
    try:
      return float(element.attrib["lat"]), float(element.attrib["lon"])
    except (KeyError, ValueError):
      return None
  for child in element.iter():
    child_local_name = child.tag.rsplit("}", 1)[-1].casefold()
    if child_local_name == "coordinates":
      return _parse_kml_coordinates(child.text)
  return None


def _xml_entry_exists(
    xml_text: str,
    *,
    name: str | None = None,
    place: str | None = None,
    require_coords: bool = False,
    match_name_or_coords: bool = False,
) -> bool:
  if not xml_text.strip():
    return False
  if name is None and place is None:
    return False
  name_lower = name.casefold() if name else None
  expected_coords = _coords(place) if place else None
  try:
    root = ElementTree.fromstring(xml_text)
  except ElementTree.ParseError:
    return (
        name_lower is not None
        and name_lower in xml_text.casefold()
        and not require_coords
    )
  for element in root.iter():
    local_name = element.tag.rsplit("}", 1)[-1].casefold()
    if local_name not in ("wpt", "placemark"):
      continue
    entry_name = _entry_name(element).casefold()
    name_ok = name_lower is not None and name_lower in entry_name
    actual_coords = _entry_coords(element)
    coord_ok = (
        actual_coords is not None
        and expected_coords is not None
        and _coords_match(actual_coords, expected_coords)
    )
    if match_name_or_coords:
      if name_ok or coord_ok:
        return True
      continue
    if name_lower is not None and not name_ok:
      continue
    if name_lower is None and expected_coords is not None and not coord_ok:
      continue
    if require_coords and not coord_ok:
      continue
    if name_lower is not None or expected_coords is not None:
      return True
  return False


def _remove_xml_entries(
    xml_text: str,
    names: tuple[str, ...],
    places: tuple[str, ...] = (),
) -> str:
  """Remove GPX/KML entries whose names or coordinates are CATBench-owned."""
  if not xml_text.strip():
    return xml_text
  lowered_names = tuple(name.casefold() for name in names)
  target_coords = tuple(
      coords for place in places if (coords := _coords(place)) is not None
  )
  try:
    root = ElementTree.fromstring(xml_text)
  except ElementTree.ParseError:
    if any(name in xml_text.casefold() for name in lowered_names):
      return ""
    return xml_text

  removed = False

  def should_remove(element: ElementTree.Element) -> bool:
    local_name = element.tag.rsplit("}", 1)[-1].casefold()
    if local_name not in ("wpt", "placemark"):
      return False
    entry_name = _entry_name(element).casefold()
    if any(name in entry_name for name in lowered_names):
      return True
    entry_coords = _entry_coords(element)
    return (
        entry_coords is not None
        and any(_coords_match(entry_coords, coords) for coords in target_coords)
    )

  def prune(parent: ElementTree.Element) -> None:
    nonlocal removed
    for child in list(parent):
      if should_remove(child):
        parent.remove(child)
        removed = True
      else:
        prune(child)

  prune(root)
  if not removed:
    return xml_text
  return ElementTree.tostring(root, encoding="unicode")


def _minimal_gpx(name: str, place: str) -> str:
  lat, lon = _coords(place) or (0.0, 0.0)
  return (
      '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<gpx version="1.1" creator="CATBench"'
      ' xmlns="http://www.topografix.com/GPX/1/1">\n'
      f'  <wpt lat="{lat:.7f}" lon="{lon:.7f}"><name>{name}</name></wpt>\n'
      "</gpx>\n"
  )


def _minimal_kml(name: str, place: str) -> str:
  lat, lon = _coords(place) or (0.0, 0.0)
  return (
      '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
      "  <Document>\n"
      "    <name>My Places</name>\n"
      "    <visibility>1</visibility>\n"
      "    <Placemark>\n"
      f"      <name>{name}</name>\n"
      "      <Point>\n"
      f"        <coordinates>{lon:.7f},{lat:.7f},0</coordinates>\n"
      "      </Point>\n"
      "    </Placemark>\n"
      "  </Document>\n"
      "</kml>\n"
  )


def _track_points_from_gpx(xml_text: str) -> tuple[tuple[float, float], ...]:
  if not xml_text.strip():
    return ()
  try:
    root = ElementTree.fromstring(xml_text)
  except ElementTree.ParseError:
    return ()
  points: list[tuple[float, float]] = []
  for element in root.iter():
    local_name = element.tag.rsplit("}", 1)[-1].casefold()
    if local_name not in ("trkpt", "rtept"):
      continue
    try:
      points.append((float(element.attrib["lat"]), float(element.attrib["lon"])))
    except (KeyError, ValueError):
      continue
  return tuple(points)


def _coordinates_points(text: str | None) -> tuple[tuple[float, float], ...]:
  if not text:
    return ()
  points: list[tuple[float, float]] = []
  for token in text.split():
    parts = token.split(",")
    if len(parts) < 2:
      continue
    try:
      lon = float(parts[0])
      lat = float(parts[1])
    except ValueError:
      continue
    points.append((lat, lon))
  return tuple(points)


def _track_points_from_kml(xml_text: str) -> tuple[tuple[float, float], ...]:
  if not xml_text.strip():
    return ()
  try:
    root = ElementTree.fromstring(xml_text)
  except ElementTree.ParseError:
    return ()
  points: list[tuple[float, float]] = []
  for element in root.iter():
    local_name = element.tag.rsplit("}", 1)[-1].casefold()
    if local_name == "coordinates":
      points.extend(_coordinates_points(element.text))
    elif local_name == "coord":
      parts = (element.text or "").split()
      if len(parts) < 2:
        continue
      try:
        lon = float(parts[0])
        lat = float(parts[1])
      except ValueError:
        continue
      points.append((lat, lon))
  return tuple(points)


def _track_points_from_xml_file(
    xml_text: str,
    path: str,
) -> tuple[tuple[float, float], ...]:
  lower_path = path.casefold()
  if lower_path.endswith(".kml"):
    return _track_points_from_kml(xml_text)
  if lower_path.endswith(".gpx"):
    return _track_points_from_gpx(xml_text)
  return _track_points_from_gpx(xml_text) or _track_points_from_kml(xml_text)


def _track_matches(
    track_points: tuple[tuple[float, float], ...],
    target_waypoint_coords: tuple[tuple[float, float], ...],
    delta_deg: float = _COORD_TOLERANCE_DEG,
) -> bool:
  if not target_waypoint_coords:
    return False
  target_index = 0
  for track_point in track_points:
    if _coords_match(
        track_point, target_waypoint_coords[target_index], delta_deg=delta_deg
    ):
      target_index += 1
      if target_index == len(target_waypoint_coords):
        return True
  return False


def _waypoint_coords(waypoints: list[str] | tuple[str, ...]) -> tuple[
    tuple[float, float], ...
]:
  coords: list[tuple[float, float]] = []
  for waypoint in waypoints:
    coord = _coords(waypoint)
    if coord is None:
      return ()
    coords.append(coord)
  return tuple(coords)


def _clear_osmand_tracks(env: interface.AsyncEnv) -> None:
  _su_shell(env, f"rm -rf {shlex.quote(_OSMAND_TRACKS_DIR)}/*")


def _track_files_for_package(
    env: interface.AsyncEnv,
    package_name: str,
) -> tuple[str, ...]:
  files: list[str] = []
  seen: set[str] = set()
  for root in _TRACK_STORAGE_PATHS.get(package_name, ()):
    for path in _list_root_files_by_extensions(env, root, ("gpx", "kml")):
      if path in seen:
        continue
      seen.add(path)
      files.append(path)
  return tuple(files)


def _track_file_matches_waypoints(
    env: interface.AsyncEnv,
    track_file: str,
    target_waypoint_coords: tuple[tuple[float, float], ...],
) -> bool:
  return _track_matches(
      _track_points_from_xml_file(_read_root_file(env, track_file), track_file),
      target_waypoint_coords,
  )


def _xml_file_contains_place(
    env: interface.AsyncEnv,
    path: str,
    place: str,
) -> bool:
  xml_text = _read_root_file(env, path)
  # GPX/KML entries always carry coordinates. Requiring the requested
  # coordinates prevents a same-named placemark at a different location from
  # being accepted as the exported target.
  return _xml_entry_exists(xml_text, place=place, require_coords=True)


def _link_file_contains_place(
    env: interface.AsyncEnv,
    path: str,
    place: str,
) -> bool:
  text = _read_root_file(env, path)
  return _link_text_contains_place(text, place)


def _link_text_contains_place(text: str, place: str) -> bool:
  if not text.strip():
    return False
  if any(
      _text_contains_place_candidate(text, candidate)
      for candidate in _place_candidates(place)
  ):
    return True
  coords = _coords(place)
  if coords is None:
    return False
  numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
  parsed_numbers = []
  for number in numbers:
    try:
      parsed_numbers.append(float(number))
    except ValueError:
      continue
  lat, lon = coords
  for i in range(len(parsed_numbers) - 1):
    pair = (parsed_numbers[i], parsed_numbers[i + 1])
    if _coords_match(pair, (lat, lon)):
      return True
    if _coords_match(pair, (lon, lat)):
      return True
  return False


def _clipboard_contains_place_link(env: interface.AsyncEnv, place: str) -> bool:
  try:
    text = adb_utils.get_clipboard_contents(env.controller)
  except Exception as error:
    raise base.VerifierStateReadError(
        "Could not read clipboard state for MapsExportLocation"
    ) from error
  return _link_text_contains_place(text, place)


def _export_files_for_package(
    env: interface.AsyncEnv,
    package_name: str,
) -> tuple[str, ...]:
  files: list[str] = []
  seen: set[str] = set()
  for root in _EXPORT_STORAGE_PATHS.get(package_name, ()):
    for path in _list_root_files_by_extensions(
        env, root, ("gpx", "kml", "kmz", "txt", "url", "html")
    ):
      if path in seen:
        continue
      seen.add(path)
      files.append(path)
  return tuple(files)


def _export_file_matches_place(
    env: interface.AsyncEnv,
    path: str,
    place: str,
) -> bool:
  lower_path = path.casefold()
  if lower_path.endswith((".gpx", ".kml", ".kmz")):
    return _xml_file_contains_place(env, path, place)
  return _link_file_contains_place(env, path, place)


def _export_exists(
    env: interface.AsyncEnv,
    package_name: str,
    place: str,
) -> bool:
  for path in _export_files_for_package(env, package_name):
    if _export_file_matches_place(env, path, place):
      return True
  return False


def _clear_export_files_for_place(
    env: interface.AsyncEnv,
    package_name: str,
    place: str,
) -> None:
  for path in _export_files_for_package(env, package_name):
    if _export_file_matches_place(env, path, place):
      _su_shell(env, f"rm -f {shlex.quote(path)}")


def _track_exists(
    env: interface.AsyncEnv,
    package_name: str,
    waypoints: list[str] | tuple[str, ...],
) -> bool:
  target_waypoint_coords = _waypoint_coords(waypoints)
  if len(target_waypoint_coords) < 2:
    return False
  for track_file in _track_files_for_package(env, package_name):
    if _track_file_matches_waypoints(env, track_file, target_waypoint_coords):
      return True
  return False


def _clear_track_files_for_waypoints(
    env: interface.AsyncEnv,
    package_name: str,
    waypoints: list[str] | tuple[str, ...],
) -> None:
  target_waypoint_coords = _waypoint_coords(waypoints)
  if len(target_waypoint_coords) < 2:
    return
  for track_file in _track_files_for_package(env, package_name):
    if _track_file_matches_waypoints(env, track_file, target_waypoint_coords):
      _su_shell(env, f"rm -f {shlex.quote(track_file)}")


def _sqlite_table_exists(
    env: interface.AsyncEnv, db_path: str, table_name: str
) -> bool:
  out = _sqlite_exec(
      env,
      db_path,
      (
          "SELECT name FROM sqlite_master WHERE type='table' AND name="
          f"{_sql_quote(table_name)};"
      ),
  )
  return table_name in {line.strip() for line in out.splitlines()}


def _wait_for_sqlite_table(
    env: interface.AsyncEnv,
    *,
    package_name: str,
    db_path: str,
    table_name: str,
    timeout_seconds: float = 5.0,
) -> bool:
  if _sqlite_table_exists(env, db_path, table_name):
    return True
  adb_utils.launch_app(package_name, env.controller)
  deadline = time.time() + timeout_seconds
  while time.time() < deadline:
    if _sqlite_table_exists(env, db_path, table_name):
      return True
    time.sleep(0.5)
  return _sqlite_table_exists(env, db_path, table_name)


def _mapsme_seed_bookmark(
    env: interface.AsyncEnv,
    *,
    name: str,
    place: str,
) -> None:
  if not _wait_for_sqlite_table(
      env,
      package_name=_MAPS_ME_PACKAGE_NAME,
      db_path=_MAPS_ME_GUIDES_DB,
      table_name="Bookmark",
  ):
    return
  lat, lon = _coords(place) or (0.0, 0.0)
  bookmark_id = "catbench_" + "".join(
      ch.lower() if ch.isalnum() else "_" for ch in name
  )
  guide_id = "catbench_my_places"
  _sqlite_exec(
      env,
      _MAPS_ME_GUIDES_DB,
      (
          "INSERT OR IGNORE INTO Guide"
          " (id, name, locationName, description, imageUrl,"
          " lastActionTimestamp, persistence)"
          " VALUES"
          f" ({_sql_quote(guide_id)}, 'My Places', 'CATBench', '', '',"
          " 0, 1);"
          "INSERT OR IGNORE INTO GuideVisibility (guideId, isVisible)"
          f" VALUES ({_sql_quote(guide_id)}, 1);"
          "INSERT OR REPLACE INTO Bookmark"
          " (id, customName, latitude, longitude, color, featureTypes,"
          " scale, icon, objectName, description, textureFilename)"
          " VALUES"
          f" ({_sql_quote(bookmark_id)}, {_sql_quote(name)}, {lat}, {lon},"
          " 0, '', 17, '',"
          f" {_sql_quote(place)}, '', '');"
          "INSERT OR REPLACE INTO GuideBookmarkRelations"
          " (guideId, bookmarkId)"
          f" VALUES ({_sql_quote(guide_id)}, {_sql_quote(bookmark_id)});"
      ),
  )


def _mapsme_bookmark_exists(
    env: interface.AsyncEnv,
    name: str | None = None,
    *,
    place: str | None = None,
    require_coords: bool = False,
    match_name_or_coords: bool = False,
) -> bool:
  if name is None and place is None:
    return False
  where = "1=1"
  if name and not match_name_or_coords:
    needle = f"%{name}%"
    where = (
        f"customName LIKE {_sql_quote(needle)}"
        f" OR objectName LIKE {_sql_quote(needle)}"
    )
  out = _sqlite_exec(
      env,
      _MAPS_ME_GUIDES_DB,
      (
          "SELECT COALESCE(customName, '') || char(9) ||"
          " COALESCE(objectName, '') || char(9) ||"
          " COALESCE(latitude, '') || char(9) || COALESCE(longitude, '')"
          f" FROM Bookmark WHERE {where};"
      ),
  )
  name_lower = name.casefold() if name else None
  expected_coords = _coords(place) if place else None
  for line in out.splitlines():
    parts = line.split("\t")
    if len(parts) != 4:
      continue
    entry_text = f"{parts[0]}\t{parts[1]}".casefold()
    name_ok = name_lower is not None and name_lower in entry_text
    try:
      actual_coords = (float(parts[2]), float(parts[3]))
    except ValueError:
      actual_coords = None
    coord_ok = (
        actual_coords is not None
        and expected_coords is not None
        and _coords_match(actual_coords, expected_coords)
    )
    if match_name_or_coords:
      if name_ok or coord_ok:
        return True
      continue
    if name_lower is not None and not name_ok:
      continue
    if name_lower is None and expected_coords is not None and not coord_ok:
      continue
    if require_coords and not coord_ok:
      continue
    if name_lower is not None or expected_coords is not None:
      return True
  return False


def _seed_favorite(env: interface.AsyncEnv, package_name: str, place: str) -> None:
  if package_name == _OSMAND_PACKAGE_NAME:
    _write_root_file(env, _OSMAND_FAVORITES_PATH, _minimal_gpx(place, place))
  elif package_name in _KML_PACKAGES_TO_PATHS:
    _write_root_file(
        env, _KML_PACKAGES_TO_PATHS[package_name], _minimal_kml(place, place)
    )
  elif package_name == _MAPS_ME_PACKAGE_NAME:
    _mapsme_seed_bookmark(env, name=place, place=place)


def _seed_marker(
    env: interface.AsyncEnv,
    package_name: str,
    *,
    place: str,
    label: str,
) -> None:
  if package_name == _OSMAND_PACKAGE_NAME:
    if not _wait_for_sqlite_table(
        env,
        package_name=_OSMAND_PACKAGE_NAME,
        db_path=_OSMAND_MARKERS_DB,
        table_name="map_markers",
    ):
      return
    lat, lon = _coords(place) or (0.0, 0.0)
    _sqlite_exec(
        env,
        _OSMAND_MARKERS_DB,
        (
            "INSERT OR REPLACE INTO map_markers"
            " (marker_id, marker_lat, marker_lon, marker_description,"
            " marker_active, marker_added, marker_visited, group_name,"
            " group_key, marker_color, marker_next_key, marker_disabled,"
            " marker_selected, marker_map_object_name)"
            " VALUES"
            f" ({_sql_quote('catbench_' + label)}, {lat}, {lon},"
            f" {_sql_quote(label)}, 1, 0, 0, '', '', 0, '', 0, 0,"
            f" {_sql_quote(place)});"
        ),
    )
  elif package_name in _KML_PACKAGES_TO_PATHS:
    _write_root_file(
        env, _KML_PACKAGES_TO_PATHS[package_name], _minimal_kml(label, place)
    )
  elif package_name == _MAPS_ME_PACKAGE_NAME:
    _mapsme_seed_bookmark(env, name=label, place=place)


def _clear_xml_entries(
    env: interface.AsyncEnv,
    path: str,
    names: tuple[str, ...] = _CATBENCH_STORAGE_NAMES,
    places: tuple[str, ...] = _PLACES,
) -> None:
  xml_text = _read_root_file(env, path)
  cleaned = _remove_xml_entries(xml_text, names, places)
  if cleaned != xml_text:
    _write_root_file(env, path, cleaned)


def _clear_osmand_catbench_storage(env: interface.AsyncEnv) -> None:
  _clear_xml_entries(env, _OSMAND_FAVORITES_PATH)
  names = _CATBENCH_STORAGE_NAMES
  _sqlite_exec(
      env,
      _OSMAND_MARKERS_DB,
      (
          "DELETE FROM map_markers WHERE marker_id LIKE 'catbench_%'"
          f" OR marker_description IN {_sql_in(names)}"
          f" OR marker_map_object_name IN {_sql_in(names)}"
          " OR "
          f"{_sqlite_coords_filter('marker_lat', 'marker_lon', _PLACES)};"
      ),
  )


def _clear_mapsme_catbench_storage(env: interface.AsyncEnv) -> None:
  if not _sqlite_table_exists(env, _MAPS_ME_GUIDES_DB, "Bookmark"):
    return
  names = _CATBENCH_STORAGE_NAMES
  bookmark_filter = (
      "id LIKE 'catbench_%'"
      f" OR customName IN {_sql_in(names)}"
      f" OR objectName IN {_sql_in(names)}"
      " OR "
      f"{_sqlite_coords_filter('latitude', 'longitude', _PLACES)}"
  )
  if _sqlite_table_exists(
      env, _MAPS_ME_GUIDES_DB, "GuideBookmarkRelations"
  ):
    _sqlite_exec(
        env,
        _MAPS_ME_GUIDES_DB,
        (
            "DELETE FROM GuideBookmarkRelations WHERE bookmarkId IN"
            f" (SELECT id FROM Bookmark WHERE {bookmark_filter});"
        ),
    )
  _sqlite_exec(
      env,
      _MAPS_ME_GUIDES_DB,
      f"DELETE FROM Bookmark WHERE {bookmark_filter};",
  )
  if _sqlite_table_exists(env, _MAPS_ME_GUIDES_DB, "GuideVisibility"):
    _sqlite_exec(
        env,
        _MAPS_ME_GUIDES_DB,
        "DELETE FROM GuideVisibility WHERE guideId = 'catbench_my_places';",
    )
  if _sqlite_table_exists(env, _MAPS_ME_GUIDES_DB, "Guide"):
    _sqlite_exec(
        env,
        _MAPS_ME_GUIDES_DB,
        "DELETE FROM Guide WHERE id = 'catbench_my_places';",
    )


def _clear_catbench_map_storage(
    env: interface.AsyncEnv, package_name: str
) -> None:
  if package_name == _OSMAND_PACKAGE_NAME:
    _clear_osmand_catbench_storage(env)
  elif package_name in _KML_PACKAGES_TO_PATHS:
    _clear_xml_entries(env, _KML_PACKAGES_TO_PATHS[package_name])
  elif package_name == _MAPS_ME_PACKAGE_NAME:
    _clear_mapsme_catbench_storage(env)


def _favorite_exists(env: interface.AsyncEnv, package_name: str, place: str) -> bool:
  if package_name == _OSMAND_PACKAGE_NAME:
    return _xml_entry_exists(
        _read_root_file(env, _OSMAND_FAVORITES_PATH),
        place=place,
        require_coords=True,
    )
  if package_name in _KML_PACKAGES_TO_PATHS:
    return _xml_entry_exists(
        _read_root_file(env, _KML_PACKAGES_TO_PATHS[package_name]),
        place=place,
        require_coords=True,
    )
  if package_name == _MAPS_ME_PACKAGE_NAME:
    return _mapsme_bookmark_exists(
        env, name=place, place=place, match_name_or_coords=True
    )
  return False


def _marker_exists(
    env: interface.AsyncEnv,
    package_name: str,
    *,
    label: str | None = None,
    place: str,
) -> bool:
  if package_name == _OSMAND_PACKAGE_NAME:
    where = "1=1"
    if label:
      needle = f"%{label}%"
      where = (
          f"marker_description LIKE {_sql_quote(needle)}"
          f" OR marker_map_object_name LIKE {_sql_quote(needle)}"
      )
    out = _sqlite_exec(
        env,
        _OSMAND_MARKERS_DB,
        (
            "SELECT marker_lat || char(9) || marker_lon FROM map_markers WHERE"
            f" {where};"
        ),
    )
    expected_coords = _coords(place)
    for line in out.splitlines():
      parts = line.split("\t")
      if len(parts) != 2:
        continue
      try:
        actual_coords = (float(parts[0]), float(parts[1]))
      except ValueError:
        continue
      if expected_coords is not None and _coords_match(actual_coords, expected_coords):
        return True
    return False
  if package_name in _KML_PACKAGES_TO_PATHS:
    return _xml_entry_exists(
        _read_root_file(env, _KML_PACKAGES_TO_PATHS[package_name]),
        name=label,
        place=place,
        require_coords=True,
    )
  if package_name == _MAPS_ME_PACKAGE_NAME:
    return _mapsme_bookmark_exists(
        env, label, place=place, require_coords=True
    )
  return False


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _MapsAppBase(base.PackageAppEval):
  """Maps lifecycle that preserves first-run/offline-map app setup.

  Clearing app data reopens first-run map-download screens in Organic Maps,
  CoMaps, and MAPS.ME. Instead, preserve app-private setup and remove only the
  CATBench-owned favorite/marker artifacts that can affect validation.

  All Maps validators raise ``base._EnvironmentNetworkError`` from
  ``is_successful`` when a known connectivity-error dialog is on screen
  (Google Maps shows ``Something went wrong`` whenever Play Services can't
  reach its backend). The runner converts the exception to
  ``exception_info``, excluding the episode from the success-rate
  denominator (MA5 in the senior review).
  """

  clear_data_on_init = False
  clear_data_on_teardown = False

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _clear_catbench_map_storage(env, self.package_name)
    _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # Defensive: every Maps subclass calls super().is_successful first, so
    # raising here propagates cleanly. Subclasses should still pull
    # ``env.get_state().ui_elements`` themselves; we re-fetch here because
    # ``task_eval.TaskEval.is_successful`` does not.
    base.raise_if_network_error(
        env.get_state().ui_elements, self.package_name
    )
    return super().is_successful(env)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      _clear_catbench_map_storage(env, self.package_name)
      _adb_shell(env, f"am force-stop {shlex.quote(self.package_name)}")
    finally:
      super().tear_down(env)


class _MapsSearchLocationBase(_MapsAppBase):
  """Search for a named place and expose a matching result/detail surface."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {"place": {"type": "string"}},
      "required": ["place"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
      1.0
      if _place_search_result_visible(
          env.get_state().ui_elements, self._params["place"]
      )
      else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsAddFavoriteBase(_MapsAppBase):
  """Add a place to favorites; place name + favorite marker visible."""

  complexity = 1.3
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.package_name in _STORAGE_FAVORITE_PACKAGES:
      return (
          1.0
          if _favorite_exists(env, self.package_name, self._params["place"])
          else 0.0
      )
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    fav_ok = base.element_text_contains(
        ui, ("favorite", "favourite", "saved", "starred", "bookmark")
    )
    return 1.0 if place_ok and fav_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": _random_location_str()[0]}


class _MapsViewPlaceInfoBase(_MapsAppBase):
  """Search for a place and open its detail/info panel."""

  complexity = 1.8
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    detail_ok = base.element_text_contains(
        ui,
        (
            "directions",
            "route",
            "share",
            "save",
            "address",
            "website",
            "phone",
            "reviews",
        ),
    )
    return 1.0 if place_ok and detail_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsRemoveFavoriteBase(_MapsAppBase):
  """Add a favorite for a place, then remove it."""

  complexity = 2.8
  schema = _MapsSearchLocationBase.schema

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    if self.package_name in _STORAGE_FAVORITE_PACKAGES:
      _seed_favorite(env, self.package_name, self._params["place"])
      self._storage_seed_ok = _favorite_exists(
          env, self.package_name, self._params["place"]
      )
      _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.package_name in _STORAGE_FAVORITE_PACKAGES:
      if not getattr(self, "_storage_seed_ok", False):
        return 0.0
      return (
          0.0
          if _favorite_exists(env, self.package_name, self._params["place"])
          else 1.0
      )
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    removed_ok = base.element_text_contains(
        ui,
        (
            "removed",
            "removed from favorites",
            "removed from favourites",
            "unsaved",
            "unstarred",
        ),
    )
    still_saved = base.element_text_contains(
        ui, ("remove favorite", "remove from favorites", "saved")
    )
    return 1.0 if place_ok and removed_ok and not still_saved else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsAddMarkerBase(_MapsAppBase):
  """Drop a marker on the searched place."""

  complexity = 2.0
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.package_name in _STORAGE_MARKER_PACKAGES:
      return (
          1.0
          if _marker_exists(
              env,
              self.package_name,
              place=self._params["place"],
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    marker_ok = base.element_text_contains(
        ui, ("marker", "pin", "waypoint", "pushpin")
    )
    return 1.0 if place_ok and marker_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": _random_location_str()[0]}


class _MapsDownloadOfflineAreaBase(_MapsAppBase):
  """Trigger an offline-map download for a named region."""

  complexity = 2.8
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    offline_ok = base.element_text_contains(
        ui, ("download", "offline", "downloading")
    )
    return 1.0 if place_ok and offline_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsShareLocationBase(_MapsAppBase):
  """Open the share sheet for the requested location.

  Android text/location sharing places the shared text or map link in
  ``Intent.EXTRA_TEXT``.  IntentResolver displays that value in its
  ``content_preview_text`` view.  Success therefore requires the requested
  place name or coordinates in that chooser-owned payload preview.  A generic
  chooser plus the requested place still visible in the source map UI is not
  payload evidence and fails closed.
  """

  complexity = 1.8
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    return (
        1.0
        if _chooser_payload_contains_place(ui, self._params["place"])
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsGetDirectionsBase(_MapsAppBase):
  """Show a computed route between two named places."""

  complexity = 3.2
  schema = {
      "type": "object",
      "properties": {
          "origin": {"type": "string"},
          "destination": {"type": "string"},
      },
      "required": ["origin", "destination"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    origin_ok = _place_visible(ui, self._params["origin"])
    dest_ok = _place_visible(ui, self._params["destination"])
    route_ok = _computed_route_visible(ui)
    return 1.0 if origin_ok and dest_ok and route_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    a, b = _choose_two_places()
    return {"origin": a, "destination": b}


class _MapsMeasureDistanceBase(_MapsAppBase):
  """Measure-distance tool active with at least one distance readout.

  Previously matched bare ``m`` anywhere, which trips on virtually every
  UI element ("m" appears in "menu", "more", "map", ...). We now require a
  numeric readout: a digit followed by a units word as whole-word match
  (MA4 in the senior review).
  """

  complexity = 2.2
  schema = {"type": "object", "properties": {}}

  _READOUT_RE: Final[re.Pattern[str]] = re.compile(
      r"\d[\d.,\s]*\s*(?:km|m|mi|ft|miles?|meters?|kilometers?)\b",
      re.IGNORECASE,
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    tool_ok = base.element_text_contains_word(
        ui, ("measure", "distance", "ruler")
    )
    if not tool_ok:
      return 0.0
    for element in ui:
      for field in (element.text, element.content_description):
        if field and self._READOUT_RE.search(field):
          return 1.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _MapsRecordTrackBase(_MapsAppBase):
  """Save a track whose track points pass through listed waypoints."""

  complexity = 12
  schema = {
      "type": "object",
      "properties": {
          "waypoints": {
              "type": "array",
              "items": {"type": "string"},
          },
          "track_name": {"type": "string"},
          "waypoints_text": {"type": "string"},
      },
      "required": ["waypoints"],
  }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    if self.package_name == _OSMAND_PACKAGE_NAME:
      _clear_osmand_tracks(env)
    _clear_track_files_for_waypoints(
        env, self.package_name, self._params["waypoints"]
    )
    _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _track_exists(env, self.package_name, self._params["waypoints"])
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      if self.package_name == _OSMAND_PACKAGE_NAME:
        _clear_osmand_tracks(env)
      _clear_track_files_for_waypoints(
          env, self.package_name, self._params["waypoints"]
      )
    finally:
      super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    waypoints = random.sample(_PLACES, random.randint(2, 4))
    track_name = (
        f"{waypoints[0].split(',', 1)[0]} to"
        f" {waypoints[-1].split(',', 1)[0]}"
    )
    return {
        "track_name": track_name,
        "waypoints": waypoints,
        "waypoints_text": ", ".join(waypoints),
    }


class _MapsRenameMarkerBase(_MapsAppBase):
  """Rename a marker from one label to a new label.

  For storage-backed apps the old-label marker is seeded in
  ``initialize_task`` and success is read from the app's real storage: the
  new label must exist at the place and the old label must be gone. Without
  seeding, an agent could pass by simply creating a fresh marker with the
  new label. UI-only apps keep the visibility heuristic as a fallback.
  """

  complexity = 2.8
  schema = {
      "type": "object",
      "properties": {
          "place": {"type": "string"},
          "old_label": {"type": "string"},
          "new_label": {"type": "string"},
      },
      "required": ["place", "old_label", "new_label"],
  }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    if self.package_name in _STORAGE_MARKER_PACKAGES:
      _seed_marker(
          env,
          self.package_name,
          place=self._params["place"],
          label=self._params["old_label"],
      )
      self._storage_seed_ok = _marker_exists(
          env,
          self.package_name,
          label=self._params["old_label"],
          place=self._params["place"],
      )
      _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.package_name in _STORAGE_MARKER_PACKAGES:
      if not getattr(self, "_storage_seed_ok", False):
        return 0.0
      new_exists = _marker_exists(
          env,
          self.package_name,
          label=self._params["new_label"],
          place=self._params["place"],
      )
      old_exists = _marker_exists(
          env,
          self.package_name,
          label=self._params["old_label"],
          place=self._params["place"],
      )
      return 1.0 if new_exists and not old_exists else 0.0
    ui = env.get_state().ui_elements
    new_ok = base.element_text_contains(ui, (self._params["new_label"],))
    old_present = base.element_text_contains(ui, (self._params["old_label"],))
    return 1.0 if new_ok and not old_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    old_label, new_label = random.sample(
        ("Meetup", "Picnic", "Viewpoint", "Hotel", "Trail", "Lookout"), 2
    )
    return {
        "place": random.choice(_PLACES),
        "old_label": old_label,
        "new_label": new_label,
    }


class _MapsDeleteMarkerBase(_MapsAppBase):
  """Drop a labelled marker, then delete it.

  Success heuristic: the label is no longer visible anywhere in the app and
  the app shows a deletion/undo confirmation. This avoids passing when the
  marker was never created.
  """

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "place": {"type": "string"},
          "label": {"type": "string"},
      },
      "required": ["place", "label"],
  }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    if self.package_name in _STORAGE_MARKER_PACKAGES:
      _seed_marker(
          env,
          self.package_name,
          place=self._params["place"],
          label=self._params["label"],
      )
      self._storage_seed_ok = _marker_exists(
          env,
          self.package_name,
          label=self._params["label"],
          place=self._params["place"],
      )
      _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.package_name in _STORAGE_MARKER_PACKAGES:
      if not getattr(self, "_storage_seed_ok", False):
        return 0.0
      return (
          0.0
          if _marker_exists(
              env,
              self.package_name,
              label=self._params["label"],
              place=self._params["place"],
          )
          else 1.0
      )
    ui = env.get_state().ui_elements
    label_present = base.element_text_contains(ui, (self._params["label"],))
    deletion_marker = base.element_text_contains(
        ui, ("deleted", "removed", "undo")
    )
    return 1.0 if deletion_marker and not label_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "place": random.choice(_PLACES),
        "label": random.choice(("Meetup", "Picnic", "Viewpoint", "Hotel")),
    }


class _MapsNavigateToPlaceBase(_MapsAppBase):
  """Search a single named destination and start turn-by-turn navigation.

  Success heuristic: place name visible AND a navigation indicator is up
  (``navigate``, ``start``, ``directions``, ``min``, ``km``, ...).
  """

  # Trail Sense is a field-tools app and does not expose turn-by-turn routing.
  excluded_packages = ("com.kylecorry.trail_sense",)
  complexity = 2.6
  schema = _MapsSearchLocationBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    place_ok = _place_visible(ui, self._params["place"])
    nav_ok = base.element_text_contains(
        ui,
        ("navigate", "directions", "route"),
    ) or _ui_text_contains_word(
        ui,
        ("go", "start", "min", "km"),
    )
    return 1.0 if place_ok and nav_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsExportOrShareLocationBase(_MapsAppBase):
  """Export/share a location with exact artifact or chooser-payload evidence."""

  complexity = 2.4
  schema = _MapsSearchLocationBase.schema

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _clear_export_files_for_place(env, self.package_name, self._params["place"])
    # The success predicate accepts a matching map link from the clipboard.
    # Reset it before each episode so a prior run cannot make a no-op pass.
    # ``set_clipboard_contents`` raises if Clipper is unavailable, correctly
    # invalidating an environment where this reset contract cannot be met.
    adb_utils.set_clipboard_contents(
        _EXPORT_CLIPBOARD_SENTINEL, env.controller
    )
    _force_stop_and_launch(self.package_name, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _export_exists(env, self.package_name, self._params["place"]):
      return 1.0
    if _clipboard_contains_place_link(env, self._params["place"]):
      return 1.0
    return (
        1.0
        if _chooser_payload_contains_place(
            env.get_state().ui_elements, self._params["place"]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      _clear_export_files_for_place(env, self.package_name, self._params["place"])
    finally:
      super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"place": random.choice(_PLACES)}


class _MapsSearchNearbyPlaceBase(_MapsAppBase):
  """Show populated results for a nearby category around a named place."""

  complexity = 2.2
  schema = {
      "type": "object",
      "properties": {
          "place": {"type": "string"},
          "category": {"type": "string"},
      },
      "required": ["place", "category"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    category_ok = base.element_text_contains_word(
        ui, (self._params["category"],)
    )
    # Generic words such as "nearby" or "results" do not bind the search to
    # the requested anchor and previously accepted e.g. hotel results around
    # Schaan for a Balzers task. Require explicit accessible anchor evidence;
    # an app that omits it needs a different validated artifact, not a generic
    # UI fallback that can certify the wrong place.
    anchor_ok = _place_visible(ui, self._params["place"])
    results_ok = _populated_nearby_result_visible(
        ui,
        place=self._params["place"],
        category=self._params["category"],
    )
    return 1.0 if category_ok and anchor_ok and results_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "place": random.choice(_PLACES),
        "category": random.choice(("coffee", "restaurant", "hotel", "parking")),
    }


class _MapsShowCurrentLocationBase(_MapsAppBase):
  """Activate the map's current-location / my-location control."""

  complexity = 1.6
  schema = {"type": "object", "properties": {}}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    return (
        1.0
        if base.element_text_contains(
            ui,
            (
                "current location",
                "my location",
                "you are here",
                "gps",
                "centered",
            ),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


# -----------------------------------------------------------------------------
# Per-app packages and generated ports.
# -----------------------------------------------------------------------------

_OSMAND_PACKAGE: Final[str] = _OSMAND_PACKAGE_NAME
_ORGANIC_MAPS_PACKAGE: Final[str] = _ORGANIC_MAPS_PACKAGE_NAME
_GOOGLE_MAPS_PACKAGE: Final[str] = _GOOGLE_MAPS_PACKAGE_NAME
_COMAPS_PACKAGE: Final[str] = _COMAPS_PACKAGE_NAME
_MAPS_ME_PACKAGE: Final[str] = _MAPS_ME_PACKAGE_NAME


_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    _OSMAND_PACKAGE: "OsmAnd",
    _ORGANIC_MAPS_PACKAGE: "Organic Maps",
    _GOOGLE_MAPS_PACKAGE: "Google Maps",
    _COMAPS_PACKAGE: "CoMaps",
    _MAPS_ME_PACKAGE: "MAPS.ME",
}
_APP_GOAL_NAMES: Final[dict[str, str]] = {
    _OSMAND_PACKAGE: "OsmAnd maps",
    _ORGANIC_MAPS_PACKAGE: "Organic Maps",
    _GOOGLE_MAPS_PACKAGE: "Google Maps",
    _COMAPS_PACKAGE: "CoMaps",
    _MAPS_ME_PACKAGE: "MAPS.ME",
}


def _class_suffix(display_name: str) -> str:
  return "".join(ch for ch in display_name if ch.isalnum())


_TEMPLATES: Final[dict[type, str]] = {
    _MapsSearchLocationBase: (
        "In the {app} app, search for the location `{{place}}`."
    ),
    _MapsAddFavoriteBase: (
        "Add a favorite location marker for {{place}} in the {app} app."
    ),
    _MapsViewPlaceInfoBase: (
        "In the {app} app, search for `{{place}}` and open the place"
        " information / details panel."
    ),
    _MapsRemoveFavoriteBase: (
        "In the {app} app, search for `{{place}}`, add it to favorites,"
        " then remove it from favorites."
    ),
    _MapsAddMarkerBase: (
        "Add a location marker for {{place}} in the {app} app."
    ),
    _MapsDownloadOfflineAreaBase: (
        "In the {app} app, download an offline map area that covers"
        " `{{place}}`."
    ),
    _MapsShareLocationBase: (
        "In the {app} app, share the location `{{place}}`."
    ),
    _MapsGetDirectionsBase: (
        "In the {app} app, show directions from `{{origin}}` to"
        " `{{destination}}`."
    ),
    _MapsSearchNearbyPlaceBase: (
        "In the {app} app, search for `{{category}}` near `{{place}}` and"
        " show the nearby search results."
    ),
    _MapsShowCurrentLocationBase: (
        "In the {app} app, tap the current-location / my-location control so"
        " the map centers on the device location."
    ),
    _MapsMeasureDistanceBase: (
        "In the {app} app, activate the measure-distance tool and record a"
        " distance by tapping two points on the map."
    ),
    _MapsRecordTrackBase: (
        "Save a track with waypoints {{waypoints_text}} in the {app}"
        " app in the same order as listed."
    ),
    _MapsRenameMarkerBase: (
        "In the {app} app, drop a marker labelled `{{old_label}}` at"
        " `{{place}}`, then rename that marker to `{{new_label}}`."
    ),
    _MapsDeleteMarkerBase: (
        "In the {app} app, drop a marker labelled `{{label}}` at"
        " `{{place}}` and then delete that marker."
    ),
    _MapsNavigateToPlaceBase: (
        "In the {app} app, search for `{{place}}` and start turn-by-turn"
        " navigation to it."
    ),
    _MapsExportOrShareLocationBase: (
        "In the {app} app, look up `{{place}}` and export or share it as"
        " a GPX / KML file or a map link."
    ),
}


_SEEDED_STORAGE_TEMPLATES: Final[dict[type, str]] = {
    _MapsRemoveFavoriteBase: (
        "In the {app} app, remove the existing favorite for `{{place}}`"
        " from favorites."
    ),
    _MapsDeleteMarkerBase: (
        "In the {app} app, delete the existing marker labelled `{{label}}`"
        " at `{{place}}`."
    ),
}


# Cross-app Maps task templates. The registry keeps these generated task
# classes importable for old artifacts and targeted debugging, but the clean
# C1/C2 matrix is scheduled from benchmark/app_generalization_profiles.py,
# which excludes app/task pairs without stable validators.
_BASE_SHORT_NAMES: Final[dict[type, str]] = {
    _MapsSearchLocationBase: "MapsSearchPlace",
    _MapsAddFavoriteBase: "MapsAddFavorite",
    _MapsRemoveFavoriteBase: "MapsRemoveFavorite",
    _MapsAddMarkerBase: "MapsAddMarker",
    _MapsDeleteMarkerBase: "MapsDeleteMarker",
    _MapsRecordTrackBase: "MapsRecordTrack",
    _MapsGetDirectionsBase: "MapsGetDirections",
    _MapsSearchNearbyPlaceBase: "MapsSearchNearbyPlace",
    _MapsExportOrShareLocationBase: "MapsExportLocation",
    _MapsShareLocationBase: "MapsShareLocation",
}


_PACKAGES = (
    _OSMAND_PACKAGE,
    _ORGANIC_MAPS_PACKAGE,
    _GOOGLE_MAPS_PACKAGE,
    _COMAPS_PACKAGE,
    _MAPS_ME_PACKAGE,
)


for _base_cls, _short in _BASE_SHORT_NAMES.items():
  excluded = getattr(_base_cls, "excluded_packages", ())
  for _pkg in _PACKAGES:
    if _pkg in excluded:
      continue
    _display = _APP_DISPLAY_NAMES[_pkg]
    _suffix = _class_suffix(_display)
    _cls_name = f"{_short}For{_suffix}"
    _template = _TEMPLATES[_base_cls]
    if (
        _pkg in _STORAGE_FAVORITE_PACKAGES
        and _base_cls == _MapsRemoveFavoriteBase
    ) or (
        _pkg in _STORAGE_MARKER_PACKAGES
        and _base_cls == _MapsDeleteMarkerBase
    ):
      _template = _SEEDED_STORAGE_TEMPLATES[_base_cls]
    _validation_mode = "UI heuristic"
    if _base_cls in (_MapsAddFavoriteBase, _MapsRemoveFavoriteBase):
      if _pkg in _STORAGE_FAVORITE_PACKAGES:
        _validation_mode = "Filesystem/SQLite"
      elif _pkg == _GOOGLE_MAPS_PACKAGE:
        _validation_mode = "UI heuristic (Google Maps opaque storage)"
    elif _base_cls in (_MapsAddMarkerBase, _MapsDeleteMarkerBase):
      if _pkg in _STORAGE_MARKER_PACKAGES:
        _validation_mode = "Filesystem/SQLite"
      elif _pkg == _GOOGLE_MAPS_PACKAGE:
        _validation_mode = "UI heuristic (Google Maps opaque storage)"
    elif _base_cls == _MapsRecordTrackBase:
      _validation_mode = "Filesystem GPX/KML"
    elif _base_cls == _MapsExportOrShareLocationBase:
      _validation_mode = "Filesystem GPX/KML/link"
    _attrs = {
        "app_names": (_pkg,),
        "package_name": _pkg,
        "catbench_semantic_id": _short,
        "catbench_app_display_name": _APP_GOAL_NAMES[_pkg],
        "template": _template.format(app=_APP_GOAL_NAMES[_pkg]),
        "validation_mode": _validation_mode,
    }
    globals()[_cls_name] = type(_cls_name, (_base_cls,), _attrs)


del _base_cls, _short, _pkg, _display, _suffix, _cls_name, _template, _validation_mode, _attrs
