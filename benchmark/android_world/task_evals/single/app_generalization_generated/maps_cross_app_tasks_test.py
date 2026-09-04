from unittest import mock

from absl.testing import absltest

from android_world.env import representation_utils
from android_world.task_evals.single.app_generalization_generated import (
    maps_cross_app_tasks as maps_tasks,
)
from app_generalization_profiles import get_domain_profiles


def _element(
    text: str | None = None,
    content_description: str | None = None,
    *,
    class_name: str | None = None,
    is_clickable: bool | None = None,
    is_editable: bool | None = None,
    package_name: str | None = None,
    resource_id: str | None = None,
) -> representation_utils.UIElement:
  return representation_utils.UIElement(
      text=text,
      content_description=content_description,
      class_name=class_name,
      is_clickable=is_clickable,
      is_editable=is_editable,
      package_name=package_name,
      resource_id=resource_id,
  )


class _FakeState:

  def __init__(self, ui_elements):
    self.ui_elements = ui_elements


class _FakeEnv:

  def __init__(self, ui_elements):
    self._state = _FakeState(ui_elements)

  def get_state(self):
    return self._state


def _score(task_cls, params, ui_elements):
  task = task_cls(params)
  task.initialized = True
  return task.is_successful(_FakeEnv(ui_elements))


class MapsCrossAppStorageHelpersTest(absltest.TestCase):

  def test_checked_storage_read_distinguishes_empty_state_from_failure(self):
    marker = maps_tasks._SHELL_STATUS_MARKER
    with mock.patch.object(
        maps_tasks,
        "_adb_shell",
        return_value=f"\n{marker}0\n",
    ):
      self.assertEqual(
          maps_tasks._su_shell_checked(
              mock.Mock(), "true", operation="read test state"
          ),
          "",
      )

    with mock.patch.object(
        maps_tasks,
        "_adb_shell",
        return_value=f"permission denied\n{marker}1\n",
    ), self.assertRaises(maps_tasks.base.VerifierStateReadError):
      maps_tasks._su_shell_checked(
          mock.Mock(), "false", operation="read test state"
      )

  def test_checked_storage_read_rejects_missing_status_marker(self):
    with mock.patch.object(maps_tasks, "_adb_shell", return_value=""):
      with self.assertRaises(maps_tasks.base.VerifierStateReadError):
        maps_tasks._su_shell_checked(
            mock.Mock(), "true", operation="read test state"
        )

  def test_generated_places_stay_on_offline_liechtenstein_map(self):
    self.assertTrue(maps_tasks._PLACES)
    self.assertTrue(
        all(place.endswith(", Liechtenstein") for place in maps_tasks._PLACES)
    )
    self.assertSameElements(maps_tasks._PLACES, maps_tasks._PLACE_COORDS)

  def test_kml_marker_requires_matching_label_and_coordinates(self):
    kml = maps_tasks._minimal_kml("Meetup", "Triesen, Liechtenstein")

    self.assertTrue(
        maps_tasks._xml_entry_exists(
            kml,
            name="Meetup",
            place="Triesen, Liechtenstein",
            require_coords=True,
        )
    )
    self.assertFalse(
        maps_tasks._xml_entry_exists(
            kml,
            name="Meetup",
            place="Balzers, Liechtenstein",
            require_coords=True,
        )
    )

  def test_aw_favorite_style_accepts_matching_coordinates_without_name(self):
    gpx = maps_tasks._minimal_gpx("Dropped pin", "Triesen, Liechtenstein")

    self.assertTrue(
        maps_tasks._xml_entry_exists(
            gpx,
            name="Triesen, Liechtenstein",
            place="Triesen, Liechtenstein",
            match_name_or_coords=True,
        )
    )
    self.assertFalse(
        maps_tasks._xml_entry_exists(
            gpx,
            name="Triesen, Liechtenstein",
            place="Balzers, Liechtenstein",
            match_name_or_coords=True,
        )
    )

  def test_favorite_rejects_expected_name_at_wrong_coordinates(self):
    wrong = maps_tasks._minimal_gpx(
        "Balzers, Liechtenstein", "Triesen, Liechtenstein"
    )
    with mock.patch.object(maps_tasks, "_read_root_file", return_value=wrong):
      self.assertFalse(
          maps_tasks._favorite_exists(
              mock.Mock(),
              maps_tasks._OSMAND_PACKAGE_NAME,
              "Balzers, Liechtenstein",
          )
      )

  def test_xml_export_rejects_expected_name_at_wrong_coordinates(self):
    wrong = maps_tasks._minimal_kml(
        "Balzers, Liechtenstein", "Triesen, Liechtenstein"
    )
    with mock.patch.object(maps_tasks, "_read_root_file", return_value=wrong):
      self.assertFalse(
          maps_tasks._xml_file_contains_place(
              mock.Mock(), "/tmp/export.kml", "Balzers, Liechtenstein"
          )
      )

  def test_aw_marker_style_accepts_matching_coordinates_without_label(self):
    kml = maps_tasks._minimal_kml("Any label", "Triesen, Liechtenstein")

    self.assertTrue(
        maps_tasks._xml_entry_exists(
            kml,
            place="Triesen, Liechtenstein",
            require_coords=True,
        )
    )
    self.assertFalse(
        maps_tasks._xml_entry_exists(
            kml,
            place="Balzers, Liechtenstein",
            require_coords=True,
      )
    )

  def test_coordinate_string_params_resolve_like_aw_osmand(self):
    self.assertEqual(
        maps_tasks._coords("47.1303814, 9.5930117"),
        maps_tasks._PLACE_COORDS["Schönberg, Liechtenstein"],
    )

  def test_add_favorite_generator_keeps_aw_name_or_coordinate_style(self):
    with mock.patch.object(
        maps_tasks.random,
        "sample",
        return_value=["Balzers, Liechtenstein"],
    ), mock.patch.object(maps_tasks.random, "getrandbits", return_value=0):
      params = maps_tasks.MapsAddFavoriteForOsmAnd.generate_random_params()

    self.assertEqual(params["place"], "47.0688832, 9.5061564")

  def test_remove_xml_entries_removes_catbench_entry_only(self):
    kml = """
<kml>
  <Document>
    <Placemark><name>Meetup</name></Placemark>
    <Placemark><name>Personal Place</name></Placemark>
  </Document>
</kml>
"""

    cleaned = maps_tasks._remove_xml_entries(
        kml, maps_tasks._CATBENCH_STORAGE_NAMES
    )

    self.assertNotIn("Meetup", cleaned)
    self.assertIn("Personal Place", cleaned)

  def test_remove_xml_entries_removes_catbench_coordinate_leftovers(self):
    kml = maps_tasks._minimal_kml("Dropped pin", "Triesen, Liechtenstein")

    cleaned = maps_tasks._remove_xml_entries(kml, (), maps_tasks._PLACES)

    self.assertNotIn("Dropped pin", cleaned)

  def test_generated_maps_tasks_preserve_app_data(self):
    self.assertFalse(
        maps_tasks.MapsSearchPlaceForOrganicMaps.clear_data_on_init
    )
    self.assertFalse(
        maps_tasks.MapsSearchPlaceForOrganicMaps.clear_data_on_teardown
    )

  def test_profiles_exclude_unstable_google_maps_and_mapsme_tasks(self):
    profile = get_domain_profiles()["maps"]
    apps = {app.app_id: app for app in profile.apps}

    # Google Maps and MAPS.ME are fully descheduled: their saved-place state
    # is opaque (synced or binary My Places.kmb), so they cannot match
    # OsmAnd's storage-validated task depth. Keeping partially-covered apps
    # would make the maps matrix inconsistent with the paper's 10-task grid.
    self.assertEqual(apps["maps_google_maps"].implemented_tasks, ())
    self.assertTrue(apps["maps_google_maps"].optional)
    self.assertEqual(apps["maps_maps_me"].implemented_tasks, ())
    self.assertTrue(apps["maps_maps_me"].optional)

    scheduled = {
        app_id for app_id, app in apps.items() if app.implemented_tasks
    }
    self.assertEqual(
        scheduled, {"maps_osmand", "maps_organic_maps", "maps_comaps"}
    )
    for app_id in scheduled:
      self.assertLen(apps[app_id].implemented_tasks, 10)

  def test_mapsme_not_in_storage_validated_package_sets(self):
    self.assertNotIn(
        maps_tasks._MAPS_ME_PACKAGE_NAME,
        maps_tasks._STORAGE_FAVORITE_PACKAGES,
    )
    self.assertNotIn(
        maps_tasks._MAPS_ME_PACKAGE_NAME,
        maps_tasks._STORAGE_MARKER_PACKAGES,
    )

  def test_osmand_marker_requires_matching_coordinates(self):
    output = "47.106997\t9.5274854\n"
    with mock.patch.object(maps_tasks, "_sqlite_exec", return_value=output):
      self.assertTrue(
          maps_tasks._marker_exists(
              mock.Mock(),
              maps_tasks._OSMAND_PACKAGE_NAME,
              place="Triesen, Liechtenstein",
          )
      )
      self.assertTrue(
          maps_tasks._marker_exists(
              mock.Mock(),
              maps_tasks._OSMAND_PACKAGE_NAME,
              label="Meetup",
              place="Triesen, Liechtenstein",
          )
      )
      self.assertFalse(
          maps_tasks._marker_exists(
              mock.Mock(),
              maps_tasks._OSMAND_PACKAGE_NAME,
              label="Meetup",
              place="Balzers, Liechtenstein",
          )
      )

  def test_mapsme_marker_requires_matching_coordinates(self):
    output = "Meetup\tTriesen, Liechtenstein\t47.106997\t9.5274854\n"
    with mock.patch.object(maps_tasks, "_sqlite_exec", return_value=output):
      self.assertTrue(
          maps_tasks._mapsme_bookmark_exists(
              mock.Mock(),
              place="Triesen, Liechtenstein",
              require_coords=True,
          )
      )
      self.assertTrue(
          maps_tasks._mapsme_bookmark_exists(
              mock.Mock(),
              "Meetup",
              place="Triesen, Liechtenstein",
              require_coords=True,
          )
      )
      self.assertFalse(
          maps_tasks._mapsme_bookmark_exists(
              mock.Mock(),
              "Meetup",
              place="Balzers, Liechtenstein",
              require_coords=True,
          )
      )

  def test_short_route_token_does_not_match_google_substring(self):
    ui = [_element(text="Google Maps")]

    self.assertFalse(maps_tasks._ui_text_contains_word(ui, ("go",)))

  def test_place_visible_accepts_city_without_country(self):
    ui = [_element(text="Balzers")]

    self.assertTrue(
        maps_tasks._place_visible(ui, "Balzers, Liechtenstein")
    )

  def test_place_visible_rejects_different_city(self):
    ui = [_element(text="Schaan")]

    self.assertFalse(
        maps_tasks._place_visible(ui, "Balzers, Liechtenstein")
    )

  def test_place_visible_rejects_prefix_city_collisions(self):
    self.assertFalse(
        maps_tasks._place_visible(
            [_element(text="Schaanwald")], "Schaan, Liechtenstein"
        )
    )
    self.assertFalse(
        maps_tasks._place_visible(
            [_element(text="Oberplanken")], "Planken, Liechtenstein"
        )
    )

  def test_link_coordinates_reject_nearby_wrong_place(self):
    # Planken and Oberplanken are only ~0.007 degrees apart. The old 0.01
    # tolerance accepted an Oberplanken URL as a Planken export.
    lat, lon = maps_tasks._PLACE_COORDS["Oberplanken, Liechtenstein"]
    link = f"https://example.test/map?lat={lat}&lon={lon}"

    self.assertTrue(
        maps_tasks._link_text_contains_place(
            link, "Oberplanken, Liechtenstein"
        )
    )
    self.assertFalse(
        maps_tasks._link_text_contains_place(link, "Planken, Liechtenstein")
    )

  def test_search_place_requires_noneditable_result_evidence(self):
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchPlaceForOsmAnd,
            {"place": "Balzers, Liechtenstein"},
            [
                _element(
                    text="Balzers",
                    is_clickable=True,
                    resource_id="net.osmand.plus:id/search_result",
                )
            ],
        ),
        1.0,
    )
    # Merely typing the target leaves it visible but does not execute/select
    # a search result.
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchPlaceForOsmAnd,
            {"place": "Balzers, Liechtenstein"},
            [
                _element(
                    text="Balzers, Liechtenstein",
                    class_name="android.widget.EditText",
                    is_editable=True,
                )
            ],
        ),
        0.0,
    )
    # A plain, non-editable echo is also insufficient without result/detail
    # semantics.
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchPlaceForOsmAnd,
            {"place": "Balzers, Liechtenstein"},
            [_element(text="Balzers")],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchPlaceForOsmAnd,
            {"place": "Balzers, Liechtenstein"},
            [_element(text="Schaan")],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchPlaceForOsmAnd,
            {"place": "Balzers, Liechtenstein"},
            [],
        ),
        0.0,
    )

  def test_storage_favorite_tasks_positive_noop_wrong_and_unseeded(self):
    params = {"place": "Balzers, Liechtenstein"}
    ui = [_element(text="Search maps")]

    with mock.patch.object(maps_tasks, "_favorite_exists", return_value=True):
      self.assertEqual(
          _score(maps_tasks.MapsAddFavoriteForOsmAnd, params, ui), 1.0
      )
    with mock.patch.object(maps_tasks, "_favorite_exists", return_value=False):
      self.assertEqual(
          _score(maps_tasks.MapsAddFavoriteForOsmAnd, params, ui), 0.0
      )

    task = maps_tasks.MapsRemoveFavoriteForOsmAnd(params)
    task.initialized = True
    task._storage_seed_ok = True
    with mock.patch.object(maps_tasks, "_favorite_exists", return_value=False):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 1.0)
    with mock.patch.object(maps_tasks, "_favorite_exists", return_value=True):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 0.0)
    task._storage_seed_ok = False
    with mock.patch.object(maps_tasks, "_favorite_exists", return_value=False):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 0.0)

  def test_storage_marker_tasks_positive_noop_wrong_and_unseeded(self):
    add_params = {"place": "Balzers, Liechtenstein"}
    ui = [_element(text="Search maps")]
    with mock.patch.object(maps_tasks, "_marker_exists", return_value=True):
      self.assertEqual(
          _score(maps_tasks.MapsAddMarkerForOsmAnd, add_params, ui), 1.0
      )
    with mock.patch.object(maps_tasks, "_marker_exists", return_value=False):
      self.assertEqual(
          _score(maps_tasks.MapsAddMarkerForOsmAnd, add_params, ui), 0.0
      )

    delete_params = {"place": "Balzers, Liechtenstein", "label": "Meetup"}
    task = maps_tasks.MapsDeleteMarkerForOsmAnd(delete_params)
    task.initialized = True
    task._storage_seed_ok = True
    with mock.patch.object(maps_tasks, "_marker_exists", return_value=False):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 1.0)
    with mock.patch.object(maps_tasks, "_marker_exists", return_value=True):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 0.0)
    task._storage_seed_ok = False
    with mock.patch.object(maps_tasks, "_marker_exists", return_value=False):
      self.assertEqual(task.is_successful(_FakeEnv(ui)), 0.0)

  def test_get_directions_ui_requires_places_and_route_evidence(self):
    params = {
        "origin": "Balzers, Liechtenstein",
        "destination": "Triesen, Liechtenstein",
    }

    self.assertEqual(
        _score(
            maps_tasks.MapsGetDirectionsForOsmAnd,
            params,
            [
                _element(text="Balzers"),
                _element(text="Triesen"),
                _element(text="Route 7 km"),
            ],
        ),
        1.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsGetDirectionsForOsmAnd,
            params,
            [_element(text="Balzers"), _element(text="Route 7 km")],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsGetDirectionsForOsmAnd,
            params,
            [_element(text="Balzers"), _element(text="Triesen")],
        ),
        0.0,
    )
    # Filled endpoint inputs plus a route control precede route computation.
    self.assertEqual(
        _score(
            maps_tasks.MapsGetDirectionsForOsmAnd,
            params,
            [
                _element(text="Balzers", is_editable=True),
                _element(text="Triesen", is_editable=True),
                _element(text="Directions"),
                _element(text="Start"),
            ],
        ),
        0.0,
    )
    # A bare distance can be the map scale and is not computed-route proof.
    self.assertEqual(
        _score(
            maps_tasks.MapsGetDirectionsForOsmAnd,
            params,
            [
                _element(text="Balzers"),
                _element(text="Triesen"),
                _element(text="200 m"),
            ],
        ),
        0.0,
    )

  def test_search_nearby_requires_populated_result_evidence(self):
    params = {"place": "Balzers, Liechtenstein", "category": "hotel"}

    self.assertEqual(
        _score(
            maps_tasks.MapsSearchNearbyPlaceForOsmAnd,
            params,
            [
                _element(text="Hotel results near Balzers"),
                _element(
                    text="Hotel Gutenberg",
                    is_clickable=True,
                    resource_id="net.osmand.plus:id/search_result_item",
                ),
            ],
        ),
        1.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchNearbyPlaceForOsmAnd,
            params,
            [
                _element(text="Restaurants near Balzers"),
                _element(
                    text="Restaurant Adler",
                    is_clickable=True,
                    resource_id="net.osmand.plus:id/search_result_item",
                ),
            ],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchNearbyPlaceForOsmAnd,
            params,
            [
                _element(text="Hotel results near Schaan"),
                _element(
                    text="Hotel Linde",
                    is_clickable=True,
                    resource_id="net.osmand.plus:id/search_result_item",
                ),
            ],
        ),
        0.0,
    )
    # Category and anchor search controls are not a populated result surface.
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchNearbyPlaceForOsmAnd,
            params,
            [
                _element(text="hotel", is_editable=True),
                _element(text="Balzers", is_editable=True),
                _element(text="Search"),
            ],
        ),
        0.0,
    )
    # An explicit empty-result row must not count as a concrete result item.
    self.assertEqual(
        _score(
            maps_tasks.MapsSearchNearbyPlaceForOsmAnd,
            params,
            [
                _element(text="Hotel results near Balzers"),
                _element(
                    text="No results found",
                    is_clickable=True,
                    resource_id="net.osmand.plus:id/search_result_item",
                ),
            ],
        ),
        0.0,
    )

  def test_share_location_requires_exact_chooser_payload(self):
    params = {"place": "Balzers, Liechtenstein"}
    chooser_package = "com.android.intentresolver"
    preview_id = "com.android.intentresolver:id/content_preview_text"

    # Source-app place/share text is not system chooser payload evidence.
    self.assertEqual(
        _score(
            maps_tasks.MapsShareLocationForOsmAnd,
            params,
            [_element(text="Balzers"), _element(text="Share")],
        ),
        0.0,
    )
    # A chooser target alone does not say which location is being shared.
    self.assertEqual(
        _score(
            maps_tasks.MapsShareLocationForOsmAnd,
            params,
            [
                _element(text="Messages", package_name=chooser_package),
                _element(text="Balzers"),
            ],
        ),
        0.0,
    )
    # EXTRA_TITLE is optional metadata, not the EXTRA_TEXT share payload.
    self.assertEqual(
        _score(
            maps_tasks.MapsShareLocationForOsmAnd,
            params,
            [
                _element(
                    text="Balzers, Liechtenstein",
                    package_name=chooser_package,
                    resource_id=(
                        "com.android.intentresolver:id/content_preview_title"
                    ),
                ),
            ],
        ),
        0.0,
    )
    # The expected place in the source app cannot rescue a wrong-place link.
    triesen_lat, triesen_lon = maps_tasks._PLACE_COORDS[
        "Triesen, Liechtenstein"
    ]
    self.assertEqual(
        _score(
            maps_tasks.MapsShareLocationForOsmAnd,
            params,
            [
                _element(text="Balzers, Liechtenstein"),
                _element(
                    text=(
                        "https://www.openstreetmap.org/"
                        f"?mlat={triesen_lat}&mlon={triesen_lon}"
                    ),
                    package_name=chooser_package,
                    resource_id=preview_id,
                ),
            ],
        ),
        0.0,
    )
    balzers_lat, balzers_lon = maps_tasks._PLACE_COORDS[
        "Balzers, Liechtenstein"
    ]
    self.assertEqual(
        _score(
            maps_tasks.MapsShareLocationForOsmAnd,
            params,
            [
                _element(
                    text=(
                        "https://www.openstreetmap.org/"
                        f"?mlat={balzers_lat}&mlon={balzers_lon}"
                    ),
                    package_name=chooser_package,
                    resource_id=preview_id,
                ),
            ],
        ),
        1.0,
    )

  def test_share_location_network_error_raises(self):
    """Google Maps "Something went wrong" must raise to exclude episode
    from the SR denominator (MA5 fix)."""
    params = {"place": "Balzers, Liechtenstein"}
    with self.assertRaises(Exception):
      _score(
          maps_tasks.MapsShareLocationForGoogleMaps,
          params,
          [
              _element(text="Something went wrong"),
              _element(text="Check your connection and try again"),
          ],
      )

  def test_export_location_requires_artifact_or_exact_result_payload(self):
    params = {"place": "Balzers, Liechtenstein"}

    with mock.patch.object(maps_tasks, "_export_exists", return_value=True):
      self.assertEqual(
          _score(
              maps_tasks.MapsExportLocationForOsmAnd,
              params,
              [_element(text="Search maps")],
          ),
          1.0,
      )
    with mock.patch.object(
        maps_tasks, "_export_exists", return_value=False
    ), mock.patch.object(
        maps_tasks, "_clipboard_contains_place_link", return_value=False
    ):
      self.assertEqual(
          _score(
              maps_tasks.MapsExportLocationForOsmAnd,
              params,
              [_element(text="Search maps")],
          ),
          0.0,
      )
      # Place + export-format/options UI is an intermediate screen, not an
      # exported artifact.
      self.assertEqual(
          _score(
              maps_tasks.MapsExportLocationForOsmAnd,
              params,
              [
                  _element(text="Balzers"),
                  _element(text="Export GPX"),
                  _element(text="Export KML"),
                  _element(text="Copy to clipboard"),
              ],
          ),
          0.0,
      )

      balzers_lat, balzers_lon = maps_tasks._PLACE_COORDS[
          "Balzers, Liechtenstein"
      ]
      self.assertEqual(
          _score(
              maps_tasks.MapsExportLocationForOsmAnd,
              params,
              [
                  _element(
                      text=(
                          "https://www.openstreetmap.org/"
                          f"?mlat={balzers_lat}&mlon={balzers_lon}"
                      ),
                      package_name="com.android.intentresolver",
                      resource_id=(
                          "com.android.intentresolver:id/content_preview_text"
                      ),
                  )
              ],
          ),
          1.0,
      )

  def test_export_location_resets_clipboard_before_episode(self):
    task = maps_tasks.MapsExportLocationForOsmAnd(
        {"place": "Balzers, Liechtenstein"}
    )
    env = mock.Mock()

    with mock.patch.object(
        maps_tasks._MapsAppBase, "initialize_task"
    ) as parent_initialize, mock.patch.object(
        maps_tasks, "_clear_export_files_for_place"
    ) as clear_exports, mock.patch.object(
        maps_tasks.adb_utils, "set_clipboard_contents"
    ) as set_clipboard, mock.patch.object(
        maps_tasks, "_force_stop_and_launch"
    ) as relaunch:
      task.initialize_task(env)

    parent_initialize.assert_called_once_with(env)
    clear_exports.assert_called_once_with(
        env,
        maps_tasks._OSMAND_PACKAGE_NAME,
        "Balzers, Liechtenstein",
    )
    set_clipboard.assert_called_once_with(
        maps_tasks._EXPORT_CLIPBOARD_SENTINEL, env.controller
    )
    relaunch.assert_called_once_with(maps_tasks._OSMAND_PACKAGE_NAME, env)

  def test_export_clipboard_sentinel_is_not_matching_link(self):
    with mock.patch.object(
        maps_tasks.adb_utils,
        "get_clipboard_contents",
        return_value=maps_tasks._EXPORT_CLIPBOARD_SENTINEL,
    ):
      self.assertFalse(
          maps_tasks._clipboard_contains_place_link(
              mock.Mock(), "Balzers, Liechtenstein"
          )
      )

  def test_export_clipboard_read_error_is_not_semantic_failure(self):
    with mock.patch.object(
        maps_tasks.adb_utils,
        "get_clipboard_contents",
        side_effect=RuntimeError("clipper unavailable"),
    ), self.assertRaises(maps_tasks.base.VerifierStateReadError):
      maps_tasks._clipboard_contains_place_link(
          mock.Mock(), "Balzers, Liechtenstein"
      )

  def test_export_location_accepts_matching_clipboard_link(self):
    params = {"place": "Balzers, Liechtenstein"}

    with mock.patch.object(maps_tasks, "_export_exists", return_value=False):
      with mock.patch.object(
          maps_tasks,
          "_clipboard_contains_place_link",
          return_value=True,
      ):
        self.assertEqual(
            _score(
                maps_tasks.MapsExportLocationForOsmAnd,
                params,
                [_element(text="Search maps")],
            ),
            1.0,
        )

  def test_add_marker_goal_matches_aw_coordinate_marker_style(self):
    params = maps_tasks.MapsAddMarkerForOsmAnd.generate_random_params()
    task = maps_tasks.MapsAddMarkerForOsmAnd(params)

    self.assertIn("place", params)
    self.assertNotIn("label", params)
    self.assertIn("Add a location marker", task.goal)
    self.assertNotIn("labelled", task.goal)

  def test_record_track_goal_matches_aw_waypoint_style(self):
    params = maps_tasks.MapsRecordTrackForOsmAnd.generate_random_params()
    task = maps_tasks.MapsRecordTrackForOsmAnd(params)

    self.assertGreaterEqual(len(params["waypoints"]), 2)
    self.assertLessEqual(len(params["waypoints"]), 4)
    self.assertIn("Save a track with waypoints", task.goal)
    self.assertIn("same order as listed", task.goal)

  def test_record_track_task_positive_noop_and_wrong_order(self):
    params = {
        "waypoints": ["Balzers, Liechtenstein", "Triesen, Liechtenstein"],
        "track_name": "Balzers to Triesen",
        "waypoints_text": "Balzers, Liechtenstein, Triesen, Liechtenstein",
    }
    task_cls = maps_tasks.MapsRecordTrackForOsmAnd
    with mock.patch.object(maps_tasks, "_track_exists", return_value=True):
      self.assertEqual(_score(task_cls, params, []), 1.0)
    with mock.patch.object(maps_tasks, "_track_exists", return_value=False):
      self.assertEqual(_score(task_cls, params, []), 0.0)

  def test_track_matches_waypoints_in_order(self):
    balzers = maps_tasks._PLACE_COORDS["Balzers, Liechtenstein"]
    triesen = maps_tasks._PLACE_COORDS["Triesen, Liechtenstein"]
    gpx = f"""
<gpx xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="{balzers[0]}" lon="{balzers[1]}" />
    <trkpt lat="{triesen[0]}" lon="{triesen[1]}" />
  </trkseg></trk>
</gpx>
"""
    points = maps_tasks._track_points_from_gpx(gpx)

    self.assertTrue(
        maps_tasks._track_matches(
            points,
            maps_tasks._waypoint_coords(
                ("Balzers, Liechtenstein", "Triesen, Liechtenstein")
            ),
        )
    )
    self.assertFalse(
        maps_tasks._track_matches(
            points,
            maps_tasks._waypoint_coords(
                ("Triesen, Liechtenstein", "Balzers, Liechtenstein")
            ),
        )
    )

  def test_track_exists_accepts_kml_linestring_for_new_map_apps(self):
    balzers = maps_tasks._PLACE_COORDS["Balzers, Liechtenstein"]
    triesen = maps_tasks._PLACE_COORDS["Triesen, Liechtenstein"]
    kml = f"""
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <LineString>
        <coordinates>
          {balzers[1]},{balzers[0]},0 {triesen[1]},{triesen[0]},0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    with mock.patch.object(
        maps_tasks, "_track_files_for_package", return_value=("/tmp/track.kml",)
    ), mock.patch.object(maps_tasks, "_read_root_file", return_value=kml):
      self.assertTrue(
          maps_tasks._track_exists(
              mock.Mock(),
              maps_tasks._COMAPS_PACKAGE_NAME,
              ("Balzers, Liechtenstein", "Triesen, Liechtenstein"),
          )
      )
      self.assertFalse(
          maps_tasks._track_exists(
              mock.Mock(),
              maps_tasks._COMAPS_PACKAGE_NAME,
              ("Triesen, Liechtenstein", "Balzers, Liechtenstein"),
          )
      )

  def test_export_exists_accepts_gpx_kml_and_link_files(self):
    gpx = maps_tasks._minimal_gpx("Balzers, Liechtenstein", "Balzers, Liechtenstein")
    link = "https://www.openstreetmap.org/?mlat=47.0688832&mlon=9.5061564"

    with mock.patch.object(
        maps_tasks,
        "_export_files_for_package",
        return_value=("/tmp/export.gpx", "/tmp/export.txt"),
    ), mock.patch.object(
        maps_tasks, "_read_root_file", side_effect=(gpx, link)
    ):
      self.assertTrue(
          maps_tasks._export_exists(
              mock.Mock(),
              maps_tasks._ORGANIC_MAPS_PACKAGE_NAME,
              "Balzers, Liechtenstein",
          )
      )
    with mock.patch.object(
        maps_tasks,
        "_export_files_for_package",
        return_value=("/tmp/export.txt",),
    ), mock.patch.object(maps_tasks, "_read_root_file", return_value=link):
      self.assertFalse(
          maps_tasks._export_exists(
              mock.Mock(),
              maps_tasks._ORGANIC_MAPS_PACKAGE_NAME,
              "Triesen, Liechtenstein",
          )
      )

  def test_record_track_uses_filesystem_validation_for_all_map_apps(self):
    self.assertEqual(
        maps_tasks.MapsRecordTrackForOsmAnd.validation_mode,
        "Filesystem GPX/KML",
    )
    self.assertEqual(
        maps_tasks.MapsRecordTrackForGoogleMaps.validation_mode,
        "Filesystem GPX/KML",
    )
    self.assertEqual(
        maps_tasks.MapsRecordTrackForOrganicMaps.validation_mode,
        "Filesystem GPX/KML",
    )

  def test_export_location_replaces_current_location_in_active_maps_set(self):
    self.assertFalse(hasattr(maps_tasks, "MapsShowCurrentLocationForOsmAnd"))
    self.assertTrue(hasattr(maps_tasks, "MapsExportLocationForOsmAnd"))
    self.assertEqual(
        maps_tasks.MapsExportLocationForOsmAnd.validation_mode,
        "Filesystem GPX/KML/link",
    )


if __name__ == "__main__":
  absltest.main()
