# Validator strategy per cross-app category

This is a per-category audit of which `app_generalization_generated/*_cross_app_tasks.py`
modules follow `tasks_guide.md` (SQLite / file-system validators) and which
deliberately remain on UI-text heuristics.

The user's correction was: "you do not follow the guide ... why you break
the rules? If there are some special cases that you cannot implement it
you have to note it why." So every UI-heuristic row below has an explicit
reason, and any port that *can* follow the guide is required to.

## Status

| Category   | Validator                | Notes                                                                              |
|------------|--------------------------|------------------------------------------------------------------------------------|
| Calendar   | **SQLite** (per guide)   | Simple/Fossify own DB; Etar/KashCal/GoogleCal share Android `CalendarProvider`.    |
| SMS        | **System SQLite / provider** (per guide) | All apps write to Telephony storage; seeding/validation uses `/data/data/com.android.providers.telephony/databases/mmssms.db`, with UI markers only for app-private mute/archive/open states. |
| Files      | **File-system** (per guide) | Apps operate on shared external storage; success = expected file present/absent.|
| Notes (Markor) | **File-system** (per guide) | Markor stores plain `.md` files under `Documents/markor`.                       |
| ToDo (Tasks.org / ntodotxt) | **SQLite** (per guide)| Backed by upstream `Tasks*` evaluators that read each app's DB.                  |
| Clock      | **SQLite where exposed; UI otherwise** | Fossify Clock exposes `alarms.db` and timer `app.db`, so create/edit/enable/delete alarm and create-timer read real DB state. Other clock apps in this image expose no stable alarm/timer table; stopwatch/world-clock/transient timer controls remain explicitly UI-only. |
| Contacts   | **ContactsProvider / SQLite / documented UI fallback** | Google Contacts uses `ContactsProvider`. Fossify Contacts and Simple Contacts Pro SE use `local_contacts.db`; Connect You uses `com.bnyro.contacts` SQLite tables. Under the pinned clean-reset protocol, Right Contact 8.2.3 uses public phone storage through `ContactsProvider`; initialization must fail closed unless the contact permissions are granted and the selected source is public. Its optional private Room store is not the pinned protocol, so neither UI-only nor Room-only validation is correct for this cohort. |
| Finance    | UI-heuristic (not yet conformance-qualified) | Each app has its own SQLite schema (Oinkoin: `transactions`; My Expenses: `transactions`; OpenMoneyBox: `movement`; ...). A valid revision needs per-app adapters that map those schemas to one abstract transaction predicate; schema diversity is not a reason to avoid durable-state validation. |
| Maps       | **File-system / SQLite where exposed; UI otherwise** | Favorite/marker mutation tasks read real storage for OsmAnd (`favorites.gpx`, `map_markers_db`) and Organic Maps/CoMaps (`My Places.kml`). Record-track tasks read GPX/KML exports and validate waypoint order only for apps that expose stable artifacts. Export-location tasks read GPX/KML exports or saved map-link text only for apps with stable artifact/link paths. Google Maps saved-place/marker/track/export state is opaque or synced in this image. MAPS.ME stores user saved places in binary `My Places.kmb` while its `guides` DB is only a downloaded-guides catalog. The clean matrix therefore schedules only search/directions/share UI tasks for Google Maps and MAPS.ME, and the submission tables exclude those two maps apps. |
| Music      | UI-heuristic (not yet conformance-qualified) | Library state is computed at scan time and playlists live in app-specific DBs. App-specific adapters and positive/near-miss calibration are required before this category supports a semantic-equivalence claim. |

## What "follows the guide" means here

Concretely:

1. The base class inherits `sqlite_validators.SQLiteApp` (or `file_validators.*`).
2. `db_path` / `table_name` / `row_type` are pinned to a value that matches
   the YAML in `docs/storage_profiles/<package>.yaml`.
3. `validate_addition_integrity` / `validate_deletion_integrity` is wired to
   the upstream helpers, with `compare_fields` listing schema columns.
4. For Android framework providers (`content://sms`,
   `content://com.android.contacts`, CalendarProvider), validators query the
   provider with `adb shell content query` instead of screen text.
5. Apps that store data encrypted-at-rest (Notesnook, Proton Calendar) are
   in `excluded_packages` with an inline reason.

## What "UI-heuristic" means here

The category-level `_*_cross_app_tasks.py` module is a `PackageAppEval`
subclass and `is_successful` reads `env.get_state().ui_elements`. This is
a deliberate trade-off, not an oversight:

- Cross-app equivalence requires one abstract postcondition, not one physical
  storage schema. Per-app SQLite/provider/file adapters are desirable when each
  maps native state to that same postcondition. Their conformance must be
  checked with gold completions and semantically wrong near misses.

- A UI heuristic is retained only when the app exposes no stable native state
  for the target predicate. Such a task–app pair is explicitly marked as a
  weaker evidence tier and is not silently treated as equivalent to a durable
  validator. If it cannot pass calibration, the pair is ineligible.

Where a shared schema DOES exist (Calendar via CalendarProvider, SMS via
SmsProvider, Contacts via ContactsProvider, Files via shared storage,
Notes/Markor via plain text on disk), we use it.

## Files cleaned up

The 5 stub files that the prior scaffold generator dropped are dead code:

  * `notes_my_brain_tasks.py`
  * `notes_neutrinote_tasks.py`
  * `notes_notallyx_tasks.py`
  * `notes_notesnook_tasks.py`
  * `notes_orgzly_revived_tasks.py`

Each contained one `NotImplementedError`-only `task_eval.TaskEval`
subclass per canonical Markor/Notes task and was never imported by the
registry. They should be deleted in a follow-up commit; cross-app notes
ports are tracked at `notes_cross_app_tasks.py` (TBD), not in per-app
files. We left them in place in this change so the user can confirm the
plan before deletion.
