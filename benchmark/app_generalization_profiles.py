"""Profiles for cross-app generalization studies in AndroidWorld.

This module intentionally uses canonical task names that already exist in
AndroidWorld. New-app extensions should port these task definitions instead of
inventing new task names.
"""

from dataclasses import dataclass


NOTES_CANONICAL_TASKS = (
    "MarkorAddNoteHeader",
    "MarkorChangeNoteContent",
    "MarkorCreateFolder",
    "MarkorCreateNote",
    "MarkorCreateNoteAndSms",
    "MarkorCreateNoteFromClipboard",
    "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "MarkorEditNote",
    "MarkorMergeNotes",
    "MarkorMoveNote",
    "MarkorTranscribeReceipt",
    "MarkorTranscribeVideo",
    "NotesIsTodo",
    "NotesMeetingAttendeeCount",
    "NotesRecipeIngredientCount",
    "NotesTodoItemCount",
    "NotesCreateFolder",
)

TODO_CANONICAL_TASKS = (
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
    "TasksCreateTask",
    "TasksEditTask",
    "TasksCompleteTask",
    "TasksDeleteTask",
    "TasksAddTaskWithPriority",
)

SMS_CANONICAL_TASKS = (
    "SmsSend",
    "SmsReply",
    "SmsReplyMostRecent",
    "SmsResend",
    # SmsSendToContact replaced SmsSendClipboard: the clipboard flow requires
    # the AW clipper helper app (absent from this image) and the emulator
    # clipboard is overwritten by host window focus, so it is not reliably
    # reproducible. SendToContact seeds a named contact the agent must look
    # up; the send is validated against the telephony sent box.
    "SmsSendToContact",
    "SmsSendReceivedAddress",
    "SmsCreateDraftMessage",
    "SmsEditDraftMessage",
    "SmsDeleteConversation",
    # SmsForwardMessage replaced SmsArchiveConversation: archive state has no
    # durable artifact on QUIK (Realm-only), whereas forwarding is validated
    # against the shared telephony sent box on all four apps.
    "SmsForwardMessage",
)

FILES_CANONICAL_TASKS = (
    "FilesCreateFolder",
    "FilesRenameFile",
    "FilesDeleteFile",
    "FilesMoveFile",
    "FilesSaveCopyOfFile",
    "FilesSearchFile",
    "FilesCompressFiles",
    "FilesExtractArchive",
    "FilesViewFileInfo",
    "FilesShareFile",
)

MAPS_CANONICAL_TASKS = (
    "MapsSearchPlace",
    "MapsAddFavorite",
    "MapsRemoveFavorite",
    "MapsAddMarker",
    "MapsDeleteMarker",
    "MapsRecordTrack",
    "MapsGetDirections",
    "MapsSearchNearbyPlace",
    "MapsExportLocation",
    "MapsShareLocation",
)

# Registry-compatible transient adapters. These are intentionally not used by
# the frozen equal-depth profile below: Google Maps and MAPS.ME have no stable
# validators for the other six Maps semantics, so their ``implemented_tasks``
# remain empty rather than creating an app-specific four-task submatrix.
MAPS_GOOGLE_MAPS_SUPPORTED_TASKS = (
    "MapsSearchPlace",
    "MapsGetDirections",
    "MapsSearchNearbyPlace",
    "MapsShareLocation",
)

# MAPS.ME has the same transient generated-adapter subset as Google Maps.
# Its `guides` SQLite Bookmark table is the downloaded-guides catalog, not
# user favorites — real bookmarks live in the binary `My Places.kmb`, which
# is not a stable validation artifact, so storage-mutation tasks are
# infeasible to validate per the AW durable-state guideline.
MAPS_MAPS_ME_SUPPORTED_TASKS = MAPS_GOOGLE_MAPS_SUPPORTED_TASKS

CONTACTS_CANONICAL_TASKS = (
    "ContactsAddContact",
    "ContactsNewContactDraft",
    "ContactsEditContact",
    "ContactsSearchContact",
    "ContactsViewContactDetails",
    "ContactsAddFavoriteContact",
    "ContactsRemoveFavoriteContact",
    "ContactsDeleteContact",
    "ContactsCallContact",
    "ContactsMessageContact",
)

CLOCK_CANONICAL_TASKS = (
    "ClockCreateAlarm",
    "ClockEditAlarm",
    "ClockEnableAlarm",
    "ClockDeleteAlarm",
    "ClockCreateTimer",
    "ClockStartTimer",
    "ClockStopwatchRunning",
    "ClockPauseStopwatch",
    "ClockStopwatchReset",
    "ClockAddWorldClock",
)

# Calendar canonical tasks. The first three are cross-app portable when the
# target app writes to an inspectable calendar DB/provider. The remaining five
# are information-retrieval
# tasks that require a pre-populated Simple Calendar Pro SQLite DB and are not
# portable across third-party calendar apps; they are listed here to document
# the full canonical set.
CALENDAR_CANONICAL_TASKS = (
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarDeleteEvents",
    "SimpleCalendarEventsOnDate",
    "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SimpleCalendarEventsInNextWeek",
    "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarEditEvent",
    "SimpleCalendarViewMonthAgenda",
)

CALENDAR_PORTABLE_TASKS = (
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarDeleteEvents",
)

# Finance canonical task list. The first seven mirror the AndroidWorld
# Pro Expense (`expense.py`) tasks; the last three are cross-app tasks
# added for category coverage.
FINANCE_CANONICAL_TASKS = (
    "ExpenseAddSingle",
    "ExpenseAddMultiple",
    "ExpenseDeleteSingle",
    "ExpenseDeleteMultiple",
    "ExpenseDeleteDuplicates",
    "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor",
    "ExpenseEditExpense",
    "ExpenseAddCategory",
    "ExpenseViewMonthlyTotal",
)

# Music canonical task list. The first six mirror the AndroidWorld
# Retro Music (`retro_music.py`) tasks; the last four are cross-app tasks
# added for category coverage.
MUSIC_CANONICAL_TASKS = (
    "RetroCreatePlaylist",
    "RetroPlayingQueue",
    "RetroSavePlaylist",
    "RetroPlaylistDuration",
    "RetroCreateTwoPlaylists",
    "RetroAddToQueue",
    "RetroPlaySong",
    "RetroPauseSong",
    "RetroShuffleQueue",
    "RetroNextTrack",
)


@dataclass(frozen=True)
class AppProfile:
    """Metadata for one app variant in the study."""

    app_id: str
    display_name: str
    optional: bool
    package_name: str | None
    implemented_tasks: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class DomainProfile:
    """A domain cohort (for example Notes or To-Do)."""

    domain: str
    task_family: str
    intents: tuple[str, ...]
    canonical_tasks: tuple[str, ...]
    apps: tuple[AppProfile, ...]


def _names(prefixes: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}For{suffix}" for prefix in prefixes)


def _for_suffix(tasks: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    return _names(tasks, suffix)


TODO_PAPER_PREFIXES = (
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
    "TasksDueWithTime",
    "TasksRecurring",
    "TasksEditTask",
    "TasksCompleteTask",
)

TODO_AW_ORIGINAL_TASKS = (
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
)

TODO_TASKS_ORG_TASKS = (
    *TODO_AW_ORIGINAL_TASKS,
    "TasksDueWithTimeForTasksOrg",
    "TasksRecurringForTasksOrg",
    "TasksEditTaskForTasksOrg",
    "TasksCompleteTaskForTasksOrg",
)

NOTES_PAPER_PREFIXES = (
    "NotesCreateNote",
    "NotesCreateChecklist",
    "NotesEditNote",
    "NotesMergeNotes",
    "NotesDeleteNote",
    "NotesSearchNote",
    "NotesShareImport",
    "NotesCreateFolder",
    "NotesAttachContent",
    "NotesCountTodoItems",
)

FINANCE_PAPER_PREFIXES = (
    "ExpenseAddSingle",
    "ExpenseAddIncome",
    "ExpenseAddMultiple",
    "ExpenseEditExpense",
    "ExpenseDeleteSingle",
    "ExpenseDeleteDuplicates",
    "ExpenseCategorySummary",
    "ExpenseDateRangeTotal",
    "ExpenseAttachReceipt",
    "ExpenseTransferBetweenWallets",
)

MUSIC_PAPER_PREFIXES = (
    "RetroCreatePlaylist",
    "RetroRenamePlaylist",
    "RetroAddToPlaylist",
    "RetroRemoveFromPlaylist",
    "RetroAddToQueue",
    "RetroReorderQueue",
    "RetroSavePlaylist",
    "RetroPlaylistDuration",
    "RetroSleepTimer",
    "RetroSearchAndPlay",
)

CALENDAR_PAPER_EXTRA_PREFIXES = (
    "SimpleCalendarAddTimedEvent",
    "SimpleCalendarEventsOnDate",
    "SimpleCalendarNextEvent",
    "SimpleCalendarEventsInRange",
    "SimpleCalendarAddReminder",
)


def _calendar_paper_tasks(suffix: str, *, android_world_original: bool = False) -> tuple[str, ...]:
    first_three = (
        "SimpleCalendarAddOneEvent",
        "SimpleCalendarAddRepeatingEvent",
        "SimpleCalendarDeleteEvents",
    )
    if not android_world_original:
        first_three = _names(first_three, suffix)
    return (
        *first_three,
        f"SimpleCalendarEditEventFor{suffix}",
        f"SimpleCalendarMoveEventFor{suffix}",
        *_names(CALENDAR_PAPER_EXTRA_PREFIXES, suffix),
    )


NOTES_PROFILE = DomainProfile(
    domain="notes",
    task_family="single",
    intents=(
        "create_note",
        "edit_note",
        "delete_note",
        "organize_note",
        "retrieve_note_information",
        "count_todos",
    ),
    canonical_tasks=NOTES_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="markor",
            display_name="Markor",
            optional=False,
            package_name="net.gsantner.markor",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "Markor"),
            notes="Full paper template set; folder creation uses durable filesystem validation.",
        ),
        AppProfile(
            app_id="joplin",
            display_name="Joplin",
            optional=False,
            package_name="net.cozic.joplin",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "Joplin"),
            notes="Full paper template set; notebook creation uses durable SQLite validation.",
        ),
        AppProfile(
            app_id="notallyx",
            display_name="NotallyX",
            optional=False,
            package_name="com.philkes.notallyx",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "NotallyX"),
            notes="Full paper template set with UI-text validation for opaque app state.",
        ),
        AppProfile(
            app_id="neutrinote",
            display_name="neutriNote CE",
            optional=False,
            package_name="com.appmindlab.nano",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "NeutriNote"),
            notes="Full paper template set with UI-text validation for opaque app state.",
        ),
        AppProfile(
            app_id="notesnook",
            display_name="Notesnook",
            optional=False,
            package_name="com.streetwriters.notesnook",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "Notesnook"),
            notes="Full paper template set with UI-text validation for opaque app state.",
        ),
        AppProfile(
            app_id="orgzly_revived",
            display_name="Orgzly Revived",
            optional=False,
            package_name="com.orgzlyrevived",
            implemented_tasks=_names(NOTES_PAPER_PREFIXES, "OrgzlyRevived"),
            notes="Full paper template set; notebook creation uses SQLite-backed validation.",
        ),
        AppProfile(
            app_id="my_brain",
            display_name="My Brain",
            optional=True,
            package_name="com.mhss.app.mybrain",
            implemented_tasks=(),
            notes=(
                "Installed candidate, but de-scoped from robust evaluation:"
                " the inspected app state does not expose a durable category"
                " container matching NotesCreateFolder."
            ),
        ),
    ),
)


TODO_PROFILE = DomainProfile(
    domain="todo",
    task_family="information_retrieval",
    intents=(
        "create_task",
        "edit_task",
        "complete_task",
        "delete_task",
        "organize_task",
        "filter_or_search_tasks",
    ),
    canonical_tasks=TODO_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="tasks_org",
            display_name="Tasks.org",
            optional=False,
            package_name="org.tasks",
            implemented_tasks=TODO_TASKS_ORG_TASKS,
            notes="Full paper template set; write tasks use SQLite-backed validation where available.",
        ),
        AppProfile(
            app_id="cfait",
            display_name="Cfait",
            optional=False,
            package_name="com.trougnouf.cfait",
            implemented_tasks=_names(TODO_PAPER_PREFIXES, "Cfait"),
            notes="Full paper template set with UI-text validation for opaque app state.",
        ),
        AppProfile(
            app_id="trudido",
            display_name="Trudido",
            optional=True,
            package_name="com.trudido.app",
            implemented_tasks=(),
            notes=(
                "De-scoped from runnable CATBench: the available F-Droid APK"
                " only ships arm64-v8a Flutter native libraries, so it is not"
                " installable on the x86_64 AndroidWorld emulator image."
            ),
        ),
        AppProfile(
            app_id="todo_list_pfa",
            display_name="Todo List (PFA)",
            optional=False,
            package_name="org.secuso.privacyfriendlytodolist",
            implemented_tasks=_names(TODO_PAPER_PREFIXES, "TodoListPfa"),
            notes="Full paper template set; write tasks use SQLite-backed validation where available.",
        ),
        AppProfile(
            app_id="ntodotxt",
            display_name="ntodotxt",
            optional=False,
            package_name="de.tnmgl.ntodotxt",
            implemented_tasks=_names(TODO_PAPER_PREFIXES, "Ntodotxt"),
            notes="Full paper template set; supported write tasks use todo.txt validation.",
        ),
        AppProfile(
            app_id="taskmate",
            display_name="TaskMate",
            optional=False,
            package_name="com.amirsteinbeck.taskmate",
            implemented_tasks=_names(TODO_PAPER_PREFIXES, "Taskmate"),
            notes="Full paper template set with UI-text validation for opaque app state.",
        ),
        AppProfile(
            app_id="super_productivity",
            display_name="Super Productivity",
            optional=True,
            package_name="com.superproductivity.superproductivity",
            implemented_tasks=(),
            notes=(
                "Installed candidate, but de-scoped from robust evaluation:"
                " task state is stored in WebView IndexedDB/LevelDB, which"
                " needs a dedicated validator rather than UI-text checks."
            ),
        ),
        AppProfile(
            app_id="grit",
            display_name="Grit",
            optional=False,
            package_name="com.shub39.grit",
            implemented_tasks=_names(TODO_PAPER_PREFIXES, "Grit"),
            notes="Full paper template set; write tasks use SQLite-backed validation where available.",
        ),
    ),
)


CLOCK_PROFILE = DomainProfile(
    domain="clock",
    task_family="single",
    intents=(
        "create_alarm",
        "edit_alarm",
        "enable_alarm",
        "delete_alarm",
        "create_timer",
        "start_timer",
        "run_stopwatch",
        "pause_stopwatch",
        "reset_stopwatch",
        "add_world_clock",
    ),
    canonical_tasks=CLOCK_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="clock_clock",
            display_name="Clock",
            optional=False,
            package_name="com.best.deskclock",
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "Clock"),
            notes=(
                "UI-only: no stable SQLite/file store was exposed."
            ),
        ),
        AppProfile(
            app_id="clock_simple_clock",
            display_name="Simple Clock",
            optional=False,
            package_name="com.simplemobiletools.clock",
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "SimpleClock"),
            notes="UI-only: no stable SQLite/file store was exposed.",
        ),
        AppProfile(
            app_id="clock_google_clock",
            display_name="Google Clock",
            optional=False,
            package_name="com.google.android.deskclock",
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "GoogleClock"),
            notes=(
                "AndroidWorld-original baseline for the CATBench clock split;"
                " UI-only because alarm/timer state is not exposed as a stable"
                " app DB."
            ),
        ),
        AppProfile(
            app_id="clock_clockyou",
            display_name="Clock You",
            optional=True,
            package_name="com.bnyro.clock",
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "ClockYou"),
            notes=(
                "Device-qualified on pinned 9.1/API-33: Room-backed exact "
                "alarm and world-clock checks plus app-specific timer/"
                "stopwatch state checks passed 42/42 positive, no-op, wrong, "
                "partial, and lifecycle cases."
            ),
        ),
        AppProfile(
            app_id="clock_chrono",
            display_name="Chrono",
            optional=False,
            package_name="com.vicolo.chrono",
            # Flutter merges the stopwatch elapsed time into a multiline
            # semantics node; validators split merged nodes per line, so
            # stopwatch state is a11y-validated like every other clock app.
            # World-clock city pool excludes Chrono's preloaded defaults
            # (New York, London, Paris, Tokyo) to prevent no-op passes.
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "Chrono"),
            notes=(
                "UI-only: Flutter/Awesome Notifications stores schedule blobs"
                " without a stable alarm/timer schema for CATBench seeding."
            ),
        ),
        AppProfile(
            app_id="clock_fossify_clock",
            display_name="Fossify Clock",
            optional=False,
            package_name="org.fossify.clock",
            implemented_tasks=_for_suffix(CLOCK_CANONICAL_TASKS, "FossifyClock"),
            notes=(
                "SQLite-backed for alarm mutation tasks and create-timer;"
                " UI-only for stopwatch/world-clock/transient timer controls."
            ),
        ),
    ),
)


CALENDAR_PROFILE = DomainProfile(
    domain="calendar",
    task_family="single",
    intents=(
        "create_event",
        "create_recurring_event",
        "delete_events",
        "query_events_on_date",
        "query_next_event",
        "query_next_meeting_with_person",
        "query_events_in_next_week",
        "query_events_in_time_range",
    ),
    canonical_tasks=CALENDAR_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="calendar_simple_calendar_pro",
            display_name="Simple Calendar Pro",
            optional=False,
            package_name="com.simplemobiletools.calendar.pro",
            implemented_tasks=_calendar_paper_tasks(
                "SimpleCalendarPro", android_world_original=True
            ),
            notes="Full paper template set; canonical write tasks use SQLite-backed validation.",
        ),
        AppProfile(
            app_id="calendar_etar",
            display_name="Etar",
            optional=False,
            package_name="ws.xsoh.etar",
            implemented_tasks=_calendar_paper_tasks("Etar"),
            notes="Full paper template set; write/edit/delete tasks use CalendarProvider validation.",
        ),
        AppProfile(
            app_id="calendar_fossify_calendar",
            display_name="Fossify Calendar",
            optional=False,
            package_name="org.fossify.calendar",
            implemented_tasks=_calendar_paper_tasks("FossifyCalendar"),
            notes="Full paper template set; write/edit/delete tasks use private events.db validation.",
        ),
        AppProfile(
            app_id="calendar_calendar",
            display_name="Calendar",
            optional=False,
            package_name="com.vayunmathur.calendar",
            implemented_tasks=_calendar_paper_tasks("Calendar"),
            notes="Full paper template set; write/edit/delete tasks use CalendarProvider validation.",
        ),
        AppProfile(
            app_id="calendar_kashcal",
            display_name="KashCal",
            optional=False,
            package_name="org.onekash.kashcal",
            implemented_tasks=_calendar_paper_tasks("Kashcal"),
            notes="Full paper template set; write/edit/delete tasks use private kashcal.db validation.",
        ),
        AppProfile(
            app_id="calendar_google_calendar",
            display_name="Google Calendar",
            optional=True,
            package_name="com.google.android.calendar",
            implemented_tasks=(),
            notes=(
                "Installed candidate, but de-scoped from robust evaluation:"
                " the emulator has no writable Google Calendar provider"
                " calendar/account rows to seed and verify events."
            ),
        ),
        AppProfile(
            app_id="calendar_samsung_calendar",
            display_name="Samsung Calendar",
            optional=True,
            package_name="com.samsung.android.calendar",
            implemented_tasks=(),
            notes=(
                "Optional Samsung-only candidate; de-scoped from generic"
                " emulator runs because the app is unavailable without a"
                " Samsung system image."
            ),
        ),
    ),
)


CONTACTS_PROFILE = DomainProfile(
    domain="contacts",
    task_family="single",
    intents=(
        "create_contact",
        "new_contact_draft",
        "edit_contact",
        "search_contact",
        "view_contact_details",
        "add_favorite_contact",
        "remove_favorite_contact",
        "delete_contact",
        "call_contact",
        "message_contact",
    ),
    canonical_tasks=CONTACTS_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="contacts_google_contacts",
            display_name="Google Contacts",
            optional=False,
            package_name="com.google.android.contacts",
            implemented_tasks=_for_suffix(CONTACTS_CANONICAL_TASKS, "GoogleContacts"),
            notes="AndroidWorld-original baseline; ContactsProvider-backed.",
        ),
        AppProfile(
            app_id="contacts_fossify_contacts",
            display_name="Fossify Contacts",
            optional=False,
            package_name="org.fossify.contacts",
            implemented_tasks=_for_suffix(CONTACTS_CANONICAL_TASKS, "FossifyContacts"),
            notes="SQLite-backed via local_contacts.db where possible.",
        ),
        AppProfile(
            app_id="contacts_connect_you",
            display_name="Connect You",
            optional=False,
            package_name="com.bnyro.contacts",
            implemented_tasks=_for_suffix(CONTACTS_CANONICAL_TASKS, "ConnectYou"),
            notes="SQLite-backed via com.bnyro.contacts DB where possible.",
        ),
        AppProfile(
            app_id="contacts_simple_contacts_pro_se",
            display_name="Simple Contacts Pro SE",
            optional=False,
            package_name="com.simplemobiletools.contacts.pro",
            implemented_tasks=_for_suffix(
                CONTACTS_CANONICAL_TASKS, "SimpleContactsProSE"
            ),
            notes="SQLite-backed via local_contacts.db where possible.",
        ),
        AppProfile(
            app_id="contacts_right_contact",
            display_name="Right Contact",
            optional=False,
            package_name="com.goodwy.contacts",
            implemented_tasks=_for_suffix(CONTACTS_CANONICAL_TASKS, "RightContact"),
            notes=(
                "ContactsProvider-backed in the pinned clean image "
                "(last_used_contact_source is Android phone storage); exact "
                "positive/negative live conformance remains required."
            ),
        ),
    ),
)


FINANCE_PROFILE = DomainProfile(
    domain="finance",
    task_family="single",
    intents=(
        "add_expense",
        "add_multiple_expenses",
        "delete_expense",
        "delete_multiple_expenses",
        "deduplicate_expenses",
        "add_expense_from_image",
        "add_expense_from_note",
        "edit_expense",
        "manage_expense_categories",
        "view_monthly_total",
    ),
    canonical_tasks=FINANCE_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="finance_oinkoin",
            display_name="Oinkoin",
            optional=False,
            package_name="com.github.emavgl.piggybankpro",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "Oinkoin"),
            notes="Full paper template set with generic finance validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="finance_openmoneybox",
            display_name="OpenMoneyBox",
            optional=False,
            package_name="com.igisw.openmoneybox",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "OpenMoneyBox"),
            notes="Full paper template set with generic finance validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="finance_my_expenses",
            display_name="My Expenses",
            optional=False,
            package_name="org.totschnig.myexpenses",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "MyExpenses"),
            notes=(
                "SQLite-backed transaction validators, including source-data"
                " gallery and Markor import setup."
            ),
        ),
        AppProfile(
            app_id="finance_finance_manager",
            display_name="Finance Manager",
            optional=False,
            package_name="org.secuso.privacyfriendlyfinancemanager",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "FinanceManager"),
            notes="Full paper template set with generic finance validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="finance_sushi",
            display_name="Sushi",
            optional=False,
            package_name="com.jerameeldelosreyes.sushi",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "Sushi"),
            notes="Full paper template set with generic finance validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="finance_pro_expense",
            display_name="Pro Expense",
            optional=False,
            package_name="com.arduia.expense",
            implemented_tasks=_names(FINANCE_PAPER_PREFIXES, "ProExpense"),
            notes=(
                "AndroidWorld-original app. accounting.db-backed validators"
                " match the active My Expenses task families."
            ),
        ),
    ),
)


MUSIC_PROFILE = DomainProfile(
    domain="music",
    task_family="single",
    intents=(
        "create_playlist",
        "view_playing_queue",
        "save_playlist",
        "view_playlist_duration",
        "create_two_playlists",
        "add_to_queue",
        "play_song",
        "pause_song",
        "shuffle_queue",
        "next_track",
    ),
    canonical_tasks=MUSIC_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="music_retro_music",
            display_name="Retro Music",
            optional=False,
            package_name="code.name.monkey.retromusic",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "RetroMusic"),
            notes=(
                "MediaProvider-backed playlist creation validators. Queue and"
                " playback tasks are de-scoped because they do not produce a"
                " durable DB state."
            ),
        ),
        AppProfile(
            app_id="music_fossify_music",
            display_name="Fossify Music",
            optional=False,
            package_name="org.fossify.musicplayer",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "FossifyMusic"),
            notes="MediaProvider-backed playlist creation validators.",
        ),
        AppProfile(
            app_id="music_apollo",
            display_name="Apollo",
            optional=False,
            package_name="org.nuclearfog.apollo",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "Apollo"),
            notes="Full paper template set with UI/media-state validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="music_sicmu_neo",
            display_name="SicMu Neo",
            optional=False,
            package_name="xyz.mordorx.sicmu",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "SicMuNeo"),
            notes="Full paper template set with UI/media-state validation where durable DB state is unavailable.",
        ),
        AppProfile(
            app_id="music_phonograph_plus",
            display_name="Phonograph Plus",
            optional=False,
            package_name="player.phonograph.plus",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "PhonographPlus"),
            notes="MediaProvider-backed playlist creation validators.",
        ),
        AppProfile(
            app_id="music_monstermusic",
            display_name="MonsterMusic",
            optional=False,
            package_name="com.ztftrue.music",
            implemented_tasks=_names(MUSIC_PAPER_PREFIXES, "MonsterMusic"),
            notes="Full paper template set with UI/media-state validation where durable DB state is unavailable.",
        ),
    ),
)


SMS_PROFILE = DomainProfile(
    domain="sms",
    task_family="single",
    intents=(
        "send",
        "reply",
        "reply_most_recent",
        "resend",
        "send_to_contact",
        "send_received_address",
        "create_draft_message",
        "edit_draft_message",
        "delete_conversation",
        "forward_message",
    ),
    canonical_tasks=SMS_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="sms_simple_sms_messenger",
            display_name="Simple SMS Messenger",
            optional=False,
            package_name="com.simplemobiletools.smsmessenger",
            implemented_tasks=_for_suffix(SMS_CANONICAL_TASKS, "SimpleSMSMessenger"),
            notes=(
                "AndroidWorld-original baseline; SmsProvider-backed for durable sends/drafts/deletes and"
                " forwarded messages (validated via the shared telephony sent"
                " box)."
            ),
        ),
        AppProfile(
            app_id="sms_fossify_messages",
            display_name="Fossify Messages",
            optional=False,
            package_name="org.fossify.messages",
            implemented_tasks=_for_suffix(SMS_CANONICAL_TASKS, "FossifyMessages"),
            notes=(
                "SmsProvider-backed for durable sends/drafts/deletes and"
                " forwarded messages (validated via the shared telephony sent"
                " box)."
            ),
        ),
        AppProfile(
            app_id="sms_quik_sms",
            display_name="QUIK SMS",
            optional=False,
            package_name="dev.octoshrimpy.quik.fdroid",
            implemented_tasks=_for_suffix(SMS_CANONICAL_TASKS, "QUIKSMS"),
            notes=(
                "SmsProvider-backed for durable sends/drafts/deletes and"
                " forwarded messages (validated via the shared telephony sent"
                " box)."
            ),
        ),
        AppProfile(
            app_id="sms_google_messages",
            display_name="Messages",
            optional=False,
            package_name="com.google.android.apps.messaging",
            implemented_tasks=_for_suffix(SMS_CANONICAL_TASKS, "Messages"),
            notes=(
                "SmsProvider-backed for durable sends/drafts/deletes and"
                " forwarded messages (validated via the shared telephony sent"
                " box)."
            ),
        ),
    ),
)


FILES_PROFILE = DomainProfile(
    domain="files",
    task_family="single",
    intents=(
        "create_folder",
        "rename_file",
        "delete_file",
        "move_file",
        "save_copy_of_file",
        "search_file",
        "compress_files",
        "extract_archive",
        "view_file_info",
        "share_file",
    ),
    canonical_tasks=FILES_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="files_material_files",
            display_name="Material Files",
            optional=False,
            package_name="me.zhanghai.android.files",
            implemented_tasks=_for_suffix(FILES_CANONICAL_TASKS, "MaterialFiles"),
            notes="AndroidWorld-original file-manager baseline; filesystem-backed.",
        ),
        AppProfile(
            app_id="files_amaze",
            display_name="Amaze File Manager",
            optional=False,
            package_name="com.amaze.filemanager",
            implemented_tasks=_for_suffix(FILES_CANONICAL_TASKS, "AmazeFileManager"),
            notes="Filesystem-backed except non-mutating info/share UI checks.",
        ),
        AppProfile(
            app_id="files_fossify_file_manager",
            display_name="Fossify File Manager",
            optional=False,
            package_name="org.fossify.filemanager",
            implemented_tasks=_for_suffix(FILES_CANONICAL_TASKS, "FossifyFileManager"),
            notes="Filesystem-backed except non-mutating info/share UI checks.",
        ),
        AppProfile(
            app_id="files_total_commander",
            display_name="Total Commander",
            optional=False,
            package_name="com.ghisler.android.TotalCommander",
            implemented_tasks=_for_suffix(FILES_CANONICAL_TASKS, "TotalCommander"),
            notes="Filesystem-backed except non-mutating info/share UI checks.",
        ),
        AppProfile(
            app_id="files_x_plore_file_manager",
            display_name="X-plore File Manager",
            optional=False,
            package_name="com.lonelycatgames.Xplore",
            implemented_tasks=_for_suffix(FILES_CANONICAL_TASKS, "XploreFileManager"),
            notes="Filesystem-backed except non-mutating info/share UI checks.",
        ),
    ),
)


MAPS_PROFILE = DomainProfile(
    domain="maps",
    task_family="single",
    intents=(
        "search_place",
        "add_favorite",
        "remove_favorite",
        "add_marker",
        "delete_marker",
        "record_track",
        "get_directions",
        "search_nearby_place",
        "export_location",
        "share_location",
    ),
    canonical_tasks=MAPS_CANONICAL_TASKS,
    apps=(
        AppProfile(
            app_id="maps_osmand",
            display_name="OsmAnd~",
            optional=False,
            package_name="net.osmand.plus",
            implemented_tasks=_for_suffix(MAPS_CANONICAL_TASKS, "OsmAnd"),
            notes=(
                "AndroidWorld-original maps baseline; GPX/SQLite-backed for"
                " favorite, marker, and track tasks, UI-only for transient"
                " search/route/share tasks."
            ),
        ),
        AppProfile(
            app_id="maps_organic_maps",
            display_name="Organic Maps",
            optional=False,
            package_name="app.organicmaps",
            implemented_tasks=_for_suffix(MAPS_CANONICAL_TASKS, "OrganicMaps"),
            notes=(
                "KML-backed for favorite, marker, and exported track/route"
                " tasks, UI-only for transient search/route/share tasks."
            ),
        ),
        AppProfile(
            app_id="maps_google_maps",
            display_name="Google Maps",
            optional=True,
            package_name="com.google.android.apps.maps",
            implemented_tasks=(),
            notes=(
                "Generated transient UI adapters remain importable, but this"
                " app is fully descheduled from the frozen equal-depth grid."
                " Saved-place/marker state is opaque or synced in this build,"
                " and track/export artifacts are not exposed as stable local"
                " validation targets."
            ),
        ),
        AppProfile(
            app_id="maps_comaps",
            display_name="CoMaps",
            optional=False,
            package_name="app.comaps.fdroid",
            implemented_tasks=_for_suffix(MAPS_CANONICAL_TASKS, "CoMaps"),
            notes=(
                "KML-backed for favorite, marker, and exported track/route"
                " tasks, UI-only for transient search/route/share tasks."
            ),
        ),
        AppProfile(
            app_id="maps_maps_me",
            display_name="MAPS.ME",
            optional=True,
            package_name="com.mapswithme.maps.pro",
            implemented_tasks=(),
            notes=(
                "Generated transient UI adapters remain importable, but this"
                " app is fully descheduled from the frozen equal-depth grid."
                " The guides SQLite Bookmark table is the downloaded-guides"
                " catalog, not user favorites; real bookmarks live in the"
                " binary My Places.kmb, so favorite/marker mutations have no"
                " stable local validation target."
            ),
        ),
    ),
)

def get_domain_profiles() -> dict[str, DomainProfile]:
    """Returns all app-generalization domain profiles."""
    return {
        NOTES_PROFILE.domain: NOTES_PROFILE,
        TODO_PROFILE.domain: TODO_PROFILE,
        CLOCK_PROFILE.domain: CLOCK_PROFILE,
        CALENDAR_PROFILE.domain: CALENDAR_PROFILE,
        CONTACTS_PROFILE.domain: CONTACTS_PROFILE,
        SMS_PROFILE.domain: SMS_PROFILE,
        FILES_PROFILE.domain: FILES_PROFILE,
        MAPS_PROFILE.domain: MAPS_PROFILE,
        FINANCE_PROFILE.domain: FINANCE_PROFILE,
        MUSIC_PROFILE.domain: MUSIC_PROFILE,
    }
