"""Deterministic conformance tests for the frozen Contacts task adapters."""

from unittest import mock

from absl.testing import absltest

from android_world.env import representation_utils
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as cross_app_base,
)
from android_world.task_evals.single.app_generalization_generated import (
    contacts_cross_app_tasks as contacts_tasks,
)


class _InMemoryState:

  def __init__(self, ui_elements=()):
    self.ui_elements = list(ui_elements)


class _InMemoryEnv:

  def __init__(self, ui_elements=()):
    self._state = _InMemoryState(ui_elements)
    self.controller = object()

  def get_state(self):
    return self._state


def _element(
    text: str | None = None,
    *,
    content_description: str | None = None,
    hint_text: str | None = None,
    package_name: str = "com.google.android.contacts",
    is_editable: bool | None = None,
    class_name: str | None = None,
) -> representation_utils.UIElement:
  return representation_utils.UIElement(
      text=text,
      content_description=content_description,
      hint_text=hint_text,
      package_name=package_name,
      is_editable=is_editable,
      class_name=class_name,
  )


def _score(task_cls, params, ui_elements=()) -> float:
  task = task_cls(params)
  task.initialized = True
  return task.is_successful(_InMemoryEnv(ui_elements))


_CONTACT = {
    "first": "Alice",
    "last": "Johnson",
    "name": "Alice Johnson",
    "number": "202-555-0147",
}


class ContactsFrozenVerifierConformanceTest(absltest.TestCase):

  def test_native_state_read_distinguishes_absence_from_command_failure(self):
    marker = contacts_tasks._VERIFIER_READ_RC_MARKER
    with mock.patch.object(
        contacts_tasks,
        "_adb_shell",
        return_value=f"No result found.\n{marker}0\n",
    ):
      self.assertEqual(
          contacts_tasks._adb_shell_read(object(), "content query ..."),
          "No result found.",
      )
    with mock.patch.object(contacts_tasks, "_adb_shell", return_value=""):
      with self.assertRaises(cross_app_base.VerifierStateReadError):
        contacts_tasks._adb_shell_read(object(), "content query ...")
    with mock.patch.object(
        contacts_tasks,
        "_adb_shell",
        return_value=f"provider failure\n{marker}1\n",
    ):
      with self.assertRaises(cross_app_base.VerifierStateReadError):
        contacts_tasks._adb_shell_read(object(), "content query ...")

  def test_absent_local_database_reads_as_empty_not_as_read_failure(self):
    """An app that has stored nothing yet has no database file.

    ``sqlite3`` exits 1 both when the file is missing and when a real read
    fails, so the command must skip a missing file itself; otherwise every
    "agent stored nothing" episode is scored as an infrastructure error and
    silently leaves the denominator.
    """
    marker = contacts_tasks._VERIFIER_READ_RC_MARKER
    captured = {}

    def fake_adb_shell(_env, cmd):
      captured["cmd"] = cmd
      return f"{marker}0\n"

    with mock.patch.object(contacts_tasks, "_adb_shell", fake_adb_shell):
      self.assertEqual(
          contacts_tasks._sqlite_exec(object(), "/data/db.sqlite", "SELECT 1;"),
          "",
      )
    self.assertIn("if [ ! -f", captured["cmd"])
    self.assertIn("/data/db.sqlite", captured["cmd"])
    with mock.patch.object(contacts_tasks, "_sqlite_exec", return_value=""):
      self.assertEqual(
          contacts_tasks._local_sqlite_rows(object(), "org.fossify.contacts"),
          [],
      )

  def test_existing_database_that_cannot_be_read_still_fails_closed(self):
    marker = contacts_tasks._VERIFIER_READ_RC_MARKER
    with mock.patch.object(
        contacts_tasks,
        "_adb_shell",
        return_value=f"disk image is malformed\n{marker}1\n",
    ):
      with self.assertRaises(cross_app_base.VerifierStateReadError):
        contacts_tasks._sqlite_exec(object(), "/data/db.sqlite", "SELECT 1;")

  def test_local_state_reader_rejects_unparseable_rows(self):
    with mock.patch.object(
        contacts_tasks, "_sqlite_exec", return_value="malformed"
    ):
      with self.assertRaises(cross_app_base.VerifierStateReadError):
        contacts_tasks._local_sqlite_rows(
            object(), "org.fossify.contacts"
        )

  def test_right_contact_uses_provider_in_pinned_clean_configuration(self):
    package = "com.goodwy.contacts"
    self.assertNotIn(package, contacts_tasks._UI_FALLBACK_PACKAGES)
    self.assertFalse(contacts_tasks._uses_local_sqlite(package))
    self.assertNotIn(package, contacts_tasks._LOCAL_CONTACTS_DB)
    self.assertStartsWith(
        contacts_tasks.ContactsAddContactForRightContact.validation_mode,
        "ContactsProvider",
    )

  def test_right_contact_provider_contract_accepts_empty_or_absent_source(self):
    granted = (
        "android.permission.WRITE_CONTACTS: granted=true\n"
        "android.permission.READ_CONTACTS: granted=true"
    )
    empty_source = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<map><string name='last_used_contact_source'></string></map>"
    )
    for prefs in (
        contacts_tasks._RIGHT_CONTACTS_PREFS_ABSENT,
        empty_source,
        "<map></map>",
    ):
      with self.subTest(prefs=prefs):
        with mock.patch.object(
            contacts_tasks,
            "_adb_shell_read",
            side_effect=[granted, prefs],
        ):
          contacts_tasks._assert_right_contact_provider_contract(object())

  def test_right_contact_provider_contract_rejects_permission_or_source_drift(self):
    granted = (
        "android.permission.WRITE_CONTACTS: granted=true\n"
        "android.permission.READ_CONTACTS: granted=true"
    )
    invalid_cases = (
        (
            "android.permission.WRITE_CONTACTS: granted=false\n"
            "android.permission.READ_CONTACTS: granted=true",
            "<map></map>",
        ),
        (
            granted,
            "<map><string name='last_used_contact_source'>"
            "smt_private</string></map>",
        ),
        (granted, "<map>"),
    )
    for permissions, prefs in invalid_cases:
      with self.subTest(permissions=permissions, prefs=prefs):
        with mock.patch.object(
            contacts_tasks,
            "_adb_shell_read",
            side_effect=[permissions, prefs],
        ):
          with self.assertRaises(cross_app_base.VerifierStateReadError):
            contacts_tasks._assert_right_contact_provider_contract(object())

  def test_provider_phone_match_allows_only_one_leading_country_code(self):
    rows = [("Alice Johnson", "+1 (202) 555-0147")]
    with mock.patch.object(
        contacts_tasks, "_provider_phone_pairs", return_value=rows
    ), mock.patch.object(contacts_tasks, "_provider_contacts", return_value=[]):
      self.assertTrue(
          contacts_tasks._provider_has_phone(
              object(), "Alice Johnson", "202-555-0147"
          )
      )
      self.assertTrue(
          contacts_tasks._provider_has_phone(
              object(), "Alice Johnson", "+1-202-555-0147"
          )
      )
      self.assertFalse(
          contacts_tasks._provider_has_phone(
              object(), "Bob Johnson", "+1-202-555-0147"
          )
      )
      self.assertFalse(
          contacts_tasks._provider_has_phone(
              object(), "Alice Johnson", "991-202-555-0147"
          )
      )

  def test_serialized_local_phone_field_rejects_arbitrary_prefix(self):
    self.assertTrue(
        contacts_tasks._phone_field_contains_equivalent(
            '["+1 (202) 555-0147"]', "202-555-0147"
        )
    )
    self.assertFalse(
        contacts_tasks._phone_field_contains_equivalent(
            '["991-202-555-0147"]', "202-555-0147"
        )
    )
    self.assertFalse(
        contacts_tasks._phone_field_contains_equivalent(
            '["991 202 555 0147"]', "202-555-0147"
        )
    )
    self.assertTrue(
        contacts_tasks._phone_field_contains_equivalent(
            "303-555-0100\x1e202-555-0147", "202-555-0147"
        )
    )
    # Frozen parameter generation can produce a ten-digit number beginning
    # with 1; it must not be misread as a nine-digit number plus country code.
    self.assertTrue(
        contacts_tasks._phone_field_contains_equivalent(
            '["129-227-5252"]', "129-227-5252"
        )
    )

  def test_provider_reset_runs_for_google_and_right_contact_tasks(self):
    params = {
        "first": "Alice",
        "last": "Johnson",
        "phone": "202-555-0147",
        "phone_label": "Work",
    }
    for task_cls in (
        contacts_tasks.ContactsNewContactDraftForGoogleContacts,
        contacts_tasks.ContactsNewContactDraftForRightContact,
    ):
      with self.subTest(task=task_cls.__name__):
        task = task_cls(params)
        env = _InMemoryEnv()
        with mock.patch.object(
            contacts_tasks.contacts_utils, "clear_contacts"
        ) as clear_contacts, mock.patch.object(
            cross_app_base.PackageAppEval, "initialize_task"
        ) as parent_initialize, mock.patch.object(
            contacts_tasks, "_assert_right_contact_provider_contract"
        ) as assert_right_contract:
          task.initialize_task(env)

        clear_contacts.assert_called_once_with(env.controller)
        parent_initialize.assert_called_once_with(env)
        if task_cls is contacts_tasks.ContactsNewContactDraftForRightContact:
          assert_right_contract.assert_called_once_with(env)
        else:
          assert_right_contract.assert_not_called()

  def test_provider_reset_is_not_applied_to_app_private_contacts_db(self):
    task = contacts_tasks.ContactsAddContactForFossifyContacts(_CONTACT)
    env = _InMemoryEnv()
    with mock.patch.object(
        contacts_tasks.contacts_utils, "clear_contacts"
    ) as clear_contacts, mock.patch.object(
        cross_app_base.PackageAppEval, "initialize_task"
    ) as parent_initialize:
      task.initialize_task(env)

    clear_contacts.assert_not_called()
    parent_initialize.assert_called_once_with(env)

  def test_destructive_transition_latches_reset_on_initialize(self):
    remove_task = (
        contacts_tasks.ContactsRemoveFavoriteContactForGoogleContacts(_CONTACT)
    )
    delete_task = contacts_tasks.ContactsDeleteContactForGoogleContacts(
        _CONTACT
    )
    remove_task._catbench_seen_target_starred = True
    delete_task._catbench_seen_exact_target_contact = True
    env = _InMemoryEnv()

    with mock.patch.object(
        contacts_tasks.contacts_utils, "clear_contacts"
    ), mock.patch.object(cross_app_base.PackageAppEval, "initialize_task"):
      remove_task.initialize_task(env)
      delete_task.initialize_task(env)

    self.assertFalse(remove_task._catbench_seen_target_starred)
    self.assertFalse(delete_task._catbench_seen_exact_target_contact)

  def test_add_contact_provider_positive_wrong_and_partial(self):
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsAddContactForGoogleContacts, _CONTACT),
          1.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsAddContactForGoogleContacts, _CONTACT),
          0.0,
      )

  def test_unsaved_draft_requires_values_in_editable_controls(self):
    params = {
        "first": "Eva",
        "last": "Lee",
        "phone": "202-555-0147",
        "phone_label": "Work",
    }
    editable_form = [
        _element("Eva", is_editable=True),
        _element("Lee", is_editable=True),
        _element("202 555 0147", is_editable=True),
        _element("Work"),
    ]
    saved_detail = [
        _element("Eva"),
        _element("Lee"),
        _element("202 555 0147"),
        _element("Work"),
    ]
    save_only_collision = [
        _element("Save", is_editable=False),
        _element("Lee", is_editable=True),
        _element("202 555 0147", is_editable=True),
        _element("Work"),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForGoogleContacts,
              params,
              editable_form,
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForGoogleContacts,
              params,
              saved_detail,
          ),
          0.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForGoogleContacts,
              params,
              save_only_collision,
          ),
          0.0,
      )

  def test_draft_rejects_saved_contact_reopened_in_edit_form(self):
    params = {
        "first": "Eva",
        "last": "Lee",
        "phone": "202-555-0147",
        "phone_label": "Work",
    }
    reopened_edit_form = [
        _element("Eva", is_editable=True),
        _element("Lee", is_editable=True),
        _element("202 555 0147", is_editable=True),
        _element("Work"),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForGoogleContacts,
              params,
              reopened_edit_form,
          ),
          0.0,
      )

  def test_edit_requires_new_pair_and_old_contact_absent(self):
    params = {
        "first": "Alice",
        "old_last": "Johnson",
        "new_last": "Smith",
        "old_number": "202-555-0147",
        "new_number": "202-555-0188",
    }

    def successful_phone(unused_env, name, number):
      return name == "Alice Smith" and number == "202-555-0188"

    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", side_effect=successful_phone
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForGoogleContacts, params),
          1.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", side_effect=successful_phone
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForGoogleContacts, params),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForGoogleContacts, params),
          0.0,
      )

  def test_provider_edit_rejects_old_phone_retained_under_new_name(self):
    params = {
        "first": "Alice",
        "old_last": "Johnson",
        "new_last": "Smith",
        "old_number": "202-555-0147",
        "new_number": "202-555-0188",
    }
    renamed_row_with_both_numbers = [
        ("Alice Smith", "202-555-0147"),
        ("Alice Smith", "202-555-0188"),
    ]
    with mock.patch.object(
        contacts_tasks,
        "_provider_phone_pairs",
        return_value=renamed_row_with_both_numbers,
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(contacts_tasks, "_provider_contacts", return_value=[]):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForGoogleContacts, params),
          0.0,
      )

  def test_local_edit_rejects_old_phone_retained_under_new_name(self):
    params = {
        "first": "Alice",
        "old_last": "Johnson",
        "new_last": "Smith",
        "old_number": "202-555-0147",
        "new_number": "202-555-0188",
    }
    renamed_row_with_both_numbers = [{
        "name": "Alice Smith",
        "phones": "202-555-0147\x1e202-555-0188",
        "starred": "0",
    }]
    with mock.patch.object(
        contacts_tasks,
        "_local_sqlite_rows",
        return_value=renamed_row_with_both_numbers,
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForFossifyContacts, params),
          0.0,
      )

  def test_right_contact_edit_requires_durable_new_values(self):
    params = {
        "first": "Alice",
        "old_last": "Johnson",
        "new_last": "Smith",
        "old_number": "202-555-0147",
        "new_number": "202-555-0188",
    }
    unsaved_edit_form = [
        _element(
            "Smith", package_name="com.goodwy.contacts", is_editable=True
        ),
        _element(
            "202-555-0188",
            package_name="com.goodwy.contacts",
            is_editable=True,
        ),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsEditContactForRightContact,
              params,
              unsaved_edit_form,
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsEditContactForRightContact, params),
          1.0,
      )

  def test_search_requires_all_contacts_and_open_target_detail(self):
    params = {
        "target_name": "Alice Johnson",
        "target_number": "202-555-0147",
        "decoy_name_a": "Bob Smith",
        "decoy_number_a": "202-555-0101",
        "decoy_name_b": "Grace Brown",
        "decoy_number_b": "202-555-0199",
    }
    detail_ui = [_element("Alice Johnson"), _element("202 555 0147")]
    filtered_list_ui = [_element("Alice Johnson")]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsSearchContactForGoogleContacts,
              params,
              detail_ui,
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsSearchContactForGoogleContacts,
              params,
              filtered_list_ui,
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", side_effect=[True, False, True]
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsSearchContactForGoogleContacts,
              params,
              detail_ui,
          ),
          0.0,
      )

  def test_local_search_requires_target_and_both_decoys_persisted(self):
    params = {
        "target_name": "Alice Johnson",
        "target_number": "202-555-0147",
        "decoy_name_a": "Bob Smith",
        "decoy_number_a": "202-555-0101",
        "decoy_name_b": "Grace Brown",
        "decoy_number_b": "202-555-0199",
    }
    ui = [_element("Alice Johnson"), _element("202 555 0147")]
    with mock.patch.object(
        contacts_tasks, "_local_has_phone", side_effect=[True, False, True]
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsSearchContactForFossifyContacts,
              params,
              ui,
          ),
          0.0,
      )

  def test_view_details_requires_target_name_phone_and_storage(self):
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsViewContactDetailsForGoogleContacts,
              _CONTACT,
              [_element("Alice Johnson"), _element("202 555 0147")],
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsViewContactDetailsForGoogleContacts,
              _CONTACT,
              [_element("Alice Johnson")],
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsViewContactDetailsForGoogleContacts,
              _CONTACT,
              [_element("Alice Johnson"), _element("202 555 0147")],
          ),
          0.0,
      )

  def test_add_favorite_requires_contact_and_starred_state(self):
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(contacts_tasks, "_provider_is_starred", return_value=True):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsAddFavoriteContactForGoogleContacts,
              _CONTACT,
          ),
          1.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(contacts_tasks, "_provider_is_starred", return_value=False):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsAddFavoriteContactForGoogleContacts,
              _CONTACT,
          ),
          0.0,
      )

  def test_add_favorite_marker_rejects_bare_or_unstarred_favorite_text(self):
    self.assertFalse(
        contacts_tasks._favorite_state_visible([_element("Favorite")])
    )
    self.assertFalse(
        contacts_tasks._favorite_state_visible([_element("Add to favorites")])
    )
    self.assertTrue(
        contacts_tasks._favorite_state_visible(
            [_element("Remove from favorites")]
        )
    )

  def test_right_contact_storage_tasks_reject_ui_only_states(self):
    exact_visible = [
        _element(
            _CONTACT["name"], package_name="com.goodwy.contacts"
        ),
        _element(
            _CONTACT["number"], package_name="com.goodwy.contacts"
        ),
    ]
    search_params = {
        "target_name": _CONTACT["name"],
        "target_number": _CONTACT["number"],
        "decoy_name_a": "Bob Smith",
        "decoy_number_a": "202-555-0101",
        "decoy_name_b": "Grace Brown",
        "decoy_number_b": "202-555-0199",
    }
    for task_cls, params, ui in (
        (
            contacts_tasks.ContactsAddContactForRightContact,
            _CONTACT,
            exact_visible,
        ),
        (
            contacts_tasks.ContactsAddFavoriteContactForRightContact,
            _CONTACT,
            exact_visible + [_element("Remove from favorites")],
        ),
        (
            contacts_tasks.ContactsSearchContactForRightContact,
            search_params,
            exact_visible,
        ),
        (
            contacts_tasks.ContactsViewContactDetailsForRightContact,
            _CONTACT,
            exact_visible,
        ),
    ):
      with self.subTest(task=task_cls.__name__):
        with mock.patch.object(
            contacts_tasks, "_provider_has_phone", return_value=False
        ):
          self.assertEqual(_score(task_cls, params, ui), 0.0)

  def test_right_contact_storage_tasks_accept_exact_durable_rows(self):
    exact_visible = [
        _element(_CONTACT["name"], package_name="com.goodwy.contacts"),
        _element(_CONTACT["number"], package_name="com.goodwy.contacts"),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(contacts_tasks.ContactsAddContactForRightContact, _CONTACT),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsViewContactDetailsForRightContact,
              _CONTACT,
              exact_visible,
          ),
          1.0,
      )

    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_is_starred", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsAddFavoriteContactForRightContact,
              _CONTACT,
          ),
          1.0,
      )

    search_params = {
        "target_name": _CONTACT["name"],
        "target_number": _CONTACT["number"],
        "decoy_name_a": "Bob Smith",
        "decoy_number_a": "202-555-0101",
        "decoy_name_b": "Grace Brown",
        "decoy_number_b": "202-555-0199",
    }
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsSearchContactForRightContact,
              search_params,
              exact_visible,
          ),
          1.0,
      )

  def test_right_contact_draft_requires_durable_absence(self):
    params = {
        "first": "Eva",
        "last": "Lee",
        "phone": "202-555-0147",
        "phone_label": "Work",
    }
    editable_form = [
        _element("Eva", package_name="com.goodwy.contacts", is_editable=True),
        _element("Lee", package_name="com.goodwy.contacts", is_editable=True),
        _element(
            "202 555 0147",
            package_name="com.goodwy.contacts",
            is_editable=True,
        ),
        _element("Work", package_name="com.goodwy.contacts"),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForRightContact,
              params,
              editable_form,
          ),
          1.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_phone_present", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsNewContactDraftForRightContact,
              params,
              editable_form,
          ),
          0.0,
      )

  def test_right_contact_call_and_message_reject_manual_external_flows(self):
    active_call = [
        _element(
            _CONTACT["name"], package_name="com.google.android.dialer"
        ),
        _element(
            content_description="Speaker",
            package_name="com.google.android.dialer",
        ),
    ]
    message_compose = [
        _element(
            _CONTACT["name"],
            package_name="com.google.android.apps.messaging",
        ),
        _element(
            hint_text="Text message",
            package_name="com.google.android.apps.messaging",
            is_editable=True,
        ),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForRightContact,
              _CONTACT,
              active_call,
          ),
          0.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForRightContact,
              _CONTACT,
              message_compose,
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForRightContact,
              _CONTACT,
              active_call,
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForRightContact,
              _CONTACT,
              message_compose,
          ),
          1.0,
      )

  def test_remove_favorite_requires_exact_target_seen_starred_latch(self):
    task = contacts_tasks.ContactsRemoveFavoriteContactForGoogleContacts(
        _CONTACT
    )
    task.initialized = True
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(
        contacts_tasks,
        "_provider_is_starred",
        side_effect=[False, True, False],
    ):
      # A toast from another contact cannot make an initially unstarred target
      # pass.
      self.assertEqual(
          task.is_successful(
              _InMemoryEnv([_element("Removed from favorites")])
          ),
          0.0,
      )
      # Observe this exact persisted contact in its starred state.
      self.assertEqual(
          task.is_successful(_InMemoryEnv()),
          0.0,
      )
      # Durable unstarred state then succeeds without racing a transient toast.
      self.assertEqual(
          task.is_successful(_InMemoryEnv()),
          1.0,
      )

  def test_right_contact_remove_favorite_uses_durable_transition_latch(self):
    task = contacts_tasks.ContactsRemoveFavoriteContactForRightContact(
        _CONTACT
    )
    task.initialized = True
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ), mock.patch.object(
        contacts_tasks, "_provider_is_starred", side_effect=[False, True, False]
    ):
      self.assertEqual(task.is_successful(_InMemoryEnv()), 0.0)
      self.assertEqual(task.is_successful(_InMemoryEnv()), 0.0)
      self.assertEqual(task.is_successful(_InMemoryEnv()), 1.0)

  def test_delete_requires_exact_target_seen_present_latch(self):
    task = contacts_tasks.ContactsDeleteContactForGoogleContacts(_CONTACT)
    task.initialized = True
    with mock.patch.object(
        contacts_tasks,
        "_provider_has_phone",
        side_effect=[False, True, False],
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ):
      # Generic deletion text cannot pass from the initial absent state.
      self.assertEqual(
          task.is_successful(
              _InMemoryEnv([_element("Contact deleted — Undo")])
          ),
          0.0,
      )
      # Persisting the exact name+number pair updates the episode latch.
      self.assertEqual(
          task.is_successful(_InMemoryEnv()),
          0.0,
      )
      # Its later durable absence succeeds without requiring a snackbar.
      self.assertEqual(
          task.is_successful(_InMemoryEnv()),
          1.0,
      )

  def test_right_contact_delete_uses_durable_transition_latch(self):
    task = contacts_tasks.ContactsDeleteContactForRightContact(_CONTACT)
    task.initialized = True

    with mock.patch.object(
        contacts_tasks,
        "_provider_has_phone",
        side_effect=[False, True, False],
    ), mock.patch.object(
        contacts_tasks, "_provider_name_present", return_value=False
    ):
      self.assertEqual(task.is_successful(_InMemoryEnv()), 0.0)
      self.assertEqual(task.is_successful(_InMemoryEnv()), 0.0)
      self.assertEqual(task.is_successful(_InMemoryEnv()), 1.0)

  def test_call_requires_external_active_call_surface_and_target(self):
    detail_only = [
        _element("Alice Johnson"),
        _element("Phone", content_description="Phone"),
    ]
    active_call = [
        _element(
            "Alice Johnson",
            package_name="com.google.android.dialer",
        ),
        _element(
            content_description="Speaker",
            package_name="com.google.android.dialer",
        ),
    ]
    wrong_call = [
        _element("Bob Smith", package_name="com.google.android.dialer"),
        _element(
            content_description="Speaker",
            package_name="com.google.android.dialer",
        ),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForGoogleContacts,
              _CONTACT,
              detail_only,
          ),
          0.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForGoogleContacts,
              _CONTACT,
              active_call,
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForGoogleContacts,
              _CONTACT,
              wrong_call,
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsCallContactForGoogleContacts,
              _CONTACT,
              active_call,
          ),
          0.0,
      )

  def test_message_requires_external_editable_composer_and_target(self):
    detail_only = [
        _element("Alice Johnson"),
        _element("Message", content_description="Message"),
    ]
    compose = [
        _element(
            "Alice Johnson", package_name="com.google.android.apps.messaging"
        ),
        _element(
            hint_text="Text message",
            package_name="com.google.android.apps.messaging",
            is_editable=True,
        ),
    ]
    wrong_compose = [
        _element("Bob Smith", package_name="com.google.android.apps.messaging"),
        _element(
            hint_text="Text message",
            package_name="com.google.android.apps.messaging",
            is_editable=True,
        ),
    ]
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=True
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForGoogleContacts,
              _CONTACT,
              detail_only,
          ),
          0.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForGoogleContacts,
              _CONTACT,
              compose,
          ),
          1.0,
      )
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForGoogleContacts,
              _CONTACT,
              wrong_compose,
          ),
          0.0,
      )
    with mock.patch.object(
        contacts_tasks, "_provider_has_phone", return_value=False
    ):
      self.assertEqual(
          _score(
              contacts_tasks.ContactsMessageContactForGoogleContacts,
              _CONTACT,
              compose,
          ),
          0.0,
      )


if __name__ == "__main__":
  absltest.main()
