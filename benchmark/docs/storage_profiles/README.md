# Storage profiles

One YAML per Android package recording where the app keeps its persistent
state. These files are the ground truth that cross-app SQLite / filesystem
validators pin to (per `tasks_guide.md` §5.2).

## How to (re)generate

With a debuggable AVD running and the package installed:

```bash
# one app:
bash benchmark/scripts/discover_app_storage.sh com.vicolo.chrono

# every app in the CSV:
bash benchmark/scripts/discover_app_storage.sh --all
```

The script writes `<package>.yaml` here, including:

- `private_databases:`  filenames in `/data/data/<pkg>/databases/` plus
  `.schema` dumps (when readable via `run-as`).
- `private_files:`  filenames in `/data/data/<pkg>/files/`.
- `public_files:`  filenames in `/sdcard/Android/data/<pkg>/files/`.

## How to use the YAML in code

When writing a cross-app validator (per `tasks_guide.md` §5.3):

1. Read the YAML for the target package.
2. Choose a database file whose schema names tables/columns relevant to the
   task semantics (e.g. `events`, `notes`, `playlists`, `transactions`).
3. Pin `db_path` / `table_name` / `row_type` on the per-app `SQLiteApp`
   subclass to those exact values.
4. List the same column names in `compare_fields` of
   `sqlite_validators.validate_rows_addition_integrity`.

If the YAML reports `<unreadable>` for a schema, re-run the discovery on a
debuggable build (`adb root`-able), or on an AVD with `userdebug` system
image. If the storage is encrypted at rest (Notesnook, Proton Calendar),
flag the package in `excluded_packages` of the task base class and document
the reason in the docstring -- per §5.1 case (4) of `tasks_guide.md`.
