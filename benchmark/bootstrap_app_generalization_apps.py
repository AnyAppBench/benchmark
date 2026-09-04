#!/usr/bin/env python3
"""Bootstraps app-generalization apps on a fresh emulator.

This script can automatically:
1) resolve APK URLs (from catalog `apk_url` or F-Droid by package name),
    and fallback to APKMirror search,
2) download APKs,
3) install APKs via adb,
4) grant common runtime permissions.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Iterable
from urllib.parse import quote_plus
from urllib.parse import unquote
from urllib.parse import urljoin
from urllib.parse import urlparse

import requests


FDROID_PACKAGE_API = "https://f-droid.org/api/v1/packages/{package_name}"
FDROID_REPO_APK = "https://f-droid.org/repo/{package_name}_{version_code}.apk"
FDROID_ARCHIVE_APK = "https://f-droid.org/archive/{package_name}_{version_code}.apk"
# IzzyOnDroid is a third-party FDroid-compatible repo. Its index page lists
# every published APK for a package, so we can scrape version codes and
# construct direct download URLs against the same repo. No Cloudflare gate.
IZZY_INDEX_URL = "https://apt.izzysoft.de/fdroid/index/apk/{package_name}"
IZZY_REPO_APK = "https://apt.izzysoft.de/fdroid/repo/{package_name}_{version_code}.apk"
APKMIRROR_SEARCH_URL = "https://www.apkmirror.com/?post_type=app_release&searchtype=apk&s={query}"
APKMIRROR_BASE_URL = "https://www.apkmirror.com"
APKCOMBO_BASE_URL = "https://apkcombo.com"
DEFAULT_TIMEOUT = 30
REQUEST_HEADERS = {
    # Some hosts block default Python user-agent for file downloads.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
}

COMMON_PERMISSIONS = (
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.READ_CALENDAR",
    "android.permission.WRITE_CALENDAR",
    "android.permission.READ_SMS",
    "android.permission.SEND_SMS",
    "android.permission.READ_PHONE_STATE",
    "android.permission.CALL_PHONE",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.ACCESS_BACKGROUND_LOCATION",
    "android.permission.POST_NOTIFICATIONS",
    "android.permission.READ_MEDIA_IMAGES",
    "android.permission.READ_MEDIA_VIDEO",
    "android.permission.READ_MEDIA_AUDIO",
)


@dataclasses.dataclass(frozen=True)
class AppRow:
    app_id: str
    display_name: str
    package_name: str
    apk_filename: str
    apk_url: str
    optional: bool


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _load_catalog(path: pathlib.Path) -> list[AppRow]:
    rows: list[AppRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                AppRow(
                    app_id=(raw.get("app_id") or "").strip(),
                    display_name=(raw.get("display_name") or "").strip(),
                    package_name=(raw.get("package_name") or "").strip(),
                    apk_filename=(raw.get("apk_filename") or "").strip(),
                    apk_url=(raw.get("apk_url") or "").strip(),
                    optional=_parse_bool(raw.get("optional") or "false"),
                )
            )
    return rows


def _split_apk_sources(apk_url: str) -> list[str]:
    """Returns candidate APK sources from `apk_url`.

    Multiple sources can be provided in the CSV field separated by `|`.
    """
    if not apk_url:
        return []
    sources: list[str] = []
    seen: set[str] = set()
    for raw in apk_url.replace("\n", "|").split("|"):
        source = raw.strip()
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def _is_http_source(source: str) -> bool:
    lowered = source.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _is_github_source(source: str) -> bool:
    return source.lower().startswith("github://")


def _resolve_github_release_urls(
    spec: str, timeout_sec: int
) -> tuple[list[str], str]:
    """Resolves APK download URLs from a ``github://owner/repo[:filter]`` spec.

    Calls ``api.github.com/repos/<owner>/<repo>/releases/latest`` and returns
    every release asset whose name ends in ``.apk``. If ``:filter`` is given
    (case-insensitive substring), only assets whose name contains it are kept.
    """
    body = spec[len("github://"):] if spec.lower().startswith("github://") else spec
    if ":" in body:
        repo_part, asset_filter = body.split(":", 1)
    else:
        repo_part, asset_filter = body, ""
    repo_part = repo_part.strip().strip("/")
    if repo_part.count("/") != 1:
        return [], f"github spec must be 'owner/repo[:filter]', got {spec!r}"

    api_url = f"https://api.github.com/repos/{repo_part}/releases/latest"
    headers = dict(REQUEST_HEADERS)
    headers["Accept"] = "application/vnd.github+json"
    try:
        response = requests.get(api_url, timeout=timeout_sec, headers=headers)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return [], f"GitHub API request failed: {exc}"
    if response.status_code != 200:
        return [], f"GitHub API returned HTTP {response.status_code}"
    try:
        payload = response.json()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return [], f"Invalid GitHub API JSON: {exc}"

    needle = asset_filter.lower()
    urls: list[str] = []
    for asset in payload.get("assets") or []:
        name = (asset.get("name") or "").lower()
        if not name.endswith(".apk"):
            continue
        if needle and needle not in name:
            continue
        url = asset.get("browser_download_url")
        if url:
            urls.append(url)
    if not urls:
        return [], "No matching .apk assets in latest release"
    return urls, ""


def _resolve_local_source_path(source: str, catalog_dir: pathlib.Path) -> pathlib.Path:
    """Resolves a local path candidate from CSV or CLI.

    Supports:
    - absolute paths
    - paths relative to catalog directory
    - file:// URLs
    """
    if source.startswith("file://"):
        parsed = urlparse(source)
        return pathlib.Path(unquote(parsed.path)).expanduser().resolve()

    candidate = pathlib.Path(source).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    catalog_relative = (catalog_dir / candidate).resolve()
    if catalog_relative.exists():
        return catalog_relative

    return candidate.resolve()


def _collect_preferred_sources(
    app: AppRow,
    local_apk_dir: str,
    catalog_dir: pathlib.Path,
    timeout_sec: int = 30,
) -> list[str]:
    """Collects explicit non-F-Droid sources before falling back to F-Droid.

    Recognised source forms in the CSV ``apk_url`` field:
      * ``http(s)://...``               -- direct download URL.
      * ``file://...`` or local path    -- local APK file.
      * ``github://owner/repo[:filter]`` -- expanded via the GitHub Releases
                                            API. ``filter`` is an optional
                                            case-insensitive substring of the
                                            asset name.
    """
    sources: list[str] = []
    seen: set[str] = set()

    for source in _split_apk_sources(app.apk_url):
        if _is_github_source(source):
            gh_urls, gh_err = _resolve_github_release_urls(source, timeout_sec)
            if gh_urls:
                print(f"  [INFO] Resolved via GitHub Releases: {gh_urls[0]}")
                for url in gh_urls:
                    if url not in seen:
                        seen.add(url)
                        sources.append(url)
            else:
                print(f"  [WARN] GitHub source {source!r}: {gh_err}")
            continue
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)

    if local_apk_dir:
        base = pathlib.Path(local_apk_dir).expanduser()
        local_names = [app.apk_filename]
        if app.package_name:
            local_names.append(f"{app.package_name}.apk")
        if app.app_id:
            local_names.append(f"{app.app_id}.apk")

        for name in local_names:
            if not name:
                continue
            candidate = (base / name).resolve()
            source = str(candidate)
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)

    # Resolve local relative paths against catalog location for convenience.
    resolved_sources: list[str] = []
    for source in sources:
        if _is_http_source(source):
            resolved_sources.append(source)
        else:
            resolved_sources.append(str(_resolve_local_source_path(source, catalog_dir)))

    return resolved_sources


def _adb_base(adb_bin: str, serial: str | None) -> list[str]:
    base = [adb_bin]
    if serial:
        base.extend(["-s", serial])
    return base


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    timeout_sec = int(os.environ.get("CATBENCH_BOOTSTRAP_ADB_TIMEOUT_SEC", "300"))
    try:
        return subprocess.run(
            cmd,
            check=check,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output if isinstance(exc.output, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=output,
            stderr=stderr or f"Timed out after {timeout_sec}s",
        )


def _detect_serial(adb_bin: str, preferred: str = "emulator-5554") -> tuple[str | None, str | None]:
    result = _run([adb_bin, "devices"])
    if result.returncode != 0:
        return None, "Failed to run adb devices"

    serials: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])

    if not serials:
        return None, "No adb device detected"

    if len(serials) == 1:
        return serials[0], None

    if preferred in serials:
        return preferred, None

    return None, (
        "Multiple adb devices detected and no --serial provided. "
        f"Devices: {serials}"
    )


def _resolve_fdroid_urls(package_name: str, timeout_sec: int) -> tuple[list[str], str]:
    api_url = FDROID_PACKAGE_API.format(package_name=package_name)
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                api_url,
                timeout=timeout_sec,
                headers=REQUEST_HEADERS,
            )
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt < 3:
                time.sleep(attempt)

    if response is None:
        return [], f"F-Droid API request failed: {last_exc}"

    if response.status_code != 200:
        return [], f"F-Droid API returned HTTP {response.status_code}"

    try:
        payload = response.json()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return [], f"Invalid F-Droid API JSON: {exc}"

    version_codes: list[int] = []

    suggested = payload.get("suggestedVersionCode")
    if isinstance(suggested, int) and suggested > 0:
        version_codes.append(suggested)

    packages = payload.get("packages") or []
    for package in packages:
        code = package.get("versionCode")
        if isinstance(code, int) and code > 0:
            version_codes.append(code)

    # Keep insertion order while de-duplicating.
    unique_codes: list[int] = []
    seen: set[int] = set()
    for code in version_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    if not unique_codes:
        return [], "No version code found in F-Droid metadata"

    urls = [
        FDROID_REPO_APK.format(package_name=package_name, version_code=code)
        for code in unique_codes
    ]
    # F-Droid moves older or unmaintained builds to /archive/.  Surface those
    # as fallbacks so apps removed from the live repo (legacy Simple Mobile
    # Tools, etc.) still install when only an archive copy exists.
    urls.extend(
        FDROID_ARCHIVE_APK.format(package_name=package_name, version_code=code)
        for code in unique_codes
    )
    return urls, ""


def _resolve_izzyondroid_urls(
    package_name: str, timeout_sec: int
) -> tuple[list[str], str]:
    """Resolves APK URLs from IzzyOnDroid's third-party F-Droid repo.

    IzzyOnDroid serves an HTML index page per package that lists every
    published APK with its version code in the filename. We fetch the page
    and extract every ``/fdroid/repo/<pkg>_<vc>.apk`` link, sorted newest
    first. No Cloudflare gate, so scraping is reliable.
    """
    page_url = IZZY_INDEX_URL.format(package_name=package_name)
    text, err = _fetch_text_with_retries(page_url, timeout_sec)
    if text is None:
        return [], f"IzzyOnDroid index fetch failed: {err}"

    pattern = re.compile(
        rf"/fdroid/repo/{re.escape(package_name)}_(\d+)\.apk"
    )
    version_codes = sorted(
        {int(match.group(1)) for match in pattern.finditer(text)},
        reverse=True,
    )
    if not version_codes:
        return [], "no APK files listed on IzzyOnDroid for this package"

    return [
        IZZY_REPO_APK.format(package_name=package_name, version_code=code)
        for code in version_codes
    ], ""


def _fetch_text_with_retries(url: str, timeout_sec: int) -> tuple[str | None, str]:
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                timeout=timeout_sec,
                headers=REQUEST_HEADERS,
            )
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt < 3:
                time.sleep(attempt)

    if response is None:
        return None, f"request failed: {last_exc}"

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    return response.text, ""


def _extract_apkmirror_release_links(html_text: str) -> list[str]:
    """Extracts APKMirror release page links from search HTML."""
    links = re.findall(
        r'href="(/apk/[^\"#?]+-release/)"',
        html_text,
        flags=re.IGNORECASE,
    )
    normalized: list[str] = []
    seen: set[str] = set()
    for link in links:
        url = urljoin(APKMIRROR_BASE_URL, html.unescape(link)).split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        normalized.append(url)
    return normalized


def _extract_apkmirror_download_pages(html_text: str) -> list[str]:
    """Extracts APKMirror intermediate download pages from release HTML."""
    links = re.findall(
        r'href="(/apk/[^\"]+-android-apk-download/)"',
        html_text,
        flags=re.IGNORECASE,
    )
    pages: list[str] = []
    seen: set[str] = set()
    for link in links:
        url = urljoin(APKMIRROR_BASE_URL, html.unescape(link)).split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        pages.append(url)
    return pages


def _is_cloudflare_challenge_page(html_text: str) -> bool:
    lowered = html_text.lower()
    return (
        "just a moment" in lowered
        and "cf_chl" in lowered
        and "enable javascript and cookies to continue" in lowered
    )


def _resolve_apkmirror_direct_urls(
    page_url: str,
    timeout_sec: int,
) -> tuple[list[str], str]:
    """Resolves direct APK file URLs from an APKMirror download page.

    The expected flow is:
    1) *-android-apk-download/ page
    2) download.php?key=... page
    3) direct file host URL ending with .apk
    """
    html_text, err = _fetch_text_with_retries(page_url, timeout_sec)
    if html_text is None:
        return [], f"APKMirror page fetch failed: {err}"

    if _is_cloudflare_challenge_page(html_text):
        return [], (
            "APKMirror blocked automated download-page access (Cloudflare challenge)."
        )

    key_links = re.findall(
        r'href="([^\"]*download\.php\?key=[^\"]+)"',
        html_text,
        flags=re.IGNORECASE,
    )
    if not key_links:
        key_links = re.findall(
            r'href="([^\"]*download\.php\?id=[^\"]+)"',
            html_text,
            flags=re.IGNORECASE,
        )
    if not key_links:
        direct = re.findall(
            r'href="(https?://[^\"]+\.apk(?:\?[^\"]*)?)"',
            html_text,
            flags=re.IGNORECASE,
        )
        if direct:
            return direct, ""
        return [], "Could not locate APKMirror download token"

    direct_urls: list[str] = []
    seen: set[str] = set()
    for link in key_links[:3]:
        key_url = urljoin(APKMIRROR_BASE_URL, html.unescape(link))
        key_html, key_err = _fetch_text_with_retries(key_url, timeout_sec)
        if key_html is None:
            continue
        if _is_cloudflare_challenge_page(key_html):
            return [], (
                "APKMirror blocked automated download token resolution "
                "(Cloudflare challenge)."
            )
        for direct in re.findall(
            r'href="(https?://[^\"]+\.apk(?:\?[^\"]*)?)"',
            key_html,
            flags=re.IGNORECASE,
        ):
            if direct in seen:
                continue
            seen.add(direct)
            direct_urls.append(direct)
        if direct_urls:
            break

    if not direct_urls:
        return [], "Could not resolve direct APK from APKMirror token page"

    return direct_urls, ""


def _resolve_apkmirror_urls(
    package_name: str,
    display_name: str,
    timeout_sec: int,
) -> tuple[list[str], str]:
    """Resolves APKMirror candidate URLs for a package/app.

    Returns a prioritized list of URLs. These may be direct `.apk` URLs or
    APKMirror intermediate download pages that are further resolved during
    `_download_file`.
    """
    queries: list[str] = []
    if package_name:
        queries.append(package_name)
    if display_name:
        queries.append(display_name)

    release_links: list[str] = []
    seen_release: set[str] = set()
    errors: list[str] = []

    for query in queries:
        search_url = APKMIRROR_SEARCH_URL.format(query=quote_plus(query))
        search_html, search_err = _fetch_text_with_retries(search_url, timeout_sec)
        if search_html is None:
            errors.append(f"query={query!r}: {search_err}")
            continue
        for release_url in _extract_apkmirror_release_links(search_html):
            if release_url in seen_release:
                continue
            seen_release.add(release_url)
            release_links.append(release_url)

    if not release_links:
        reason = "; ".join(errors) if errors else "No release links found"
        return [], reason

    candidate_download_pages: list[str] = []
    seen_pages: set[str] = set()
    per_release_limit = 2
    release_limit = 8

    for release_url in release_links[:release_limit]:
        rel_html, rel_err = _fetch_text_with_retries(release_url, timeout_sec)
        if rel_html is None:
            continue
        pages = _extract_apkmirror_download_pages(rel_html)
        for page_url in pages[:per_release_limit]:
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            candidate_download_pages.append(page_url)

    if not candidate_download_pages:
        return [], "Release pages found but no APK download pages extracted"

    return candidate_download_pages, ""


def _resolve_apkcombo_direct_urls(
    page_url: str,
    timeout_sec: int,
) -> tuple[list[str], str]:
    """Resolves APKCombo's signed `/r2?u=...` download redirects."""
    html_text, err = _fetch_text_with_retries(page_url, timeout_sec)
    if html_text is None:
        return [], f"APKCombo page fetch failed: {err}"

    candidates: list[str] = []
    seen: set[str] = set()
    for href in re.findall(r'href="([^"]+)"', html_text, flags=re.IGNORECASE):
        lowered = href.lower()
        if "/r2?u=" not in href:
            continue
        if ".apk" not in lowered and ".xapk" not in lowered and ".apks" not in lowered:
            continue
        url = urljoin(APKCOMBO_BASE_URL, html.unescape(href))
        if url in seen:
            continue
        seen.add(url)
        candidates.append(url)

    if not candidates:
        return [], "Could not locate APKCombo signed download link"

    # Prefer single APKs when available, otherwise return the first split bundle.
    candidates.sort(
        key=lambda url: (".xapk" in url.lower() or ".apks" in url.lower(), url)
    )
    return candidates, ""


def _is_no_matching_abis_error(output: str) -> bool:
    return "INSTALL_FAILED_NO_MATCHING_ABIS" in output


def _pick_target_name(app: AppRow, resolved_url: str) -> str:
    if app.apk_filename:
        return app.apk_filename
    parsed = urlparse(resolved_url)
    return os.path.basename(parsed.path)


def _download_file(url: str, target: pathlib.Path, timeout_sec: int, force: bool) -> str:
    if target.exists() and not force:
        return "exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            download_url = url
            parsed_source = urlparse(url)
            if "apkmirror.com" in parsed_source.netloc and not download_url.lower().endswith(".apk"):
                direct_urls, resolve_err = _resolve_apkmirror_direct_urls(download_url, timeout_sec)
                if not direct_urls:
                    raise RuntimeError(f"APKMirror direct resolution failed: {resolve_err}")
                download_url = direct_urls[0]
            if "apkcombo.com" in parsed_source.netloc and "/r2" not in parsed_source.path:
                direct_urls, resolve_err = _resolve_apkcombo_direct_urls(download_url, timeout_sec)
                if not direct_urls:
                    raise RuntimeError(f"APKCombo direct resolution failed: {resolve_err}")
                download_url = direct_urls[0]

            with requests.get(
                download_url,
                stream=True,
                timeout=timeout_sec,
                headers=REQUEST_HEADERS,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "").lower()
                final_url = str(response.url)
                if "text/html" in content_type and not final_url.lower().endswith((".apk", ".xapk", ".apks")):
                    raise RuntimeError(
                        "URL resolved to an HTML page, not a direct APK/bundle "
                        "file. The host may be blocking automated access; retry "
                        "later or provide a local APK in --local-apk-dir."
                    )
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temp.replace(target)
            return "downloaded"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if temp.exists():
                temp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt)

    raise RuntimeError(f"failed after retries: {last_exc}")


def _is_installed(adb_bin: str, serial: str | None, package_name: str) -> bool:
    cmd = [*_adb_base(adb_bin, serial), "shell", "pm", "path", package_name]
    result = _run(cmd)
    return result.returncode == 0 and "package:" in result.stdout


def _install_apk(adb_bin: str, serial: str | None, apk_path: pathlib.Path) -> tuple[bool, str]:
    if apk_path.suffix.lower() in {".xapk", ".apks"}:
        with tempfile.TemporaryDirectory(prefix="catbench_xapk_") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            try:
                with zipfile.ZipFile(apk_path) as archive:
                    apk_members = [
                        member
                        for member in archive.namelist()
                        if member.lower().endswith(".apk")
                    ]
                    if not apk_members:
                        return False, f"No .apk entries found inside {apk_path}"
                    extracted: list[pathlib.Path] = []
                    for member in apk_members:
                        target = tmp_path / pathlib.PurePosixPath(member).name
                        with archive.open(member) as src, target.open("wb") as dst:
                            dst.write(src.read())
                        extracted.append(target)
            except zipfile.BadZipFile as exc:
                return False, f"Invalid APK bundle {apk_path}: {exc}"

            extracted.sort(
                key=lambda path: (
                    path.name != "base.apk" and path.name.startswith("config."),
                    path.name,
                )
            )
            cmd = [
                *_adb_base(adb_bin, serial),
                "install-multiple",
                "-r",
                "-t",
                "-g",
                *(str(path) for path in extracted),
            ]
            result = _run(cmd)
            ok = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
            return ok, output

    cmd = [*_adb_base(adb_bin, serial), "install", "-r", "-t", "-g", str(apk_path)]
    result = _run(cmd)
    ok = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    return ok, output


def _grant_permissions(adb_bin: str, serial: str | None, package_name: str) -> None:
    for permission in COMMON_PERMISSIONS:
        cmd = [
            *_adb_base(adb_bin, serial),
            "shell",
            "pm",
            "grant",
            package_name,
            permission,
        ]
        _run(cmd)


def _iter_selected_apps(rows: Iterable[AppRow], include_optional: bool) -> Iterable[AppRow]:
    for row in rows:
        if row.optional and not include_optional:
            continue
        yield row


def main() -> int:
    script_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Bootstrap app-generalization APKs and permissions.")
    parser.add_argument(
        "--catalog-file",
        default=str(script_dir / "app_generalization_apps.csv"),
        help="Path to app generalization CSV catalog.",
    )
    parser.add_argument(
        "--output-dir",
        default="$HOME/anyappbench_apks",
        help="Directory to store downloaded APKs.",
    )
    parser.add_argument(
        "--local-apk-dir",
        default="",
        help=(
            "Optional local directory of pre-downloaded APK files. "
            "Checked before F-Droid fallback."
        ),
    )
    parser.add_argument("--include-optional", action="store_true", help="Include optional apps.")
    parser.add_argument("--force-download", action="store_true", help="Redownload APK even if it exists.")
    parser.add_argument("--adb", default="adb", help="ADB executable path.")
    parser.add_argument("--serial", default="", help="ADB device serial (optional).")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step.")
    parser.add_argument("--skip-install", action="store_true", help="Skip install step.")
    parser.add_argument("--skip-grant", action="store_true", help="Skip permission grant step.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    args = parser.parse_args()

    needs_adb = not args.skip_install or not args.skip_grant
    selected_serial = args.serial.strip()
    if needs_adb and not selected_serial:
        selected_serial, detect_error = _detect_serial(args.adb)
        if detect_error:
            print(f"ADB device selection error: {detect_error}")
            return 2

    catalog_path = pathlib.Path(args.catalog_file).expanduser().resolve()
    if not catalog_path.exists():
        print(f"Catalog file not found: {catalog_path}")
        return 2

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if not args.skip_download:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_catalog(catalog_path)

    considered = 0
    resolved = 0
    downloaded = 0
    installed = 0
    granted = 0
    skipped = 0
    failed = 0
    incompatible_abis = 0

    print("========================================")
    print("Bootstrap App-Generalization Apps")
    print(f"Catalog: {catalog_path}")
    print(f"Output:  {output_dir}")
    if args.local_apk_dir:
        print(f"Local APK dir: {pathlib.Path(args.local_apk_dir).expanduser().resolve()}")
    if needs_adb:
        print(f"ADB:     {args.adb} (serial: {selected_serial})")
    else:
        print("ADB:     not required (--skip-install --skip-grant)")
    print("========================================")

    for app in _iter_selected_apps(rows, args.include_optional):
        considered += 1
        print(f"\n[{app.app_id}] {app.display_name}")

        already_installed = False
        if needs_adb and app.package_name:
            already_installed = _is_installed(args.adb, selected_serial, app.package_name)

        # Preinstalled/system apps may not have an APK source in catalog/F-Droid.
        # If app already exists on device, skip download/install and only grant.
        if already_installed and not args.force_download:
            skipped += 1
            print(f"  [SKIP] Package already installed: {app.package_name}")
            if not args.skip_grant:
                if args.dry_run:
                    print(f"  [DRY ] Grant runtime permissions to {app.package_name}")
                else:
                    _grant_permissions(args.adb, selected_serial, app.package_name)
                    granted += 1
                    print(f"  [GRNT] Permissions granted for {app.package_name}")
            continue

        cached_apk = output_dir / app.apk_filename if app.apk_filename else None
        if cached_apk is not None and cached_apk.exists() and not args.force_download:
            resolved_sources = [str(cached_apk)]
            print(f"  [SKIP] APK already exists {cached_apk.name}")
        else:
            resolved_sources = _collect_preferred_sources(
                app,
                args.local_apk_dir,
                catalog_path.parent,
                timeout_sec=args.timeout,
            )
        if not resolved_sources:
            if app.package_name:
                auto_sources: list[str] = []
                resolve_failures: list[str] = []

                fdroid_sources, fdroid_reason = _resolve_fdroid_urls(
                    app.package_name,
                    args.timeout,
                )
                if fdroid_sources:
                    print(f"  [INFO] Resolved via F-Droid: {fdroid_sources[0]}")
                    auto_sources.extend(fdroid_sources)
                else:
                    resolve_failures.append(f"F-Droid: {fdroid_reason}")

                izzy_sources, izzy_reason = _resolve_izzyondroid_urls(
                    app.package_name,
                    args.timeout,
                )
                if izzy_sources:
                    print(f"  [INFO] Resolved via IzzyOnDroid: {izzy_sources[0]}")
                    for source in izzy_sources:
                        if source not in auto_sources:
                            auto_sources.append(source)
                else:
                    resolve_failures.append(f"IzzyOnDroid: {izzy_reason}")

                # APKMirror is behind Cloudflare and frequently returns 403 to
                # plain HTTP clients; keep it as a last resort, not the primary
                # fallback.
                if not auto_sources:
                    apkmirror_sources, apkmirror_reason = _resolve_apkmirror_urls(
                        app.package_name,
                        app.display_name,
                        args.timeout,
                    )
                    if apkmirror_sources:
                        print(
                            f"  [INFO] Resolved via APKMirror: {apkmirror_sources[0]}"
                        )
                        for source in apkmirror_sources:
                            if source not in auto_sources:
                                auto_sources.append(source)
                    else:
                        resolve_failures.append(f"APKMirror: {apkmirror_reason}")

                resolved_sources = auto_sources
                if not resolved_sources:
                    print(
                        "  [FAIL] Cannot resolve APK URL: "
                        + " | ".join(resolve_failures)
                    )
                    failed += 1
                    continue
            else:
                print("  [FAIL] apk_url and package_name are both empty")
                failed += 1
                continue
        resolved += 1

        install_success = args.skip_install
        install_error = ""
        no_matching_abis = False
        apk_path: pathlib.Path | None = None

        max_candidates = len(resolved_sources)
        if not args.skip_install:
            # Older version codes can sometimes be ABI-compatible where latest is not.
            max_candidates = min(max_candidates, 6)

        for idx, resolved_source in enumerate(resolved_sources[:max_candidates], start=1):
            if _is_http_source(resolved_source):
                target_name = _pick_target_name(app, resolved_source)
                apk_path = output_dir / target_name

                if not args.skip_download:
                    if args.dry_run:
                        print(f"  [DRY ] Download {resolved_source} -> {apk_path}")
                    else:
                        try:
                            state = _download_file(
                                resolved_source,
                                apk_path,
                                timeout_sec=args.timeout,
                                force=args.force_download or idx > 1,
                            )
                            if state == "downloaded":
                                downloaded += 1
                                print(f"  [GET ] Downloaded {apk_path.name}")
                            else:
                                print(f"  [SKIP] APK already exists {apk_path.name}")
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            install_error = f"Download failed: {exc}"
                            if idx < max_candidates:
                                print(
                                    f"  [WARN] {install_error}; "
                                    f"trying next source ({idx + 1}/{max_candidates})"
                                )
                                continue
                            apk_path = None
                            break
                else:
                    if not apk_path.exists() and not args.dry_run:
                        install_error = f"APK missing on disk: {apk_path}"
                        if idx < max_candidates:
                            print(
                                f"  [WARN] {install_error}; "
                                f"trying next source ({idx + 1}/{max_candidates})"
                            )
                            continue
                        apk_path = None
                        break
            else:
                apk_path = pathlib.Path(resolved_source)
                if args.dry_run:
                    print(f"  [DRY ] Use local APK {apk_path}")
                elif not apk_path.exists():
                    install_error = f"Local APK not found: {apk_path}"
                    if idx < max_candidates:
                        print(
                            f"  [WARN] {install_error}; "
                            f"trying next source ({idx + 1}/{max_candidates})"
                        )
                        continue
                    apk_path = None
                    break

            if apk_path is not None and apk_path.suffix.lower() not in {".apk", ".xapk", ".apks"}:
                install_error = (
                    f"Unsupported package format '{apk_path.suffix}'. "
                    "Provide a direct .apk file or split bundle (.xapk/.apks)."
                )
                if idx < max_candidates:
                    print(
                        f"  [WARN] {install_error}; "
                        f"trying next source ({idx + 1}/{max_candidates})"
                    )
                    continue
                apk_path = None
                break

            if args.skip_install:
                install_success = True
                break

            if args.dry_run:
                print(f"  [DRY ] Install {apk_path}")
                install_success = True
                break

            if apk_path is None:
                break

            ok, output = _install_apk(args.adb, selected_serial, apk_path)
            if ok:
                installed += 1
                print(f"  [INST] Installed {apk_path.name}")
                install_success = True
                break

            install_error = output
            if _is_no_matching_abis_error(output):
                no_matching_abis = True
                if idx < max_candidates:
                    print(
                        f"  [WARN] ABI mismatch for candidate {idx}/{max_candidates};"
                        " trying older F-Droid build..."
                    )
                    continue

            break

        if not install_success:
            if no_matching_abis:
                incompatible_abis += 1
                skipped += 1
                print("  [SKIP] No compatible ABI build found for this emulator")
                continue

            print(f"  [FAIL] Install failed: {install_error}")
            failed += 1
            continue

        if not args.skip_grant:
            if not app.package_name:
                print("  [SKIP] Cannot grant permissions: package_name is empty")
                skipped += 1
                continue
            if args.dry_run:
                print(f"  [DRY ] Grant runtime permissions to {app.package_name}")
            else:
                if _is_installed(args.adb, selected_serial, app.package_name):
                    _grant_permissions(args.adb, selected_serial, app.package_name)
                    granted += 1
                    print(f"  [GRNT] Permissions granted for {app.package_name}")
                else:
                    print(f"  [SKIP] Package not installed: {app.package_name}")
                    skipped += 1

    print("\nDone")
    print(f"Considered: {considered}")
    print(f"Resolved:   {resolved}")
    print(f"Downloaded: {downloaded}")
    print(f"Installed:  {installed}")
    print(f"Granted:    {granted}")
    print(f"Incompat ABI skipped: {incompatible_abis}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
